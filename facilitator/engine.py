"""実時間の統合エンジン: マイク → VAD/STT/声の照合 → 議論状態 → エージェント → TTS。

- voice モード: マイク音声で動く。テキスト入力も併用できる(デモの保険)
- text モード: 話者を選んでテキスト入力。STT/照合は使わない
- エコー対策: AI の発話中(+余韻)はマイク音声を無音に置き換えて STT に送り、VAD も止める
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import Settings
from facilitator.agent.agent import FacilitatorAgent
from facilitator.llm.factory import make_llm
from facilitator.logging import TickLogger
from facilitator.models import AI_SPEAKER, UNKNOWN, Action, TickRecord
from facilitator.state import DiscussionState
from facilitator.stt.base import TranscriptEvent, is_noise_text

ECHO_TAIL_SEC = 0.6  # AI の発話終了後もしばらくマイクを無視する(残響対策)
INTRO_MIN_VOICED_SEC = 2.0
INTRO_END_SILENCE_SEC = 2.5
INTRO_TIMEOUT_SEC = 25.0
FORCED_PAUSE_SEC = 0.6  # 強制イベントはこの長さの間で話し始める
FORCED_WAIT_MAX_SEC = 12.0  # これ以上間がなければ、割り込んででも話す


@dataclass
class SessionConfig:
    theme: str
    minutes: float
    names: list[str]
    mode: str = "voice"  # voice | text
    intro: bool = True  # 自己紹介で声を登録する(voice モードのみ)
    tts: bool = True
    use_speaker_info: bool = True


@dataclass
class IntroState:
    index: int = -1
    turns: list[TranscriptEvent] = field(default_factory=list)
    enrolled: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


class Engine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.status = "idle"  # idle | intro | running | finished
        self.cfg: SessionConfig | None = None
        self.state: DiscussionState | None = None
        self.agent: FacilitatorAgent | None = None
        self.partial = ""
        self.level_db = -100.0
        self.ai_speaking = False
        self._ai_mute_until = 0.0
        self._agent_busy = False
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self.intro = IntroState()
        self.events: list[str] = []  # UI に出すシステムメッセージ
        self._mic = self._stt = self._vid = self._buf = self._vad = self._tts = None
        self._q: queue.Queue = queue.Queue()

    # ================= 時計 =================
    def now(self) -> float:
        return time.monotonic() - self._t0

    def _note(self, msg: str) -> None:
        self.events.append(f"[{self.now():6.1f}s] {msg}")
        self.events = self.events[-50:]

    # ================= 開始・終了 =================
    def start(self, cfg: SessionConfig) -> None:
        self.stop()
        self._stop = threading.Event()
        self.cfg = cfg
        s = self.settings
        s.num_participants = len(cfg.names)
        s.use_speaker_info = cfg.use_speaker_info
        self.state = DiscussionState(settings=s, theme=cfg.theme, duration_sec=cfg.minutes * 60)
        for i, name in enumerate(cfg.names):
            # text モードは入力者が話者を選ぶので名前は確定。voice モードは自己紹介で声を登録できた人だけ確定にする
            self.state.set_speaker_name(f"speaker_{i + 1}", name, confirmed=cfg.mode == "text")
        self.agent = FacilitatorAgent(
            self.state, make_llm(s), s, logger=TickLogger(s.log_dir, "live", include_speech=True)
        )
        self.intro = IntroState()
        self.events = []
        self._t0 = time.monotonic()
        self._tts = None
        if cfg.tts:
            if sys.platform == "win32":
                from facilitator.tts.windows_tts import WindowsTTS

                self._tts = WindowsTTS()
            else:
                from facilitator.tts.say_tts import SayTTS

                self._tts = SayTTS()

        if cfg.mode == "voice":
            self._start_audio()
        self.status = "intro" if (cfg.mode == "voice" and cfg.intro) else "running"
        threading.Thread(target=self._agent_loop, daemon=True).start()
        if self.status == "intro":
            threading.Thread(target=self._intro_flow, daemon=True).start()
        self._note(f"セッション開始({cfg.mode} モード)")

    def stop(self) -> None:
        self._stop.set()
        if self._mic:
            self._mic.stop()
            self._mic = None
        if self._stt:
            self._stt.stop()
            self._stt = None
        if self._vid:
            self._vid.clear()  # 声の特徴量を破棄
            self._vid = None
        if self._tts:
            self._tts.stop()
        if self.status in ("intro", "running"):
            self.status = "finished"

    # ================= 音声入力 =================
    def _start_audio(self) -> None:
        from facilitator.audio.mic import Mic
        from facilitator.audio.vad import EnergyVAD
        from facilitator.speaker.voice_id import AudioBuffer, VoiceIdentifier
        from facilitator.stt.assemblyai_stt import AssemblyAIStream

        self._vid = VoiceIdentifier()
        self._buf = AudioBuffer()
        self._vad = EnergyVAD()
        self._stt = AssemblyAIStream(speaker_labels=False, keyterms=list(self.cfg.names) if self.cfg else None)
        self._stt.start(self._on_stt)
        self._q = queue.Queue()
        self._mic = Mic(self._q.put)
        self._mic.start()
        self._t0 = time.monotonic()  # STT のストリーム時刻と時計の原点をそろえる
        threading.Thread(target=self._audio_loop, daemon=True).start()

    def _audio_loop(self) -> None:
        while not self._stop.is_set():
            try:
                block = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            muted = self.ai_speaking or self.now() < self._ai_mute_until
            if muted:
                block = np.zeros_like(block)
                self.state.speaking_now = False
            else:
                self._vad.process(block)
                self.level_db = self._vad.level_db
                self.state.speaking_now = self._vad.speaking
            self._buf.append(block)
            self._stt.send_audio(block.tobytes())

    def _on_stt(self, e: TranscriptEvent) -> None:
        if e.kind == "partial":
            self.partial = e.text
            return
        if e.kind == "error":
            self._note(f"STT エラー: {e.text[:80]}")
            return
        if e.kind != "final" or e.start is None or is_noise_text(e.text):
            return
        self.partial = ""
        if self.status == "intro":
            self.intro.turns.append(e)
            return
        sid, conf = UNKNOWN, 0.0
        if self._vid and self._vid.enrolled:
            r = self._vid.identify(self._buf.segment(e.start - 0.1, e.end + 0.1))
            sid, conf = r.speaker_id, r.confidence
        self.state.add_utterance(sid, e.text, e.start, e.end, conf)

    # ================= 自己紹介(声の登録) =================
    def _intro_flow(self) -> None:
        st, cfg = self.state, self.cfg
        assert st and cfg and self._vid and self._buf
        time.sleep(1.0)
        for i, name in enumerate(cfg.names):
            if self._stop.is_set():
                return
            sid = f"speaker_{i + 1}"
            self.intro.index = i
            self.intro.turns = []
            if i == 0:
                prompt = (
                    "はじめに、お一人ずつ、お名前と一言自己紹介をお願いします。"
                    f"私が声を覚えますので、5秒ほど話してください。では、{name}さんからどうぞ。"
                )
            else:
                prompt = f"ありがとうございます。次は{name}さん、お願いします。"
            self._say(prompt, record=True)
            t_start = self.now()
            while not self._stop.is_set():
                time.sleep(0.2)
                voiced = sum(t.end - t.start for t in self.intro.turns)
                last_end = max((t.end for t in self.intro.turns), default=None)
                quiet = not st.speaking_now and last_end is not None and self.now() - last_end >= INTRO_END_SILENCE_SEC
                if (voiced >= INTRO_MIN_VOICED_SEC and quiet) or self.now() - t_start > INTRO_TIMEOUT_SEC:
                    break
            turns = list(self.intro.turns)
            if turns:
                audio = np.concatenate([self._buf.segment(t.start, t.end) for t in turns])
                self._vid.enroll(sid, audio)
                st.set_speaker_name(sid, name, confirmed=True)
                self.intro.enrolled.append(name)
                for t in turns:
                    st.add_utterance(sid, t.text, t.start, t.end, 1.0)
                self._note(f"{name}さんの声を登録しました({len(audio) / 16000:.1f}秒)")
            else:
                self.intro.failed.append(name)
                self._note(f"{name}さんの自己紹介を取得できませんでした(匿名のまま進行)")
        self.intro.index = -1
        self.status = "running"  # 次のエージェントのサイクルで start_gd が発動する

    # ================= エージェント =================
    def _agent_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(0.25)
            if self.status != "running" or self._agent_busy or self.ai_speaking:
                continue
            if self.now() < self._ai_mute_until:
                continue
            trig = self.agent.should_tick(self.now())
            if trig:
                self._agent_busy = True
                threading.Thread(target=self._run_tick, args=(trig,), daemon=True).start()

    def _run_tick(self, trigger: str) -> None:
        try:
            rec = self.agent.tick(self.now(), trigger, act=False)
            d = rec.decision
            if d.action == Action.no_intervention:
                return
            # LLM の応答を待つ間に状況が変わっていないか、話す直前にもう一度確認する
            now = self.now()
            st = self.state
            forced = rec.gate.startswith("forced")
            if forced:
                # 時間告知・終了は取りやめない。次の短い間を待って話し、それでも間がなければ話す
                deadline = now + FORCED_WAIT_MAX_SEC
                while d.action != Action.start_gd and not self._stop.is_set():
                    now = self.now()
                    if not st.speaking_now and st.silence_sec(now) >= FORCED_PAUSE_SEC:
                        break
                    if now >= deadline:
                        self._note(f"{d.action.value}: 間がないため発話を優先")
                        break
                    time.sleep(0.1)
                self._speak_decision(rec)
                if d.action == Action.close_gd:
                    self.status = "finished"
                    self._note("GD を終了しました")
                    self.stop()
                return
            if st.is_speaking(now):
                self.agent.cancel(rec, "誰かが話し始めた")
                self._note(f"{d.action.value} を取りやめ(誰かが話している)")
                return
            if st.silence_sec(now) < self.settings.min_silence_for_intervention_sec:
                self.agent.cancel(rec, "沈黙が短い")
                self._note(f"{d.action.value} を取りやめ(会話が続いている)")
                return
            self._speak_decision(rec)
        finally:
            self._agent_busy = False

    def _speak_decision(self, rec: TickRecord) -> None:
        text = rec.decision.speech
        est = len(text) / self.settings.tts_chars_per_sec
        self.agent.commit(rec, self.now(), est)
        self._say(text, record=False)

    def _say(self, text: str, record: bool = True) -> None:
        """AI の発話。発話中はマイクをミュートする。record=True なら議論ログにも追加する。"""
        if record:
            now = self.now()
            self.state.add_utterance(AI_SPEAKER, text, now, now + len(text) / 5.6, 1.0, source="ai")
        self.ai_speaking = True
        try:
            spoken = False
            if self._tts:
                try:
                    self._tts.speak(text)
                    spoken = True
                except Exception as e:  # TTS が使えなくても進行は止めない(以降は声なしで続ける)
                    self._note(f"音声出力に失敗したため、声なしで続けます({type(e).__name__})")
                    self._tts = None
            if not spoken:
                time.sleep(min(3.0, len(text) / self.settings.tts_chars_per_sec))
        finally:
            self.ai_speaking = False
            self._ai_mute_until = self.now() + ECHO_TAIL_SEC

    # ================= UI からの操作 =================
    def add_text(self, speaker_id: str, text: str) -> None:
        if self.status not in ("running", "intro") or not text.strip():
            return
        now = self.now()
        dur = max(1.0, len(text) / 6.0)
        conf = 1.0 if speaker_id != UNKNOWN else 0.0
        self.state.add_utterance(speaker_id, text.strip(), now - dur, now, conf)

    def relabel(self, utterance_id: int, speaker_id: str) -> None:
        if self.state:
            self.state.update_speaker(utterance_id, speaker_id, 0.0 if speaker_id == UNKNOWN else 1.0)

    def set_use_speaker_info(self, on: bool) -> None:
        self.settings.use_speaker_info = on

    # ================= UI への状態 =================
    def snapshot(self) -> dict[str, Any]:
        st, cfg, agent = self.state, self.cfg, self.agent
        if not st or not cfg or not agent:
            return {"status": self.status}
        now = self.now()
        names = {f"speaker_{i + 1}": n for i, n in enumerate(cfg.names)}
        utts = []
        for u in st.utterances[-120:]:
            utts.append(
                {
                    "id": u.id,
                    "t": round(st.elapsed(u.start_time) if st.started else u.start_time, 1),
                    "speaker_id": u.speaker_id,
                    "who": "AI" if u.source == "ai" else (names.get(u.speaker_id) or "不明"),
                    "text": u.text,
                    "source": u.source,
                    "conf": round(u.speaker_confidence, 2),
                }
            )
        ticks = []
        for r in agent.history[-40:]:
            ticks.append(
                {
                    "t": round(st.elapsed(r.time), 1),
                    "trigger": r.trigger,
                    "gate": r.gate,
                    "action": r.decision.action.value,
                    "reasoning": r.decision.reasoning,
                    "speech": r.decision.speech,
                    "overridden": r.overridden,
                    "latency": r.latency_sec,
                    "health": r.analysis.health,
                }
            )
        return {
            "status": self.status,
            "llm": agent.llm.name,
            "mode": cfg.mode,
            "theme": cfg.theme,
            "names": names,
            "started": st.started,
            "elapsed": round(st.elapsed(now), 1),
            "remaining": round(st.remaining(now), 1),
            "duration": st.duration_sec,
            "phase": st.phase,
            "analysis": st.analysis.model_dump(),
            "silence": round(st.silence_sec(now), 1),
            "speaking": st.speaking_now,
            "ai_speaking": self.ai_speaking,
            "thinking": self._agent_busy,
            "level_db": round(self.level_db, 1),
            "partial": self.partial,
            "use_speaker_info": self.settings.use_speaker_info,
            "reliability": round(st.speaker_reliability(), 2),
            "stats": [x.model_dump() for x in st.speaker_stats(now)],
            "utterances": utts,
            "ticks": ticks,
            "interventions": len(st.interventions),
            "intro": {
                "current": cfg.names[self.intro.index] if 0 <= self.intro.index < len(cfg.names) else None,
                "enrolled": self.intro.enrolled,
                "failed": self.intro.failed,
            },
            "events": self.events[-8:],
        }
