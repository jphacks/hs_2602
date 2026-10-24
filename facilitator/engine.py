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

from config import CUSTOM_KEYS, PRESETS, Settings, load_settings
from facilitator.agent.agent import FacilitatorAgent
from facilitator.agent.topic_return import detect_topic_return
from facilitator.discussion_types import resolve_session_type
from facilitator.llm.factory import make_llm
from facilitator.logging import TickLogger
from facilitator.llm.base import LLMError
from facilitator.models import AI_SPEAKER, UNKNOWN, Action, TickRecord
from facilitator.participants import (
    LEVELS,
    Persona,
    level_interval,
    bot_mode_for,
    choose_and_generate,
    generate_personas,
    generate_utterance,
    pick_bot,
    transcript_for_bot,
)
from facilitator.report import RecordingLLM, TestReport
from facilitator.state import DiscussionState
from facilitator.stt.base import TranscriptEvent, is_noise_text

ECHO_TAIL_SEC = 0.6  # AI の発話終了後もしばらくマイクを無視する(残響対策)
INTRO_MIN_VOICED_SEC = 2.0
INTRO_END_SILENCE_SEC = 2.5
INTRO_TIMEOUT_SEC = 25.0
FORCED_PAUSE_SEC = 0.6  # 強制イベントはこの長さの間で話し始める
FORCED_WAIT_MAX_SEC = 12.0  # これ以上間がなければ、割り込んででも話す

MAC_BOT_VOICES = ["Eddy (日本語（日本）)", "Shelley (日本語（日本）)", "Reed (日本語（日本）)"]
WIN_BOT_VOICES = ["Microsoft Ichiro Desktop", "Microsoft Ayumi Desktop", "Microsoft Haruka Desktop"]


@dataclass
class SessionConfig:
    theme: str
    minutes: float
    names: list[str]
    mode: str = "voice"  # voice | text
    intro: bool = True  # 自己紹介で声を登録する(voice モードのみ)
    tts: bool = True
    use_speaker_info: bool = True
    purpose: str = "job"  # job(就活) | general(就活以外)
    discussion_type: str | None = None  # 議論タイプ。就活GDは必須(4種類から選択)、一般GWは省略時に汎用型
    preset: str = "A"  # A | B | custom
    custom: dict[str, float] = field(default_factory=dict)
    ai_count: int = 0  # AI 参加者の人数(names は人間の参加者のみ)
    ai_level: int | None = 2  # AI 参加者のレベル(1〜3)。思考力と発言の間隔が変わる。None なら従来の間隔


@dataclass
class IntroState:
    index: int = -1
    turns: list[TranscriptEvent] = field(default_factory=list)
    enrolled: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


class Engine:
    def __init__(self, settings: Settings, clock: Any = None, **base_overrides: Any) -> None:
        self.settings = settings
        self.clock = clock or time  # monotonic() / sleep() を持つ時計。テストでは疑似時計に差し替える
        self._base_overrides = base_overrides  # 起動時に指定された設定(セッションごとに引き継ぐ)
        self.paused = False
        self._ended_at: float | None = None  # 終了した時刻(終了後は時計を止める)
        self.status = "idle"  # idle | intro | running | finished
        self.cfg: SessionConfig | None = None
        self.state: DiscussionState | None = None
        self.agent: FacilitatorAgent | None = None
        self.partial = ""
        self.level_db = -100.0
        self.ai_speaking = False
        self._ai_mute_until = 0.0
        self._voice_end: float | None = None  # VAD で声が止まった時刻(音声モードの区切り判定に使う)
        self._agent_busy = False
        self.redirect_log: list[dict[str, Any]] = []  # 軌道修正の結果(話した / 取り消し)の計測
        self.skipped_sec: dict[str, float] = {}  # 定期分析の時刻なのに見送った秒数(理由別)の計測
        self._stop = threading.Event()
        self._t0 = self.clock.monotonic()
        self.intro = IntroState()
        self.events: list[str] = []  # UI に出すシステムメッセージ
        self._mic = self._stt = self._vid = self._buf = self._vad = self._tts = None
        self._q: queue.Queue = queue.Queue()
        # AI 参加者
        self.personas: list[Persona] = []
        self.bot_mode = ""  # "" | active | reserved
        self._bot_tts: dict[str, Any] = {}
        self._bot_busy = False
        self._bot_attempt_at = -1e9
        self._bot_answered_id = -1  # 名指しに答えた発言 ID(二重に答えない)
        self._last_bot_end = -1e9
        self._speaking_as: str | None = None  # 今声を出している話者(AI_SPEAKER / AI 参加者の ID)
        self._speak_lock = threading.Lock()
        self.ai_level: int | None = None
        self._interval: tuple[float, float] | None = None
        self._threshold_cache: dict[int, float] = {}
        # まとめログ(セッション終了時に logs/report-*.md へ書き出す)
        self.report: TestReport | None = None
        self._sec = None
        self._report_written = True

    # ================= 時計 =================
    def now(self) -> float:
        return self.clock.monotonic() - self._t0

    def _note(self, msg: str) -> None:
        self.events.append(f"[{self.now():6.1f}s] {msg}")
        self.events = self.events[-50:]

    # ================= 開始・終了 =================
    def start(self, cfg: SessionConfig, background: bool = True) -> None:
        """background=False ならスレッドを起動しない(テストで _agent_step を直接回すため)。"""
        dtype = resolve_session_type(cfg.discussion_type, cfg.purpose)  # 不正なら、走っているセッションを止める前に拒否する
        self.stop()
        self._stop = threading.Event()
        self.cfg = cfg
        overrides: dict[str, Any] = dict(PRESETS.get(cfg.preset, {}))
        if cfg.preset == "custom":
            overrides.update({k: float(v) for k, v in cfg.custom.items() if k in CUSTOM_KEYS})
        self.settings = s = load_settings(**{**self._base_overrides, **overrides})
        self.paused = False
        self._ended_at = None
        humans, bots = len(cfg.names), max(0, min(3, cfg.ai_count))
        s.num_participants = humans + bots
        s.use_speaker_info = cfg.use_speaker_info
        self.state = DiscussionState(settings=s, theme=cfg.theme, duration_sec=cfg.minutes * 60)
        for i, name in enumerate(cfg.names):
            # text モードは入力者が話者を選ぶので名前は確定。voice モードは自己紹介で声を登録できた人だけ確定にする
            self.state.set_speaker_name(f"speaker_{i + 1}", name, confirmed=cfg.mode == "text")
        llm = RecordingLLM(make_llm(s))  # 入力と返事をまとめログに残す
        self.report = TestReport("アプリ(app/server.py)", "live", {"LLM": llm.name})
        self._sec = sec = self.report.section(cfg.theme)
        self._report_written = False
        m = llm.mark()
        # AI 参加者: 人数分の異なる立場を生成して割り当てる
        self.ai_level = cfg.ai_level if cfg.ai_level in LEVELS else None
        self._interval = level_interval(self.ai_level)
        self._threshold_cache: dict[int, float] = {}
        self.personas = generate_personas(
            llm, cfg.theme, cfg.purpose, bots, list(cfg.names), humans + 1, self.ai_level
        )
        for p in self.personas:
            self.state.set_speaker_name(p.speaker_id, p.name, confirmed=True)
            self.state.bot_ids.add(p.speaker_id)
        self.bot_mode = bot_mode_for(humans, bots)
        sec.state, sec.personas, sec.persona_call = self.state, self.personas, llm.since(m)
        sec.hide_human_text = not s.log_transcript_text  # 実際の人の発言本文は既定で残さない(プライバシー)
        sec.info = {
            "テーマ": cfg.theme,
            "用途": "就活" if cfg.purpose == "job" else "就活以外",
            "入力方法": "音声" if cfg.mode == "voice" else "テキスト",
            "進行スタイル": cfg.preset,
            "人数": f"人間 {humans}人 / AI参加者 {bots}人",
            "AI参加者のモード": {"active": "練習相手モード", "reserved": "控えめモード"}.get(self.bot_mode, "なし"),
            **({"AI参加者のレベル": LEVELS[self.ai_level]["label"]} if bots and self.ai_level else {}),
            "制限時間": f"{cfg.minutes:g}分",
        }
        self._bot_busy, self._bot_attempt_at, self._bot_answered_id, self._last_bot_end = False, -1e9, -1, -1e9
        self.agent = FacilitatorAgent(
            self.state,
            llm,
            s,
            logger=TickLogger(s.log_dir, "live", include_speech=True),
            purpose=cfg.purpose,
            discussion_type=dtype.key,
            clock=self.clock,
        )
        self.intro = IntroState()
        self.events = []
        self._t0 = self.clock.monotonic()
        self._ai_mute_until = 0.0  # 時計を0に戻すので、前のセッションの時計でのミュート時刻を持ち越さない
        self._voice_end = None
        self.redirect_log = []
        self.skipped_sec = {}
        self._tts = None
        self._bot_tts = {}
        if cfg.tts:
            if sys.platform == "win32":
                from facilitator.tts.windows_tts import WindowsTTS

                self._tts = WindowsTTS()
                for i, p in enumerate(self.personas):
                    self._bot_tts[p.speaker_id] = WindowsTTS(voice=WIN_BOT_VOICES[i % len(WIN_BOT_VOICES)])
            else:
                from facilitator.tts.say_tts import SayTTS

                self._tts = SayTTS()
                for i, p in enumerate(self.personas):
                    self._bot_tts[p.speaker_id] = SayTTS(voice=MAC_BOT_VOICES[i % len(MAC_BOT_VOICES)])

        if cfg.mode == "voice":
            self._start_audio()
        self.status = "intro" if (cfg.mode == "voice" and cfg.intro) else "running"
        if background:
            threading.Thread(target=self._agent_loop, daemon=True).start()
        if self.personas:
            if background:
                threading.Thread(target=self._bot_loop, daemon=True).start()
            self._note(
                "AI参加者: " + "、".join(f"{p.name}({p.stance})" for p in self.personas)
                + f" / {'練習相手モード' if self.bot_mode == 'active' else '控えめモード'}"
            )
        if self.status == "intro" and background:
            threading.Thread(target=self._intro_flow, daemon=True).start()
        self._note(f"セッション開始({cfg.mode} モード)")

    def stop(self) -> None:
        self._stop.set()
        if self.report and not self._report_written and self.state:
            self._report_written = True
            try:
                path = self.report.write(self.settings.log_dir)
                self._note(f"まとめログ: {path.name}")
            except OSError as e:
                self._note(f"まとめログを書き出せませんでした({type(e).__name__})")
        if self.state and self._ended_at is None:
            self._ended_at = self.now()
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
        for t in self._bot_tts.values():
            t.stop()
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
        self._t0 = self.clock.monotonic()  # STT のストリーム時刻と時計の原点をそろえる
        threading.Thread(target=self._audio_loop, daemon=True).start()

    def _audio_loop(self) -> None:
        while not self._stop.is_set():
            try:
                block = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            muted = self.paused or self.ai_speaking or self.now() < self._ai_mute_until
            if muted:
                block = np.zeros_like(block)
                self.state.speaking_now = False
            else:
                if self._vad.process(block) == "end":
                    # VAD は無音が hangover 続いてから終了と判定するので、その分を引いた時刻を声の終わりとする
                    self._voice_end = self.now() - self._vad.hang_blocks * self._vad.block_ms / 1000
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
        if e.kind != "final" or e.start is None or is_noise_text(e.text) or self.paused:
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
        self.clock.sleep(1.0)
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
                self.clock.sleep(0.2)
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
            self.clock.sleep(0.25)
            trig = self._agent_step()
            if trig:
                threading.Thread(target=self._run_tick, args=(trig,), daemon=True).start()

    def _agent_step(self) -> str | None:
        """判断ループの1回分。判断を始めるならトリガー名を返し、_agent_busy を立てる(実行は呼び出し側)。"""
        if self.status != "running" or self.paused or self._agent_busy or self.ai_speaking:
            return None
        if self.now() < self._ai_mute_until:
            return None
        trig = self.agent.should_tick(self.now())
        if self.agent.skipped_due:
            r = self.agent.skipped_due
            self.skipped_sec[r] = self.skipped_sec.get(r, 0.0) + 0.25
        if trig:
            self._agent_busy = True
        return trig

    def _run_tick(self, trigger: str) -> None:
        try:
            m = self.agent.llm.mark()
            rec = self.agent.tick(self.now(), trigger, act=False)
            if self._sec is not None:
                self._sec.decision(rec, self.agent.llm.since(m))  # 取りやめた場合も rec に追記されて残る
            d = rec.decision
            if d.action == Action.no_intervention:
                return
            # LLM の応答を待つ間に状況が変わっていないか、話す直前にもう一度確認する
            if self.paused:
                self.agent.cancel(rec, "一時停止中")
                return
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
                    self.clock.sleep(0.1)
                self._speak_decision(rec)
                if d.action == Action.close_gd:
                    self.status = "finished"
                    self._note("GD を終了しました")
                    self.stop()
                return
            if d.action == Action.redirect_topic:
                self._await_redirect(rec)
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

    def _quiet_sec(self, now: float) -> float:
        """いまの静けさの長さ。発話中は0。音声モードは VAD で声が止まった時刻も見る(STT の確定は遅れるため)。"""
        st = self.state
        if st.speaking_now:
            return 0.0
        quiet = st.silence_sec(now)
        if self._voice_end is not None:
            quiet = min(quiet, now - self._voice_end)
        return quiet

    def _await_redirect(self, rec: TickRecord) -> None:
        """脱線の軌道修正: 判断後、短い発話の区切り(redirect_pause_sec)を待って話す。
        発話中には話さず、区切りがないまま redirect_wait_max_sec たつか、本題に戻ったら取り消す。"""
        st, s = self.state, self.settings
        decided = self.now()
        cut = rec.context_last_utt_id

        def human(after: bool) -> list[str]:
            return [u.text for u in st.utterances if u.source == "human" and (cut is None or (u.id > cut) == after)]

        drift = human(after=False)[-4:]  # 判断に使った直近の発言(脱線部分)。この語は本題の語から除く
        while True:
            now = self.now()
            new = human(after=True)
            reason = None
            if self._stop.is_set() or self.status != "running":
                reason = "セッション終了"
            elif self.paused:
                reason = "一時停止"
            elif self.agent.forced_event_pending(now):
                reason = "強制イベントを優先"
            elif back := detect_topic_return(st.theme, rec.analysis, drift, new):
                reason = f"本題に戻った({back})"
            elif self._quiet_sec(now) >= s.redirect_pause_sec:
                self._log_redirect(rec, "spoken", None, now - decided, len(new))
                self._speak_decision(rec)
                return
            elif now - decided >= s.redirect_wait_max_sec:
                reason = "区切りなし"
            if reason:
                self.agent.cancel(rec, reason)
                self._note(f"redirect_topic を取りやめ({reason})")
                self._log_redirect(rec, "canceled", reason, now - decided, len(new))
                return
            self.clock.sleep(0.1)

    def _log_redirect(self, rec: TickRecord, outcome: str, reason: str | None, wait: float, new: int) -> None:
        entry = {"time": round(self.state.elapsed(rec.time), 1), "outcome": outcome, "reason": reason,
                 "wait_sec": round(wait, 2), "new_utterances": new}
        self.redirect_log.append(entry)
        if self.agent.logger:
            self.agent.logger.log_event({"redirect_topic": entry})

    def metrics(self) -> dict[str, Any]:
        """計測値: LLM の呼び出し・トークン・応答時間、軌道修正の結果、分析を見送った秒数。"""
        hist = self.agent.history if self.agent else []
        calls = [r for r in hist if r.llm_called]
        tokens: dict[str, int] = {}
        for r in calls:
            for k, v in (r.llm_usage or {}).items():
                tokens[k] = tokens.get(k, 0) + v
        canceled: dict[str, int] = {}
        for e in self.redirect_log:
            if e["outcome"] == "canceled":
                key = e["reason"].split("(")[0]
                canceled[key] = canceled.get(key, 0) + 1
        # 判断の出どころ: ルール層の見送り / LLM の見守り / LLM の介入(話した・後段で取りやめ・検証で書き換え)
        decisions = {"rule": 0, "llm_no_intervention": 0, "llm_intervention_spoken": 0, "llm_intervention_canceled": 0, "llm_overridden": 0}
        for r in hist:
            if not r.llm_called:
                decisions["rule"] += 1
            elif r.llm_raw_action is not None and r.llm_raw_action != r.decision.action.value:
                decisions["llm_overridden"] += 1
            elif r.decision.action == Action.no_intervention:
                decisions["llm_no_intervention"] += 1
            elif (r.overridden or "").startswith("発話取りやめ"):
                decisions["llm_intervention_canceled"] += 1
            else:
                decisions["llm_intervention_spoken"] += 1
        return {
            "decisions": decisions,
            "llm_calls": len(calls),
            "llm_tokens": tokens,
            "llm_latency_avg": round(sum(r.latency_sec for r in calls) / len(calls), 2) if calls else None,
            "redirect_chosen": sum(1 for r in hist if r.decision.action == Action.redirect_topic),
            "redirect_spoken": sum(1 for e in self.redirect_log if e["outcome"] == "spoken"),
            "redirect_canceled": canceled,
            "redirect_wait_sec": [e["wait_sec"] for e in self.redirect_log if e["outcome"] == "spoken"],
            "analysis_skipped_sec": {k: round(v, 2) for k, v in self.skipped_sec.items()},
        }

    def _speak_decision(self, rec: TickRecord) -> None:
        text = rec.decision.speech
        est = len(text) / self.settings.tts_chars_per_sec
        self.agent.commit(rec, self.now(), est)
        utt = next((u for u in reversed(self.state.utterances) if u.source == "ai"), None)
        self._say(text, record=False, utt=utt)

    def _say(self, text: str, record: bool = True, speaker: str = AI_SPEAKER, utt: Any = None) -> Any:
        """AI(進行役 / AI 参加者)の発話。発話中はマイクをミュートする。record=True なら議論ログにも追加する。
        複数の AI が同時に話さないよう、発話はロックで1つずつにする。"""
        is_bot = speaker != AI_SPEAKER
        tts = self._bot_tts.get(speaker) if is_bot else self._tts
        with self._speak_lock:
            if record:
                now = self.now()
                dur = len(text) / 5.6 if tts else min(3.0, len(text) / self.settings.tts_chars_per_sec)
                utt = self.state.add_utterance(speaker, text, now, now + dur, 1.0, source="bot" if is_bot else "ai")
            self.ai_speaking = True
            self._speaking_as = speaker
            try:
                spoken = False
                if tts:
                    try:
                        tts.speak(text)
                        spoken = True
                    except Exception as e:  # TTS が使えなくても進行は止めない(以降は声なしで続ける)
                        self._note(f"音声出力に失敗したため、声なしで続けます({type(e).__name__})")
                        if is_bot:
                            self._bot_tts.pop(speaker, None)
                        else:
                            self._tts = None
                if not spoken:
                    self.clock.sleep(min(3.0, len(text) / self.settings.tts_chars_per_sec))
            finally:
                if utt is not None:
                    utt.end_time = max(utt.start_time, self.now())  # 実際に読み上げが終わった時刻を記録
                self.ai_speaking = False
                self._speaking_as = None
                self._ai_mute_until = self.now() + ECHO_TAIL_SEC
        return utt

    # ================= AI 参加者 =================
    def _bot_loop(self) -> None:
        while not self._stop.is_set():
            self.clock.sleep(0.3)
            if self.status != "running" or self.paused or self._bot_busy or self.ai_speaking:
                continue
            now = self.now()
            if now < self._ai_mute_until or now - self._bot_attempt_at < 3.0:
                continue
            picked = self._bot_pick(now)
            if picked:
                persona, reason, addressed_id = picked
                self._bot_busy = True
                self._bot_attempt_at = now
                threading.Thread(target=self._bot_turn, args=(persona, reason, addressed_id), daemon=True).start()

    def _bot_pick(self, now: float) -> tuple[Persona, str, int] | None:
        return pick_bot(
            self.state, self.personas, self.bot_mode, now, self._last_bot_end, self._bot_answered_id, len(self.cfg.names),
            self._interval, self._threshold_cache,
        )

    def _bot_turn(self, p: Persona, reason: str, addressed_id: int) -> None:
        st = self.state
        try:
            last_id = st.utterances[-1].id if st.utterances else -1
            transcript = transcript_for_bot(st)
            m = self.agent.llm.mark()
            try:
                if addressed_id >= 0:  # 名指しされた人が答える
                    text = generate_utterance(
                        self.agent.llm, p, self.cfg.purpose, st.theme, transcript, reason, self.ai_level
                    )
                else:  # 誰が話すかは文脈で選ぶ(直前の話者は除く)
                    last_sid = st.utterances[-1].speaker_id if st.utterances else ""
                    cands = [x for x in self.personas if x.speaker_id != last_sid] or self.personas
                    counts = {x.speaker_id: x.spoken for x in self.personas}
                    p, text, why = choose_and_generate(
                        self.agent.llm, cands, self.cfg.purpose, st.theme, transcript, reason, self.ai_level, counts, p
                    )
                    if why:
                        reason += f" / {p.name}が話した理由: {why}"
            except LLMError as e:
                self._note(f"{p.name}の発言を生成できませんでした({type(e).__name__})")
                return
            if not text or self._stop.is_set() or self.paused or self.status != "running":
                return
            # 生成を待つ間に人間が話し始めた/新しい発言があった場合は取りやめる(名指しへの応答は除く)
            newer = st.utterances and st.utterances[-1].id != last_id
            if st.speaking_now or (newer and addressed_id < 0):
                return
            if addressed_id >= 0:
                self._bot_answered_id = addressed_id
            p.spoken += 1
            call = self.agent.llm.since(m)
            utt = self._say(text, speaker=p.speaker_id)
            if utt is not None and self._sec is not None:
                self._sec.bot(p.speaker_id, utt.start_time, reason, call)
            self._last_bot_end = self.now()
        finally:
            self._bot_busy = False

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

    def pause(self) -> None:
        if self.status == "running" and self.state and not self.paused:
            self.paused = True
            self.state.pause(self.now())
            self.state.speaking_now = False
            self._note("一時停止")

    def resume(self) -> None:
        if self.paused and self.state:
            self.paused = False
            self.state.resume(self.now())
            self._note("再開")

    def set_use_speaker_info(self, on: bool) -> None:
        self.settings.use_speaker_info = on

    # ================= UI への状態 =================
    def _active_speaker(self, now: float) -> str | None:
        """直近(3秒以内)に発言を終えた人。発話中の人は確定するまで分からないため、直近の話者で近似する。"""
        if self.ai_speaking and self._speaking_as:
            return self._speaking_as
        for u in reversed(self.state.utterances):
            if u.source == "ai":
                return None
            if now - u.end_time <= 3.0:
                return u.speaker_id
            return None
        return None

    def snapshot(self) -> dict[str, Any]:
        st, cfg, agent = self.state, self.cfg, self.agent
        if not st or not cfg or not agent:
            return {"status": self.status}
        now = self._ended_at if self._ended_at is not None else self.now()
        names = {f"speaker_{i + 1}": n for i, n in enumerate(cfg.names)}
        names.update({p.speaker_id: p.name for p in self.personas})
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
            "paused": self.paused,
            "purpose": cfg.purpose,
            "preset": cfg.preset,
            "active_speaker": self._active_speaker(now),
            "llm": agent.llm.name,
            "mode": cfg.mode,
            "theme": cfg.theme,
            "names": names,
            "humans": [f"speaker_{i + 1}" for i in range(len(cfg.names))],
            "participants": [
                {"id": p.speaker_id, "name": p.name, "stance": p.stance, "opinion": p.opinion} for p in self.personas
            ],
            "bot_mode": self.bot_mode,
            "ai_level": self.ai_level if self.personas else None,
            "started": st.started,
            "elapsed": round(st.elapsed(now), 1),
            "remaining": round(st.remaining(now), 1),
            "duration": st.duration_sec,
            "discussion_type": agent.dtype.label,
            "discussion_type_key": agent.dtype.key,
            "phases": [[p.key, p.label] for p in agent.dtype.phases],
            "phase": st.observed_phase,  # 実際のフェーズ(LLM の判断)
            "target_phase": agent.phase_status(now)[0] if st.started else "",
            "analysis": st.analysis.model_dump(),
            "metrics": self.metrics(),
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
