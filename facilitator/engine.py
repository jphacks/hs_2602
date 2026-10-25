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

from config import POLICY_PRESET_LABELS, PURPOSE_PRESET, Settings, load_settings
from facilitator.agent.agent import FacilitatorAgent
from facilitator.agent.topic_return import detect_topic_return
from facilitator.discussion_types import resolve_session_type
from facilitator.feedback import build_feedback
from facilitator.llm.factory import make_llm
from facilitator.logging import TickLogger
from facilitator.llm.base import LLMError, supports_stream
from facilitator.llm.stream import SentenceFeeder, partial_string
from facilitator.models import AI_SPEAKER, UNKNOWN, Action, TickRecord
from facilitator.policy import InterventionPolicy, from_knobs, preset_policy, request_policy
from facilitator.participants import (
    LEVELS,
    Persona,
    level_interval,
    bot_mode_for,
    choose_and_generate,
    discussion_flow,
    generate_personas,
    generate_utterance,
    pick_bot,
    transcript_for_bot,
)
from facilitator.report import RecordingLLM, TestReport
from facilitator.speaker.split import fill_labels, per_sentence, sentence_spans, split_sentences
from facilitator.state import DiscussionState
from facilitator.stt.base import TranscriptEvent, is_noise_text

ECHO_TAIL_SEC = 0.6  # AI の発話終了後もしばらくマイクを無視する(残響対策)
INTRO_MIN_VOICED_SEC = 2.0
INTRO_END_SILENCE_SEC = 2.5
INTRO_TIMEOUT_SEC = 25.0
FORCED_PAUSE_SEC = 0.6  # 強制イベントはこの長さの間で話し始める
FORCED_WAIT_MAX_SEC = 12.0  # これ以上間がなければ、割り込んででも話す
# 生成しながら話してよい進行役の行動(個人指名・軌道修正は検証や待ち合わせで内容が変わりうるので、全文を待ってから話す)
NO_LIVE_ACTIONS = {Action.no_intervention, Action.invite_person, Action.redirect_topic,
                   Action.start_gd, Action.time_warning, Action.close_gd}
ECHO_SIMILARITY = 0.45  # AI の発話中の文字起こしが、AI の発言とこれ以上似ていれば AI 自身の声(回り込み)とみなす

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
    preset: str = "A"  # 旧進行スタイル A | B | custom(互換のため受け付ける。B は「⑤停滞への介入: 積極的」)
    custom: dict[str, float] = field(default_factory=dict)  # 旧カスタムの数値(廃止。受け付けるが使わない)
    ai_count: int = 0  # AI 参加者の人数(names は人間の参加者のみ)
    ai_level: int | None = 2  # AI 参加者のレベル(1〜3)。思考力と発言の間隔が変わる。None なら従来の間隔
    knobs: dict[str, int] = field(default_factory=dict)  # 詳細の5項目(0=控えめ / 1=標準 / 2=積極的)
    policy: InterventionPolicy | None = None  # 介入ポリシー。None ならプリセットと旧設定(preset / knobs)から作る(knobs は上に重ねる)
    policy_preset: str | None = None  # 介入ポリシーのプリセット(job_gd / general_gw)。None なら用途の既定


def _bigrams(text: str) -> set[str]:
    t = "".join(ch for ch in text if ch.isalnum())
    return {t[i : i + 2] for i in range(len(t) - 1)} or ({t} if t else set())


def _overlap(heard: str, said: str) -> float:
    """聞き取った文のうち、AI が話した文にも含まれる2文字の組の割合(0〜1)。"""
    h = _bigrams(heard)
    return len(h & _bigrams(said)) / len(h) if h else 0.0


class LiveSpeech:
    """生成中の AI の発言を、一文できるたびに議論ログへ出して読み上げる(進行役・AI 参加者で共通)。
    最初の一文ができた時点で話し始めてよいか(can_start)を確かめ、だめならこの発言は全文を待つ従来の流れに任せる。"""

    def __init__(self, engine: "Engine", gen: int, can_start: Any, max_sentences: int | None = None) -> None:
        self.engine, self.gen, self.can_start, self.max = engine, gen, can_start, max_sentences
        self.speaker: str | None = None
        self.persona: Persona | None = None
        self.action: Action | None = None
        self.q: queue.Queue[str | None] = queue.Queue()
        self.started = False
        self.declined = False
        self.count = 0
        self.utt: Any = None
        self.spoken: list[str] = []
        self.done = threading.Event()

    def push(self, sentence: str) -> None:
        if self.declined or (self.max and self.count >= self.max):
            return
        if not self.started:
            if self.speaker is None or not self.can_start():
                self.declined = True
                return
            self.started = True
            threading.Thread(target=self.engine._speak_live, args=(self,), daemon=True).start()
        self.count += 1
        self.q.put(sentence)

    def close(self) -> None:
        if self.started:
            self.q.put(None)

    def wait(self) -> None:
        if self.started:
            self.done.wait()

    @property
    def text(self) -> str:
        return "".join(self.spoken)


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
        self.policy: InterventionPolicy | None = None
        self.policy_preset = PURPOSE_PRESET["job"]
        self.partial = ""
        self.level_db = -100.0
        self.ai_speaking = False
        self._ai_mute_until = 0.0
        self.partial_speaker: str | None = None  # 認識中の発言の話者の推定
        self.partial_parts: list[dict[str, Any]] = []  # 認識中の発言を「。」で分けたもの [{text, speaker}]
        self._partial_labels: list[tuple[str | None, bool]] = []  # 文ごとの (話者, 書き終わった文の判定か)
        self._partial_turn: int | None = None
        self._partial_id_at = 0.0
        self._voice_end: float | None = None  # VAD で声が止まった時刻(音声モードの区切り判定に使う)
        self._agent_busy = False
        self.redirect_log: list[dict[str, Any]] = []  # 軌道修正の結果(話した / 取り消し)の計測
        self.skipped_sec: dict[str, float] = {}  # 定期分析の時刻なのに見送った秒数(理由別)の計測
        self._stop = threading.Event()  # 現在のセッションの停止イベント(セッションごとに作り直す)
        self._gen = 0  # セッションの世代番号。古いセッションのスレッドが新しいセッションを変更しないための識別
        self._session_lock = threading.Lock()  # 開始・終了の状態の切り替えだけを守る(時間のかかる処理では持たない)
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
        self._feedback: tuple[int, dict[str, Any]] | None = None  # (セッションの世代, 振り返り)
        self._feedback_lock = threading.Lock()
        self._enroll_stop = threading.Event()  # 声の登録を画面のボタンで終える合図
        self._speak_lock = threading.RLock()  # 発話を1つずつにする(_speak_decision → _say の入れ子で取り直す)
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
        with self._session_lock:
            self._gen += 1
            gen = self._gen
            self._stop = threading.Event()
            self._agent_busy = False  # 前のセッションの判断が残っていても、このセッションの判断は止めない
        self.cfg = cfg
        # 介入ポリシー: 指定がなければ用途と画面の「詳細」から作る。①〜⑥の効果は進行役(FacilitatorAgent)がポリシーから決める
        if cfg.policy is not None:
            policy = from_knobs(cfg.knobs, cfg.policy, cfg.purpose)
        else:
            policy, _ = request_policy(cfg.purpose, cfg.policy_preset, None, cfg.preset, cfg.knobs)
        self.policy = policy
        self.policy_preset = cfg.policy_preset or PURPOSE_PRESET.get(cfg.purpose, PURPOSE_PRESET["job"])
        self.settings = s = load_settings(**self._base_overrides)
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
            "介入プリセット": POLICY_PRESET_LABELS.get(self.policy_preset, self.policy_preset),
            "介入ポリシー": policy.summary(),
            **({"注意": " / ".join(policy.notices(cfg.purpose))} if policy.notices(cfg.purpose) else {}),
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
            policy=policy,
        )
        self.intro = IntroState()
        self.partial, self.partial_speaker, self.partial_parts, self._partial_labels = "", None, [], []
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
            self._start_audio(gen)
        self.status = "intro" if (cfg.mode == "voice" and cfg.intro) else "running"
        if background:
            threading.Thread(target=self._agent_loop, args=(gen,), daemon=True).start()
        if self.personas:
            if background:
                threading.Thread(target=self._bot_loop, args=(gen,), daemon=True).start()
            self._note(
                "AI参加者: " + "、".join(f"{p.name}({p.stance})" for p in self.personas)
                + f" / {'練習相手モード' if self.bot_mode == 'active' else '控えめモード'}"
            )
        self._note(f"セッション開始({cfg.mode} モード)")

    def _alive(self, gen: int) -> bool:
        """gen のセッションがまだ現在のセッションで、止められていないか。"""
        return gen == self._gen and not self._stop.is_set()

    def stop(self, gen: int | None = None) -> None:
        """現在のセッションを終了する。gen を渡すと、そのセッションが現在のものであるときだけ終了する。
        状態の切り替えと機器の取り外しは排他の中で一度に行い、機器の停止やまとめログの書き出しは
        取り外したものに対して排他の外で行う(途中で次のセッションが始まっても、そのセッションを変更しない)。"""
        with self._session_lock:
            if gen is not None and gen != self._gen:
                return
            gen = self._gen
            self._stop.set()
            report = self.report if self.report and not self._report_written and self.state else None
            self._report_written = True
            log_dir = self.settings.log_dir
            if self.state and self._ended_at is None:
                self._ended_at = self.now()
            mic, stt, vid = self._mic, self._stt, self._vid
            self._mic = self._stt = self._vid = None
            tts = [t for t in (self._tts, *self._bot_tts.values()) if t]
            if self.status in ("intro", "running"):
                self.status = "finished"
        if mic:
            mic.stop()
        if stt:
            stt.stop()
        if vid:
            vid.clear()  # 声の特徴量を破棄
        for t in tts:
            t.stop()
        if report:
            try:
                path = report.write(log_dir)
                msg = f"まとめログ: {path.name}"
            except OSError as e:
                msg = f"まとめログを書き出せませんでした({type(e).__name__})"
            if gen == self._gen:  # 次のセッションが始まっていれば、そのセッションのメッセージには出さない
                self._note(msg)

    # ================= 音声入力 =================
    def _start_audio(self, gen: int) -> None:
        from facilitator.audio.mic import Mic
        from facilitator.audio.vad import EnergyVAD
        from facilitator.speaker.voice_id import AudioBuffer, VoiceIdentifier
        from facilitator.stt.assemblyai_stt import AssemblyAIStream

        self._vid = VoiceIdentifier()
        self._buf = AudioBuffer()
        self._vad = EnergyVAD()
        # 一文ずつ確定させる: Universal-3.x Pro は「文末の句読点 + min_turn_silence の無音」でターンを区切る
        self._stt = AssemblyAIStream(speaker_labels=False, keyterms=list(self.cfg.names) if self.cfg else None,
                                     extra_params={"min_turn_silence": "300"})
        self._stt.start(lambda e: self._on_stt(e) if gen == self._gen else None)
        self._q = queue.Queue()
        self._mic = Mic(self._q.put)
        self._mic.start()
        self._t0 = self.clock.monotonic()  # STT のストリーム時刻と時計の原点をそろえる
        threading.Thread(target=self._audio_loop, args=(gen, self._q, self._vad, self._buf, self._stt), daemon=True).start()

    def _audio_loop(self, gen: int, q: queue.Queue, vad: Any, buf: Any, stt: Any) -> None:
        """このセッションの音声だけを処理する(機器はセッション開始時のものを使う)。"""
        while self._alive(gen):
            try:
                block = q.get(timeout=0.5)
            except queue.Empty:
                continue
            ai_turn = self.ai_speaking or self.now() < self._ai_mute_until
            if self.paused or (ai_turn and self.status == "intro"):  # 自己紹介中は AI の声を登録しないよう従来どおり無音にする
                block = np.zeros_like(block)
                self.state.speaking_now = False
            elif ai_turn:
                # AI の発話中も文字起こしには実際の音を送る(人の発言を議事録に残すため)。
                # ただし発話中かどうかの判定(介入や AI 参加者のタイミング)には使わない。AI 自身の声の回り込みは _on_stt で除く
                self.state.speaking_now = False
            else:
                if vad.process(block) == "end":
                    # VAD は無音が hangover 続いてから終了と判定するので、その分を引いた時刻を声の終わりとする
                    self._voice_end = self.now() - vad.hang_blocks * vad.block_ms / 1000
                self.level_db = vad.level_db
                self.state.speaking_now = vad.speaking
            buf.append(block)
            stt.send_audio(block.tobytes())

    def _on_stt(self, e: TranscriptEvent) -> None:
        if e.kind == "partial":
            if not self.ai_speaking:  # AI の発話中の途中経過は、ほとんどが AI 自身の声なので出さない
                self.partial = e.text
                if self.state and e.text.strip():
                    self.state.heard_at = self.now()  # 認識中は沈黙を数えない(確定は話し終わってから届くため)
                    self._guess_partial_speaker(e)
            return
        if e.kind == "error":
            self._note(f"STT エラー: {e.text[:80]}")
            return
        if e.kind != "final" or e.start is None or is_noise_text(e.text) or self.paused:
            return
        self.partial = ""
        self.partial_speaker, self.partial_parts, self._partial_labels = None, [], []
        if self.status == "intro":
            self.intro.turns.append(e)
            return
        for text, start, end, sid, conf in self._identify_by_sentence(e):
            echo = self._echo_of_ai(text, start, end)
            if echo:
                self._note(f"AI の声の回り込みとみなして除外: {echo}")
                continue
            self.state.add_utterance(sid, text, start, end, conf)

    def _guess_partial_speaker(self, e: TranscriptEvent) -> None:
        """認識中の発言を「。」ごとに分け、文ごとに話者を推定する(画面の「認識中」の吹き出し)。
        書き終わった文はその文の声で1回判定し、書きかけの文は新しい文が始まった時と約1秒ごとに判定し直す。
        確定したときは文ごとの判定(_identify_by_sentence)で置き換わる。"""
        parts = split_sentences(e.text) or [e.text]
        if e.turn_order != self._partial_turn:  # 新しいターン
            self._partial_turn, self._partial_labels = e.turn_order, []
        labels = self._partial_labels
        vid, buf = self._vid, self._buf
        if vid and vid.enrolled and buf and e.start is not None and e.end is not None:
            spans = sentence_spans(parts, e.start, e.end, (e.raw or {}).get("words"))
            now = self.now()
            for i, (s, t) in enumerate(spans):
                done = i < len(parts) - 1
                if i < len(labels) and (done and labels[i][1] or not done and now - self._partial_id_at < 1.0):
                    continue  # 書き終わった文は判定済み / 書きかけの文は1秒以内に判定済み
                if t - s >= vid.min_sec:
                    r = vid.identify(buf.segment(max(s, t - 4.0), t))
                    lab = (r.speaker_id if r.speaker_id != UNKNOWN else None, done)
                else:  # 短すぎる文は前の文の話者を引き継ぐ(書き終わっていれば確定扱い)
                    lab = (labels[i - 1][0] if 0 < i <= len(labels) else None, done)
                if i < len(labels):
                    labels[i] = lab
                else:
                    labels.append(lab)
                if not done:
                    self._partial_id_at = now
        self.partial_parts = [{"text": p, "speaker": labels[i][0] if i < len(labels) else None} for i, p in enumerate(parts)]
        self.partial_speaker = self.partial_parts[-1]["speaker"]

    def _identify_by_sentence(self, e: TranscriptEvent) -> list[tuple[str, float, float, str, float]]:
        """誰が話したかを文ごとに判定する。複数人が間をあけずに話して1つのターンになった場合も、人ごとの発言に分ける。
        短すぎて判定できない文・不確かな文は前後の文の話者を引き継ぎ、同じ人が続く文はまとめる。"""
        assert e.start is not None and e.end is not None
        if not (self._vid and self._vid.enrolled):
            return [(e.text, e.start, e.end, UNKNOWN, 0.0)]
        parts = split_sentences(e.text)
        if len(parts) <= 1:
            r = self._vid.identify(self._buf.segment(e.start - 0.1, e.end + 0.1))
            return [(e.text, e.start, e.end, r.speaker_id, r.confidence)]
        spans = sentence_spans(parts, e.start, e.end, (e.raw or {}).get("words"))
        # 文(「。」)ごとに声を照合し直し、文ごとに1つの発言にする
        labels: list[tuple[str, float] | None] = []
        for s, t in spans:
            if t - s >= self._vid.min_sec:
                r = self._vid.identify(self._buf.segment(s - 0.05, t + 0.05))
                labels.append((r.speaker_id, r.confidence))
            else:
                labels.append(None)
        if all(lb is None or lb[0] == UNKNOWN for lb in labels):  # どの文も判定できなければ、ターン全体の判定を各文に使う
            r = self._vid.identify(self._buf.segment(e.start - 0.1, e.end + 0.1))
            labels = [(r.speaker_id, r.confidence)] * len(parts)
        return per_sentence(parts, spans, fill_labels(labels, UNKNOWN))

    def _echo_of_ai(self, text: str, start: float, end: float) -> str | None:
        """AI の発話と時間が重なる文字起こしが、AI の発言とよく似ていれば(スピーカーの声をマイクが拾った)理由を返す。"""
        tail = ECHO_TAIL_SEC + 0.5
        ai = [u for u in self.state.utterances if u.source in ("ai", "bot") and u.start_time - 0.5 <= end and start <= u.end_time + tail]
        if not ai and not self.ai_speaking:
            return None
        if self.ai_speaking and self._speaking_as is not None:  # 読み上げ中の発言(終了時刻がまだ確定していない)も比べる
            cur = next((u for u in reversed(self.state.utterances) if u.speaker_id == self._speaking_as), None)
            if cur is not None and cur not in ai:
                ai.append(cur)
        sim = max((_overlap(text, u.text) for u in ai), default=0.0)
        return f"一致度 {sim:.2f}" if sim >= ECHO_SIMILARITY else None

    # ================= 自己紹介(声の登録) =================
    def enroll(self, index: int) -> dict[str, Any]:
        """画面の「声を登録」: index 番目の人間の参加者の声を、今から話す自己紹介で登録する(やり直しもできる)。"""
        cfg = self.cfg
        with self._session_lock:
            if self.status != "intro" or not cfg or not (0 <= index < len(cfg.names)):
                return {"ok": False, "error": "今は声を登録できません"}
            if self.intro.index >= 0:
                return {"ok": False, "error": f"{cfg.names[self.intro.index]}さんの登録中です"}
            self.intro.index = index
            self.intro.turns = []
            self._enroll_stop.clear()
            gen = self._gen
        threading.Thread(target=self._enroll_one, args=(gen, index), daemon=True).start()
        return {"ok": True}

    def finish_enroll(self) -> dict[str, Any]:
        """画面の「登録を終了」: 今の人の声の登録を、ここまでに話した声で終える。"""
        if self.status != "intro" or self.intro.index < 0:
            return {"ok": False, "error": "登録中の人はいません"}
        self._enroll_stop.set()
        return {"ok": True}

    def begin(self) -> dict[str, Any]:
        """画面の「議論を始める」: 声の登録を終えて議論に入る(次のエージェントのサイクルで開始の説明をする)。"""
        with self._session_lock:
            if self.status != "intro":
                return {"ok": False, "error": "今は始められません"}
            if self.intro.index >= 0:
                return {"ok": False, "error": "声の登録が終わるまでお待ちください"}
            self.status = "running"
        return {"ok": True}

    def _enroll_one(self, gen: int, i: int) -> None:
        st, cfg = self.state, self.cfg
        assert st and cfg and self._vid and self._buf
        sid, name = f"speaker_{i + 1}", cfg.names[i]
        t_start = self.now()
        self._voice_end = None

        def mine() -> list[TranscriptEvent]:
            # ボタンを押した後に話し始めた声だけを使う(前の人の遅れて届いた文字起こしを除く)
            return [t for t in self.intro.turns if t.start is not None and t.start >= t_start - 0.3]

        try:
            while self._alive(gen):
                self.clock.sleep(0.2)
                if self._enroll_stop.is_set():  # 「登録を終了」が押された: 話し終わりの文字起こしが届くのを少し待つ
                    deadline = self.now() + 2.0
                    while self._alive(gen) and self.now() < deadline and (st.speaking_now or self.partial):
                        self.clock.sleep(0.1)
                    break
                turns = mine()
                voiced = sum(t.end - t.start for t in turns)
                last_end = max((t.end for t in turns), default=None)
                # 区切りは「マイクが実際に静かになってからの時間」で判断する(文字起こしの確定は遅れて届くため)
                heard_end = max(x for x in (last_end, self._voice_end) if x is not None) if last_end is not None else None
                quiet = not st.speaking_now and heard_end is not None and self.now() - heard_end >= INTRO_END_SILENCE_SEC
                if (voiced >= INTRO_MIN_VOICED_SEC and quiet) or self.now() - t_start > INTRO_TIMEOUT_SEC:
                    break
            if not self._alive(gen):
                return
            turns = [t for t in mine() if not self._is_enrolled_voice(t, sid)]
            if turns:
                audio = np.concatenate([self._buf.segment(t.start, t.end) for t in turns])
                self._vid.enrolled.pop(sid, None)  # 登録し直しのときは前の声を捨てる
                self._vid.enroll(sid, audio)
                st.set_speaker_name(sid, name, confirmed=True)
                if name not in self.intro.enrolled:
                    self.intro.enrolled.append(name)
                if name in self.intro.failed:
                    self.intro.failed.remove(name)
                for t in turns:
                    st.add_utterance(sid, t.text, t.start, t.end, 1.0)
                self._note(f"{name}さんの声を登録しました({len(audio) / 16000:.1f}秒)")
            else:
                if name not in self.intro.failed and name not in self.intro.enrolled:
                    self.intro.failed.append(name)
                self._note(f"{name}さんの声を取得できませんでした")
        finally:
            if gen == self._gen:
                self.intro.index = -1

    def _is_enrolled_voice(self, t: TranscriptEvent, sid: str) -> bool:
        """自己紹介の声が、すでに登録した別の人の声とはっきり一致するか(前の人がまだ話していた場合など)。"""
        if not (self._vid and self._vid.enrolled) or t.start is None or t.end is None:
            return False
        r = self._vid.identify(self._buf.segment(t.start, t.end))
        return r.speaker_id not in (UNKNOWN, sid) and r.confidence >= 0.8

    # ================= エージェント =================
    def _agent_loop(self, gen: int) -> None:
        while self._alive(gen):
            self.clock.sleep(0.25)
            if not self._alive(gen):
                return
            trig = self._agent_step()
            if trig:
                threading.Thread(target=self._run_tick, args=(trig, gen), daemon=True).start()

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

    def _run_tick(self, trigger: str, gen: int | None = None) -> None:
        gen = self._gen if gen is None else gen
        live: LiveSpeech | None = None
        try:
            m = self.agent.llm.mark()
            on_partial = None
            if supports_stream(self.agent.llm):
                live = LiveSpeech(self, gen, lambda: self._facilitator_can_start(gen), max_sentences=3)
                live.speaker = AI_SPEAKER

                def push_checked(sentence: str) -> None:
                    """一文ごとに、全文の場合と同じ検証(個人名・文数)を通す。通らなければ、その文から先は話さない。"""
                    if live.action is None or live.declined:
                        return
                    ok = self.agent.live_sentence(live.action, sentence, live.count)
                    if ok is None:
                        if live.started:
                            live.max = live.count  # 話した分で止める
                        else:
                            live.declined = True  # まだ話していなければ全文を待つ(全文の検証に任せる)
                        return
                    live.push(ok)

                feeder = SentenceFeeder("speech", push_checked)

                def on_partial(acc: str, allowed: list[Action], gate: str) -> None:
                    if live.declined:
                        return
                    if live.action is None:  # 行動が決まるまでは話し始めない
                        a, closed = partial_string(acc, "action")
                        if not closed:
                            return
                        act = next((x for x in Action if x.value == a), None)
                        if (act is None or act in NO_LIVE_ACTIONS or act not in allowed or gate.startswith("forced")
                                or not self.agent.live_ok(act, self.now())):
                            live.declined = True
                            return
                        live.action = act
                    feeder.feed(acc)

            failed: Exception | None = None
            try:
                rec = self.agent.tick(self.now(), trigger, act=False, on_partial=on_partial)
            except Exception as e:  # LLM の失敗は tick の中で見送りになる。ここに来るのはそれ以外の想定外の失敗
                failed = e
            finally:
                if live is not None:
                    if live.started:
                        feeder.finish()  # 句点のない最後の文
                    live.close()
            if failed is not None:
                if live is not None and live.started:  # 話した分は介入として記録する(クールダウンが効くように)
                    live.wait()
                    if live.utt is not None and live.action is not None and gen == self._gen:
                        self.agent.record_spoken(live.utt.start_time, live.action, live.text)
                raise failed
            if gen != self._gen:  # LLM の応答待ちの間に次のセッションが始まった。古い判断は捨てる
                return
            if self._sec is not None:
                self._sec.decision(rec, self.agent.llm.since(m))  # 取りやめた場合も rec に追記されて残る
            if live is not None and live.started:  # 生成しながら話した
                live.wait()
                if live.utt is not None and gen == self._gen and live.action is not None:
                    self.agent.commit_spoken(rec, live.utt.start_time, live.action, live.text)
                return
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
                while d.action != Action.start_gd and self._alive(gen):
                    now = self.now()
                    if not st.speaking_now and st.silence_sec(now) >= FORCED_PAUSE_SEC:
                        break
                    if now >= deadline:
                        self._note(f"{d.action.value}: 間がないため発話を優先")
                        break
                    self.clock.sleep(0.1)
                if not self._alive(gen):
                    return
                self._speak_decision(rec, gen)  # 発話の排他を待つ間に終わった場合も、排他の中でもう一度確認する
                if d.action == Action.close_gd and gen == self._gen:
                    self._note("GD を終了しました")
                    self.stop(gen)  # このセッションが現在のものであるときだけ終了する
                return
            if d.action == Action.redirect_topic:
                self._await_redirect(rec, gen)
                return
            if st.is_speaking(now):
                self.agent.cancel(rec, "誰かが話し始めた")
                self._note(f"{d.action.value} を取りやめ(誰かが話している)")
                return
            if st.silence_sec(now) < self.settings.min_silence_for_intervention_sec:
                self.agent.cancel(rec, "沈黙が短い")
                self._note(f"{d.action.value} を取りやめ(会話が続いている)")
                return
            self._speak_decision(rec, gen)
        finally:
            if gen == self._gen:  # 次のセッションの「判断中」の印は消さない
                self._agent_busy = False

    def _facilitator_can_start(self, gen: int) -> bool:
        """進行役が生成途中の発言を話し始めてよいか(全文を待つ場合の、話す直前の確認と同じ条件)。"""
        if not self._alive(gen) or self.paused or self.status != "running" or self.ai_speaking:
            return False
        now, st = self.now(), self.state
        return not st.is_speaking(now) and st.silence_sec(now) >= self.settings.min_silence_for_intervention_sec

    def _speak_live(self, live: LiveSpeech) -> None:
        """LiveSpeech の文を順に議論ログへ足して読み上げる。最初の文で発言を追加し、以降は同じ発言に書き足す。"""
        speaker = live.speaker or AI_SPEAKER
        is_bot = speaker != AI_SPEAKER
        try:
            with self._speak_lock:
                if not self._alive(live.gen):
                    return
                tts = self._bot_tts.get(speaker) if is_bot else self._tts
                self.ai_speaking = True
                self._speaking_as = speaker
                try:
                    while True:
                        sent = live.q.get()
                        if sent is None or not self._alive(live.gen):
                            break
                        now = self.now()
                        est = len(sent) / 5.6 if tts else min(3.0, len(sent) / self.settings.tts_chars_per_sec)
                        if live.utt is None:
                            live.utt = self.state.add_utterance(speaker, sent, now, now + est, 1.0, source="bot" if is_bot else "ai")
                        else:
                            live.utt.text += sent
                            live.utt.end_time = now + est
                        live.spoken.append(sent)
                        spoken = False
                        if tts:
                            try:
                                tts.speak(sent)
                                spoken = True
                            except Exception as e:  # TTS が使えなくても進行は止めない(以降は声なしで続ける)
                                self._note(f"音声出力に失敗したため、声なしで続けます({type(e).__name__})")
                                if is_bot:
                                    self._bot_tts.pop(speaker, None)
                                else:
                                    self._tts = None
                                tts = None
                        if not spoken:
                            self.clock.sleep(est)
                finally:
                    if live.utt is not None:
                        live.utt.end_time = max(live.utt.start_time, self.now())
                    self.ai_speaking = False
                    self._speaking_as = None
                    self._ai_mute_until = self.now() + ECHO_TAIL_SEC
        finally:
            live.done.set()

    def _quiet_sec(self, now: float) -> float:
        """いまの静けさの長さ。発話中は0。音声モードは VAD で声が止まった時刻も見る(STT の確定は遅れるため)。"""
        st = self.state
        if st.speaking_now:
            return 0.0
        quiet = st.silence_sec(now)
        if self._voice_end is not None:
            quiet = min(quiet, now - self._voice_end)
        return quiet

    def _await_redirect(self, rec: TickRecord, gen: int | None = None) -> None:
        """脱線の軌道修正: 判断後、短い発話の区切り(redirect_pause_sec)を待って話す。
        発話中には話さず、区切りがないまま redirect_wait_max_sec たつか、本題に戻ったら取り消す。"""
        st, s = self.state, self.settings
        gen = self._gen if gen is None else gen
        decided = self.now()
        cut = rec.context_last_utt_id

        def human(after: bool) -> list[str]:
            return [u.text for u in st.utterances if u.source == "human" and (cut is None or (u.id > cut) == after)]

        drift = human(after=False)[-4:]  # 判断に使った直近の発言(脱線部分)。この語は本題の語から除く
        while True:
            now = self.now()
            new = human(after=True)
            reason = None
            if not self._alive(gen) or self.status != "running":
                reason = "セッション終了"
            elif self.paused:
                reason = "一時停止"
            elif self.agent.forced_event_pending(now):
                reason = "強制イベントを優先"
            elif back := detect_topic_return(st.theme, rec.analysis, drift, new):
                reason = f"本題に戻った({back})"
            elif self._quiet_sec(now) >= s.redirect_pause_sec:
                self._log_redirect(rec, "spoken", None, now - decided, len(new))
                self._speak_decision(rec, gen)
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

    def _speak_decision(self, rec: TickRecord, gen: int | None = None) -> None:
        text = rec.decision.speech
        with self._speak_lock:  # 介入の記録と読み上げを、世代の確認と同じ排他の中で行う
            if gen is not None and not self._alive(gen):
                return
            est = len(text) / self.settings.tts_chars_per_sec
            self.agent.commit(rec, self.now(), est)
            utt = next((u for u in reversed(self.state.utterances) if u.source == "ai"), None)
            self._say(text, record=False, utt=utt, gen=gen)

    def _say(self, text: str, record: bool = True, speaker: str = AI_SPEAKER, utt: Any = None, gen: int | None = None) -> Any:
        """AI(進行役 / AI 参加者)の発話。発話中はマイクをミュートする。record=True なら議論ログにも追加する。
        複数の AI が同時に話さないよう、発話はロックで1つずつにする。gen を渡すと、ロックを取った時点で
        そのセッションが終わっていれば話さずに None を返す(ロック待ちの間に次のセッションが始まった場合)。"""
        is_bot = speaker != AI_SPEAKER
        with self._speak_lock:
            if gen is not None and not self._alive(gen):
                return None
            tts = self._bot_tts.get(speaker) if is_bot else self._tts  # ロックの中で取る(現在のセッションの声)
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
    def _bot_loop(self, gen: int) -> None:
        while self._alive(gen):
            self.clock.sleep(0.3)
            if not self._alive(gen):
                return
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
                threading.Thread(target=self._bot_turn, args=(persona, reason, addressed_id, gen), daemon=True).start()

    def _bot_pick(self, now: float) -> tuple[Persona, str, int] | None:
        if self.agent and self.agent.has_pending_request(now):
            return None  # 進行役への依頼には進行役が答える(AI 参加者が先に答えない)
        return pick_bot(
            self.state, self.personas, self.bot_mode, now, self._last_bot_end, self._bot_answered_id, len(self.cfg.names),
            self._interval, self._threshold_cache,
        )

    def _bot_turn(self, p: Persona, reason: str, addressed_id: int, gen: int | None = None) -> None:
        st = self.state
        gen = self._gen if gen is None else gen
        try:
            last_id = st.latest_id
            transcript = transcript_for_bot(st)
            flow = discussion_flow(self.agent, self.now())  # 進行役と同じ議論の流れ(いまの段階)
            m = self.agent.llm.mark()
            llm = self.agent.llm
            live: LiveSpeech | None = None
            on_partial = None
            if supports_stream(llm):
                live = LiveSpeech(self, gen, lambda: self._bot_can_start(gen, last_id, addressed_id))
                feeder = SentenceFeeder("text", live.push)
                if addressed_id >= 0:
                    live.speaker, live.persona = p.speaker_id, p
                    on_partial = feeder.feed
                else:
                    by_name = {x.name: x for x in self.personas}

                    def on_partial(acc: str) -> None:
                        if live.declined:
                            return
                        if live.speaker is None:  # 誰が話すかが決まってから文を受け取る
                            name, closed = partial_string(acc, "speaker")
                            if not closed:
                                return
                            live.persona = by_name.get(name or "")
                            if live.persona is None:
                                live.declined = True
                                return
                            live.speaker = live.persona.speaker_id
                        feeder.feed(acc)
            try:
                if addressed_id >= 0:  # 名指しされた人が答える
                    text = generate_utterance(
                        llm, p, self.cfg.purpose, st.theme, transcript, reason, self.ai_level, on_partial, flow
                    )
                else:  # 誰が話すかは文脈で選ぶ(直前の話者は除く)
                    last_sid = st.utterances[-1].speaker_id if st.utterances else ""
                    cands = [x for x in self.personas if x.speaker_id != last_sid] or self.personas
                    counts = {x.speaker_id: x.spoken for x in self.personas}
                    p, text, why = choose_and_generate(
                        llm, cands, self.cfg.purpose, st.theme, transcript, reason, self.ai_level, counts, p, on_partial, flow
                    )
                    if why:
                        reason += f" / {p.name}が話した理由: {why}"
            except LLMError as e:
                self._note(f"{p.name}の発言を生成できませんでした({type(e).__name__})")
                text = ""
            finally:
                if live is not None:
                    if live.started:
                        feeder.finish()
                    live.close()
            if live is not None and live.started:  # 生成しながら話した
                live.wait()
                if live.utt is None or gen != self._gen:
                    return
                p = live.persona or p
                if addressed_id >= 0:
                    self._bot_answered_id = addressed_id
                p.spoken += 1
                if self._sec is not None:
                    self._sec.bot(p.speaker_id, live.utt.start_time, reason, self.agent.llm.since(m))
                self._last_bot_end = self.now()
                return
            if not text or not self._alive(gen) or self.paused or self.status != "running":
                return
            # 生成を待つ間に人間が話し始めた/新しい発言があった場合は取りやめる(名指しへの応答は除く)
            newer = st.latest_id != last_id
            if st.speaking_now or (newer and addressed_id < 0):
                return
            if addressed_id >= 0:
                self._bot_answered_id = addressed_id
            p.spoken += 1
            call = self.agent.llm.since(m)
            utt = self._say(text, speaker=p.speaker_id, gen=gen)
            if utt is None:  # 発話の排他を待つ間にセッションが終わった
                return
            if self._sec is not None:
                self._sec.bot(p.speaker_id, utt.start_time, reason, call)
            self._last_bot_end = self.now()
        finally:
            if gen == self._gen:
                self._bot_busy = False

    def _bot_can_start(self, gen: int, last_id: int, addressed_id: int) -> bool:
        """AI 参加者が生成途中の発言を話し始めてよいか(全文を待つ場合の、話す直前の確認と同じ条件)。"""
        st = self.state
        if not self._alive(gen) or self.paused or self.status != "running" or st.speaking_now or self.ai_speaking:
            return False
        return addressed_id >= 0 or st.latest_id == last_id

    # ================= UI からの操作 =================
    def add_text(self, speaker_id: str, text: str) -> None:
        if self.status not in ("running", "intro") or not text.strip():
            return
        now = self.now()
        dur = max(1.0, len(text) / 6.0)
        conf = 1.0 if speaker_id != UNKNOWN else 0.0
        # 入力した発言には本当の話し始めがないので、既にある発言より前には入れない(送った順に並べる)
        latest = max((u.start_time for u in self.state.utterances), default=now - dur)
        start = min(now, max(now - dur, latest + 0.001))
        self.state.add_utterance(speaker_id, text.strip(), start, now, conf)

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

    def feedback(self) -> dict[str, Any]:
        """終了後の振り返り(練習の補助)と発言の種類。セッションごとに1回だけ作って使い回す。"""
        r = self.result()
        if not r.get("ok"):
            return {"ok": False}
        key = self._gen
        with self._feedback_lock:
            if self._feedback is not None and self._feedback[0] == key:
                return self._feedback[1]
            humans = [{"id": p["id"], "name": p["name"]} for p in r["people"] if p["kind"] == "human"]
            # 振り返りの内容はファイルに保存しない(人の発言の本文を含むため)
            fb = build_feedback(self.agent.llm, r["theme"], r["discussion_type"], r["log"], humans)
            if key == self._gen:
                self._feedback = (key, fb)
            return fb

    def result(self) -> dict[str, Any]:
        """終了後の結果画面用: 全部の発言と、参加者ごとの発言量(回数・秒数・文字数)。"""
        st, cfg, agent = self.state, self.cfg, self.agent
        if not st or not cfg or not agent:
            return {"ok": False}
        now = self._ended_at if self._ended_at is not None else self.now()
        names = {f"speaker_{i + 1}": n for i, n in enumerate(cfg.names)}
        names.update({p.speaker_id: p.name for p in self.personas})
        done = [u for u in st.utterances if u.start_time <= now]

        def amount(us: list[Any]) -> dict[str, Any]:
            return {"turns": len(us), "sec": round(sum(max(0.0, u.end_time - u.start_time) for u in us), 1),
                    "chars": sum(len(u.text) for u in us)}

        people = []
        order = list(names) + ([UNKNOWN] if any(u.speaker_id == UNKNOWN for u in done) else [])
        for sid in order:
            mine = [u for u in done if u.speaker_id == sid and u.source != "ai"]
            kind = "bot" if sid in st.bot_ids else ("unknown" if sid == UNKNOWN else "human")
            people.append({"id": sid, "name": names.get(sid, "不明"), "kind": kind, **amount(mine)})
        return {
            "ok": True,
            "theme": cfg.theme,
            "purpose": cfg.purpose,
            "discussion_type": agent.dtype.label,
            "elapsed": round(st.elapsed(now), 1) if st.started else 0.0,
            "duration": st.duration_sec,
            "people": people,
            "facilitator": amount([u for u in done if u.source == "ai"]),
            "interventions": len(st.interventions),
            "log": [
                {
                    "id": u.id,
                    "t": round(st.elapsed(u.start_time) if st.started else u.start_time, 1),
                    "speaker_id": u.speaker_id,
                    "who": "AI ファシリテーター" if u.source == "ai" else (names.get(u.speaker_id) or "不明"),
                    "text": u.text,
                    "source": u.source,
                }
                for u in done
            ],
        }

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
            "policy_preset": self.policy_preset,
            "policy_overrides": self.policy.overrides_from(preset_policy(self.policy_preset)),  # 再読み込みで画面を復元する
            "policy": self.policy.model_dump(mode="json"),
            "policy_notices": self.policy.notices(cfg.purpose),
            "active_speaker": self._active_speaker(now),
            "llm": agent.llm.name,
            "mode": cfg.mode,
            "theme": cfg.theme,
            "names": names,
            "humans": [f"speaker_{i + 1}" for i in range(len(cfg.names))],
            "participants": [
                # 立場・意見は画面に送らない(参加者からは見えないようにする。ログには残る)
                {"id": p.speaker_id, "name": p.name} for p in self.personas
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
            "partial_speaker": self.partial_speaker,
            "partial_parts": self.partial_parts,
            "use_speaker_info": self.settings.use_speaker_info,
            "reliability": round(st.speaker_reliability(), 2),
            "stats": [x.model_dump() for x in st.speaker_stats(now)],
            "utterances": utts,
            "ticks": ticks,
            "interventions": len(st.interventions),
            "intro": {
                "current": cfg.names[self.intro.index] if 0 <= self.intro.index < len(cfg.names) else None,
                "recording": f"speaker_{self.intro.index + 1}" if 0 <= self.intro.index < len(cfg.names) else None,
                "enrolled": self.intro.enrolled,
                "failed": self.intro.failed,
            },
            "events": self.events[-8:],
        }
