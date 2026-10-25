"""声の登録: 参加者カードの「声を登録」で1人ずつ(順番自由・やり直し可)、「議論を始める」で議論に入る。"""

from __future__ import annotations

import numpy as np

import facilitator.engine as engine_mod
from config import load_settings
from facilitator.engine import Engine, SessionConfig
from facilitator.llm.mock_llm import MockLLM
from facilitator.speaker.voice_id import IdResult
from facilitator.stt.base import TranscriptEvent
from tests.sim import SimClock


class FakeBuf:
    def segment(self, s: float, e: float) -> np.ndarray:
        return np.zeros(int(max(0.0, e - s) * 16000), dtype=np.int16)


class FakeVid:
    min_sec = 1.0

    def __init__(self) -> None:
        self.enrolled: dict[str, list] = {}

    def enroll(self, sid: str, pcm: np.ndarray) -> None:
        self.enrolled.setdefault(sid, []).append(len(pcm))

    def identify(self, pcm: np.ndarray) -> IdResult:
        return IdResult("unknown", 0.0, 0.0, 0.0, {})

    def clear(self) -> None:
        self.enrolled.clear()


def make(monkeypatch, tmp_path) -> Engine:
    monkeypatch.setattr(engine_mod, "make_llm", lambda settings: MockLLM())
    eng = Engine(load_settings(), clock=SimClock(), log_dir=tmp_path, tts_chars_per_sec=1000.0)
    eng.start(SessionConfig("テーマ", 5, ["たろう", "はなこ"], mode="text", intro=False, tts=False,
                            discussion_type="problem_solving"), background=False)
    eng.status = "intro"  # 音声モードで自己紹介をする状態(マイクなしで再現)
    eng._vid, eng._buf = FakeVid(), FakeBuf()
    return eng


def speak(eng: Engine, index: int, text: str) -> None:
    """index 番目の人のボタンを押して話す(押した後に話し始めた声だけが使われる)。"""
    with eng._session_lock:
        eng.intro.index = index
        eng.intro.turns = []
    now = eng.now()
    eng.intro.turns.append(TranscriptEvent("final", text=text, start=now + 0.2, end=now + 3.2))
    eng._enroll_one(eng._gen, index)


def test_enroll_in_any_order_and_redo(monkeypatch, tmp_path):
    eng = make(monkeypatch, tmp_path)
    try:
        speak(eng, 1, "はなこです。よろしくお願いします。")  # 2人目から先に登録してもよい
        assert eng.intro.enrolled == ["はなこ"] and eng.intro.index == -1
        assert "speaker_2" in eng._vid.enrolled and "speaker_1" not in eng._vid.enrolled
        speak(eng, 0, "たろうです。よろしくお願いします。")
        speak(eng, 0, "たろうです。もう一度登録します。")  # やり直しは前の声を捨てて登録し直す
        assert eng.intro.enrolled == ["はなこ", "たろう"] and len(eng._vid.enrolled["speaker_1"]) == 1
    finally:
        eng.stop()


def test_old_turns_before_the_button_are_not_used(monkeypatch, tmp_path):
    eng = make(monkeypatch, tmp_path)
    try:
        with eng._session_lock:
            eng.intro.index = 1
        now = eng.now()
        # ボタンを押す前に話していた前の人の声(遅れて届いた文字起こし)だけしかない
        eng.intro.turns = [TranscriptEvent("final", text="たろうの続き", start=now - 5.0, end=now - 1.0)]
        eng._enroll_one(eng._gen, 1)
        assert "speaker_2" not in eng._vid.enrolled and eng.intro.failed == ["はなこ"]
    finally:
        eng.stop()


def test_begin_starts_discussion_only_when_not_recording(monkeypatch, tmp_path):
    eng = make(monkeypatch, tmp_path)
    monkeypatch.setattr(eng, "_enroll_one", lambda gen, i: None)  # 録音は終わらないままにする(登録中の状態を保つ)
    try:
        assert eng.enroll(0)["ok"] is True  # 登録中
        assert eng.enroll(1)["ok"] is False  # 同時に2人は登録しない
        assert eng.begin()["ok"] is False  # 登録が終わるまで始めない
        eng.intro.index = -1
        assert eng.begin()["ok"] is True and eng.status == "running"
        assert eng.enroll(0)["ok"] is False  # 始まった後は登録しない
    finally:
        eng.stop()


def test_finish_button_ends_enrollment_with_what_was_said(monkeypatch, tmp_path):
    eng = make(monkeypatch, tmp_path)
    try:
        with eng._session_lock:
            eng.intro.index = 0
            eng.intro.turns = []
        now = eng.now()
        eng.intro.turns.append(TranscriptEvent("final", text="たろうです", start=now + 0.2, end=now + 1.2))  # 2秒に満たない
        eng._enroll_stop.set()  # 「登録を終了」(自動で終わる長さに達していなくても終える)
        eng._enroll_one(eng._gen, 0)
        assert eng.intro.enrolled == ["たろう"] and "speaker_1" in eng._vid.enrolled
        assert eng.finish_enroll()["ok"] is False  # 登録中でなければ何もしない
    finally:
        eng.stop()
