"""文ごとの話者判定: 1つのターンに複数人の発言がまとまっても、人ごとの発言に分ける。"""

from __future__ import annotations

from facilitator.speaker.split import fill_labels, group_by_speaker, sentence_spans, split_sentences


def test_split_sentences():
    assert split_sentences("昼休みをずらすのがいいと思います。でも先生の負担が心配です。") == [
        "昼休みをずらすのがいいと思います。", "でも先生の負担が心配です。"]
    assert split_sentences("そうですね") == ["そうですね"]


def test_spans_by_character_share():
    parts = ["あいうえお。", "かき。"]  # 6文字と3文字
    spans = sentence_spans(parts, 10.0, 19.0)
    assert spans[0] == (10.0, 16.0) and spans[1] == (16.0, 19.0)


def test_spans_use_word_times_when_available():
    words = [{"text": "あいうえお。", "start": 10000, "end": 12000}, {"text": "かき。", "start": 15000, "end": 16000}]
    spans = sentence_spans(["あいうえお。", "かき。"], 10.0, 16.0, words)
    assert spans[0] == (10.0, 12.0) and spans[1][0] == 15.0 and spans[1][1] == 16.0


def test_short_or_unsure_sentences_inherit_neighbors():
    labels = fill_labels([("speaker_1", 0.9), None, ("unknown", 0.2), ("speaker_2", 0.85)])
    assert [sid for sid, _ in labels] == ["speaker_1", "speaker_1", "speaker_1", "speaker_2"]
    assert fill_labels([None, ("speaker_2", 0.8)])[0][0] == "speaker_2"
    assert fill_labels([None, None]) == [("unknown", 0.0), ("unknown", 0.0)]


def test_group_merges_same_speaker_and_splits_different():
    parts = ["Aの一文目。", "Aの二文目。", "Bの発言。"]
    spans = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]
    labels = [("speaker_1", 0.9), ("speaker_1", 0.8), ("speaker_2", 0.95)]
    out = group_by_speaker(parts, spans, labels)
    assert out == [("Aの一文目。Aの二文目。", 0.0, 4.0, "speaker_1", 0.8), ("Bの発言。", 4.0, 6.0, "speaker_2", 0.95)]


# ---------- エンジン: 1つのターンを人ごとの発言に分ける ----------
import numpy as np  # noqa: E402

import facilitator.engine as engine_mod  # noqa: E402
from config import load_settings  # noqa: E402
from facilitator.engine import Engine, SessionConfig  # noqa: E402
from facilitator.llm.mock_llm import MockLLM  # noqa: E402
from facilitator.speaker.voice_id import IdResult  # noqa: E402
from facilitator.stt.base import TranscriptEvent  # noqa: E402
from tests.sim import SimClock  # noqa: E402


class FakeBuf:
    def segment(self, s: float, e: float) -> np.ndarray:
        return np.full(int(max(0.0, e - s) * 16000), s, dtype=np.float32)  # 区間の開始時刻を音として渡す


class FakeVid:
    """区間の開始時刻で話者を返す(2秒より前はたろう、後ははなこ)。"""

    min_sec = 1.0
    enrolled = {"speaker_1": [], "speaker_2": []}

    def clear(self) -> None:
        pass

    def identify(self, pcm: np.ndarray) -> IdResult:
        sid = "speaker_1" if float(pcm[0]) < 1.9 else "speaker_2"
        return IdResult(sid, 0.9, 0.7, 0.3, {})


def test_one_turn_with_two_speakers_becomes_two_utterances(monkeypatch, tmp_path):
    monkeypatch.setattr(engine_mod, "make_llm", lambda settings: MockLLM())
    eng = Engine(load_settings(), clock=SimClock(), log_dir=tmp_path, tts_chars_per_sec=1000.0)
    try:
        eng.start(SessionConfig("テーマ", 5, ["たろう", "はなこ"], mode="text", intro=False, tts=False,
                                discussion_type="problem_solving"), background=False)
        eng._vid, eng._buf = FakeVid(), FakeBuf()
        eng._on_stt(TranscriptEvent("final", text="ずらすのがいいです。でも負担が心配です。", start=0.0, end=4.0))
        got = [(u.speaker_id, u.text) for u in eng.state.utterances if u.source == "human"]
        assert got == [("speaker_1", "ずらすのがいいです。"), ("speaker_2", "でも負担が心配です。")]
    finally:
        eng.stop()
