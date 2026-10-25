"""生成しながら話す仕組み(途中の JSON から文を取り出す)・AI の声の回り込みの除外・発言の並び順。"""

from __future__ import annotations

import json
import time

from config import load_settings
from facilitator.engine import _overlap
from facilitator.llm.stream import SentenceFeeder, partial_string, simulate_stream
from facilitator.state import DiscussionState


def test_partial_string_reads_unfinished_value_and_escapes():
    assert partial_string('{"a": 1, "speech": "こんに', "speech") == ("こんに", False)
    assert partial_string('{"speech": "a\\"b\\nc"}', "speech") == ('a"b\nc', True)
    assert partial_string('{"speech": "ab\\', "speech") == ("ab", False)  # エスケープの途中で切れている
    assert partial_string('{"action": "ask', "speech") == (None, False)


def test_sentences_are_emitted_as_soon_as_they_end():
    got: list[str] = []
    f = SentenceFeeder("speech", got.append)
    text = json.dumps({"decision": {"action": "ask_question", "speech": "一つ目です。二つ目ですか?三つ目"}}, ensure_ascii=False)
    seen = []
    for i in range(1, len(text) + 1):
        f.feed(text[:i])
        seen.append(len(got))
    assert got == ["一つ目です。", "二つ目ですか?", "三つ目"]  # 閉じた時点で句点のない最後の文も出る
    assert seen.index(1) < seen.index(2)  # 一文ずつ、順に出る


def test_finish_flushes_last_sentence():
    got: list[str] = []
    f = SentenceFeeder("text", got.append)
    f.feed('{"text": "まだ途中')
    assert got == []
    f.finish()
    assert got == ["まだ途中"]


def test_simulate_stream_feeds_growing_prefixes():
    parts: list[str] = []
    simulate_stream({"text": "こんにちは。"}, parts.append, chunk=4)
    assert parts[-1] == json.dumps({"text": "こんにちは。"}, ensure_ascii=False)
    assert all(parts[i + 1].startswith(parts[i]) for i in range(len(parts) - 1))


def test_echo_overlap_detects_ai_voice_but_not_different_speech():
    said = "皆さんの意見を整理すると、コストと手間のどちらを重視するかで分かれています。"
    assert _overlap("皆さんの意見を整理するとコストと手間の", said) > 0.8
    assert _overlap("いや、私は利用者の満足度が一番大事だと思います", said) < 0.3


def test_utterances_are_ordered_by_start_time():
    st = DiscussionState(load_settings(), "テーマ", 300.0)
    a = st.add_utterance("ai", "AI の発言", 10.0, 14.0, 1.0, source="ai")
    b = st.add_utterance("speaker_1", "先に話し始めた人", 9.5, 15.0, 1.0)  # 確定は後だが話し始めは早い
    c = st.add_utterance("speaker_2", "後から話し始めた人", 12.0, 13.0, 1.0)
    assert [u.id for u in st.utterances] == [b.id, a.id, c.id]
    assert st.latest_id == c.id


# ---------- エンジン: 生成しながら話す ----------
import facilitator.engine as engine_mod  # noqa: E402
from facilitator.engine import Engine, SessionConfig  # noqa: E402
from facilitator.llm.mock_llm import MockLLM  # noqa: E402
from facilitator.models import Action  # noqa: E402
from tests.sim import SimClock  # noqa: E402


class SpyMock(MockLLM):
    """生成途中の JSON を受け取ったときに、すでに話し始めていたかを記録する。"""

    stream_delay = 0.0

    def __init__(self) -> None:
        self.eng: Engine | None = None
        self.spoke_before_done = False

    def generate_json(self, system, user, schema, context, on_partial=None):
        def spy(acc: str) -> None:
            on_partial(acc)
            time.sleep(0.005)  # 実際のストリーミングと同じく、少しずつ届く
            if self.eng and any(u.source == "ai" and u.id > 0 for u in self.eng.state.utterances):
                self.spoke_before_done = True

        return super().generate_json(system, user, schema, context, spy if on_partial else None)


def test_facilitator_speaks_while_generating(monkeypatch, tmp_path):
    spy = SpyMock()
    monkeypatch.setattr(engine_mod, "make_llm", lambda settings: spy)
    eng = Engine(load_settings(), clock=SimClock(), log_dir=tmp_path, tts_chars_per_sec=1000.0, cooldown_sec=0.0)
    spy.eng = eng
    try:
        eng.start(SessionConfig("テーマ", 5, ["A", "B"], mode="text", intro=False, tts=False, discussion_type="problem_solving"), background=False)
        eng._run_tick(eng._agent_step())  # 開始の説明
        assert eng.state.utterances[0].text == "テーマは「テーマ」、時間は5分です。始めてください。"
        eng.add_text("speaker_1", "原因は昼の集中です。")
        eng.clock.sleep(15.0)  # 長い沈黙 → モックは問いかける
        eng._run_tick(eng._agent_step())
        ai = [u for u in eng.state.utterances if u.source == "ai"]
        assert len(ai) == 2 and ai[1].text.endswith("?")
        assert spy.spoke_before_done  # 返事を全部受け取る前に、議論ログに出ていた
        assert eng.state.interventions[-1].action == Action.ask_question  # 介入として記録される
    finally:
        eng.stop()
