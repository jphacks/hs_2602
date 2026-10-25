"""owaki さんの「生成しながら話す」(engine.LiveSpeech)と介入ポリシーの検証の接続のテスト。API は呼ばない。

全文を待つ場合の検証(個人名・⑤控えめの見送り・文数)を、生成しながら話す場合にも通すこと、
生成しながら話した内容支援も内容支援として記録されること(_record の content_reason)を確かめる。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from facilitator.agent.agent import FacilitatorAgent  # noqa: E402
from facilitator.llm.stream import simulate_stream  # noqa: E402
from facilitator.models import Action  # noqa: E402
from facilitator.policy import InterventionPolicy  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402
from tests.sim import SimSession  # noqa: E402
from tests.test_content import StubLLM  # noqa: E402


def _agent(policy=None):
    s = load_settings()
    st = DiscussionState(settings=s, theme="t", duration_sec=600)
    st.started, st.start_time = True, 0.0
    for sid, n in {"speaker_1": "たろう", "speaker_2": "はなこ"}.items():
        st.set_speaker_name(sid, n)
    st.add_utterance("speaker_1", "うーん。", 100, 103, 0.9)
    return st, FacilitatorAgent(st, StubLLM(), s, policy=policy or InterventionPolicy())


def test_live_sentence_applies_name_rule_and_sentence_limit():
    st, ag = _agent()
    assert ag.live_sentence(Action.ask_question, "はなこさん、どう思いますか?", 0) == "どう思いますか?"
    assert ag.live_sentence(Action.ask_question, "はなこさんの案はどうでしょう。", 0) == "先ほど出た案はどうでしょう。"
    assert ag.live_sentence(Action.ask_question, "はなこさんは賛成でしたね。", 0) is None  # 言い換えられない → 話さない
    assert ag.live_sentence(Action.ask_question, "三つ目です。", 2) == "三つ目です。"
    assert ag.live_sentence(Action.ask_question, "四つ目です。", 3) is None  # 3文まで
    assert ag.live_sentence(Action.offer_hint, "三つ目です。", 2) is None  # ヒントは2文まで


def test_live_ok_defers_possible_stall_hold_to_full_validation():
    """⑤控えめで、見送りの対象になりうる行動は、沈黙が基準(18秒)未満なら生成しながら話さない(全文を待って検証する)。"""
    st, ag = _agent(InterventionPolicy(stall="low"))
    assert not ag.live_ok(Action.ask_question, 110.0)  # 沈黙7秒
    assert not ag.live_ok(Action.organize_issues, 110.0)
    assert ag.live_ok(Action.compare_criteria, 110.0)  # 膠着など、見送りの対象でない行動はそのまま
    assert ag.live_ok(Action.invite_all, 110.0)
    assert ag.live_ok(Action.ask_question, 122.0)  # 沈黙19秒
    st2, ag2 = _agent()  # 標準(GD 既定)は従来どおり生成しながら話せる
    assert ag2.live_ok(Action.ask_question, 110.0)


def test_commit_spoken_records_content_support():
    """生成しながら話したヒントも内容支援として記録する(マージで _record が content_reason を失っていた)。"""
    st, ag = _agent(InterventionPolicy(content_scope="hint", content_initiative="proactive"))
    ag.llm.action, ag.llm.speech = "offer_hint", "利用する人の立場から見るとどうでしょう?"
    rec = ag.tick(110.0, act=False)
    assert rec.content_reason == "proactive"
    ag.commit_spoken(rec, 110.5, Action.offer_hint, rec.decision.speech)
    assert ag.content_log == [(110.5, Action.offer_hint, "proactive")]


class StreamingStub(StubLLM):
    """生成途中の JSON を少しずつ渡すスタブ(owaki さんのモックと同じ simulate_stream を使う)。"""

    streams = True

    def generate_json(self, system, user, schema, context, on_partial=None):
        out = super().generate_json(system, user, schema, context)
        if on_partial is not None:
            simulate_stream(out, on_partial)
        return out


def _session(monkeypatch, speech):
    llm = StreamingStub("ask_question", speech, health="stalled")
    sim = SimSession(monkeypatch, llm, names=("たろう", "はなこ", "けんた"))
    sim.run(5)  # 開始の発言
    sim.say(10, "speaker_1", "原因は時間の集中ですね。", 2)
    sim.run(60)
    return sim


def test_engine_live_speech_rephrases_names(monkeypatch):
    sim = _session(monkeypatch, "はなこさん、どの原因から考えますか?")
    said = [u.text for u in sim.eng.state.utterances if u.source == "ai"][1:]
    assert said and all("はなこさん" not in t for t in said) and "どの原因から考えますか?" in said[0]


def test_engine_live_speech_stops_before_unrephrasable_name(monkeypatch):
    sim = _session(monkeypatch, "はなこさんは賛成でしたね。皆さんはどうですか?")
    said = [u.text for u in sim.eng.state.utterances if u.source == "ai"][1:]
    assert all("はなこ" not in t for t in said)  # 生成しながらも、全文を待っても、名前を含む発話は話さない
