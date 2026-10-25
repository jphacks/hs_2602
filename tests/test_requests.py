"""介入ポリシー⑥ 内容支援のテスト(参加者からの依頼の検知、自発性と停滞時の段階的な支援)。API は呼ばない。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ROOT, load_settings  # noqa: E402
from facilitator.agent.agent import FacilitatorAgent  # noqa: E402
from facilitator.agent.request_detect import detect_request  # noqa: E402
from facilitator.llm.mock_llm import MockLLM  # noqa: E402
from facilitator.models import Action  # noqa: E402
from facilitator.policy import InterventionPolicy  # noqa: E402
from facilitator.scenario import load_scenario, run_scenario, scenario_policy  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402
from tests.test_content import StubLLM  # noqa: E402

HINT = InterventionPolicy(content_scope="hint")  # 依頼されたときだけ
GW = InterventionPolicy(content_scope="hint", content_initiative="stalled")
CONCRETE = InterventionPolicy(content_scope="concrete", content_initiative="stalled")


# ---------- 依頼の検知(参加者どうしの質問と区別する) ----------
@pytest.mark.parametrize(
    "text",
    [
        "AIさん、何かヒントもらえますか？",
        "進行役さん、どう思いますか?",
        "ファシリテーターさん、アイデアをください。",
        "エーアイさん、助けてください。",
        "ちょっと行き詰まったので、AIさんに聞きたいです。",
        "司会の方、何か考える切り口ありますか？",
    ],
)
def test_detects_explicit_request_to_facilitator(text):
    assert detect_request(text)


@pytest.mark.parametrize(
    "text",
    [
        "けんたさん、どう思いますか？",  # 参加者どうしの質問
        "何かいいアイデアありますかね？",  # 誰に向けたかわからない(全体への問いかけ)
        "AIを使って予約できるようにしたらどうですか？",  # AI を話題にしているだけ
        "AIさんが言ってたように、時間の集中が大きいですね。",  # 伝聞
        "さっき進行役さんの質問にあった、優先順位の話ですけど？",  # 言及
        "AIさん、ありがとうございます。",  # 依頼の表現がない
        "AIさんたちはどう思いますか？",  # AI 参加者たちへの呼びかけ
    ],
)
def test_does_not_detect_non_requests(text):
    assert detect_request(text) is None


def test_message_naming_an_ai_participant_is_not_a_facilitator_request():
    assert detect_request("サクラさん、AIさんとしてはどう思いますか？", ["サクラ"]) is None
    assert detect_request("AIさん、どう思いますか？", ["サクラ"])


# ---------- 依頼への応答(進行役) ----------
def make(policy: InterventionPolicy, llm=None):
    s = load_settings()
    st = DiscussionState(settings=s, theme="次の学園祭の企画を決めよう", duration_sec=600)
    st.started, st.start_time = True, 0.0
    st.observed_phase, st.phase_observed_at = "cause_analysis", 0.0
    for sid, n in {"speaker_1": "たろう", "speaker_2": "はなこ", "speaker_3": "けんた"}.items():
        st.set_speaker_name(sid, n)
    st.add_utterance("speaker_1", "企画、何がいいですかね。", 200, 204, 0.9)
    return st, FacilitatorAgent(st, llm or StubLLM(), s, purpose="general", policy=policy)


def ask(st, t: float = 210.0, text: str = "AIさん、何かヒントもらえますか？", source: str = "human"):
    if source == "bot":
        st.bot_ids.add("speaker_3")
        return st.add_utterance("speaker_3", text, t, t + 3, 1.0, source="bot")
    return st.add_utterance("speaker_2", text, t, t + 3, 0.9)


def test_scope_none_does_not_detect_requests():
    """就活GD(内容支援なし)は依頼の検知もしない(従来どおり、通常の判断のタイミングで発話ログとして見るだけ)。"""
    st, agent = make(InterventionPolicy())
    ask(st)
    assert agent.should_tick(216.0) != "request" and agent.pending_request is None


def test_request_triggers_after_short_pause_and_allows_scope_actions():
    llm = StubLLM("offer_hint", "体験する人の立場から考えるとどうでしょう?")
    st, agent = make(HINT, llm)
    agent.last_tick_time = 212.0
    ask(st)
    assert agent.should_tick(214.0) is None  # 2秒の間がない(発話の直後)
    assert agent.should_tick(215.5) == "request"
    rec = agent.tick(215.5, "request")
    assert rec.content_reason == "request" and rec.decision.action == Action.offer_hint
    assert "offer_hint" in rec.allowed_actions and "propose_idea" not in rec.allowed_actions  # 範囲の外は許可しない
    assert "# 進行役への依頼" in llm.user and "はなこ: AIさん、何かヒントもらえますか？" in llm.user
    assert agent.pending_request is None  # 1つの依頼には1回だけ
    st.add_utterance("speaker_1", "なるほど。", 230, 232, 0.9)
    assert "offer_hint" not in agent._gate(300.0)[1]  # 依頼されたときだけ: 依頼がなければ許可しない


def test_request_bypasses_cooldown_but_not_short_pause():
    st, agent = make(HINT, StubLLM("ask_question", "どんな企画が浮かびそうですか?"))
    agent.tick(206.0)  # 介入した直後(クールダウン45秒の中)
    ask(st, 210.0)  # 213 秒に言い終わる
    agent._scan_requests(214.5)
    assert agent.pending_request is not None
    assert agent._gate(214.5)[0] == "blocked:short_pause"  # 依頼でも、言い終わってから2秒は待つ
    gate, allowed, note = agent._gate(215.5)
    # 直前の介入から10秒(クールダウン中)でも依頼には応じる。許可は見送りと範囲の中の内容支援だけ
    assert gate == "llm:request" and allowed == [Action.no_intervention, Action.offer_hint] and "依頼に応じることだけ" in note


def test_non_request_is_cleared_after_one_llm_judgment():
    """依頼の候補でも、LLM が依頼ではないと判断すれば(no_intervention)、その後は通常どおり。"""
    st, agent = make(HINT, StubLLM())
    ask(st, text="AIさん、それってどういう意味ですか？")
    rec = agent.tick(215.5, "request")
    assert rec.decision.action == Action.no_intervention and agent.pending_request is None
    assert agent.should_tick(216.0) is None


def test_old_request_expires_and_bot_messages_are_ignored():
    st, agent = make(HINT)
    ask(st)
    agent._scan_requests(250.0)  # 30秒以上たった依頼には応じない
    assert agent.pending_request is None
    st2, agent2 = make(HINT)
    ask(st2, source="bot")  # AI 参加者の発言は依頼として扱わない
    agent2._scan_requests(215.5)
    assert agent2.pending_request is None


def test_request_for_concrete_allows_idea_without_ladder():
    st, agent = make(CONCRETE, StubLLM("propose_idea", "例えば、謎解きの企画という案も考えられます。皆さんはどう思いますか?"))
    ask(st, text="AIさん、具体的な案を何か出してもらえますか？")
    rec = agent.tick(215.5, "request")
    assert rec.decision.action == Action.propose_idea and agent.content_log[-1][2] == "request"


def test_requests_do_not_count_toward_spontaneous_cap():
    st, agent = make(GW, StubLLM("offer_hint", "体験する人の立場から考えるとどうでしょう?"))
    for i in range(4):
        t = 210.0 + i * 20
        ask(st, t)
        assert agent.tick(t + 5.5, "request").decision.action == Action.offer_hint
    assert agent._spontaneous_ok(400.0)


# ---------- 自発性と停滞時の段階的な支援 ----------
def stall_tick(agent, st, t: float, llm, action: str, speech: str = "どんな企画が浮かびそうですか?"):
    llm.action, llm.speech, llm.health = action, speech, "stalled"
    rec = agent.tick(t)
    st.add_utterance("speaker_1", "うーん…。", t + 10, t + 12, 0.9)
    return rec


def test_stalled_ladder_process_then_hint_then_idea():
    llm = StubLLM()
    st, agent = make(CONCRETE, llm)
    r1 = stall_tick(agent, st, 220.0, llm, "ask_question")
    assert not {"offer_hint", "propose_idea"} & set(r1.allowed_actions)  # 1回目は進行支援だけ
    r2 = stall_tick(agent, st, 270.0, llm, "offer_hint", "体験する人の立場から考えるとどうでしょう?")
    assert "offer_hint" in r2.allowed_actions and "propose_idea" not in r2.allowed_actions
    assert r2.decision.action == Action.offer_hint and r2.content_reason == "stalled"
    r3 = stall_tick(agent, st, 330.0, llm, "propose_idea")
    assert "propose_idea" not in r3.allowed_actions  # 内容支援どうしの間隔(120秒)の中
    r4 = stall_tick(agent, st, 400.0, llm, "propose_idea", "例えば、謎解きの企画という案も考えられます。皆さんはどう思いますか?")
    assert r4.decision.action == Action.propose_idea


def test_ladder_resets_when_discussion_moves_again():
    llm = StubLLM()
    st, agent = make(CONCRETE, llm)
    stall_tick(agent, st, 220.0, llm, "ask_question")
    assert agent.support_step == 1
    llm.action, llm.health = "no_intervention", "progressing"
    agent.tick(270.0)
    assert agent.support_step == 0
    st.add_utterance("speaker_1", "うーん…。", 280, 282, 0.9)
    r = stall_tick(agent, st, 320.0, llm, "ask_question")  # 前回は進んでいた → また進行支援から
    assert "offer_hint" not in r.allowed_actions


@pytest.mark.parametrize("initiative,expect_hint", [("none", False), ("stalled", True), ("proactive", True)])
def test_initiative_controls_spontaneous_hint_in_stall(initiative, expect_hint):
    llm = StubLLM()
    st, agent = make(InterventionPolicy(content_scope="hint", content_initiative=initiative), llm)
    stall_tick(agent, st, 220.0, llm, "ask_question")
    r = stall_tick(agent, st, 270.0, llm, "no_intervention")
    assert ("offer_hint" in r.allowed_actions) == expect_hint


def test_scope_hint_never_allows_idea_even_in_long_stall():
    llm = StubLLM()
    st, agent = make(GW, llm)
    stall_tick(agent, st, 220.0, llm, "ask_question")
    stall_tick(agent, st, 270.0, llm, "offer_hint", "体験する人の立場から考えるとどうでしょう?")
    r = stall_tick(agent, st, 400.0, llm, "propose_idea", "例えば謎解きの企画はどうでしょう。")
    assert "propose_idea" not in r.allowed_actions and r.decision.action == Action.no_intervention


# ---------- シナリオ(モック): 依頼と段階的な支援の配線 ----------
@pytest.mark.parametrize("scope", ["none", "hint", "concrete"])
@pytest.mark.parametrize("name", ["23_request", "24_stall_ladder"])
def test_content_scenarios_with_mock(name, scope):
    """モックは「依頼か停滞なら、許可されている中で一番踏み込んだ支援を選ぶ」ので、ルール層の許可の配線と一致する。"""
    sc = load_scenario(ROOT / "scenarios" / f"{name}.yaml")
    res = run_scenario(sc, MockLLM(), load_settings(llm_provider="mock"), policy=scenario_policy(sc, overrides={"content_scope": scope}))
    assert res.checks
    assert all(c.passed for c in res.checks), [(c.at, c.record.decision.action.value, c.fail_reason) for c in res.checks]
