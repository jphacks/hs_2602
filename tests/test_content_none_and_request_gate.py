"""レビューの高-1(内容支援「なし」の仕様の統一)と高-2(依頼の候補によるクールダウン回避)のテスト。API は呼ばない。

プロンプトの禁止事項は、プロンプトに入っていることまでしか確かめられない。LLM が実際の発話でそれを守るかは、
Claude での評価(シナリオの再生)で別に確かめる必要がある。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from facilitator.agent.agent import FacilitatorAgent  # noqa: E402
from facilitator.agent.prompts import (  # noqa: E402
    CONTENT_NONE_RULES,
    JOB_NEUTRAL_RULES,
    build_system_prompt,
)
from facilitator.discussion_types import DISCUSSION_TYPES  # noqa: E402
from facilitator.llm.base import LLMError  # noqa: E402
from facilitator.llm.mock_llm import MockLLM  # noqa: E402
from facilitator.models import Action  # noqa: E402
from facilitator.policy import OPTION_HELP, InterventionPolicy, from_knobs, levels, preset_policy  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402
from tests.test_content import StubLLM  # noqa: E402

GW_NONE = preset_policy("general_gw").updated(content_scope="none")


def _prompt(purpose: str, dtype: str, policy: InterventionPolicy) -> str:
    return build_system_prompt(purpose, dtype, None, levels(policy), (policy.content_scope.value, policy.content_initiative.value))


# ---------- 高-1: 内容支援「なし」の仕様を就活GD・一般GW でそろえる ----------
@pytest.mark.parametrize("dtype", list(DISCUSSION_TYPES))
def test_general_scope_none_forbids_new_viewpoints(dtype):
    p = _prompt("general", dtype, GW_NONE)
    assert p.count(CONTENT_NONE_RULES) == 1
    assert "問いかけの形でも同じです" in p  # 問いの形での持ち込みも禁止
    assert "まずは思いついたことから挙げてみませんか?" in p  # 発言がまだないときの進め方の問いかけ
    assert "# 内容支援(主催者の設定" not in p and "offer_hint:" not in p


@pytest.mark.parametrize("dtype", ["problem_solving", "idea_generation", "decision_making", "planning"])
def test_job_scope_none_keeps_existing_neutral_rules(dtype):
    """就活GDは用途別ルールが同じ趣旨を定めているので、プロンプトは変えない。"""
    p = _prompt("job", dtype, InterventionPolicy())
    assert p == build_system_prompt("job", dtype)
    assert JOB_NEUTRAL_RULES in p and CONTENT_NONE_RULES not in p


def test_ui_description_of_none_matches_both_purposes():
    """画面の説明「新しい案や観点は出しません」と、どちらの用途のプロンプトも一致する。"""
    assert "新しい案や観点は出しません" in OPTION_HELP["content_scope"]["none"]
    assert "参加者が出していない基準や観点を持ち込まないでください" in _prompt("job", "problem_solving", InterventionPolicy())
    assert "参加者が出していないアイデア・観点・切り口を持ち込まないでください" in _prompt("general", "generic", GW_NONE)


@pytest.mark.parametrize("scope,initiative", [("hint", "stalled"), ("concrete", "proactive"), ("hint", "none")])
@pytest.mark.parametrize("purpose", ["job", "general"])
def test_content_support_keeps_hint_and_idea(purpose, scope, initiative):
    p = _prompt(purpose, "idea_generation", InterventionPolicy(content_scope=scope, content_initiative=initiative))
    assert CONTENT_NONE_RULES not in p and JOB_NEUTRAL_RULES not in p
    assert "offer_hint:" in p and ("propose_idea:" in p) == (scope == "concrete")


def test_agent_and_legacy_knob_use_none_rules_for_general():
    s = load_settings()
    st = DiscussionState(settings=s, theme="t", duration_sec=600)
    assert CONTENT_NONE_RULES in FacilitatorAgent(st, StubLLM(), s, purpose="general", policy=GW_NONE).system_prompt
    legacy = from_knobs({"suggest": 0}, preset_policy("general_gw"), "general")  # 旧「提案度合い: 控えめ」
    assert CONTENT_NONE_RULES in FacilitatorAgent(st, StubLLM(), s, purpose="general", policy=legacy).system_prompt
    assert CONTENT_NONE_RULES not in FacilitatorAgent(st, StubLLM(), s, purpose="general").system_prompt  # GW 既定はヒントまで


# ---------- 高-2: 依頼の候補によるクールダウン回避は、依頼への応答だけ ----------
HINT = InterventionPolicy(content_scope="hint")
CONCRETE = InterventionPolicy(content_scope="concrete", content_initiative="stalled")
HINT_SPEECH = "体験する人の立場から考えるとどうでしょう?"


def make(policy: InterventionPolicy, llm):
    s = load_settings()
    st = DiscussionState(settings=s, theme="次の学園祭の企画を決めよう", duration_sec=600)
    st.started, st.start_time = True, 0.0
    st.observed_phase, st.phase_observed_at = "divergence", 0.0
    for sid, n in {"speaker_1": "たろう", "speaker_2": "はなこ", "speaker_3": "けんた"}.items():
        st.set_speaker_name(sid, n)
    st.add_utterance("speaker_1", "企画、何がいいですかね。", 190, 194, 0.9)
    agent = FacilitatorAgent(st, llm, s, purpose="general", policy=policy)
    return st, agent


def intervene_then_ask(st, agent, llm, text="AIさん、何かヒントもらえますか？"):
    """200 秒に介入(クールダウン開始)し、210〜213 秒に依頼の候補。"""
    llm.action, llm.speech = "ask_question", "どんな企画が浮かびそうですか?"
    assert agent.tick(200.0).decision.action == Action.ask_question
    st.add_utterance("speaker_2", text, 210, 213, 0.9)


@pytest.mark.parametrize("unrelated", ["ask_question", "redirect_topic", "organize_issues", "compare_criteria", "invite_all", "invite_person"])
def test_unrelated_action_cannot_bypass_cooldown(unrelated):
    llm = StubLLM()
    st, agent = make(HINT, llm)
    intervene_then_ask(st, agent, llm)
    llm.action, llm.speech = unrelated, "皆さんはどう思いますか?"
    rec = agent.tick(215.5, "request")
    assert rec.gate == "llm:request" and rec.allowed_actions == ["no_intervention", "offer_hint"]
    assert rec.decision.action == Action.no_intervention and "許可外" in rec.overridden  # 話さない
    assert len(st.interventions) == 1 and not agent.has_pending_request(215.6)
    assert agent._gate(230.0)[0] == "blocked:cooldown"  # 候補が消えたら通常のクールダウン


def test_legitimate_request_is_answered_within_scope():
    llm = StubLLM()
    st, agent = make(HINT, llm)
    intervene_then_ask(st, agent, llm)
    llm.action, llm.speech = "offer_hint", HINT_SPEECH
    rec = agent.tick(215.5, "request")
    assert rec.decision.action == Action.offer_hint and rec.content_reason == "request"
    assert "依頼に応じることだけ" in llm.user  # 依頼でなければ見送るよう伝える
    llm.action = "propose_idea"  # ヒントまでの設定では具体案は出さない(次の依頼でも)
    st.add_utterance("speaker_3", "AIさん、具体的な案もありますか？", 220, 223, 0.9)
    rec = agent.tick(225.5, "request")
    assert rec.decision.action == Action.no_intervention and "propose_idea" not in rec.allowed_actions


def test_concrete_request_in_cooldown_allows_idea():
    llm = StubLLM()
    st, agent = make(CONCRETE, llm)
    intervene_then_ask(st, agent, llm, "AIさん、具体的な案を1つ出してもらえますか？")
    llm.action, llm.speech = "propose_idea", "例えば、謎解きの企画という案も考えられます。皆さんはどう思いますか?"
    rec = agent.tick(215.5, "request")
    assert rec.allowed_actions == ["no_intervention", "offer_hint", "propose_idea"]
    assert rec.decision.action == Action.propose_idea


def test_false_positive_request_is_released_by_llm_judgment():
    """依頼の候補でも、LLM が依頼ではないと判断すれば見送り、以後は通常のクールダウン(LLM も呼ばない)。"""
    llm = StubLLM()
    st, agent = make(HINT, llm)
    intervene_then_ask(st, agent, llm, "AIさん、って呼び方でいいんですかね？")
    llm.action, llm.speech = "no_intervention", ""
    calls = llm.calls
    rec = agent.tick(215.5, "request")
    assert rec.decision.action == Action.no_intervention and llm.calls == calls + 1
    assert not agent.has_pending_request(215.6) and agent.should_tick(216.0) is None
    assert agent.tick(230.0).gate == "blocked:cooldown" and llm.calls == calls + 1


def test_llm_error_in_cooldown_request_releases_state():
    class Failing(StubLLM):
        def generate_json(self, *a):
            if self.action == "fail":
                raise LLMError("boom")
            return super().generate_json(*a)

    llm = Failing()
    st, agent = make(HINT, llm)
    intervene_then_ask(st, agent, llm)
    llm.action = "fail"
    rec = agent.tick(215.5, "request")
    assert rec.gate == "llm:request" and rec.decision.action == Action.no_intervention and rec.overridden.startswith("LLM失敗")
    assert not agent.has_pending_request(215.6) and agent._gate(230.0)[0] == "blocked:cooldown"


def test_forced_event_keeps_priority_over_request_in_cooldown():
    llm = StubLLM()
    st, agent = make(HINT, llm)
    st.duration_sec = 330  # 時間告知の閾値は min(120, 330×0.35=115.5) 秒 → 214.5 秒から
    intervene_then_ask(st, agent, llm)  # 200 秒の介入(強制イベントの最低間隔20秒は 220 秒まで)
    llm.action, llm.speech = "time_warning", "残り約2分です。"
    agent._scan_requests(215.5)
    assert agent.has_pending_request(215.5)
    assert agent._gate(215.5)[0] == "blocked:forced_gap"  # 依頼の候補でも、時間告知の待ちを飛ばして介入しない
    st.add_utterance("speaker_2", "AIさん、ヒントありますか？", 219, 222, 0.9)
    rec = agent.tick(224.5)
    assert rec.gate == "forced:time_warning" and rec.decision.action == Action.time_warning
    assert not agent.has_pending_request(224.6)


def test_request_outside_cooldown_is_a_normal_judgment():
    """クールダウンの外では従来どおり通常の判断(依頼が理由で内容支援も許可される)。クールダウンは回避していない。"""
    llm = StubLLM("offer_hint", HINT_SPEECH)
    st, agent = make(HINT, llm)
    st.add_utterance("speaker_2", "AIさん、何かヒントもらえますか？", 210, 213, 0.9)
    rec = agent.tick(215.5, "request")
    assert rec.gate == "llm" and "ask_question" in rec.allowed_actions and "offer_hint" in rec.allowed_actions


def test_mock_timeline_request_in_cooldown_and_unrelated_stall():
    """モックの時系列: 介入の直後の依頼にはクールダウン中でも内容支援で応じ、そのあとの停滞(依頼なし)では
    クールダウンが明けるまで判断しない。AI 参加者を待たせる依頼の候補も残らない。"""
    s = load_settings(llm_provider="mock")
    st = DiscussionState(settings=s, theme="次の学園祭の企画を決めよう", duration_sec=900)
    st.started, st.start_time = True, 0.0
    agent = FacilitatorAgent(st, MockLLM(), s, purpose="general", policy=preset_policy("general_gw"))
    script = [(100, 103, "企画を考えましょうか。"), (104, 107, "うーん…。"),
              (125, 128, "AIさん、何かヒントもらえますか？"), (140, 142, "なるほど…。")]
    pending, t, log = list(script), 95.0, []
    while t < 200:
        while pending and pending[0][1] <= t:
            a, b, text = pending.pop(0)
            st.add_utterance("speaker_1", text, a, b, 0.9)
        st.speaking_now = any(a <= t < b for a, b, _ in pending)
        trig = agent.should_tick(t)
        if trig:
            rec = agent.tick(t, trig)
            log.append((t, trig, rec.gate, rec.decision.action.value))
        t += 0.25
    acted = [(t, trig, g, a) for t, trig, g, a in log if a != "no_intervention"]
    assert acted[0][3] == "ask_question"  # 停滞(8秒)で質問
    assert acted[1][1:] == ("request", "llm:request", "offer_hint") and acted[1][0] - acted[0][0] < 45  # クールダウン中の依頼
    first, second = acted[0][0], acted[1][0]
    assert not [x for x in log if second < x[0] < first + 45]  # その後の停滞では、クールダウン中に判断しない
    assert not agent.has_pending_request(200.0)
