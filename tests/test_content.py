"""介入ポリシー⑥ 内容支援のテスト(アクション・権限制御、プロンプトの出し分け、発話本文の検査)。API は呼ばない。"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from facilitator.agent.agent import FacilitatorAgent  # noqa: E402
from facilitator.agent.content_check import content_flags  # noqa: E402
from facilitator.agent.prompts import (  # noqa: E402
    AI_RESTRICTION,
    JOB_CONTENT_RULES,
    JOB_NEUTRAL_RULES,
    ROLE_DONT,
    build_output_schema,
    build_system_prompt,
)
from facilitator.discussion_types import DISCUSSION_TYPES, get_discussion_type  # noqa: E402
from facilitator.models import Action  # noqa: E402
from facilitator.policy import CONTENT_ACTIONS, ContentInitiative, ContentScope, InterventionPolicy  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402

ORIGINAL_ACTIONS = [  # 第3段階より前の出力スキーマの選択肢(この順)
    "no_intervention", "start_gd", "ask_question", "redirect_topic", "organize_issues", "compare_criteria",
    "invite_all", "invite_person", "return_to_speaker", "time_warning", "close_gd",
]
CONTENT_POLICIES = [  # ⑥ の有効な7通り
    InterventionPolicy(content_scope=s, content_initiative=i)
    for s, i in itertools.product(ContentScope, ContentInitiative)
    if s != ContentScope.none or i == ContentInitiative.none
]


class StubLLM:
    name = "stub"

    def __init__(self, action: str = "no_intervention", speech: str = "", health: str = "progressing") -> None:
        self.action, self.speech, self.health = action, speech, health
        self.calls, self.user, self.context, self.schema, self.system = 0, "", {}, None, ""

    def generate_json(self, system, user, schema, context):
        self.calls += 1
        self.system, self.user, self.context, self.schema = system, user, context, schema
        return {
            "analysis": {"phase": "cause_analysis", "phase_reason": "", "health": self.health, "issues": [], "opinions": [],
                         "conflicts": [], "agreements": [], "missing_perspectives": [], "recent_summary": ""},
            "decision": {"action": self.action, "reasoning": "r", "speech": self.speech, "urgency": 0.5, "target_speaker": None},
        }


def make(policy: InterventionPolicy, llm=None, purpose: str = "job"):
    s = load_settings()
    st = DiscussionState(settings=s, theme="大学の学食の混雑を改善するには？", duration_sec=600)
    st.started, st.start_time = True, 0.0
    st.observed_phase, st.phase_observed_at = "cause_analysis", 0.0
    for sid, n in {"speaker_1": "たろう", "speaker_2": "はなこ", "speaker_3": "けんた"}.items():
        st.set_speaker_name(sid, n)
    for i, t in enumerate(range(200, 260, 10)):
        st.add_utterance(f"speaker_{i % 3 + 1}", "原因は時間の集中ですね。", t, t + 5, 0.9)  # 最後は 250〜255
    return st, FacilitatorAgent(st, llm or StubLLM(), s, purpose=purpose, policy=policy)


# ---------- 出力スキーマ: 範囲の外の内容支援のアクションは選択肢にも入れない ----------
@pytest.mark.parametrize("scope,extra", [("none", []), ("hint", ["offer_hint"]), ("concrete", ["offer_hint", "propose_idea"])])
def test_output_schema_actions_by_scope(scope, extra):
    enum = build_output_schema(get_discussion_type(), scope)["properties"]["decision"]["properties"]["action"]["enum"]
    assert enum == ORIGINAL_ACTIONS + extra


def test_job_gd_agent_uses_original_schema_and_prompt():
    st, agent = make(InterventionPolicy())
    assert agent.output_schema == build_output_schema(agent.dtype)
    assert agent.output_schema["properties"]["decision"]["properties"]["action"]["enum"] == ORIGINAL_ACTIONS
    assert agent.system_prompt == build_system_prompt("job", "problem_solving")


# ---------- プロンプトの出し分け(内容支援なしなら従来どおり、ありなら中立のルールと矛盾させない) ----------
@pytest.mark.parametrize("purpose", ["job", "general"])
@pytest.mark.parametrize("dtype", list(DISCUSSION_TYPES))
def test_prompt_by_content_scope(purpose, dtype):
    base = build_system_prompt(purpose, dtype)
    assert build_system_prompt(purpose, dtype, content=("none", "none")) == base
    for scope, initiative in [("hint", "none"), ("hint", "stalled"), ("concrete", "proactive")]:
        p = build_system_prompt(purpose, dtype, content=(scope, initiative))
        assert ROLE_DONT[scope] in p and ROLE_DONT["none"] not in p
        assert not AI_RESTRICTION.search(p), "フェーズ定義に「(AI は〜しない)」が残っている"
        assert "# 内容支援(主催者の設定" in p
        assert ("propose_idea:" in p) == (scope == "concrete")
        assert JOB_NEUTRAL_RULES not in p  # 「参加者が出していない観点を持ち込まない」と矛盾させない
        assert (JOB_CONTENT_RULES in p) == (purpose == "job")
        assert "議論の進め方や結論の出来を評価・称賛しないでください" in p or purpose == "general"  # 評価しないルールは残す
        t = DISCUSSION_TYPES[dtype]
        if t.support_guidance:
            assert t.support_guidance[scope] in p and t.guidance not in p


def test_neutral_job_prompt_keeps_ai_restrictions():
    p = build_system_prompt("job", "idea_generation")
    assert "(AI は案を出さない)" in p and JOB_NEUTRAL_RULES in p and "# 内容支援" not in p


# ---------- 権限制御 ----------
def test_scope_none_never_allows_content_actions():
    st, agent = make(InterventionPolicy())
    assert not CONTENT_ACTIONS & set(agent._gate(258.0)[1])


@pytest.mark.parametrize("policy", CONTENT_POLICIES, ids=lambda p: f"{p.content_scope.value}-{p.content_initiative.value}")
def test_allowed_content_actions_stay_within_scope(policy):
    st, agent = make(policy)
    allowed = set(agent._gate(258.0)[1])
    assert allowed & CONTENT_ACTIONS <= set(policy.content_actions)
    assert Action.propose_idea not in allowed  # 具体案は、依頼か停滞の段階を踏んだときだけ(自発的にいきなり出さない)
    assert (Action.offer_hint in allowed) == (policy.content_initiative == ContentInitiative.proactive)


@pytest.mark.parametrize("chosen", ["offer_hint", "propose_idea"])
def test_validate_rejects_content_action_not_allowed(chosen):
    """LLM が許可されていない内容支援を選んでも、発話させない(範囲外・自発性の条件外)。"""
    for policy in (InterventionPolicy(), InterventionPolicy(content_scope="hint")):
        st, agent = make(policy, StubLLM(chosen, "例えば、券売機を増やす案もあります。"))
        rec = agent.tick(258.0)
        assert rec.decision.action == Action.no_intervention and "許可外" in rec.overridden


def test_proactive_hint_has_gap_and_session_cap():
    s = load_settings()
    llm = StubLLM("offer_hint", "利用する人の立場から見るとどうでしょう?")
    st, agent = make(InterventionPolicy(content_scope="hint", content_initiative="proactive"), llm)
    t, spoken = 258.0, []
    for _ in range(12):  # 50秒ごと(クールダウン45秒は越える)に判断させる
        rec = agent.tick(t)
        if rec.decision.action == Action.offer_hint:
            spoken.append(t)
        st.add_utterance("speaker_1", "なるほど。", t + 10, t + 12, 0.9)
        t += 50
    assert len(spoken) == s.content_max_spontaneous
    assert all(b - a >= s.content_cooldown_sec for a, b in zip(spoken, spoken[1:]))
    assert agent.content_log[0][2] == "proactive"


def test_hint_sentences_are_limited():
    llm = StubLLM("offer_hint", "一つ目です。二つ目です。三つ目です。")
    st, agent = make(InterventionPolicy(content_scope="hint", content_initiative="proactive"), llm)
    rec = agent.tick(258.0)
    assert rec.decision.speech == "一つ目です。二つ目です。" and "2文に短縮" in rec.overridden


# ---------- 発話本文の検査(記録だけ。判断は変えない) ----------
@pytest.mark.parametrize(
    "action,speech,expected",
    [
        (Action.ask_question, "どんな時間帯に一番混みますか?", []),
        (Action.ask_question, "例えば、券売機を増やすのはどうでしょう?", ["案の提案", "例示"]),
        (Action.organize_issues, "時間と会計の2つが出ていますね。私はキャッシュレスの方がいいと思います。", ["意見"]),
        (Action.ask_question, "利用者の立場から考えるとどうですか?", ["観点の提示"]),
        (Action.offer_hint, "利用者の立場から考えるとどうでしょう?", []),  # ヒントは観点まで
        (Action.offer_hint, "券売機を増やすのはどうでしょう?", ["案の提案"]),  # ヒントに具体案は含めない
        (Action.propose_idea, "例えば、時間帯ごとの割引という案も考えられます。皆さんはどう思いますか?", []),
        (Action.propose_idea, "時間帯割引がおすすめです。", ["推奨"]),
        (Action.time_warning, "残り2分です。例えば何を決めますか?", []),  # 強制イベントは対象外
    ],
)
def test_content_flags(action, speech, expected):
    assert content_flags(action, speech) == expected


def test_flags_are_recorded_without_changing_decision():
    st, agent = make(InterventionPolicy(), StubLLM("ask_question", "例えば、券売機を増やすのはどうでしょう?", health="stalled"))
    rec = agent.tick(270.0)
    assert rec.decision.action == Action.ask_question and rec.content_flags == ["案の提案", "例示"]
