"""プロンプトの方針(中立性・過剰介入防止)が欠けていないか、用途別に正しく組み立てられるかのテスト。API は呼ばない。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from facilitator.agent.agent import FacilitatorAgent  # noqa: E402
from facilitator.agent.prompts import (  # noqa: E402
    COMMON_PROMPT,
    PURPOSE_INTRO,
    PURPOSE_RULES,
    SYSTEM_PROMPT,
    build_phase_block,
    build_system_prompt,
)
from facilitator.discussion_types import get_discussion_type  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402

# 就活GD専用のルール(一般モードには入れない)
JOB_ONLY = ["参加者が出していない基準や観点を持ち込まないでください", "議論の進め方や結論の出来を評価・称賛しないでください"]


def _line(prefix: str) -> str:
    return next(line for line in COMMON_PROMPT.splitlines() if line.startswith(prefix))


@pytest.mark.parametrize("purpose", list(PURPOSE_INTRO))
def test_prompt_is_intro_common_phases_and_rules(purpose):
    block = build_phase_block(get_discussion_type(purpose=purpose))
    assert build_system_prompt(purpose) == PURPOSE_INTRO[purpose] + COMMON_PROMPT + block + PURPOSE_RULES[purpose]


def test_default_and_unknown_purpose_fall_back_to_job():
    assert build_system_prompt() == SYSTEM_PROMPT == build_system_prompt("unknown")


def test_common_policy_is_purpose_neutral():
    assert not [w for w in ("就活", "学生", "選考", "採点") if w in COMMON_PROMPT]


def test_no_self_contradicting_phrases():
    # 「結論への誘導」は禁止事項(特定の結論へ誘導)と矛盾し、「具体的な切り口を示す」は AI 独自の枠組みを招いた
    assert "結論への誘導" not in SYSTEM_PROMPT
    assert "具体的な切り口を示す" not in SYSTEM_PROMPT


def test_common_policy_items_present():
    warn = _line("- time_warning:")  # 1. 時間告知: 中立・簡潔
    assert "どちらかの意見を前提にせず" in warn and "残り時間と次に決めるべき事項" in warn and "要約はしません" in warn
    assert "それだけでは介入の理由になりません" in COMMON_PROMPT  # 2. 過剰介入の防止
    assert "質問や整理は参加者の発言を起点に" in COMMON_PROMPT  # 3. 中立性(共通部分)
    assert "多数派の意見へ誘導する" in COMMON_PROMPT
    assert "均等化ではなく" in _line("- 発言の偏り:")  # 4. 発言の偏り
    assert "同意は求めない" in _line("- invite_person / invite_all では")  # 中立性と具体性の両立
    assert "話題(何について話しているか)には具体的に触れ" in _line("- invite_person / invite_all では")
    assert "建設的なので見守る" in _line("- 膠着(conflicting):")  # 5. 対立・膠着
    assert "意見が多いだけでは介入しない" in _line("- 論点の散乱:")  # 6. 論点整理
    close = _line("- close_gd:")  # 7. 終了: 簡潔・締めは最後の一文
    assert "実際に合意した結論" in close and "新たな結論を付け加えて" in close
    assert "最後の一文で終了を告げます" in close and "経過は要約しません" in close and "重要な未決事項があるときだけ" in close


def test_invite_examples_separate_topic_from_majority():
    good, bad = _line("  良い例:「優先順位").split(" 悪い例:")
    assert "流れ" not in good and "第一候補" not in good  # 良い例は話題に触れるだけ
    assert "流れ" in bad  # 悪い例は多数派の結論を示している


def test_job_only_rules_not_applied_to_general():
    job, general = build_system_prompt("job"), build_system_prompt("general")
    for rule in JOB_ONLY:
        assert rule in job and rule not in general and rule not in COMMON_PROMPT
    assert "評価" not in _line("- close_gd:")  # 終了時の評価禁止は就活GD専用


class CaptureLLM:
    name = "capture"

    def __init__(self, action: str) -> None:
        self.action, self.system, self.user, self.context = action, "", "", {}

    def generate_json(self, system, user, schema, context):
        self.system, self.user, self.context = system, user, context
        return {
            "analysis": {
                "phase": "conclusion",
                "health": "conflicting",
                "issues": [],
                "opinions": [],
                "conflicts": ["全部実施するか、優先順位をつけるか"],
                "agreements": [],
                "missing_perspectives": [],
                "recent_summary": "",
            },
            "decision": {"action": self.action, "reasoning": "r", "speech": "残り2分です。", "urgency": 1.0, "target_speaker": None},
        }


@pytest.mark.parametrize("purpose", list(PURPOSE_INTRO))
def test_time_warning_prompt_carries_unresolved_conflict(purpose):
    """時間告知の判断時に、用途に応じたシステムプロンプトと未解決の対立(前回の分析)が LLM に渡ること。"""
    s = load_settings()
    st = DiscussionState(settings=s, theme="テーマ", duration_sec=600)
    st.started, st.start_time = True, 0.0
    st.add_utterance("speaker_3", "全部やればいいんじゃないですか？", 470, 473, 0.9)
    st.add_utterance("speaker_4", "でも優先順位はつけたほうがいいですよね。", 474, 477, 0.9)
    st.analysis = st.analysis.model_copy(update={"conflicts": ["全部実施するか、優先順位をつけるか"]})
    llm = CaptureLLM("time_warning")
    rec = FacilitatorAgent(st, llm, s, purpose=purpose).tick(482)
    assert rec.gate == "forced:time_warning" and rec.decision.action.value == "time_warning"
    assert llm.context["allowed_actions"] == ["time_warning"]
    assert llm.system == build_system_prompt(purpose)
    assert "どちらかの意見を前提にせず" in llm.system
    assert "全部実施するか、優先順位をつけるか" in llm.user
