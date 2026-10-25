"""議論フェーズ管理(実際のフェーズ / 目標フェーズ)のテスト。LLM は固定応答のスタブを使い、API は呼ばない。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ROOT, load_settings  # noqa: E402
from facilitator.agent.agent import FacilitatorAgent  # noqa: E402
from facilitator.agent.prompts import COMMON_PROMPT, PHASE_BEHIND_NOTE, build_system_prompt, build_user_prompt  # noqa: E402
from facilitator.discussion_types import GENERIC, PROBLEM_SOLVING, get_discussion_type  # noqa: E402
from facilitator.engine import Engine, SessionConfig  # noqa: E402
from facilitator.llm.base import LLMError  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402

PS_KEYS = ["problem_definition", "cause_analysis", "solution_ideation", "evaluation", "conclusion"]


class PhaseLLM:
    """指定したフェーズと行動を返すスタブ。渡されたプロンプトを記録する。"""

    name = "phase-stub"

    def __init__(self, phase: str = "cause_analysis", action: str = "no_intervention", fail: bool = False) -> None:
        self.phase, self.action, self.fail, self.calls, self.user, self.context = phase, action, fail, 0, "", {}

    def generate_json(self, system, user, schema, context):
        self.calls += 1
        self.system, self.user, self.context, self.schema = system, user, context, schema
        if self.fail:
            raise LLMError("boom")
        speech = "" if self.action == "no_intervention" else "ここまでの原因を確認しませんか?"
        return {
            "analysis": {
                "phase": self.phase,
                "phase_reason": "根拠",
                "health": "progressing",
                "issues": [],
                "opinions": [],
                "conflicts": [],
                "agreements": [],
                "missing_perspectives": [],
                "recent_summary": "",
            },
            "decision": {"action": self.action, "reasoning": "r", "speech": speech, "urgency": 0.3, "target_speaker": None},
        }


def make(llm, observed: str = "", purpose: str = "job", duration: float = 600, **overrides):
    s = load_settings(**overrides)
    st = DiscussionState(settings=s, theme="テーマ", duration_sec=duration)
    st.started, st.start_time = True, 0.0
    st.observed_phase, st.phase_observed_at = observed, (0.0 if observed else None)
    for i in range(4):
        st.set_speaker_name(f"speaker_{i + 1}", f"名前{i + 1}")
    return st, FacilitatorAgent(st, llm, s, purpose=purpose)


def talk(st, t_end: float, n: int = 4):
    """t_end に終わる発言を n 件(5秒間隔)追加する。"""
    for i in range(n):
        end = t_end - (n - 1 - i) * 5
        st.add_utterance(f"speaker_{i % 4 + 1}", "発言です。", end - 4, end, 0.9)


# ---------- 定義 ----------
def test_problem_solving_has_five_phases_in_order():
    assert PROBLEM_SOLVING.keys == PS_KEYS
    assert [p.label for p in PROBLEM_SOLVING.phases] == ["課題整理", "原因分析", "解決策の検討", "評価・比較", "結論・まとめ"]
    assert [p.target_end for p in PROBLEM_SOLVING.phases] == [0.15, 0.35, 0.60, 0.80, 1.0]
    assert all(p.signs and p.policy for p in PROBLEM_SOLVING.phases)


@pytest.mark.parametrize(
    "frac, expected",
    [(0.0, "problem_definition"), (0.149, "problem_definition"), (0.15, "cause_analysis"), (0.35, "solution_ideation"),
     (0.6, "evaluation"), (0.8, "conclusion"), (1.0, "conclusion")],
)
def test_target_phase_from_elapsed_fraction(frac, expected):
    assert PROBLEM_SOLVING.target_phase(frac).key == expected


def test_purpose_defaults_and_general_uses_generic():
    assert get_discussion_type(purpose="job") is PROBLEM_SOLVING
    assert get_discussion_type(purpose="general") is GENERIC
    assert GENERIC.keys == ["ideation", "comparison", "conclusion"]  # 既存の3段階
    llm = PhaseLLM("comparison")
    st, ag = make(llm, purpose="general")
    talk(st, 200)
    ag.tick(203)
    assert ag.dtype is GENERIC and llm.schema["properties"]["analysis"]["properties"]["phase"]["enum"] == GENERIC.keys
    assert "# 議論タイプ: 汎用型" in llm.system and "課題解決型" not in llm.system
    assert st.observed_phase == "comparison"


def test_job_prompt_has_problem_solving_block_and_schema():
    p = build_system_prompt("job")
    assert "# 議論タイプ: 課題解決型" in p and "汎用型" not in p
    for ph in PROBLEM_SOLVING.phases:
        assert f"- {ph.key} / {ph.label}: {ph.signs} → {ph.policy}" in p
    # 就活GD専用ルールは議論タイプのブロックの後(末尾)
    assert p.index("# 議論タイプ") < p.index("# 就活GDでの追加ルール")


# ---------- 実際のフェーズ ----------
def test_backtracking_is_allowed():
    llm = PhaseLLM("cause_analysis")
    st, ag = make(llm, observed="evaluation")
    talk(st, 400)
    ag.tick(403)
    assert st.observed_phase == "cause_analysis"  # 評価・比較 → 原因分析 に戻れる


def test_unknown_phase_is_ignored():
    llm = PhaseLLM("ideation")  # 汎用型のフェーズは課題解決型では無効
    st, ag = make(llm, observed="evaluation")
    talk(st, 400)
    ag.tick(403)
    assert st.observed_phase == "evaluation"


def test_time_does_not_force_observed_phase():
    llm = PhaseLLM(fail=True)  # LLM が判断しない(失敗)まま時間だけが進む
    st, ag = make(llm, observed="problem_definition")
    talk(st, 540)
    rec = ag.tick(543)  # 経過90%(目標は結論・まとめ)
    assert st.observed_phase == "problem_definition"
    assert rec.observed_phase == "problem_definition" and rec.target_phase == "conclusion"


def test_observed_and_target_are_independent():
    llm = PhaseLLM("solution_ideation")
    st, ag = make(llm, observed="cause_analysis")
    assert ag.phase_status(30)[0] == "problem_definition" and st.observed_phase == "cause_analysis"
    talk(st, 100)
    rec = ag.tick(103)  # LLM の判断で実際のフェーズだけが変わる
    assert st.observed_phase == "solution_ideation" and rec.target_phase == "cause_analysis"  # 経過17%
    assert ag.phase_status(500)[0] == "conclusion" and st.observed_phase == "solution_ideation"  # 時間で目標だけが変わる


def test_temporary_mention_vs_real_backtrack_instruction():
    assert "一時的に触れただけでは戻さず" in COMMON_PROMPT
    assert "議論の主な目的が前の段階に戻ったとき" in COMMON_PROMPT
    assert "いま参加者が主に何を目的に話しているか" in COMMON_PROMPT


# ---------- 遅れの判定と介入 ----------
@pytest.mark.parametrize(
    "observed, now, gap, behind",
    [
        ("", 400, None, False),  # まだ判断なし
        ("problem_definition", 240, 2, True),  # 40%: 目標は解決策、2段階遅れ
        ("cause_analysis", 300, 1, False),  # 50%: 1段階遅れだが60%未満
        ("solution_ideation", 390, 1, True),  # 65%: 1段階遅れで60%以上
        ("evaluation", 300, -1, False),  # 先行
    ],
)
def test_phase_status_behind(observed, now, gap, behind):
    _, ag = make(PhaseLLM(), observed=observed)
    _, g, b = ag.phase_status(now)
    assert (g, b) == (gap, behind)


def test_behind_alone_does_not_force_intervention():
    llm = PhaseLLM("cause_analysis", action="no_intervention")
    st, ag = make(llm, observed="cause_analysis")
    talk(st, 420)
    rec = ag.tick(423)  # 70%: 目標は評価・比較(2段階遅れ)
    assert rec.phase_behind and rec.gate == "llm"
    assert rec.decision.action.value == "no_intervention" and rec.overridden is None
    assert llm.context["phase_behind"] and PHASE_BEHIND_NOTE in llm.user
    assert "介入は必須ではありません" in PHASE_BEHIND_NOTE


def test_behind_note_only_when_behind():
    llm = PhaseLLM("cause_analysis")
    st, ag = make(llm, observed="cause_analysis")
    talk(st, 150)
    ag.tick(153)  # 25%: 目標どおり
    assert not llm.context["phase_behind"] and PHASE_BEHIND_NOTE not in llm.user


def test_stale_observed_phase_is_not_a_rule_trigger():
    """古い実際のフェーズで大きく遅れていても、ルール層の見送り・トリガーはフェーズに左右されない。"""
    llm = PhaseLLM("cause_analysis", action="ask_question")
    st, ag = make(llm, observed="problem_definition")
    talk(st, 420)
    assert ag.phase_status(421)[2]  # 遅れの条件は満たしている
    assert ag.should_tick(421) is None  # 沈黙1秒: 遅れがあってもトリガーしない
    rec = ag.tick(421)
    assert rec.gate == "blocked:short_pause" and llm.calls == 0
    st.speaking_now = True
    assert ag.tick(430).gate == "blocked:speaking" and llm.calls == 0
    st.speaking_now = False
    assert ag.tick(430).decision.action.value == "ask_question"  # 介入するかは LLM の判断
    assert ag.tick(440).gate == "blocked:cooldown"  # クールダウンも維持


def test_stale_phase_is_labeled_in_prompt():
    llm = PhaseLLM("cause_analysis")
    st, ag = make(llm, observed="cause_analysis")
    talk(st, 200)
    ag.tick(203)
    assert "前回のあなたの判断" in llm.user and "203秒前の判断" in llm.user
    assert "古い可能性があります" in COMMON_PROMPT and "今回判断し直したフェーズで考えて" in COMMON_PROMPT
    assert "目標フェーズより遅れていても見守ります" in COMMON_PROMPT


def test_forced_events_unaffected_by_phase():
    llm = PhaseLLM("problem_definition", action="time_warning")
    st, ag = make(llm, observed="problem_definition")
    talk(st, 480)
    assert ag.tick(483).decision.action.value == "time_warning"
    close = make(PhaseLLM(fail=True), observed="problem_definition")[1].tick(601)
    assert close.decision.action.value == "close_gd" and close.decision.speech


# ---------- シナリオ・UI ----------
def test_scenario_phase_keys_are_valid():
    for p in sorted((ROOT / "scenarios").glob("*.yaml")):
        sc = yaml.safe_load(p.read_text(encoding="utf-8"))
        keys = set(get_discussion_type(sc.get("discussion_type")).keys)
        assert sc.get("initial_phase", "") in keys | {""}, p.name
        for cp in sc["checkpoints"]:
            assert set(cp.get("expect_phase", [])) <= keys, (p.name, cp["at"])


def test_engine_snapshot_exposes_type_phases():
    eng = Engine(load_settings(), llm_provider="mock")
    eng.start(SessionConfig("テーマ", 1, ["A", "B"], mode="text", intro=False, tts=False, discussion_type="problem_solving"))
    try:
        deadline = time.monotonic() + 5
        while not eng.snapshot().get("started") and time.monotonic() < deadline:
            time.sleep(0.05)  # 開始の説明(start_gd)まで待つ
        s = eng.snapshot()
        assert s["phases"] == [[p.key, p.label] for p in PROBLEM_SOLVING.phases]
        assert s["discussion_type"] == "課題解決型" and s["target_phase"] in PROBLEM_SOLVING.keys
        assert "phase" in s
    finally:
        eng.stop()


def test_user_prompt_renders_without_observed_phase():
    _, ag = make(PhaseLLM())
    ctx = ag._context(10, "periodic", [], None)
    text = build_user_prompt(ctx)
    assert "まだ判断なし" in text and "目標フェーズ(経過時間から): 課題整理" in text
