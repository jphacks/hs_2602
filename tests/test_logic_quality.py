"""AI ロジックの品質改善(生成しながら話す処理の記録、内容支援の文の整理、場面が重なるときの方針、評価シナリオ)のテスト。
API は呼ばない。プロンプトに入れた方針を LLM が守るかは、Claude での評価で別に確かめる必要がある。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ROOT, load_settings  # noqa: E402
from facilitator.agent.agent import FacilitatorAgent  # noqa: E402
from facilitator.agent.prompts import (  # noqa: E402
    CONTENT_INITIATIVE_TEXT,
    OVERLAP_NOTE,
    ROLE_LINE,
    ROLE_LINE_CONCRETE,
    build_system_prompt,
)
from facilitator.llm.mock_llm import MockLLM  # noqa: E402
from facilitator.llm.stream import simulate_stream  # noqa: E402
from facilitator.models import Action  # noqa: E402
from facilitator.policy import LEVEL_ITEMS, InterventionPolicy  # noqa: E402
from facilitator.scenario import load_scenario, run_scenario, scenario_policy  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402
from tests.sim import SimSession  # noqa: E402
from tests.test_content import StubLLM  # noqa: E402

STANDARD = dict.fromkeys(LEVEL_ITEMS, "standard")


# ---------- 生成しながら話す処理の記録 ----------
def test_commit_spoken_rechecks_content_flags_on_spoken_text():
    """途中で話すのをやめたときは、LLM の全文ではなく実際に話した文で内容の混入を検査し直す。"""
    s = load_settings()
    st = DiscussionState(settings=s, theme="t", duration_sec=600)
    st.started, st.start_time = True, 0.0
    st.add_utterance("speaker_1", "うーん。", 100, 103, 0.9)
    ag = FacilitatorAgent(st, StubLLM("ask_question", "どの原因から考えますか?例えば券売機を増やすのはどうでしょう?",
                                      health="stalled"), s)
    rec = ag.tick(110.0, act=False)
    assert rec.content_flags == ["案の提案", "例示"]  # 全文には案が混ざっている
    ag.commit_spoken(rec, 110.5, Action.ask_question, "どの原因から考えますか?")  # 1文目だけ話した
    assert rec.content_flags == [] and rec.decision.speech == "どの原因から考えますか?"


class StreamThenFail(StubLLM):
    """生成途中まで流してから、LLM 以外の想定外の失敗をする。"""

    streams = True

    def generate_json(self, system, user, schema, context, on_partial=None):
        out = super().generate_json(system, user, schema, context)
        if self.action != "no_intervention" and on_partial is not None:
            simulate_stream(out, on_partial)
            raise RuntimeError("想定外の失敗")
        return out


def test_spoken_part_is_recorded_when_tick_fails_unexpectedly(monkeypatch):
    """生成しながら話した後に判断が想定外の失敗で中断しても、話した分を介入として記録する(クールダウンが効く)。"""
    llm = StreamThenFail("no_intervention")
    sim = SimSession(monkeypatch, llm, names=("たろう", "はなこ", "けんた"))
    sim.run(5)  # 開始の発言
    llm.action, llm.speech, llm.health = "ask_question", "どの原因から考えますか?", "stalled"
    sim.say(10, "speaker_1", "原因は時間の集中ですね。", 2)
    with pytest.raises(RuntimeError):
        sim.run(60)
    spoken = [iv for iv in sim.eng.state.interventions if iv.action == Action.ask_question]
    assert spoken and spoken[0].speech == "どの原因から考えますか?"
    assert not sim.eng._agent_busy  # 判断中の印も残らない


# ---------- 内容支援の文の整理 ----------
def test_concrete_scope_role_line_matches_content_support():
    p = build_system_prompt("general", "idea_generation", content=("concrete", "stalled"))
    assert ROLE_LINE_CONCRETE in p and ROLE_LINE not in p  # 「答えは与えない」と具体案の例が矛盾しない
    for scope in ("none", "hint"):
        assert ROLE_LINE in build_system_prompt("general", "idea_generation", content=(scope, "stalled" if scope != "none" else "none"))


@pytest.mark.parametrize("initiative", ["stalled", "proactive"])
def test_hint_scope_does_not_mention_ideas(initiative):
    p = build_system_prompt("general", "idea_generation", content=("hint", initiative))
    assert "propose_idea" not in p  # 使えないアクションの説明をしない
    assert "具体案" not in CONTENT_INITIATIVE_TEXT[initiative]["hint"]
    assert "ヒントまでの設定で具体案を求められた" in p  # 範囲を超える依頼への応じ方は残す
    q = build_system_prompt("general", "idea_generation", content=("concrete", initiative))
    assert "propose_idea" in q


# ---------- 場面が重なるとき ----------
def test_gd_standard_prompt_has_no_overlap_note():
    """就活GD標準(すべて標準・内容支援なし)のプロンプトは変えない。一般GW の「なし」でも①〜⑤が標準なら入れない。"""
    for dtype in ("problem_solving", "idea_generation", "decision_making", "planning"):
        assert OVERLAP_NOTE not in build_system_prompt("job", dtype)
        assert OVERLAP_NOTE not in build_system_prompt("job", dtype, levels=STANDARD)
    assert OVERLAP_NOTE not in build_system_prompt("general", "generic", levels=STANDARD)


@pytest.mark.parametrize("item", LEVEL_ITEMS)
@pytest.mark.parametrize("level", ["low", "high"])
def test_overlap_note_when_levels_or_content_change(item, level):
    p = build_system_prompt("job", "problem_solving", levels={**STANDARD, item: level})
    assert p.count(OVERLAP_NOTE) == 1 and p.index(OVERLAP_NOTE) < p.index("# 発話のルール")
    assert build_system_prompt("job", "problem_solving", content=("hint", "none")).count(OVERLAP_NOTE) == 1


# ---------- 評価シナリオ ----------
@pytest.mark.parametrize("name", ["27_content_none_empty_start", "28_gd_request_no_content", "29_proactive_smooth"])
def test_new_boundary_scenarios_reach_llm_with_mock(name):
    """新しい境界のシナリオは、ルール層を通って LLM が判断する(質の評価は Claude で行う)。"""
    sc = load_scenario(ROOT / "scenarios" / f"{name}.yaml")
    res = run_scenario(sc, MockLLM(), load_settings(llm_provider="mock"), policy=scenario_policy(sc))
    assert res.checks and all(c.record.llm_called and c.passed for c in res.checks), [
        (c.at, c.record.gate, c.record.decision.action.value, c.fail_reason) for c in res.checks
    ]


def test_scenario_28_runs_as_gd_without_content_actions():
    sc = load_scenario(ROOT / "scenarios" / "28_gd_request_no_content.yaml")
    res = run_scenario(sc, MockLLM(), load_settings(llm_provider="mock"), policy=scenario_policy(sc))
    assert scenario_policy(sc) == InterventionPolicy()  # 就活GD の既定(内容支援なし)
    assert all("offer_hint" not in c.record.allowed_actions for c in res.checks)
