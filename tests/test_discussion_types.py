"""議論タイプ(課題解決型・アイデア創出型・意思決定型・計画立案型)の選択と、プロンプト・フェーズ管理・API・画面への反映のテスト。
API は呼ばない。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.server as server  # noqa: E402
from config import ROOT, load_settings  # noqa: E402
from facilitator.agent.agent import FacilitatorAgent  # noqa: E402
from facilitator.agent.prompts import build_phase_block, build_system_prompt  # noqa: E402
from facilitator.discussion_types import (  # noqa: E402
    CASUAL_FREE_THEME,
    DISCUSSION_TYPES,
    GENERIC,
    PROBLEM_SOLVING,
    SELECTABLE_TYPES,
    resolve_session_type,
)
from facilitator.engine import Engine, SessionConfig  # noqa: E402
from facilitator.scenario import load_scenario, run_scenario  # noqa: E402
from facilitator.models import Action  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402

EXPECTED = {
    "problem_solving": ["課題整理", "原因分析", "解決策の検討", "評価・比較", "結論・まとめ"],
    "idea_generation": ["目的確認", "アイデア発散", "整理・発展", "まとめ"],
    "decision_making": ["選択肢確認", "評価基準", "比較", "決定"],
    "planning": ["目標整理", "作業洗い出し", "計画具体化", "確認"],
}
NEUTRAL_RULE = "アイデア・選択肢・作業や計画についても同じく、参加者が出していないものを持ち込まないでください"
NEW_TYPES = ["idea_generation", "decision_making", "planning", "casual"]


# ---------- 定義 ----------
def test_four_selectable_types_and_phases():
    assert SELECTABLE_TYPES == list(EXPECTED)
    for key, labels in EXPECTED.items():
        t = DISCUSSION_TYPES[key]
        assert [p.label for p in t.phases] == labels
        ends = [p.target_end for p in t.phases]
        assert ends == sorted(ends) and ends[-1] == 1.0 and len(set(ends)) == len(ends)  # 目標の終了時点は単調増加
        assert all(p.signs and p.policy for p in t.phases)
        assert len({p.key for p in t.phases}) == len(t.phases)


@pytest.mark.parametrize("key", NEW_TYPES)
def test_new_types_have_purpose_and_guidance(key):
    t = DISCUSSION_TYPES[key]
    assert t.guidance and all(p.description for p in t.phases)
    assert "参加者" in t.guidance  # 内容を決めるのは参加者


def test_problem_solving_definition_unchanged():
    """既存の課題解決型は、目的・要点の欄を空のままにして、プロンプトのフェーズ定義を従来と同じにする。"""
    assert not PROBLEM_SOLVING.guidance and not any(p.description for p in PROBLEM_SOLVING.phases)
    block = build_phase_block(PROBLEM_SOLVING)
    assert "目的:" not in block and block.count("\n- ") == 5


# ---------- 手動指定(暗黙の割り当てをしない) ----------
def test_job_requires_explicit_type():
    with pytest.raises(ValueError, match="議論タイプを選択"):
        resolve_session_type(None, "job")
    with pytest.raises(ValueError):
        resolve_session_type("generic", "job")  # 就活GDでは4種類から選ぶ
    with pytest.raises(ValueError):
        resolve_session_type("chat", "job")  # 雑談型などは対象外
    for k in SELECTABLE_TYPES:
        assert resolve_session_type(k, "job").key == k


def test_general_mode_compatible():
    assert resolve_session_type(None, "general") is GENERIC  # 従来どおり汎用型
    assert resolve_session_type("decision_making", "general").key == "decision_making"


# ---------- プロンプト・出力形式 ----------
@pytest.mark.parametrize("key", SELECTABLE_TYPES)
def test_prompt_and_schema_follow_selected_type(key):
    t = DISCUSSION_TYPES[key]
    p = build_system_prompt("job", key)
    assert f"# 議論タイプ: {t.label}" in p and NEUTRAL_RULE in p  # 中立性は全タイプ共通(就活GD)
    for ph in t.phases:
        assert f"- {ph.key} / {ph.label}" in p
    other = [DISCUSSION_TYPES[k].label for k in SELECTABLE_TYPES if k != key]
    assert not any(f"# 議論タイプ: {o}" in p for o in other)
    st = DiscussionState(settings=load_settings(), theme="テーマ", duration_sec=600)
    ag = FacilitatorAgent(st, None, load_settings(), discussion_type=key)
    assert ag.output_schema["properties"]["analysis"]["properties"]["phase"]["enum"] == t.keys


def test_general_prompt_has_no_job_rules():
    assert NEUTRAL_RULE not in build_system_prompt("general", "planning")


@pytest.mark.parametrize("key", NEW_TYPES)
def test_phase_management_shared_across_types(key):
    """後戻りあり・時間で実際のフェーズを動かさない・遅れの判定、が全タイプで同じように働く。"""
    t = DISCUSSION_TYPES[key]
    st = DiscussionState(settings=load_settings(), theme="テーマ", duration_sec=600)
    st.started, st.start_time = True, 0.0
    ag = FacilitatorAgent(st, None, load_settings(), discussion_type=key)
    st.update_phase(t.keys[2], t.keys, 100)
    st.update_phase(t.keys[1], t.keys, 110)  # 後戻り
    assert st.observed_phase == t.keys[1]
    st.update_phase("cause_analysis", t.keys, 120)  # 他タイプのフェーズは無視
    assert st.observed_phase == t.keys[1]
    target, gap, behind = ag.phase_status(590)
    assert target == t.keys[-1] and gap == len(t.keys) - 2 and behind and st.observed_phase == t.keys[1]


# ---------- エンジン・API・画面 ----------
@pytest.fixture
def engine():
    eng = Engine(load_settings(), llm_provider="mock", tts_chars_per_sec=1000.0)
    yield eng
    eng.stop()


def _cfg(**kw):
    return SessionConfig(kw.pop("theme", "テーマ"), 5, ["A", "B"], mode="text", intro=False, tts=False, **kw)


@pytest.mark.parametrize("key", SELECTABLE_TYPES)
def test_engine_snapshot_shows_selected_type(engine, key):
    engine.start(_cfg(discussion_type=key))
    s = engine.snapshot()
    assert s["discussion_type_key"] == key and s["discussion_type"] == DISCUSSION_TYPES[key].label
    assert [label for _, label in s["phases"]] == EXPECTED[key]
    assert f"# 議論タイプ: {DISCUSSION_TYPES[key].label}" in engine.agent.system_prompt


def test_engine_rejects_missing_type_without_stopping_running_session(engine):
    engine.start(_cfg(discussion_type="planning"))
    with pytest.raises(ValueError):
        engine.start(_cfg())  # 就活GDで未選択
    assert engine.status == "running" and engine.agent.dtype.key == "planning"  # 走っているセッションはそのまま


def test_api_start_validates_theme_and_type(engine, monkeypatch):
    monkeypatch.setattr(server, "engine", engine)
    base = dict(minutes=5, names=["A", "B"], mode="text", intro=False, tts=False)
    r = server.start(server.StartReq(theme="テーマ", **base))
    assert not r["ok"] and "議論タイプ" in r["error"]
    r = server.start(server.StartReq(theme="  ", discussion_type="planning", **base))
    assert not r["ok"] and "テーマ" in r["error"]
    assert server.start(server.StartReq(theme="テーマ", discussion_type="idea_generation", **base))["ok"]
    assert engine.agent.dtype.key == "idea_generation"
    assert server.start(server.StartReq(theme="テーマ", purpose="general", **base))["ok"]  # 一般GWは従来どおり
    assert engine.agent.dtype is GENERIC


def test_api_info_lists_four_types():
    types = server.info()["discussion_types"]
    assert [t["key"] for t in types] == SELECTABLE_TYPES and types[1]["phases"] == EXPECTED["idea_generation"]


def test_ui_has_type_selector_and_validation():
    html = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="seg-type"' in html and 'id="b-type"' in html
    assert "議論タイプを選択してください" in html and "discussion_type: ui.dtype || null" in html
    assert "discussion_type: s.discussion_type_key" in html  # 再読み込み時もタイプを引き継ぐ


# ---------- 評価シナリオ ----------
class FixedLLM:
    name = "fixed"

    def __init__(self, action):
        self.action, self.systems, self.schemas = action, [], []

    def generate_json(self, system, user, schema, context):
        self.systems.append(system)
        self.schemas.append(schema)
        allowed = context["allowed_actions"]
        a = allowed[0] if len(allowed) == 1 else self.action
        return {"analysis": {"phase": context["observed_phase"] or context["target_phase"], "phase_reason": "r", "health": "stalled",
                             "issues": [], "opinions": [], "conflicts": [], "agreements": [], "missing_perspectives": [], "recent_summary": ""},
                "decision": {"action": a, "reasoning": "r", "speech": "" if a == "no_intervention" else "どう進めますか?", "urgency": 0.5, "target_speaker": None}}


@pytest.mark.parametrize("name, key", [("15_idea_generation", "idea_generation"), ("16_decision_making", "decision_making"), ("17_planning", "planning")])
def test_type_scenarios_feed_selected_type_and_judge_stalls(name, key):
    sc = load_scenario(ROOT / "scenarios" / f"{name}.yaml")
    assert sc["discussion_type"] == key
    llm = FixedLLM("no_intervention")
    res = run_scenario(sc, llm, load_settings(llm_provider="mock"))
    label = DISCUSSION_TYPES[key].label
    assert all(f"# 議論タイプ: {label}" in s for s in llm.systems) and all(sch["properties"]["analysis"]["properties"]["phase"]["enum"] == DISCUSSION_TYPES[key].keys for sch in llm.schemas)
    stalls = [c for c in res.checks if "no_intervention" in c.forbid]
    assert stalls and not any(c.passed for c in stalls)  # 停滞の時点で見送ると不合格
    assert all(c.passed for c in res.checks if c.expect == ["no_intervention"])


def test_casual_type_does_not_ask_for_conclusion():
    """雑談型は結論を求めない(時間の告知・終了の発言)。"""
    st = DiscussionState(settings=load_settings(), theme="休日の過ごし方", duration_sec=600)
    ag = FacilitatorAgent(st, None, load_settings(), purpose="general", discussion_type="casual")
    warn, close = ag._template(Action.time_warning, 0.0), ag._template(Action.close_gd, 0.0)
    assert "結論" not in warn and "結論" not in close and "今日はここまで" in close
    assert "結論を出す必要はありません" in build_system_prompt("general", "casual")
    assert [p.label for p in DISCUSSION_TYPES["casual"].phases] == ["話題出し", "広げる", "締め"]


def test_casual_is_only_for_chat_mode():
    """雑談型は議論タイプの選択肢には出さず、「雑談」モード(一般)だけで使う。テーマなしなら自由に話す開始の発言。"""
    assert "casual" not in SELECTABLE_TYPES
    assert resolve_session_type("casual", "general").key == "casual"
    with pytest.raises(ValueError):
        resolve_session_type("casual", "job")
    st = DiscussionState(settings=load_settings(), theme=CASUAL_FREE_THEME, duration_sec=600)
    ag = FacilitatorAgent(st, None, load_settings(), purpose="general", discussion_type="casual")
    assert ag._template(Action.start_gd, 0.0) == "今日は自由に話しましょう。時間は10分です。始めてください。"
