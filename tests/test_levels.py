"""介入ポリシー①〜⑤の段階ごとの効果と、項目どうしの独立性のテスト。LLM は固定応答のスタブかモックで、API は呼ばない。"""

from __future__ import annotations

import dataclasses
import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import LEVEL_SETTINGS, ROOT, Settings, load_settings  # noqa: E402
from facilitator.agent.agent import FacilitatorAgent  # noqa: E402
from facilitator.agent.prompts import (  # noqa: E402
    LEVEL_TEXT,
    PHASE_BEHIND_NOTE,
    PHASE_BEHIND_NOTES,
    build_system_prompt,
    build_user_prompt,
)
from facilitator.llm.mock_llm import MockLLM  # noqa: E402
from facilitator.models import Action  # noqa: E402
from facilitator.policy import LEVEL_ITEMS, InterventionPolicy, level_settings, levels  # noqa: E402
from facilitator.scenario import applies, load_scenario, parse_sets, run_scenario, scenario_policy  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402

NAMES = {"speaker_1": "たろう", "speaker_2": "はなこ", "speaker_3": "けんた", "speaker_4": "みさき"}
ALL_COMBOS = [dict(zip(LEVEL_ITEMS, c)) for c in itertools.product(("low", "standard", "high"), repeat=len(LEVEL_ITEMS))]
# どの項目・段階でも変えない(全体の過剰介入防止と話者識別の条件)
PROTECTED = {
    "cooldown_sec", "analysis_interval_sec", "min_silence_for_intervention_sec", "min_gap_after_speech_sec",
    "person_cooldown_sec", "forced_min_gap_sec", "time_warnings_remaining_sec", "time_warning_max_fraction",
    "speaker_conf_threshold", "speaker_reliability_threshold", "use_speaker_info",
}


class StubLLM:
    """固定の analysis.health と decision を返し、渡されたプロンプトと状況を記録する。"""

    name = "stub"

    def __init__(self, action: str = "no_intervention", health: str = "progressing", target: str | None = None) -> None:
        self.action, self.health, self.target = action, health, target
        self.calls, self.user, self.context = 0, "", {}

    def generate_json(self, system, user, schema, context):
        self.calls += 1
        self.user, self.context = user, context
        return {
            "analysis": {"phase": context["observed_phase"] or "cause_analysis", "phase_reason": "", "health": self.health,
                         "issues": [], "opinions": [], "conflicts": [], "agreements": [], "missing_perspectives": [],
                         "recent_summary": ""},
            "decision": {"action": self.action, "reasoning": "r", "speech": "" if self.action == "no_intervention" else "話を戻しましょう。",
                         "urgency": 0.5, "target_speaker": self.target},
        }


def make(policy: InterventionPolicy | dict | None = None, llm=None, observed: str = "cause_analysis"):
    """経過300秒(50%)・沈黙10秒の状態。みさき(speaker_4)だけ最初に1回話したきり。"""
    if isinstance(policy, dict):
        policy = InterventionPolicy(**policy)
    s = load_settings()
    st = DiscussionState(settings=s, theme="大学の学食の混雑を改善するには？", duration_sec=600)
    st.started, st.start_time = True, 0.0
    st.observed_phase, st.phase_observed_at = observed, 0.0
    for sid, n in NAMES.items():
        st.set_speaker_name(sid, n)
    st.add_utterance("speaker_4", "原因から整理しませんか。", 10, 14, 0.9)
    for i, t in enumerate(range(240, 290, 10)):
        st.add_utterance(f"speaker_{i % 3 + 1}", "時間の集中と会計の遅さが原因ですね。", t, t + 5, 0.9)  # 最後は 280〜285
    agent = FacilitatorAgent(st, llm or StubLLM(), s, policy=policy)
    return st, agent


# ---------- 項目の独立性(構造) ----------
def test_level_settings_are_disjoint_known_and_protected():
    fields = {f.name for f in dataclasses.fields(Settings)}
    owners: dict[str, str] = {}
    for item, by_level in LEVEL_SETTINGS.items():
        assert "standard" not in by_level, item  # 標準は上書きしない(=現行の値)
        for values in by_level.values():
            for key in values:
                assert key in fields, key
                assert key not in PROTECTED, (item, key)
                assert owners.setdefault(key, item) == item, f"{key} を {owners[key]} と {item} が変えている"


@pytest.mark.parametrize("item,level", [(i, lv) for i in LEVEL_ITEMS for lv in ("low", "high")])
def test_single_item_changes_only_its_own_prompt_text(item, level):
    std = build_system_prompt("job", "problem_solving")
    p = build_system_prompt("job", "problem_solving", levels={item: level})
    assert p != std
    assert p == std.replace(LEVEL_TEXT[item]["standard"], LEVEL_TEXT[item][level])


def test_all_combinations_compose_without_interference():
    """243 通りの組み合わせで、各項目の選んだ段階の文がちょうど1回入り、ほかの段階の文は入らない。
    数値の上書きは項目ごとの上書きの和になり、守るべき設定は変わらない。"""
    for combo in ALL_COMBOS:
        p = build_system_prompt("job", "problem_solving", levels=combo)
        for item, level in combo.items():
            assert p.count(LEVEL_TEXT[item][level]) == 1, (combo, item)
            for other in {"low", "standard", "high"} - {level}:
                assert LEVEL_TEXT[item][other] not in p, (combo, item, other)
        ls = level_settings(InterventionPolicy(**combo))
        expected = {}
        for item, level in combo.items():
            expected.update(LEVEL_SETTINGS[item].get(level, {}))
        assert ls == expected and not (set(ls) & PROTECTED)


def test_rules_and_triggers_depend_only_on_own_item():
    """243 通りで、ルール層・トリガー・状況説明の各判断が、その項目の段階だけで決まる。
    ② redirect_topic の許可 / ⑤ 沈黙トリガー / ① 遅れの判定 / ④ 発言の少ない参加者の表示 / クールダウン・2秒の間は不変。"""
    for combo in ALL_COMBOS:
        st, agent = make(combo)
        now = 295.0  # 沈黙10秒
        agent.last_tick_time = now - 10
        gate, allowed, _ = agent._gate(now)
        assert gate == "llm"
        assert (Action.redirect_topic in allowed) == (combo["drift"] != "low"), combo
        assert (agent.should_tick(now) == "silence") == (combo["stall"] == "high"), combo
        assert agent.phase_status(now)[2] == (combo["progress"] == "high"), combo  # 経過49%で1段階の遅れ
        ctx = agent._context(now, "periodic", allowed, None)
        assert bool(ctx["quiet_members"]) == (combo["turns"] == "high"), combo
        assert (agent._param("cooldown_sec"), agent._param("min_silence_for_intervention_sec")) == (45.0, 2.0)
        assert agent._param("analysis_interval_sec") == 20.0


# ---------- ① 進行・時間管理 ----------
def test_progress_behind_conditions():
    for level, frac_gap_expect in {
        "low": [(0.5, 1, False), (0.8, 2, True), (0.5, 2, False)],
        "standard": [(0.5, 1, False), (0.65, 1, True), (0.4, 2, True)],
        "high": [(0.2, 1, True)],
    }.items():
        for frac, gap, expect in frac_gap_expect:
            st, agent = make({"progress": level})
            # 課題解決型の5段階で、目標フェーズから gap だけ遅れた実際のフェーズにする
            target = agent.dtype.index(agent.dtype.target_phase(frac).key)
            assert target - gap >= 0
            st.observed_phase = agent.dtype.keys[target - gap]
            assert agent.phase_status(frac * 600)[2] == expect, (level, frac, gap)


@pytest.mark.parametrize("level", ["low", "standard", "high"])
def test_progress_note_in_user_prompt(level):
    llm = StubLLM()
    st, agent = make({"progress": level}, llm, observed="problem_definition")
    agent.tick(470.0)  # 経過78%(目標は評価・比較)で3段階遅れ: どの段階でも「遅れ」
    assert llm.context["phase_behind"]
    assert PHASE_BEHIND_NOTES[level] in llm.user
    assert (PHASE_BEHIND_NOTE in llm.user) == (level == "standard")


# ---------- ② 脱線への対応 ----------
def test_drift_low_needs_previous_off_topic_judgment():
    llm = StubLLM("redirect_topic", health="off_topic")
    st, agent = make({"drift": "low"}, llm)
    rec = agent.tick(295.0)  # 初めて脱線と判断: redirect_topic は許可されていない
    assert "redirect_topic" not in rec.allowed_actions
    assert rec.decision.action == Action.no_intervention and agent.off_topic_streak == 1
    st.add_utterance("speaker_1", "駅前のラーメンおいしいですよね。", 300, 304, 0.9)
    rec = agent.tick(310.0)  # 前回も脱線: 許可される
    assert rec.decision.action == Action.redirect_topic


def test_drift_streak_resets_when_back_on_topic():
    llm = StubLLM("no_intervention", health="off_topic")
    st, agent = make({"drift": "low"}, llm)
    agent.tick(295.0)
    llm.health = "progressing"
    st.add_utterance("speaker_1", "本題に戻ると、会計の遅さですね。", 300, 304, 0.9)
    agent.tick(310.0)
    assert agent.off_topic_streak == 0
    assert Action.redirect_topic not in agent._gate(330.0)[1]


@pytest.mark.parametrize("level", ["standard", "high"])
def test_drift_standard_and_high_allow_first_time(level):
    rec = make({"drift": level}, StubLLM("redirect_topic", health="off_topic"))[1].tick(295.0)
    assert rec.decision.action == Action.redirect_topic


# ---------- ④ 発言機会の調整 ----------
def test_turns_high_lists_quiet_humans_only():
    llm = StubLLM()
    st, agent = make({"turns": "high"}, llm)
    st.bot_ids.add("speaker_4")  # AI 参加者は対象外
    agent.tick(295.0)
    assert "# 発言の少ない参加者" not in llm.user
    st.bot_ids.clear()
    agent.tick(330.0)
    assert "# 発言の少ない参加者(目安: 60秒以上発言なし)" in llm.user
    assert "みさき(speaker_4): 最後の発言から316秒" in llm.user
    assert "たろう" not in llm.user.split("# 発言の少ない参加者")[1].split("#")[0]


def test_turns_high_does_not_relax_speaker_reliability():
    """話者識別の信頼度が低ければ、積極的でも発言の少ない参加者を示さず、個人指名の条件も変わらない。"""
    llm = StubLLM("invite_person", target="speaker_4")
    st, agent = make({"turns": "high"}, llm)
    for i in range(10):  # 不明な発話を増やして信頼度を閾値未満にする
        st.add_utterance("speaker_1", "うーん。", 286 + i * 0.5, 286.2 + i * 0.5, 0.3)
    assert not st.speaker_info_usable()
    rec = agent.tick(300.0)
    assert "# 発言の少ない参加者" not in llm.user
    assert rec.decision.action == Action.invite_all  # 標準と同じく全体への呼びかけに置き換え
    st2, agent2 = make({"turns": "standard"}, StubLLM("invite_person", target="speaker_4"))
    for i in range(10):
        st2.add_utterance("speaker_1", "うーん。", 286 + i * 0.5, 286.2 + i * 0.5, 0.3)
    assert agent2.tick(300.0).decision.action == rec.decision.action


def test_turns_standard_user_prompt_unchanged():
    llm = StubLLM()
    make({}, llm)[1].tick(295.0)
    assert "発言の少ない参加者" not in llm.user and llm.context["quiet_members"] == []


def test_quiet_member_never_spoken_counts_after_threshold():
    st, agent = make({"turns": "high"})
    st.utterances = [u for u in st.utterances if u.speaker_id != "speaker_4"]
    assert any("一度も発言なし" in q for q in agent._quiet_members(295.0))
    st.start_time = 250.0  # 経過45秒(60秒未満)なら、一度も発言していなくてもまだ示さない
    assert not any("speaker_4" in q for q in agent._quiet_members(295.0))


# ---------- ⑤ 停滞への介入: LLM の呼び出し回数 ----------
def _count_llm_calls(stall: str) -> int:
    """5分間: 話しては10秒・15秒・25秒の沈黙、を繰り返す。LLM は常に見守り(介入しないのでクールダウンは効かない)。"""
    llm = StubLLM()
    s = load_settings()
    st = DiscussionState(settings=s, theme="テーマ", duration_sec=600)
    st.started, st.start_time = True, 0.0
    agent = FacilitatorAgent(st, llm, s, policy=InterventionPolicy(stall=stall))
    t, events = 0.0, []
    for gap in (10, 15, 25) * 3:
        for _ in range(4):
            events.append((t, t + 4))
            t += 5
        t += gap
    pending = list(events)
    now = 0.0
    while now < t:
        while pending and pending[0][1] <= now:
            a, b = pending.pop(0)
            st.add_utterance("speaker_1", "えっと", a, b, 0.9)
        st.speaking_now = any(a <= now < b for a, b in pending)
        trig = agent.should_tick(now)
        if trig:
            agent.tick(now, trig)
        now += 0.5
    return llm.calls


def test_stall_levels_change_llm_calls_only_moderately():
    low, std, high = (_count_llm_calls(x) for x in ("low", "standard", "high"))
    assert low <= std <= high
    assert high <= std * 1.5  # 沈黙トリガーを短くしても呼び出しは大きく増えない(仮の上限)


# ---------- シナリオ評価で介入ポリシーを指定する ----------
def test_scenario_policy_priority_and_when():
    sc = {"policy": {"preset": "general_gw", "stall": "low"}}
    p = scenario_policy(sc)
    assert p.stall.value == "low" and p.organize.value == "high"
    assert scenario_policy(sc, overrides={"stall": "high"}).stall.value == "high"  # --set が最後
    assert scenario_policy(sc, preset="job_gd") == InterventionPolicy()  # --preset はシナリオの policy の代わり
    assert scenario_policy({}) == InterventionPolicy()  # 省略時は就活GD(既存シナリオと同じ)
    assert scenario_policy({"purpose": "general"}).stall.value == "high"
    assert applies({"when": {"stall": ["low", "standard"]}}, p) and not applies({"when": {"stall": "high"}}, p)
    assert applies({}, p)
    assert parse_sets(["stall=high", " drift = low "]) == {"stall": "high", "drift": "low"}
    with pytest.raises(SystemExit):
        parse_sets(["stall"])


@pytest.mark.parametrize("stall", ["low", "standard", "high"])
def test_stall_ladder_scenario_with_mock(stall):
    """18_stall_levels をモックで流すと、段階ごとの期待(どの沈黙の長さから介入するか)と一致する。
    モックは「停滞とみなす沈黙(⑤の沈黙トリガー)以上なら質問する」だけなので、ルール・トリガーの配線の確認。"""
    sc = load_scenario(ROOT / "scenarios" / "18_stall_levels.yaml")
    res = run_scenario(sc, MockLLM(), load_settings(llm_provider="mock"), policy=scenario_policy(sc, overrides={"stall": stall}))
    assert len(res.checks) == 3
    assert all(c.passed for c in res.checks), [(c.at, c.record.decision.action.value, c.fail_reason) for c in res.checks]


def test_existing_scenarios_default_to_standard_policy():
    for path in sorted((ROOT / "scenarios").glob("*.yaml")):
        sc = load_scenario(path)
        if not sc.get("policy") and sc.get("purpose", "job") == "job":
            assert levels(scenario_policy(sc)) == dict.fromkeys(LEVEL_ITEMS, "standard"), path.name
