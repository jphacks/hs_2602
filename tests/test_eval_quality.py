"""評価の成立条件(C3)と、未評価だったシナリオ07・09・10の現状の挙動(C4)のテスト。API は呼ばない。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ROOT, load_settings  # noqa: E402
from facilitator.agent.prompts import build_user_prompt  # noqa: E402
from facilitator.llm.mock_llm import MockLLM  # noqa: E402
from facilitator.models import Decision, TickRecord  # noqa: E402
from facilitator.scenario import applies, judge, judged_by, load_scenario, run_scenario, scenario_policy  # noqa: E402
from tests.sim import SimSession  # noqa: E402

SCENARIOS = sorted((ROOT / "scenarios").glob("*.yaml"))


class FixedLLM:
    """非強制の判断では指定の行動を返す。渡された状況説明を記録する。"""

    name = "fixed"

    def __init__(self, action: str, speech: str = "皆さんはどう考えますか?", target: str | None = None, latency: float = 0.0) -> None:
        self.action, self.speech, self.target, self.contexts, self.users = action, speech, target, [], []
        self.latency, self.clock = latency, None  # clock は SimSession が疑似時計に差し替える
        self.on_call = None  # 呼ばれた直後に実行する(応答待ちの間の出来事を予定するため)

    def generate_json(self, system, user, schema, context):
        self.contexts.append(context)
        self.users.append(user)
        if self.on_call:
            self.on_call()
        if self.latency and self.clock:
            self.clock.sleep(self.latency)
        allowed = context["allowed_actions"]
        action = allowed[0] if len(allowed) == 1 else self.action
        speech = "" if action == "no_intervention" else self.speech
        analysis = {"phase": context["observed_phase"] or context["target_phase"], "phase_reason": "r", "health": "stalled",
                    "issues": [], "opinions": [], "conflicts": [], "agreements": [], "missing_perspectives": [], "recent_summary": ""}
        return {"analysis": analysis, "decision": {"action": action, "reasoning": "r", "speech": speech, "urgency": 0.5, "target_speaker": self.target}}


def run(name: str, llm):
    return run_scenario(load_scenario(ROOT / "scenarios" / f"{name}.yaml"), llm, load_settings(llm_provider="mock"))


# ================= C3: LLM 自身の見守り判断を評価できること =================
@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_require_llm_checkpoints_actually_reach_llm(path):
    """require_llm の時点は、本番の条件(2秒の沈黙・クールダウン・発話中)のままルール層を通り、LLM が判断する。"""
    sc = load_scenario(path)
    res = run_scenario(sc, MockLLM(), load_settings(llm_provider="mock"))
    policy = scenario_policy(sc)  # when が合わないチェックポイントは判断されない(結果に含まれない)
    for cp, c in zip(sorted((c for c in sc["checkpoints"] if applies(c, policy)), key=lambda c: c["at"]), res.checks):
        if cp.get("require_llm"):
            assert c.record.llm_called and c.judged_by == "llm", (path.stem, cp["at"], c.record.gate)


def test_no_intervention_scenario_has_llm_judged_checkpoints():
    """順調な議論のシナリオ14は、すべての時点で LLM が呼ばれ、見守りを LLM 自身が選んだときだけ合格する。"""
    res = run("14_smooth_with_pauses", FixedLLM("no_intervention"))
    assert len(res.checks) == 3 and all(c.passed and c.judged_by == "llm" for c in res.checks)
    res = run("14_smooth_with_pauses", FixedLLM("ask_question"))  # LLM が過剰に介入した場合は不合格
    assert not any(c.passed for c in res.checks)


def _rec(gate="llm", raw="no_intervention", final="no_intervention", called=True):
    return TickRecord(time=0, trigger="checkpoint", observed_phase="", target_phase="", remaining_sec=100, silence_sec=3,
                      gate=gate, allowed_actions=[], analysis={}, decision=Decision(action=final),
                      llm_raw_action=raw if called else None, llm_called=called)


def test_judged_by_separates_rule_llm_and_override():
    assert judged_by(_rec(gate="blocked:short_pause", called=False)) == "rule"
    assert judged_by(_rec()) == "llm"
    assert judged_by(_rec(raw="invite_person", final="no_intervention")) == "llm_overridden"


def test_require_llm_fails_when_rule_layer_handles_it():
    cp = {"expect": ["no_intervention"], "require_llm": True}
    ok, reason = judge(cp, _rec(gate="blocked:cooldown", called=False))
    assert not ok and "評価条件不成立" in reason
    assert judge({"expect": ["no_intervention"]}, _rec(gate="blocked:cooldown", called=False))[0]  # 従来の時点は従来どおり


def test_runtime_metrics_separate_canceled_intervention(monkeypatch):
    """実行時: LLM が介入を選んだが、応答後の確認(2秒の沈黙)で取りやめたものを別に数える。"""
    llm = FixedLLM("ask_question", "原因はどう整理できそうですか?", latency=1.0)
    s = SimSession(monkeypatch, llm)
    s.say(50, "speaker_1", "原因は昼の集中だと思います。")
    # 応答待ち(1秒)の間に次の発言 → 応答後の確認(2秒の沈黙)で取りやめ
    llm.on_call = lambda: s.say(s.now() + 0.3, "speaker_2", "確かに、12時台はすごいですよね。", 0.5)
    s.run(90, stop=lambda: any(r.llm_called for r in s.eng.agent.history))
    d = s.eng.metrics()["decisions"]
    assert d["llm_intervention_canceled"] >= 1 and d["rule"] >= 1


# ================= C4: シナリオ07・09・10の現状 =================
def _llm_check(res):
    return next(c for c in res.checks if c.record.llm_called)


def test_07_monologue_allows_naming_others_but_not_the_monologuer():
    sc = load_scenario(ROOT / "scenarios" / "07_monologue.yaml")
    policy = scenario_policy(sc)  # 270 の②積極的用の時点(when: drift: high)は既定のポリシーでは判断しない
    assert [cp["at"] for cp in sc["checkpoints"] if applies(cp, policy)] == [270, 286]  # 270は発話中(ルール層)、286で LLM が判断
    ok = _llm_check(run("07_monologue", FixedLLM("invite_person", "はなこさんはどう考えますか?", "speaker_2")))
    assert ok.passed and "invite_person" in ok.record.allowed_actions
    ng = _llm_check(run("07_monologue", FixedLLM("invite_person", "たろうさん、続けてください。", "speaker_1")))
    assert not ng.passed and "指名対象が期待外" in ng.fail_reason  # 独占している本人の指名は不合格
    assert not _llm_check(run("07_monologue", FixedLLM("no_intervention"))).passed  # 見送りも不合格


def test_09_return_to_speaker_unavailable_and_overridden():
    """09: return_to_speaker は無効(enable_return_to_speaker=False)。LLM が選ぶと no_intervention に置き換わり不合格(B3)。"""
    c = _llm_check(run("09_interrupted", FixedLLM("return_to_speaker", "けんたさん、先ほどの続きをどうぞ。", "speaker_3")))
    assert "return_to_speaker" not in c.record.allowed_actions and "invite_person" in c.record.allowed_actions
    assert c.judged_by == "llm_overridden" and c.record.decision.action.value == "no_intervention" and not c.passed
    alt = _llm_check(run("09_interrupted", FixedLLM("invite_person", "けんたさん、先ほどのお話を続けてもらえますか?", "speaker_3")))
    assert alt.passed  # 現状の代替手段: けんたさんへの個人指名
    llm = FixedLLM("no_intervention")
    run("09_interrupted", llm)
    assert "私は、混雑の原因って実は席じゃなくて、" in llm.users[-1]  # 遮りは文末「、」からしか読み取れない(検出機能はない)


def test_10_low_confidence_hides_names_and_blocks_invite_person():
    """10: 確信度0.7未満の発言は「不明」になり、話者統計は利用不可。invite_person は許可されない。
    LLM が選ぶと、個人名を含まない全体への呼びかけ(invite_all)に置き換わる(B2 の代替処理)。"""
    llm = FixedLLM("invite_person", "たろうさん、どうですか?", "speaker_1")
    c = _llm_check(run("10_low_confidence", llm))
    ctx = llm.contexts[-1]
    assert ctx["speaker_stats"] is None and "invite_person" not in c.record.allowed_actions
    assert sum("不明:" in line for line in ctx["transcript"]) == 7  # 8発言中7つが不明
    assert "利用不可" in build_user_prompt(ctx)
    assert c.judged_by == "llm_overridden" and c.record.decision.action.value == "invite_all" and c.passed
    assert "たろう" not in c.record.decision.speech and c.record.decision.target_speaker is None  # 誤った個人名は発話されない
    assert _llm_check(run("10_low_confidence", FixedLLM("invite_all", "まだ話していない方も、どう考えますか?"))).passed
