"""レビュー対応(個人名の使用ルール、⑤停滞への介入の控えめ、評価方法の改善)のテスト。API は呼ばない。

ここで確かめられるのはルール層(検証・見送り)と評価の仕組みまで。LLM が個人名を使わない・停滞を正しく判断するといった
発話や判断そのものの質は、Claude での評価で別に確かめる必要がある。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ROOT, load_settings  # noqa: E402
from facilitator.agent.agent import FacilitatorAgent  # noqa: E402
from facilitator.agent.name_check import strip_names  # noqa: E402
from facilitator.llm.mock_llm import MockLLM  # noqa: E402
from facilitator.models import Action  # noqa: E402
from facilitator.policy import InterventionPolicy  # noqa: E402
from facilitator.scenario import applies, judge, load_scenario, run_scenario, scenario_policy  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402
from tests.test_content import StubLLM  # noqa: E402

NAMES = ["たろう", "はなこ", "けんた", "サクラ"]


# ---------- 個人名の使用ルール ----------
@pytest.mark.parametrize(
    "speech,keep,expected,left",
    [
        ("はなこさんの体験型の案、どう広げられそうですか?", None, "先ほど出た体験型の案、どう広げられそうですか?", []),
        ("はなこさん、どう思いますか?", None, "どう思いますか?", []),
        ("3つ出ましたね。けんたさん、どれから考えますか?", None, "3つ出ましたね。どれから考えますか?", []),
        ("はなこさんは賛成でしたね。皆さんはどうですか?", None, "はなこさんは賛成でしたね。皆さんはどうですか?", ["はなこ"]),
        ("たろうさんとはなこさんの意見が出ています。", None, "たろうさんと先ほど出た意見が出ています。", ["たろう"]),
        ("サクラさんの案もありましたね。", None, "先ほど出た案もありましたね。", []),  # AI 参加者の名前も同じ
        ("たろうさんの案について、はなこさんはどう考えますか?", "はなこ", "先ほど出た案について、はなこさんはどう考えますか?", []),
        ("織田信長ならどう考えるでしょう?", None, "織田信長ならどう考えるでしょう?", []),  # テーマに出てくる人物名
        ("「はなこ案」と「たろう案」を比べると?", None, "「はなこ案」と「たろう案」を比べると?", []),  # 敬称のない引用
    ],
)
def test_strip_names(speech, keep, expected, left):
    assert strip_names(speech, NAMES, keep) == (expected, left)


def make(llm, policy=None, observed="cause_analysis"):
    s = load_settings()
    st = DiscussionState(settings=s, theme="大学の学食の混雑を改善するには？", duration_sec=600)
    st.started, st.start_time = True, 0.0
    st.observed_phase, st.phase_observed_at = observed, 0.0
    for sid, n in {"speaker_1": "たろう", "speaker_2": "はなこ", "speaker_3": "けんた"}.items():
        st.set_speaker_name(sid, n)
    for i, t in enumerate(range(200, 260, 10)):
        st.add_utterance(f"speaker_{i % 3 + 1}", "原因は時間の集中ですね。", t, t + 5, 0.9)  # 最後は 250〜255
    return st, FacilitatorAgent(st, llm, s, policy=policy or InterventionPolicy())


def test_name_in_question_is_rephrased():
    rec = make(StubLLM("ask_question", "はなこさんの案、どう広げられそうですか?", health="stalled"))[1].tick(270.0)
    assert rec.decision.action == Action.ask_question and rec.decision.speech == "先ほど出た案、どう広げられそうですか?"
    assert "個人名を言い換え" in rec.overridden


def test_name_that_cannot_be_rephrased_is_not_spoken():
    rec = make(StubLLM("organize_issues", "はなこさんは賛成でしたね。皆さんはどうですか?", health="conflicting"))[1].tick(270.0)
    assert rec.decision.action == Action.no_intervention and "個人名(はなこ)" in rec.overridden


class InviteLLM(StubLLM):
    """指名対象(target_speaker)も返すスタブ。"""

    def __init__(self, speech: str, target: str) -> None:
        super().__init__("invite_person", speech)
        self.target = target

    def generate_json(self, *a):
        out = super().generate_json(*a)
        out["decision"]["target_speaker"] = self.target
        return out


def test_invite_person_keeps_target_and_falls_back_for_other_names():
    rec = make(InviteLLM("はなこさん、この点についてどう考えますか?", "speaker_2"))[1].tick(270.0)
    assert rec.decision.action == Action.invite_person and rec.decision.speech.startswith("はなこさん")
    rec = make(InviteLLM("たろうさんも言っていましたが、はなこさんはどうですか?", "speaker_2"))[1].tick(270.0)
    assert rec.decision.action == Action.invite_all and "指名対象以外の個人名" in rec.overridden  # 指名の安全条件は従来どおり


def test_name_in_forced_event_uses_template():
    st, ag = make(StubLLM("time_warning", "残り2分です。たろうさんは案をまとめてください。"))
    rec = ag.tick(482.0)
    assert rec.gate == "forced:time_warning" and "たろう" not in rec.decision.speech and "定型文" in rec.overridden


def test_speech_without_names_is_unchanged():
    rec = make(StubLLM("ask_question", "混雑が起きるのはどの場面だと思いますか?", health="stalled"))[1].tick(270.0)
    assert rec.decision.speech == "混雑が起きるのはどの場面だと思いますか?" and rec.overridden is None


# ---------- ⑤ 停滞への介入(控えめ): 停滞を理由とする介入だけ、沈黙の長さを待つ ----------
LOW = InterventionPolicy(stall="low")


@pytest.mark.parametrize("action", ["ask_question", "organize_issues"])
def test_low_holds_stall_intervention_on_short_silence(action):
    rec = make(StubLLM(action, "どの原因から考えますか?", health="stalled"), LOW)[1].tick(265.0)  # 沈黙10秒
    assert rec.decision.action == Action.no_intervention and "⑤停滞への介入(控えめ)" in rec.overridden
    rec = make(StubLLM(action, "どの原因から考えますか?", health="stalled"), LOW)[1].tick(273.5)  # 沈黙18.5秒
    assert rec.decision.action.value == action


@pytest.mark.parametrize(
    "health,action,policy,observed,why",
    [
        ("stalled", "ask_question", InterventionPolicy(), "cause_analysis", "標準は従来どおり"),
        ("conflicting", "compare_criteria", LOW, "cause_analysis", "膠着(③)"),
        ("conflicting", "organize_issues", LOW, "cause_analysis", "膠着の整理(③)"),
        ("progressing", "organize_issues", LOW, "cause_analysis", "停滞でない整理(③)"),
        ("stalled", "invite_all", LOW, "cause_analysis", "発言機会(④)"),
        ("stalled", "organize_issues", LOW, "problem_definition", "①の遅れ(経過44%で2段階)の進行の問いかけ"),
    ],
)
def test_low_does_not_affect_other_reasons(health, action, policy, observed, why):
    rec = make(StubLLM(action, "どう比べるか決めておきませんか?", health=health), policy, observed)[1].tick(265.0)
    assert rec.decision.action.value == action, why


def test_low_holds_stall_hint_but_not_request():
    hint = InterventionPolicy(stall="low", content_scope="hint", content_initiative="stalled")
    st, ag = make(StubLLM("ask_question", "どの原因から考えますか?", health="stalled"), hint)
    ag.support_step = 1
    st.analysis = st.analysis.model_copy(update={"health": "stalled"})
    ag.llm.action, ag.llm.speech = "offer_hint", "利用する人の立場から見るとどうでしょう?"
    rec = ag.tick(265.0)
    assert rec.content_reason == "stalled" and rec.decision.action == Action.no_intervention  # 停滞を理由のヒントも待つ
    st2, ag2 = make(StubLLM("offer_hint", "利用する人の立場から見るとどうでしょう?", health="stalled"), hint)
    st2.add_utterance("speaker_2", "AIさん、何かヒントもらえますか？", 258, 261, 0.9)
    rec = ag2.tick(265.0, "request")  # 依頼への応答は待たない
    assert rec.content_reason == "request" and rec.decision.action == Action.offer_hint


def test_low_stall_in_timeline_never_intervenes_below_threshold():
    """LLM が常に「停滞、質問する」と答える最悪の場合でも、控えめは 18 秒未満の沈黙で停滞への質問をしない。
    標準は定期分析が短い沈黙で判断を始めるので、12 秒未満でも質問しうる(従来の就活GDと同じ)。"""
    def run(policy):
        s = load_settings()
        st = DiscussionState(settings=s, theme="t", duration_sec=900)
        st.started, st.start_time = True, 0.0
        ag = FacilitatorAgent(st, StubLLM("ask_question", "どの原因から考えますか?", health="stalled"), s, policy=policy)
        t, out = 0.0, []
        for gap in (6, 10, 14, 20, 25) * 2:
            st.add_utterance("speaker_1", "うーん", t, t + 3, 0.9)
            end = t + 3 + gap
            t += 3
            while t < end:
                trig = ag.should_tick(t)
                if trig:
                    r = ag.tick(t, trig)
                    if r.decision.action == Action.ask_question:
                        out.append(round(r.silence_sec, 1))
                t += 0.25
        return out

    low, std = run(LOW), run(InterventionPolicy())
    assert low and min(low) >= 18.0
    assert std and min(std) < 12.0


# ---------- 評価方法: ルール層の見送りと LLM の見守りを区別する ----------
def _all_scenarios():
    return [load_scenario(p) for p in sorted((ROOT / "scenarios").glob("*.yaml"))]


def test_every_watch_checkpoint_declares_who_decides():
    """「見守り」を正解に含む時点は、LLM の判断(require_llm)かルール層の判定(expect_gate)のどちらかを明示する
    (ルール層が見送っただけで、LLM が見守れたと数えないため)。"""
    missing = [
        (sc["id"], cp["at"]) for sc in _all_scenarios() for cp in sc["checkpoints"]
        if "no_intervention" in cp["expect"] and not cp.get("require_llm") and not cp.get("expect_gate")
    ]
    assert not missing, missing


def test_expect_gate_is_checked():
    class Rec:
        gate, llm_called = "llm", True

        class decision:
            action, target_speaker = Action.no_intervention, None

    ok, reason = judge({"expect": ["no_intervention"], "expect_gate": "blocked:short_pause"}, Rec())
    assert not ok and "ルール層の判定が期待外" in reason


def test_rule_gate_checkpoints_hold_with_mock():
    """expect_gate の時点は、モックでも(LLM の判断によらず)ルール層で決まる。既定のポリシーで確かめる。"""
    for sc in _all_scenarios():
        policy = scenario_policy(sc)
        gated = [cp["at"] for cp in sc["checkpoints"] if cp.get("expect_gate") and applies(cp, policy)]
        if not gated:
            continue
        res = run_scenario(sc, MockLLM(), load_settings(llm_provider="mock"), policy=policy)
        for c in res.checks:
            if c.at in gated:
                assert c.passed and c.judged_by == "rule", (sc["id"], c.at, c.record.gate)


@pytest.mark.parametrize("name", ["25_drift_first_chatter", "26_agreement_milestone"])
@pytest.mark.parametrize("level", ["low", "standard", "high"])
def test_boundary_scenarios_reach_llm(name, level):
    """新しい境界のシナリオは、どの段階でもルール層を通って LLM が判断する(質の評価は Claude で行う)。"""
    sc = load_scenario(ROOT / "scenarios" / f"{name}.yaml")
    item = "drift" if name.startswith("25") else "organize"
    res = run_scenario(sc, MockLLM(), load_settings(llm_provider="mock"), policy=scenario_policy(sc, overrides={item: level}))
    assert res.checks and all(c.record.llm_called and c.passed for c in res.checks)
