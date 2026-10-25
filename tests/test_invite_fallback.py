"""B2: 個人指名(invite_person)ができないときの代替処理のテスト。LLM はスタブで、API は呼ばない。

- 誰を指名すべきか確かでない(話者識別の信頼度不足・話者情報オフ・名前未確認)なら、全体への呼びかけ(invite_all)に置き換える
- 発言は個人名を含まない定型文にし、指名対象も消す
- 直近に全体への呼びかけ済みなら繰り返さず見送る。理由が話者の特定と無関係なら置き換えない
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ROOT, load_settings  # noqa: E402
from facilitator.agent.agent import INVITE_ALL_FALLBACK, FacilitatorAgent  # noqa: E402
from facilitator.models import Action, Decision  # noqa: E402
from facilitator.scenario import load_scenario, run_scenario  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402

NAMES = ["たろう", "はなこ", "けんた", "みさき"]
PERSONAL = "たろうさん、さっきの続きをどうぞ。"


class InviteLLM:
    name = "invite"

    def __init__(self, action="invite_person", target="speaker_1", speech=PERSONAL) -> None:
        self.action, self.target, self.speech, self.contexts = action, target, speech, []

    def generate_json(self, system, user, schema, context):
        self.contexts.append(context)
        allowed = context["allowed_actions"]
        action = allowed[0] if len(allowed) == 1 else self.action
        return {"analysis": {"phase": "cause_analysis", "phase_reason": "r", "health": "stalled", "issues": [], "opinions": [],
                             "conflicts": [], "agreements": [], "missing_perspectives": [], "recent_summary": ""},
                "decision": {"action": action, "reasoning": "r", "speech": self.speech, "urgency": 0.6, "target_speaker": self.target}}


def make(conf=0.9, confirmed=True, **overrides):
    s = load_settings(**overrides)
    st = DiscussionState(settings=s, theme="テーマ", duration_sec=600)
    st.started, st.start_time = True, 0.0
    for i, n in enumerate(NAMES):
        st.set_speaker_name(f"speaker_{i + 1}", n, confirmed=confirmed)
    for i in range(6):
        st.add_utterance(f"speaker_{i % 2 + 1}", "発言です。", 100 + i * 5, 104 + i * 5, conf)
    return st, FacilitatorAgent(st, InviteLLM(), s)


def assert_neutral(d: Decision):
    assert d.action == Action.invite_all and d.speech == INVITE_ALL_FALLBACK and d.target_speaker is None
    assert not any(n in d.speech for n in NAMES) and "さん" not in d.speech  # 個人名・個人を指す呼びかけを残さない


# ---------- 代替すべき場合 ----------
def test_scenario10_low_confidence_falls_back_to_neutral_invite_all():
    sc = load_scenario(ROOT / "scenarios" / "10_low_confidence.yaml")
    res = run_scenario(sc, InviteLLM(), load_settings(llm_provider="mock"))
    c = res.checks[0]
    assert "invite_person" not in c.record.allowed_actions  # 誤指名を防ぐため個人指名は許可されない
    assert c.record.llm_raw_action == "invite_person" and c.judged_by == "llm_overridden"
    assert_neutral(c.record.decision)
    assert "話者識別の信頼度が閾値未満" in c.record.overridden
    assert c.passed  # 期待(invite_all / ask_question / organize_issues)を満たす


def test_speaker_info_off_falls_back():
    st, ag = make(use_speaker_info=False)
    rec = ag.tick(200)
    assert_neutral(rec.decision)
    assert "話者情報を使わない設定" in rec.overridden


def test_no_confirmed_names_falls_back():
    st, ag = make(confirmed=False)  # 音声モードで自己紹介の登録ができなかった場合など
    rec = ag.tick(200)
    assert_neutral(rec.decision)
    assert "名前が確認済みの参加者がいない" in rec.overridden


def test_allowed_but_unconfirmed_target_falls_back():
    """個人指名は許可されているが、指名対象が確認済みでない(従来からある代替)。"""
    st, ag = make()
    ag.llm.target = "speaker_9"
    rec = ag.tick(200)
    assert_neutral(rec.decision)
    assert "指名対象の名前が未確認" in rec.overridden


# ---------- 見送るべき場合 ----------
def test_recent_group_call_is_not_repeated():
    st, ag = make(conf=0.5)
    st.record_intervention(150, Decision(action=Action.invite_all, speech="皆さんはどうですか?"))  # 50秒前に全体へ呼びかけ済み
    rec = ag.tick(200)
    assert rec.decision.action == Action.no_intervention and rec.decision.speech == ""
    assert "直近に全体への呼びかけ済み" in rec.overridden


def test_group_call_allowed_again_after_person_cooldown():
    st, ag = make(conf=0.5)
    st.record_intervention(60, Decision(action=Action.invite_all, speech="皆さんはどうですか?"))  # 140秒前(120秒を超える)
    assert ag.tick(200).decision.action == Action.invite_all


def test_not_substituted_when_reason_is_unrelated_to_speaker_identity():
    """個人指名が許可されていない理由が話者の特定と無関係なら、無条件に invite_all にしない。"""
    st, ag = make()  # 話者識別は十分
    d = Decision(action=Action.invite_person, speech=PERSONAL, target_speaker="speaker_1")
    out, note = ag._validate(d, [Action.no_intervention, Action.invite_all], 200)
    assert out.action == Action.no_intervention and "no_intervention に置換" in note


def test_not_substituted_when_invite_all_not_allowed():
    st, ag = make(conf=0.5)
    d = Decision(action=Action.invite_person, speech=PERSONAL, target_speaker="speaker_1")
    out, _ = ag._validate(d, [Action.no_intervention, Action.ask_question], 200)
    assert out.action == Action.no_intervention


# ---------- 既存ルールの維持 ----------
def test_substituted_group_call_starts_cooldown():
    st, ag = make(conf=0.5)
    assert ag.tick(200).decision.action == Action.invite_all
    assert ag.tick(220).gate == "blocked:cooldown"  # 置き換えた呼びかけも介入として数え、クールダウンが効く


def test_forced_event_unaffected():
    st, ag = make(conf=0.5)
    assert ag.tick(601).decision.action == Action.close_gd
