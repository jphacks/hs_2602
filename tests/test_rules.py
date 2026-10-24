"""ルール層(過剰介入防止・指名の制限)のテスト。LLM は固定応答のスタブを使う。"""

from __future__ import annotations

import sys

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from facilitator.agent.agent import FacilitatorAgent  # noqa: E402
from facilitator.llm.base import LLMError  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402

ANALYSIS = {
    "phase": "ideation",
    "health": "stalled",
    "issues": [],
    "opinions": [],
    "conflicts": [],
    "agreements": [],
    "missing_perspectives": [],
    "recent_summary": "",
}


class StubLLM:
    name = "stub"

    def __init__(self, decision: dict | None = None, fail: bool = False) -> None:
        self.decision, self.fail, self.calls = decision, fail, 0

    def generate_json(self, system, user, schema, context):
        self.calls += 1
        if self.fail:
            raise LLMError("boom")
        return {"analysis": ANALYSIS, "decision": self.decision}


def make(decision=None, fail=False, conf=0.9, **overrides):
    s = load_settings(**overrides)
    st = DiscussionState(settings=s, theme="テーマ", duration_sec=600)
    st.started, st.start_time = True, 0.0
    for i in range(4):
        st.set_speaker_name(f"speaker_{i + 1}", f"名前{i + 1}")
    for i in range(6):
        st.add_utterance(f"speaker_{i % 2 + 1}", "発言", 100 + i * 5, 104 + i * 5, conf)
    llm = StubLLM(decision, fail)
    return st, FacilitatorAgent(st, llm, s), llm


def d(action, speech="発話です。", target=None):
    return {"action": action, "reasoning": "r", "speech": speech, "urgency": 0.5, "target_speaker": target}


def test_no_speech_while_speaking():
    st, ag, llm = make(d("ask_question"))
    st.speaking_now = True
    rec = ag.tick(200)
    assert rec.decision.action.value == "no_intervention" and llm.calls == 0


def test_cooldown_blocks():
    st, ag, llm = make(d("ask_question"))
    ag.tick(200)
    rec = ag.tick(220)
    assert rec.gate == "blocked:cooldown" and rec.decision.action.value == "no_intervention"


def test_time_warning_forced_even_in_cooldown():
    st, ag, _ = make(d("no_intervention", ""))
    ag.tick(470)
    st2_rec = ag.tick(485)  # 残り115秒
    assert st2_rec.decision.action.value == "time_warning"


def test_invite_person_downgraded_when_low_confidence():
    st, ag, _ = make(d("invite_person", "名前4さん、どうですか?", "speaker_4"), conf=0.5)
    rec = ag.tick(200)
    assert rec.decision.action.value != "invite_person"


def test_invite_person_ok_and_person_cooldown():
    st, ag, _ = make(d("invite_person", "名前4さん、どうですか?", "speaker_4"), cooldown_sec=0)
    assert ag.tick(200).decision.action.value == "invite_person"
    rec = ag.tick(260)
    assert rec.decision.action.value == "invite_all"


def test_llm_failure_falls_back_to_no_intervention():
    _, ag, _ = make(fail=True)
    assert ag.tick(200).decision.action.value == "no_intervention"


def test_close_gd_template_when_llm_fails():
    _, ag, _ = make(fail=True)
    rec = ag.tick(601)
    assert rec.decision.action.value == "close_gd" and rec.decision.speech


def test_llm_cannot_choose_close_early():
    _, ag, _ = make(d("close_gd"))
    assert ag.tick(200).decision.action.value == "no_intervention"


def test_speaker_relabel():
    st, _, _ = make()
    st.update_speaker(0, "speaker_3", 0.95)
    assert st.utterances[0].speaker_id == "speaker_3"
    st.update_speaker(1, "speaker_3", 0.3)
    assert st.utterances[1].speaker_id == "unknown"


def test_time_warning_waits_after_recent_intervention():
    st, ag, _ = make(d("ask_question"))
    ag.tick(470)  # 質問(残り130秒)
    rec = ag.tick(482)  # 残り118秒だが直前の介入から12秒
    assert rec.gate == "blocked:forced_gap"
    rec = ag.tick(492)
    assert rec.decision.action.value == "time_warning"


def test_time_warning_threshold_scales_for_short_gd():
    s = load_settings()
    st = DiscussionState(settings=s, theme="t", duration_sec=180)
    st.started, st.start_time = True, 0.0
    ag = FacilitatorAgent(st, StubLLM(d("no_intervention", "")), s)
    assert ag._warn_thresholds() == [pytest.approx(63.0)]


def test_pause_excludes_time():
    s = load_settings()
    st = DiscussionState(settings=s, theme="t", duration_sec=600)
    st.started, st.start_time = True, 0.0
    st.pause(100)
    assert st.elapsed(150) == pytest.approx(100)
    st.resume(160)
    assert st.elapsed(200) == pytest.approx(140)
    assert st.remaining(200) == pytest.approx(460)


@pytest.mark.parametrize(
    "humans,bots,mode",
    [(1, 1, "active"), (2, 1, "active"), (3, 3, "active"), (3, 1, "reserved"), (4, 2, "reserved"), (3, 0, "")],
)
def test_bot_mode_by_headcount(humans, bots, mode):
    from facilitator.engine import Engine, SessionConfig

    e = Engine(load_settings(), llm_provider="mock")  # セッションごとの設定にもモックを引き継ぐ
    e.start(SessionConfig("t", 1, [f"h{i}" for i in range(humans)], mode="text", tts=False, ai_count=bots))
    try:
        assert e.bot_mode == mode
        assert len(e.personas) == bots
        assert e.state.bot_ids == {f"speaker_{humans + i + 1}" for i in range(bots)}
    finally:
        e.stop()


def _bot_state():
    from facilitator.participants import Persona

    s = load_settings()
    st = DiscussionState(settings=s, theme="t", duration_sec=600)
    st.started, st.start_time = True, 0.0
    st.set_speaker_name("speaker_1", "たろう")
    ps = [Persona("speaker_2", "ミサキ", "a", "a", "a"), Persona("speaker_3", "ユウ", "b", "b", "b")]
    for p in ps:
        st.set_speaker_name(p.speaker_id, p.name)
        st.bot_ids.add(p.speaker_id)
    return st, ps


def test_bot_name_mention_only_counts_in_last_sentence():
    from facilitator.participants import pick_bot

    st, ps = _bot_state()
    st.add_utterance("speaker_3", "私もミサキさんに近い考えです。たろうさんはどう思いますか？", 10, 20, 1.0, source="bot")
    # ミサキは名指しされていない。たろう(人間)が名指しされたので AI 参加者は待つ
    assert pick_bot(st, ps, "active", 22, 20, -1, 1) is None
    # 6秒待ってもたろうが答えなければ、ほかの AI 参加者が話してよい
    picked = pick_bot(st, ps, "active", 27, 20, -1, 1)
    assert picked and "名指し" not in picked[1]


def test_bot_chain_resets_after_facilitator():
    from facilitator.participants import pick_bot

    st, ps = _bot_state()
    st.add_utterance("speaker_2", "意見A", 10, 15, 1.0, source="bot")
    st.add_utterance("speaker_3", "意見B", 16, 20, 1.0, source="bot")
    assert pick_bot(st, ps, "active", 30, 20, -1, 1) is None  # AI が2回続いた
    st.add_utterance("ai", "どんな基準で比べますか？", 40, 45, 1.0, source="ai")
    assert pick_bot(st, ps, "active", 48, 20, -1, 1) is not None  # 進行役の問いかけでリセット
