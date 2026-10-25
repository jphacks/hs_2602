"""雑談の司会の会話の状態と介入のタイミング(第2段階)の時系列テスト。LLM はスタブ・モックで、API は呼ばない。

LLM には「許可されていれば必ず介入する」スタブ(最悪の場合)を使い、介入してよいタイミングをルール層が正しく
決めているかを確かめる。秒数は config.Settings の chat_*(仮の値)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import facilitator.engine as engine_mod  # noqa: E402
from config import load_settings  # noqa: E402
from facilitator.agent.agent import FacilitatorAgent  # noqa: E402
from facilitator.chat.host import FREE, LEAD, ChatHostAgent  # noqa: E402
from facilitator.engine import Engine, SessionConfig  # noqa: E402
from facilitator.models import Action  # noqa: E402
from facilitator.policy import preset_policy  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402
from tests.sim import SimClock  # noqa: E402

S = load_settings(tts_chars_per_sec=1000.0)  # AI の発話はほぼ一瞬(開始の発言の長さで時刻がずれないように)
NAMES = {"speaker_1": "たろう", "speaker_2": "はなこ", "speaker_3": "けんた"}


VARIED = ["休みの日の朝はどう過ごしていますか?", "最近よく行くお店はありますか?", "最近見て面白かったものはありますか?",
          "行ってみたい場所はどこですか?", "子どものころ好きだった遊びは何ですか?", "最近買ってよかったものは何ですか?",
          "好きな季節はいつですか?", "朝ごはんは何派ですか?", "最近覚えたことはありますか?", "得意な料理は何ですか?"]


class EagerLLM:
    """許可されていれば必ず介入する(依頼があれば内容支援、なければ質問)。health は flowing で切り替えられる。"""

    name = "eager"

    def __init__(self, health: str = "stalled") -> None:
        self.health, self.calls = health, []

    def generate_json(self, system, user, schema, ctx):
        allowed = ctx["allowed_actions"]
        if ctx.get("request") and "offer_topic" in allowed:
            a, sp = "offer_topic", "最近行ってよかった場所はありますか?"
        elif len(allowed) == 1:
            a, sp = allowed[0], ""
        elif self.health in ("progressing", "converging"):  # 会話が進んでいると判断したら見守る
            a, sp = "no_intervention", ""
        elif "ask_question" in allowed:
            a, sp = "ask_question", VARIED[len(self.calls) % len(VARIED)]  # 同じ問いは繰り返さない(検証で止まる)
        else:
            a, sp = "no_intervention", ""
        self.calls.append((ctx["elapsed_sec"], ctx["trigger"], a))
        return {"analysis": {"phase": ctx["observed_phase"] or "warm_up", "phase_reason": "", "health": self.health,
                             "issues": [], "opinions": [], "conflicts": [], "agreements": [], "missing_perspectives": [],
                             "recent_summary": ""},
                "decision": {"action": a, "reasoning": "", "speech": sp, "urgency": 0.5, "target_speaker": None}}


def make_chat(llm=None, duration=900.0, policy=None):
    st = DiscussionState(settings=S, theme="休日の過ごし方", duration_sec=duration)
    for sid, n in NAMES.items():
        st.set_speaker_name(sid, n)
    ag = ChatHostAgent(st, llm or EagerLLM(), S, purpose="general", discussion_type="casual",
                       policy=policy or preset_policy("general_gw"))
    return st, ag


def run(st, ag, script, until, start=0.0):
    """script = [(話し始め, 話し終わり, 話者, 発言)]。0.25 秒ごとに should_tick → tick(AI は話す)。
    返り値: AI の発言 [(時刻, アクション)]。AI の発話中・人の発話中は、engine と同じく判断しない。"""
    pending = sorted(script, key=lambda x: x[1])
    t, out = start, []
    while t <= until:
        while pending and pending[0][1] <= t:
            a, b, sid, text = pending.pop(0)
            st.add_utterance(sid, text, a, b, 0.9)
        st.speaking_now = any(a <= t < b for a, b, _s, _x in pending)
        ai_talking = any(u.source == "ai" and u.end_time > t for u in st.utterances)
        if not ai_talking:
            trig = ag.should_tick(t)
            if trig:
                rec = ag.tick(t, trig)
                if rec.decision.action != Action.no_intervention:
                    out.append((round(t, 2), rec.decision.action.value))
        t += 0.25
    return out


def ai_turns(st):
    return [(round(u.start_time, 2), round(u.end_time, 2)) for u in st.utterances if u.source == "ai"]


# ---------- 開始直後(AI 主導期): 答えがなければ待ってから、続けて話しすぎない ----------
def test_no_answer_waits_and_backs_off():
    st, ag = make_chat()
    out = run(st, ag, [], 60)
    assert out[0] == (0.0, "start_gd")
    ai = ai_turns(st)
    gaps = [b[0] - a[1] for a, b in zip(ai, ai[1:])]  # 前の AI の発言の終わりから次の発言まで
    assert gaps[0] >= S.chat_no_answer_sec - 0.01  # 誰も答えない: 10秒待つ
    assert gaps[1] >= 2 * S.chat_no_answer_sec - 0.01  # 答えのない発言が続くほど長く待つ(質問を連発しない)
    assert len(out) <= 3 and ag.chat_phase == LEAD


# ---------- 一人が答えたら、ほかの人の反応を待つ ----------
def test_waits_for_others_after_one_answer():
    st, ag = make_chat()
    out = run(st, ag, [(3, 7, "speaker_1", "休みの日は寝てばかりです。")], 30)
    follow = [t for t, a in out if a != "start_gd"]
    assert follow and follow[0] >= 7 + S.chat_reaction_wait_sec - 0.01  # 答え終わってから6秒の間は話さない


def test_no_intervention_when_someone_starts_during_wait():
    """反応の待ちの間に別の人が話し始めたら、介入せずに待ちを数え直す。"""
    st, ag = make_chat()
    script = [(3, 7, "speaker_1", "休みの日は寝てばかりです。"), (10, 13, "speaker_2", "わかります、私もです。")]
    out = run(st, ag, script, 30)
    follow = [t for t, a in out if a != "start_gd"]
    assert follow and follow[0] >= 13 + S.chat_reaction_wait_sec - 0.01


# ---------- 主導期 → 自走期 → 途切れたら主導期 ----------
LIVELY = [  # 別の人が短い間隔で続けて話す(つながった発言)
    (3, 7, "speaker_1", "休みの日は寝てばかりです。"),
    (8, 11, "speaker_2", "わかります、私もです。"),
    (12, 15, "speaker_3", "僕は映画を観に行くことが多いですね。"),
    (16, 19, "speaker_1", "何の映画が好きですか?"),
    (20, 24, "speaker_3", "最近はアニメ映画をよく観ます。"),
    (30, 33, "speaker_2", "アニメいいですね。"),  # 6秒の間があっても、自走期なので AI は入らない
    (40, 44, "speaker_1", "今度おすすめを教えてください。"),  # 7秒の間
]


def test_free_phase_after_linked_exchange_and_no_spontaneous_intervention():
    llm = EagerLLM()
    st, ag = make_chat(llm)
    out = run(st, ag, LIVELY, 55)
    assert ag.chat_log[0][1] == FREE and ag.chat_log[0][0] <= 15.25  # 2回目のつながった発言(15秒に確定)で自走期
    assert [a for t, a in out if t > 8] == []  # 自走期では介入しない
    assert not [c for c in llm.calls if c[0] > 12]  # LLM も呼ばない


def test_back_to_lead_after_lull_but_not_on_short_pause():
    st, ag = make_chat()
    out = run(st, ag, LIVELY, 80)
    assert ag.chat_phase == LEAD and ag.chat_log[-1][2].startswith("会話が途切れた")
    back = [t for t, a in out if t > 44]
    assert back and back[0] >= 44 + S.chat_lull_silence_sec - 0.01  # 15秒の沈黙で戻り、そこで支援する
    assert len(ag.chat_log) == 2  # 6〜7秒の間では切り替えない


def test_llm_can_confirm_flow_with_one_link():
    """つながった発言が1回でも、LLM が会話が進んでいると判断していれば自走期に移る(ルールで決めきれない分を補う)。"""
    st, ag = make_chat(EagerLLM(health="progressing"))
    script = [(3, 7, "speaker_1", "休みの日は寝てばかりです。"), (8, 11, "speaker_2", "わかります、私もです。")]
    run(st, ag, script, 25)
    assert ag.chat_phase == FREE
    st2, ag2 = make_chat(EagerLLM(health="stalled"))
    run(st2, ag2, script, 25)
    assert ag2.chat_phase == LEAD


# ---------- 自走期でも、依頼と強制イベントには応じる ----------
def test_free_phase_still_answers_request_and_time_warning():
    st, ag = make_chat(duration=120.0)  # 時間告知は min(120, 120×0.35=42) → 残り42秒(78秒)
    script = LIVELY + [(50, 54, "speaker_2", "AIさん、何か話題ありますか？")]
    out = run(st, ag, script, 110)
    phase_at_request = [p for t, p, _w in ag.chat_log if t <= 56][-1]
    assert phase_at_request == FREE  # 依頼のときは自走期
    assert any(a == "offer_topic" and 54 <= t <= 60 for t, a in out)  # 依頼に応じる(自走期でも)
    assert any(a == "time_warning" for _t, a in out)  # 時間告知(強制イベント)にも応じる


# ---------- LLM の判断中に会話が再開したら、古い状況での発話はしない ----------
def test_stale_reason_after_new_human_utterance():
    st, ag = make_chat()
    st.started, st.start_time = True, 0.0
    st.add_utterance("ai", "休みの日はどう過ごしていますか?", 0, 3, 1.0, source="ai")
    st.add_utterance("speaker_1", "寝てばかりです。", 4, 6, 0.9)
    ag.tick(20.0)
    assert ag.stale_reason(21.0) is None
    st.add_utterance("speaker_2", "私もです。", 21, 23, 0.9)  # 判断中に会話が再開した
    assert ag.stale_reason(23.5) == "判断中に会話が再開した"
    assert FacilitatorAgent.stale_reason(ag, 23.5) is None  # GD・GW の進行役は常に None(従来どおり)


def test_engine_cancels_stale_decision(monkeypatch, tmp_path):
    """engine は話す直前に stale_reason を確かめ、判断中に会話が再開していたら取りやめる。"""
    clock = SimClock()

    class SlowLLM(EagerLLM):
        def generate_json(self, system, user, schema, ctx):
            if ctx["trigger"] == "chat_lead":  # 判断している間に人が話した
                eng.state.add_utterance("speaker_2", "あ、そういえば…", eng.now() - 2, eng.now() - 0.5, 0.9)
            return super().generate_json(system, user, schema, ctx)

    llm = SlowLLM()
    monkeypatch.setattr(engine_mod, "make_llm", lambda settings: llm)
    eng = Engine(load_settings(), clock=clock, log_dir=tmp_path)
    eng.start(SessionConfig("休日の過ごし方", 15, ["たろう", "はなこ"], mode="text", intro=False, tts=False,
                            purpose="general", discussion_type="casual"), background=False)
    try:
        assert type(eng.agent) is ChatHostAgent
        eng._run_tick("start")
        clock.sleep(20)
        eng._run_tick("chat_lead")
        rec = eng.agent.history[-1]
        assert rec.gate == "llm", rec.gate
        assert rec.decision.action == Action.ask_question and rec.overridden == "発話取りやめ: 判断中に会話が再開した"
        assert len([u for u in eng.state.utterances if u.source == "ai"]) == 1  # 開始の発言だけ
    finally:
        eng.stop()


# ---------- GD・GW の進行役は変わらない ----------
def test_gd_gw_agent_timing_unchanged():
    st = DiscussionState(settings=S, theme="t", duration_sec=600)
    ag = FacilitatorAgent(st, EagerLLM(), S)
    assert ag._cooldown_sec() == S.cooldown_sec == 45.0 and S.analysis_interval_sec == 20.0
    assert ag.stale_reason(10.0, 0) is None


@pytest.mark.parametrize("key", ["chat_min_gap_sec", "chat_reaction_wait_sec", "chat_no_answer_sec", "chat_link_gap_sec",
                                 "chat_lull_silence_sec", "chat_free_min_sec"])
def test_chat_timing_values_are_settings(key):
    """秒数は設定に出してあり、上書きで調整できる(実際の音声テストで調整するため)。"""
    s = load_settings(**{key: 99.0})
    assert getattr(s, key) == 99.0
