"""就活GDで②脱線への対応が「積極的」のときだけ、脱線修正を発話中・短い間でも割り込んで話すことのテスト(API は呼ばない)。

- 疑似時計(tests/sim.py)で実際のエンジンを回す。クールダウン45秒・分析間隔20秒・2秒の間は本番どおり
- 発話中は VAD の speaking_now を立て、話し終えた時刻に発言を確定する(テキストモードで音声モードを模擬)
- 実際に声が重なったときの音声認識・回り込み除去・聞こえ方は確かめていない(実マイクでの確認が必要)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import facilitator.engine as engine_mod  # noqa: E402
from config import load_settings  # noqa: E402
from facilitator.agent.agent import FacilitatorAgent  # noqa: E402
from facilitator.chat.host import ChatHostAgent  # noqa: E402
from facilitator.engine import Engine, SessionConfig  # noqa: E402
from facilitator.models import Action  # noqa: E402
from facilitator.policy import InterventionPolicy, Level  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402
from tests.sim import THEME, SimClock, SimSession, chat_script  # noqa: E402

TOPIC = ("学食", "混雑", "レジ", "席", "対策", "原因", "キャッシュレス", "並")
ON = ["学食の混雑って、やっぱりレジが原因ですよね。", "席が足りないのも原因だと思います。"]
CHAT = ["駅前のラーメン屋、替え玉無料なんですよ。", "え、知らなかった。今度行こうかな。", "あそこの近くでバイトしてるんです。",
        "バイト何してるんですか？", "居酒屋です。まかないが出ます。", "いいなあ、唐揚げとか？"]
REDIRECT = "盛り上がっているところすみません。学食の混雑の話に戻りましょう。"
ANALYSIS = {"phase": "cause_analysis", "phase_reason": "r", "health": "off_topic", "issues": ["混雑の原因"],
            "opinions": ["キャッシュレスレジ"], "conflicts": [], "agreements": [], "missing_perspectives": [], "recent_summary": ""}
T0 = 50.0  # 開始の説明のクールダウン(45秒)明け


class JudgeLLM:
    """本題の語がない人間の発言が2つ以上続いていれば、許可されていれば redirect_topic(1回だけ)。
    action を渡すと、1回目はその行動を返す(許可の外の行動が止められるかを見る)。on_call(t) で応答待ち中の出来事を予定する。"""

    name = "mock-judge"

    def __init__(self, latency: float = 1.0, action: str | None = None, on_call=None) -> None:
        self.latency, self.action, self.on_call = latency, action, on_call
        self.redirected, self.calls, self.clock, self.last_usage = False, [], time, None
        self.now = lambda: 0.0

    def generate_json(self, system, user, schema, context):
        allowed = context["allowed_actions"]
        self.calls.append((self.now(), context["trigger"], list(allowed)))
        if len(self.calls) == 1 and self.on_call:
            self.on_call(self.now())
        self.clock.sleep(self.latency)
        human = [line.split(": ", 1)[1] for line in context["transcript"] if "AI(" not in line]
        run = 0
        for text in reversed(human):
            if any(w in text for w in TOPIC):
                break
            run += 1
        action, speech = "no_intervention", ""
        if len(allowed) == 1:
            action, speech = allowed[0], "時間になりました。"
        elif self.action and len(self.calls) == 1:
            action, speech = self.action, "原因はどう整理できそうですか?"
        elif run >= 2 and not self.redirected and "redirect_topic" in allowed:
            self.redirected = True
            action, speech = "redirect_topic", REDIRECT
        analysis = {**ANALYSIS, "health": "off_topic" if run >= 2 else "progressing"}
        return {"analysis": analysis, "decision": {"action": action, "reasoning": "r", "speech": speech, "urgency": 0.8, "target_speaker": None}}


def session(monkeypatch, llm, drift="high", purpose="job", discussion_type="problem_solving", **overrides) -> SimSession:
    """SimSession と同じ疑似時計のエンジンを、介入ポリシーの②を指定して始める。"""
    s = SimSession.__new__(SimSession)
    s.clock = SimClock()
    llm.clock, llm.now = s.clock, lambda: s.eng.now()
    monkeypatch.setattr(engine_mod, "make_llm", lambda settings: llm)
    s.eng = Engine(load_settings(), clock=s.clock, **overrides)
    s.eng.start(SessionConfig(THEME, 10.0, ["A", "B", "C"], mode="text", intro=False, tts=False, purpose=purpose,
                              discussion_type=discussion_type, policy=InterventionPolicy(drift=Level(drift))), background=False)
    s.spans = []
    return s


def continuous_chat(s: SimSession, n: int = 8, gap: float = 0.3, dur: float = 2.0) -> None:
    """本題2発言のあと、2秒の間が一度もない雑談が続く台本。"""
    s.play(chat_script(ON, start=T0, gap=gap, dur=dur) + chat_script((CHAT * 2)[:n], start=T0 + 2 * (dur + gap), gap=gap, dur=dur))


def speaking_at(s: SimSession, t: float) -> bool:
    return any(a <= t <= b for a, b in s.spans)


def run(s: SimSession, until: float) -> None:
    s.run(until, stop=lambda: bool(s.eng.redirect_log))
    s.run(s.now() + 3.0)


# 1・2. 2秒の間がない雑談でも、発話中に脱線修正を話す(定期分析・2秒の間を待たない)
def test_high_redirects_mid_speech_without_two_second_pause(monkeypatch):
    llm = JudgeLLM(latency=1.0)
    s = session(monkeypatch, llm)
    continuous_chat(s)
    run(s, T0 + 30)
    log = s.eng.redirect_log
    assert log and log[0]["outcome"] == "spoken" and log[0]["wait_sec"] < 0.05, log
    ai = s.ai("戻りましょう")
    assert len(ai) == 1
    assert speaking_at(s, ai[0].start_time), "参加者の発話中に話していない"
    rec = next(r for r in s.eng.agent.history if r.decision.action == Action.redirect_topic)
    assert rec.gate == "llm:drift_interrupt" and rec.trigger == "drift"
    assert rec.allowed_actions == ["no_intervention", "redirect_topic"]
    # 雑談2発言目が終わってから、確認どうしの最低間隔(3秒。雑談1発言目の後の確認から数える)+ LLM の応答時間(1秒)以内に話した
    # (定期分析の20秒や2秒の間は待たない)
    second_chat_end = T0 + 2 * 2.3 + 2 * 2.3 - 0.3
    assert ai[0].start_time - second_chat_end < 3.0 + 1.0 + 1e-6


# 3. LLM の判断中に別の参加者が話し始めても、それだけでは取り消さない
def test_high_does_not_cancel_when_someone_starts_talking_during_judgement(monkeypatch):
    holder = {}
    llm = JudgeLLM(latency=1.0, on_call=lambda t: holder["s"].say(t + 0.1, "speaker_3", CHAT[3], 3.0))
    s = holder["s"] = session(monkeypatch, llm)
    s.say(T0, "speaker_1", CHAT[0])
    s.say(T0 + 1.5, "speaker_2", CHAT[1])  # その後は2秒以上の間 → 通常の判断で redirect_topic
    run(s, T0 + 40)
    e = s.eng.redirect_log[0]
    assert e["outcome"] == "spoken" and e["new_utterances"] == 0 and e["wait_sec"] < 0.05, e
    assert s.eng.state.speaking_now or speaking_at(s, s.ai("戻りましょう")[0].start_time)


# 4. 本題に戻ったら見送る(確定した発言 / 発話途中の文字起こし)
def test_high_cancels_when_group_returned_to_topic(monkeypatch):
    holder = {}
    llm = JudgeLLM(latency=1.0, on_call=lambda t: holder["s"].say(t + 0.1, "speaker_3", "話を戻すと、レジの対策を考えましょう。", 0.5))
    s = holder["s"] = session(monkeypatch, llm)
    s.say(T0, "speaker_1", CHAT[0])
    s.say(T0 + 1.5, "speaker_2", CHAT[1])
    run(s, T0 + 40)
    e = s.eng.redirect_log[0]
    assert e["outcome"] == "canceled" and e["reason"].startswith("本題に戻った"), e
    assert not s.ai("戻りましょう")


def test_high_cancels_on_partial_transcript_returning_to_topic(monkeypatch):
    holder = {}

    def returning(t):  # 判断中に話し始め、確定前の途中経過で本題に戻っている(音声モードの STT の途中経過)
        holder["s"].at(t + 0.1, lambda: (setattr(holder["s"].eng.state, "speaking_now", True),
                                         setattr(holder["s"].eng, "partial", "話を戻すと、キャッシュレスの")))

    llm = JudgeLLM(latency=1.0, on_call=returning)
    s = holder["s"] = session(monkeypatch, llm)
    s.say(T0, "speaker_1", CHAT[0])
    s.say(T0 + 1.5, "speaker_2", CHAT[1])
    run(s, T0 + 40)
    e = s.eng.redirect_log[0]
    assert e["outcome"] == "canceled" and "話を戻" in e["reason"], e


# 5. 標準・控えめは従来どおり(発話中は判断せず、話すのは区切りの後)
@pytest.mark.parametrize("drift", ["standard", "low"])
def test_standard_and_low_keep_old_constraints(monkeypatch, drift):
    llm = JudgeLLM(latency=1.0)
    s = session(monkeypatch, llm, drift=drift)
    continuous_chat(s)
    s.run(T0 + 30)
    assert not any(trig == "drift" for _t, trig, _a in llm.calls)
    assert not any(r.gate == "llm:drift_interrupt" for r in s.eng.agent.history)
    assert all(not speaking_at(s, u.start_time) for u in s.eng.state.utterances if u.source == "ai")


@pytest.mark.parametrize("drift", ["standard", "low"])
def test_standard_and_low_still_wait_for_pause(monkeypatch, drift):
    holder = {}
    llm = JudgeLLM(latency=0.5, on_call=lambda t: holder["s"].say(t + 0.1, "speaker_3", CHAT[3], 2.4))
    s = holder["s"] = session(monkeypatch, llm, drift=drift)
    s.say(T0, "speaker_1", CHAT[0])
    s.say(T0 + 1.5, "speaker_2", CHAT[1])
    s.run(T0 + 90, stop=lambda: bool(s.eng.redirect_log))
    s.run(s.now() + 3.0)
    for u in s.ai("戻りましょう"):  # 控えめは2回続けて脱線と判断するまで話さない
        assert not speaking_at(s, u.start_time)
    if drift == "standard":  # 応答待ちの間に話し始めた人の声が止まり、0.8秒の区切りができてから話す
        assert s.eng.redirect_log[0]["outcome"] == "spoken"
        assert s.eng.redirect_log[0]["wait_sec"] >= 2.4 - 0.4 + 0.8 - 1e-6


# 6. ほかの介入は従来どおり(発話中の判断では脱線修正しか許可しない。2秒の間がなければ取りやめる)
def test_high_mid_speech_gate_allows_only_redirect(monkeypatch):
    llm = JudgeLLM(latency=1.0, action="ask_question")
    s = session(monkeypatch, llm)
    continuous_chat(s, n=3)
    s.run(T0 + 20)
    first = next(r for r in s.eng.agent.history if r.llm_called)
    assert first.gate == "llm:drift_interrupt" and first.llm_raw_action == "ask_question"
    assert first.decision.action == Action.no_intervention
    assert not s.ai("原因はどう整理")


def test_high_other_interventions_still_need_two_seconds(monkeypatch):
    holder = {}
    llm = JudgeLLM(latency=1.0, action="ask_question", on_call=lambda t: holder["s"].say(t + 0.5, "speaker_3", CHAT[1], 0.3))
    s = holder["s"] = session(monkeypatch, llm)
    s.say(T0, "speaker_1", ON[0])
    s.say(T0 + 1.5, "speaker_2", ON[1])
    s.run(T0 + 40, stop=lambda: bool(llm.calls))
    s.run(s.now() + 3)
    rec = next(r for r in s.eng.agent.history if r.llm_called)
    assert rec.gate == "llm" and "発話取りやめ" in (rec.overridden or "")
    assert not s.ai("原因はどう整理")


def test_high_respects_cooldown(monkeypatch):
    """クールダウン(45秒)中は発話中の判断もしない。開始の説明の直後に雑談が始まっても、45秒たつまで割り込まない。"""
    llm = JudgeLLM(latency=1.0)
    s = session(monkeypatch, llm)
    s.play(chat_script((CHAT * 4)[:18], start=5.0, gap=0.3, dur=2.0))
    s.run(60, stop=lambda: bool(s.eng.redirect_log))
    assert all(t >= 45.0 for t, trig, _a in llm.calls if trig == "drift")
    assert s.eng.redirect_log and s.eng.agent.history[-1].time >= 45.0


@pytest.mark.parametrize("purpose, dtype, cls", [("general", "general", FacilitatorAgent), ("general", "casual", ChatHostAgent)])
def test_gw_and_casual_unaffected(monkeypatch, purpose, dtype, cls):
    s = load_settings()
    st = DiscussionState(settings=s, theme=THEME, duration_sec=600)
    st.started, st.start_time = True, 0.0
    agent = cls(st, JudgeLLM(), s, purpose=purpose, discussion_type=dtype, policy=InterventionPolicy(drift=Level.high))
    assert not agent.drift_interrupt()
    for i, text in enumerate(CHAT):
        st.add_utterance(f"speaker_{i % 3 + 1}", text, 60 + i * 2.3, 62 + i * 2.3, 1.0)
    st.speaking_now = True
    assert agent._gate(80.0)[0] != "llm:drift_interrupt"
    assert agent.should_tick(80.0) != "drift"


def test_gd_gate_unit():
    """ルール層の単体: 就活GD・②積極的だけ、発話中・短い間に脱線修正の判断を許可する。"""
    s = load_settings()
    for drift, expect in (("high", "llm:drift_interrupt"), ("standard", "blocked:speaking"), ("low", "blocked:speaking")):
        st = DiscussionState(settings=s, theme=THEME, duration_sec=600)
        st.started, st.start_time = True, 0.0
        agent = FacilitatorAgent(st, JudgeLLM(), s, purpose="job", policy=InterventionPolicy(drift=Level(drift)))
        for i, text in enumerate(CHAT[:3]):
            st.add_utterance(f"speaker_{i + 1}", text, 60 + i * 2.3, 62 + i * 2.3, 1.0)
        st.speaking_now = True
        assert agent._gate(67.0)[0] == expect
        st.speaking_now = False  # 短い間(1.4秒)
        assert agent._gate(68.0)[0] == ("llm:drift_interrupt" if drift == "high" else "blocked:short_pause")
        assert (agent.should_tick(68.0) == "drift") == (drift == "high")


def test_drift_check_skips_on_topic_talk():
    """本題の語を含む発言が続いている間は、発話中の脱線の確認(LLM の呼び出し)を始めない。"""
    s = load_settings()
    st = DiscussionState(settings=s, theme=THEME, duration_sec=600)
    st.started, st.start_time = True, 0.0
    agent = FacilitatorAgent(st, JudgeLLM(), s, purpose="job", policy=InterventionPolicy(drift=Level.high))
    for i, text in enumerate(ON + ["学食の席を増やすのはどうですか。", "混雑する時間をずらすのもありですね。"]):
        st.add_utterance(f"speaker_{i % 3 + 1}", text, 60 + i * 2.3, 62 + i * 2.3, 1.0)
    st.speaking_now = True
    assert agent.should_tick(70.0) is None
