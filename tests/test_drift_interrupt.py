"""②脱線への対応が「積極的」のとき(就活GD・一般GW・カスタム。雑談型は除く)だけ、脱線修正を発話中・短い間でも割り込んで話すことのテスト(API は呼ばない)。

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
from facilitator.models import Action, Analysis, Decision  # noqa: E402
from facilitator.policy import InterventionPolicy, Level, preset_policy  # noqa: E402
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

    def __init__(self, latency: float = 1.0, action: str | None = None, on_call=None, max_redirects: int = 1) -> None:
        self.latency, self.action, self.on_call, self.max_redirects = latency, action, on_call, max_redirects
        self.redirected, self.calls, self.clock, self.last_usage = 0, [], time, None
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
        elif run >= 2 and self.redirected < self.max_redirects and "redirect_topic" in allowed:
            self.redirected += 1
            action, speech = "redirect_topic", REDIRECT
        analysis = {**ANALYSIS, "health": "off_topic" if run >= 2 else "progressing"}
        return {"analysis": analysis, "decision": {"action": action, "reasoning": "r", "speech": speech, "urgency": 0.8, "target_speaker": None}}


LEVELS = ["low", "standard", "high"]
# カスタム = 就活以外の進め方で、画面の詳細設定(7項目、0〜2)を自由に決めたもの。redirect は session() が drift から入れる
CUSTOM_KNOBS = {"progress": 1, "organize": 2, "turns": 0, "stall": 1, "content": 1, "content_when": 0}
MODES = ["gd", "gw", "custom"]


def session_config(mode: str, drift: str) -> SessionConfig:
    """gd: 就活GD(課題解決型)/ gw: 一般GWのプリセット(汎用型)に②だけ変更 / custom: 就活以外 + 詳細設定(アイデア出し型)。"""
    base = dict(theme=THEME, minutes=10.0, names=["A", "B", "C"], mode="text", intro=False, tts=False)
    if mode == "gd":
        return SessionConfig(**base, purpose="job", discussion_type="problem_solving", policy=InterventionPolicy(drift=Level(drift)))
    if mode == "gw":
        return SessionConfig(**base, purpose="general", policy_preset="general_gw",
                             policy=preset_policy("general_gw").updated(drift=Level(drift)))
    return SessionConfig(**base, purpose="general", discussion_type="idea_generation",
                         knobs={**CUSTOM_KNOBS, "redirect": LEVELS.index(drift)})


def session(monkeypatch, llm, drift="high", mode="gd", **overrides) -> SimSession:
    """SimSession と同じ疑似時計のエンジンを、用途(mode)と介入ポリシーの②を指定して始める。"""
    s = SimSession.__new__(SimSession)
    s.clock = SimClock()
    llm.clock, llm.now = s.clock, lambda: s.eng.now()
    monkeypatch.setattr(engine_mod, "make_llm", lambda settings: llm)
    s.eng = Engine(load_settings(), clock=s.clock, **overrides)
    s.eng.start(session_config(mode, drift), background=False)
    assert s.eng.policy.drift == Level(drift)
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
@pytest.mark.parametrize("mode", MODES)
def test_high_redirects_mid_speech_without_two_second_pause(monkeypatch, mode):
    llm = JudgeLLM(latency=1.0)
    s = session(monkeypatch, llm, mode=mode)
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
@pytest.mark.parametrize("mode", MODES)
def test_high_does_not_cancel_when_someone_starts_talking_during_judgement(monkeypatch, mode):
    holder = {}
    llm = JudgeLLM(latency=1.0, on_call=lambda t: holder["s"].say(t + 0.1, "speaker_3", CHAT[3], 3.0))
    s = holder["s"] = session(monkeypatch, llm, mode=mode)
    s.say(T0, "speaker_1", CHAT[0])
    s.say(T0 + 1.5, "speaker_2", CHAT[1])  # その後は2秒以上の間 → 通常の判断で redirect_topic
    run(s, T0 + 40)
    e = s.eng.redirect_log[0]
    assert e["outcome"] == "spoken" and e["new_utterances"] == 0 and e["wait_sec"] < 0.05, e
    assert s.eng.state.speaking_now or speaking_at(s, s.ai("戻りましょう")[0].start_time)


# 4. 本題に戻ったら見送る(確定した発言 / 発話途中の文字起こし)
@pytest.mark.parametrize("mode", MODES)
def test_high_cancels_when_group_returned_to_topic(monkeypatch, mode):
    holder = {}
    llm = JudgeLLM(latency=1.0, on_call=lambda t: holder["s"].say(t + 0.1, "speaker_3", "話を戻すと、レジの対策を考えましょう。", 0.5))
    s = holder["s"] = session(monkeypatch, llm, mode=mode)
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
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("drift", ["standard", "low"])
def test_standard_and_low_keep_old_constraints(monkeypatch, drift, mode):
    llm = JudgeLLM(latency=1.0)
    s = session(monkeypatch, llm, drift=drift, mode=mode)
    continuous_chat(s)
    s.run(T0 + 30)
    assert not any(trig == "drift" for _t, trig, _a in llm.calls)
    assert not any(r.gate == "llm:drift_interrupt" for r in s.eng.agent.history)
    assert all(not speaking_at(s, u.start_time) for u in s.eng.state.utterances if u.source == "ai")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("drift", ["standard", "low"])
def test_standard_and_low_still_wait_for_pause(monkeypatch, drift, mode):
    holder = {}
    llm = JudgeLLM(latency=0.5, on_call=lambda t: holder["s"].say(t + 0.1, "speaker_3", CHAT[3], 2.4))
    s = holder["s"] = session(monkeypatch, llm, drift=drift, mode=mode)
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
@pytest.mark.parametrize("mode", MODES)
def test_high_mid_speech_gate_allows_only_redirect(monkeypatch, mode):
    llm = JudgeLLM(latency=1.0, action="ask_question")
    s = session(monkeypatch, llm, mode=mode)
    continuous_chat(s, n=3)
    s.run(T0 + 20)
    first = next(r for r in s.eng.agent.history if r.llm_called)
    assert first.gate == "llm:drift_interrupt" and first.llm_raw_action == "ask_question"
    assert first.decision.action == Action.no_intervention
    assert not s.ai("原因はどう整理")


@pytest.mark.parametrize("mode", MODES)
def test_high_other_interventions_still_need_two_seconds(monkeypatch, mode):
    holder = {}
    llm = JudgeLLM(latency=1.0, action="ask_question", on_call=lambda t: holder["s"].say(t + 0.5, "speaker_3", CHAT[1], 0.3))
    s = holder["s"] = session(monkeypatch, llm, mode=mode)
    s.say(T0, "speaker_1", ON[0])
    s.say(T0 + 1.5, "speaker_2", ON[1])
    s.run(T0 + 40, stop=lambda: bool(llm.calls))
    s.run(s.now() + 3)
    rec = next(r for r in s.eng.agent.history if r.llm_called)
    assert rec.gate == "llm" and "発話取りやめ" in (rec.overridden or "")
    assert not s.ai("原因はどう整理")


# ---------- 共通のクールダウン(45秒)を越える / 脱線修正どうしの重複を防ぐ ----------
def redirects(s: SimSession) -> list[float]:
    return [iv.time for iv in s.eng.state.interventions if iv.action == Action.redirect_topic]


@pytest.mark.parametrize("mode", MODES)
def test_high_bypasses_cooldown_right_after_start(monkeypatch, mode):
    """開始の説明の直後(45秒以内)に雑談が始まっても、脱線修正はクールダウンを待たない。"""
    llm = JudgeLLM(latency=1.0)
    s = session(monkeypatch, llm, mode=mode)
    s.play(chat_script((CHAT * 4)[:18], start=5.0, gap=0.3, dur=2.0))
    s.run(45, stop=lambda: bool(s.eng.redirect_log))
    assert s.eng.redirect_log and s.eng.redirect_log[0]["outcome"] == "spoken"
    assert redirects(s)[0] < 20.0  # 雑談2発言目(9.6秒)の後、確認の間隔と応答時間のうちに話した


@pytest.mark.parametrize("mode", MODES)
def test_high_bypasses_cooldown_after_other_intervention(monkeypatch, mode):
    """ほかの介入(質問)の直後に脱線が始まっても、45秒を待たずに脱線修正する。ほかの介入はクールダウンのまま。"""
    llm = JudgeLLM(latency=1.0, action="ask_question")
    s = session(monkeypatch, llm, mode=mode)
    s.say(T0, "speaker_1", ON[0])
    s.say(T0 + 1.5, "speaker_2", ON[1])  # その後の沈黙で質問する
    s.run(T0 + 25, stop=lambda: bool(s.eng.state.interventions[1:]))
    ask = s.eng.state.interventions[-1]
    assert ask.action == Action.ask_question
    s.play(chat_script(CHAT * 2, start=s.now() + 5.0, gap=0.3, dur=2.0))
    s.run(ask.time + 44, stop=lambda: bool(redirects(s)))
    assert redirects(s) and redirects(s)[0] - ask.time < 45.0
    gates = {r.gate for r in s.eng.agent.history if r.time - ask.time < 45.0 and r.llm_called and r.time > ask.time}
    assert gates == {"llm:drift_interrupt"}  # クールダウン中に LLM に許したのは脱線修正だけ


def test_high_cooldown_gate_allows_only_redirect():
    """クールダウン中に許すのは脱線修正と見送りだけ(質問・整理などはクールダウンのまま)。"""
    s = load_settings()
    st = DiscussionState(settings=s, theme=THEME, duration_sec=600)
    st.started, st.start_time = True, 0.0
    agent = FacilitatorAgent(st, JudgeLLM(), s, purpose="job", policy=InterventionPolicy(drift=Level.high))
    st.record_intervention(50.0, Decision(action=Action.ask_question, reasoning="r", speech="?"))
    st.add_utterance("speaker_1", CHAT[0], 55, 57, 1.0)
    gate, allowed, _ = agent._gate(62.0)  # 沈黙5秒・質問から12秒
    assert gate == "llm:drift_interrupt" and allowed == [Action.no_intervention, Action.redirect_topic]
    std = FacilitatorAgent(st, JudgeLLM(), s, purpose="job", policy=InterventionPolicy(drift=Level.standard))
    assert std._gate(62.0)[0] == "blocked:cooldown"


@pytest.mark.parametrize("mode", MODES)
def test_high_same_drift_is_not_redirected_twice(monkeypatch, mode):
    """脱線修正の後も同じ雑談が続くだけなら、45秒たつまで再び修正しない(その間は脱線の確認の LLM も呼ばない)。"""
    llm = JudgeLLM(latency=1.0, max_redirects=3)
    s = session(monkeypatch, llm, mode=mode)
    continuous_chat(s, n=30)  # 約70秒続く
    s.run(T0 + 75)
    rs = redirects(s)
    assert rs, "1回目の脱線修正がない"
    assert all(b - a >= 45.0 for a, b in zip(rs, rs[1:])), rs
    assert not [t for t, trig, _a in llm.calls if rs[0] < t < rs[0] + 45.0]


@pytest.mark.parametrize("mode", MODES)
def test_high_new_drift_after_return_can_be_redirected_again(monkeypatch, mode):
    """修正の後に本題へ戻り、そこから新しい脱線が始まれば、45秒を待たずにまた修正できる(前回から10秒以上あける)。"""
    llm = JudgeLLM(latency=1.0, max_redirects=2)
    s = session(monkeypatch, llm, mode=mode)
    continuous_chat(s, n=3)
    s.run(T0 + 30, stop=lambda: bool(redirects(s)))
    first = redirects(s)[0]
    t = s.now() + 3.5  # 修正(3秒)を聞いた後
    s.say(t, "speaker_1", "話を戻すと、レジの対策を考えましょう。", 2.0)
    s.play(chat_script(["ところで週末どこか行きました？", "海に行きました。日焼けがすごくて。", "いいですね、どこの海ですか？"],
                       start=t + 2.3, gap=0.3, dur=2.0))
    s.run(first + 44)
    rs = redirects(s)
    assert len(rs) == 2, rs
    assert 10.0 <= rs[1] - first < 45.0


def test_redirect_repeat_reason_unit():
    s = load_settings()
    st = DiscussionState(settings=s, theme=THEME, duration_sec=600)
    st.started, st.start_time = True, 0.0
    agent = FacilitatorAgent(st, JudgeLLM(), s, purpose="job", policy=InterventionPolicy(drift=Level.high))
    st.add_utterance("speaker_1", CHAT[0], 55, 57, 1.0)
    st.record_intervention(60.0, Decision(action=Action.redirect_topic, reasoning="r", speech="戻りましょう。"))
    st.add_utterance("speaker_2", CHAT[1], 63, 65, 1.0)
    assert agent.redirect_repeat_reason(65.0) == "直前に脱線修正をした"
    assert agent.redirect_repeat_reason(75.0) == "同じ脱線が続いている"
    assert agent.redirect_repeat_reason(105.0) is None  # 45秒たてば従来どおり
    st.add_utterance("speaker_3", "話を戻すと、学食の混雑ですよね。", 66, 68, 1.0)
    assert agent.redirect_repeat_reason(69.0) == "直前に脱線修正をした"  # 戻っていても10秒はあける
    assert agent.redirect_repeat_reason(75.0) is None  # 本題に戻った後なので新しい脱線を修正できる


@pytest.mark.parametrize("purpose, dtype, cls, enabled", [
    ("job", "problem_solving", FacilitatorAgent, True),
    ("general", "generic", FacilitatorAgent, True),  # 一般GW(汎用型)
    ("general", "idea_generation", FacilitatorAgent, True),  # 一般GW・カスタム(議論タイプを選んだもの)
    ("general", "casual", ChatHostAgent, False),  # 雑談は対象外
])
def test_scope_gd_gw_custom_but_not_casual(purpose, dtype, cls, enabled):
    s = load_settings()
    st = DiscussionState(settings=s, theme=THEME, duration_sec=600)
    st.started, st.start_time = True, 0.0
    agent = cls(st, JudgeLLM(), s, purpose=purpose, discussion_type=dtype, policy=InterventionPolicy(drift=Level.high))
    assert agent.drift_interrupt() is enabled
    for i, text in enumerate(CHAT):
        st.add_utterance(f"speaker_{i % 3 + 1}", text, 60 + i * 2.3, 62 + i * 2.3, 1.0)
    st.speaking_now = True
    assert (agent._gate(80.0)[0] == "llm:drift_interrupt") is enabled
    assert (agent.should_tick(80.0) == "drift") is enabled


def test_casual_mode_session_is_excluded(monkeypatch):
    """画面の「雑談」モード(就活以外・雑談型)で②に積極的が送られても、雑談の司会は割り込まない。"""
    llm = JudgeLLM()
    monkeypatch.setattr(engine_mod, "make_llm", lambda settings: llm)
    eng = Engine(load_settings(), clock=SimClock())
    eng.start(SessionConfig(THEME, 10.0, ["A", "B"], mode="text", intro=False, tts=False, purpose="general",
                            discussion_type="casual", knobs={"redirect": 2}), background=False)
    assert isinstance(eng.agent, ChatHostAgent) and eng.policy.drift == Level.high
    assert not eng.agent.drift_interrupt()


@pytest.mark.parametrize("purpose, dtype", [("job", "problem_solving"), ("general", "generic"), ("general", "idea_generation")])
def test_gate_unit(purpose, dtype):
    """ルール層の単体: ②積極的だけ、発話中・短い間に脱線修正の判断を許可する(GD・GW 共通)。"""
    s = load_settings()
    for drift, expect in (("high", "llm:drift_interrupt"), ("standard", "blocked:speaking"), ("low", "blocked:speaking")):
        st = DiscussionState(settings=s, theme=THEME, duration_sec=600)
        st.started, st.start_time = True, 0.0
        agent = FacilitatorAgent(st, JudgeLLM(), s, purpose=purpose, discussion_type=dtype, policy=InterventionPolicy(drift=Level(drift)))
        for i, text in enumerate(CHAT[:3]):
            st.add_utterance(f"speaker_{i + 1}", text, 60 + i * 2.3, 62 + i * 2.3, 1.0)
        st.speaking_now = True
        assert agent._gate(67.0)[0] == expect
        st.speaking_now = False  # 短い間(1.4秒)
        assert agent._gate(68.0)[0] == ("llm:drift_interrupt" if drift == "high" else "blocked:short_pause")
        assert (agent.should_tick(68.0) == "drift") == (drift == "high")


# ---------- 事前判定(LLM を呼ぶかの粗い判定)の検証。脱線かどうかは LLM が決める ----------
ISSUES = Analysis(issues=["混雑の原因", "レジの待ち時間"], opinions=["キャッシュレスレジ", "空き教室の開放"])
PREFILTER_CASES = [
    # (名前, 直近2発言, 前回の分析, LLM を呼ぶか)。呼ぶ = 事前判定で脱線の疑いあり
    ("本題・テーマの語なし", ["それなら昼休みを学部ごとにずらせばいいんじゃない？", "確かに、時間を分ければ並ぶ人も減りそう。"], ISSUES, True),
    ("本題・言い換え", ["お昼の行列、券売機が遅いのが一番だと思う。", "支払いを電子マネーにすれば早いですよね。"], ISSUES, True),
    ("本題・テーマの語が1つずつ", ["学食の席を増やすのはどうですか。", "混雑する時間をずらすのもありですね。"], Analysis(), True),
    ("本題・本題の語が2つ", ["学食の混雑はレジが原因ですよね。", "キャッシュレスレジなら混雑も減りそう。"], ISSUES, False),
    ("脱線・テーマの語あり", ["学食のカレー、最近値上がりしましたよね。", "学食より駅前のラーメン屋の方が安いですよ。"], ISSUES, True),
    ("脱線・テーマの語あり2", ["混雑といえば、昨日の電車すごかったです。", "大学の最寄り駅、朝は地獄ですよね。"], ISSUES, True),
    ("脱線・本題の語が2つ", ["学食の混雑と関係ないけど、大学の近くにラーメン屋できましたよ。", "え、どこですか？"], ISSUES, False),
    ("一時的な雑談からすぐ本題へ", ["そういえば昨日のドラマ見ました？", "あ、でも学食の話に戻ると、レジが問題ですよね。"], ISSUES, False),
    ("一時的な雑談からすぐ本題へ(分析なし)", ["そういえば昨日のドラマ見ました？", "あ、でも学食の話に戻ると、レジが問題ですよね。"], Analysis(), True),
    ("無関係な話が続く", ["駅前のラーメン屋、替え玉無料なんですよ。", "え、知らなかった。今度行こうかな。"], ISSUES, True),
]


@pytest.mark.parametrize("name, texts, analysis, expect", PREFILTER_CASES, ids=[c[0] for c in PREFILTER_CASES])
def test_prefilter_cases(name, texts, analysis, expect):
    s = load_settings()
    st = DiscussionState(settings=s, theme=THEME, duration_sec=600)
    st.started, st.start_time = True, 0.0
    st.analysis = analysis
    agent = FacilitatorAgent(st, JudgeLLM(), s, purpose="job", policy=InterventionPolicy(drift=Level.high))
    agent.last_tick_time = 60.0  # 前回の判断から5秒(定期分析の間隔の確認は働かない)
    for i, text in enumerate(texts):
        st.add_utterance(f"speaker_{i + 1}", text, 60 + i * 2.3, 62 + i * 2.3, 1.0)
    st.speaking_now = True
    assert agent._drift_check_due(65.0) is expect


def test_prefilter_miss_is_covered_by_interval_check(monkeypatch):
    """事前判定が本題らしいとみなす脱線(本題の語を2つ含む雑談)が切れ目なく続いても、20秒の間隔の確認で LLM が判断する。"""
    drift = ["学食の混雑と関係ないけど、大学の近くにラーメン屋できましたよ。", "学食の混雑より、大学の駅前のラーメンが気になります。"]
    llm = JudgeLLM(latency=1.0)
    s = session(monkeypatch, llm)
    s.play(chat_script(ON + drift * 6, start=T0, gap=0.3, dur=2.0))
    s.run(T0 + 40)
    calls = [t for t, trig, _a in llm.calls if trig == "drift" and t > T0]
    assert calls and all(b - a >= 20.0 - 0.25 for a, b in zip(calls, calls[1:]))  # 発話中でも20秒ごとに確かめる
    assert len(calls) <= 3


def test_on_topic_continuous_talk_calls_llm_at_most_every_20s(monkeypatch):
    """本題の語を2つ含む発言が切れ目なく続く間は、脱線の確認は20秒ごとだけ(事前判定では呼ばない)。"""
    talk = ["学食の混雑はレジが原因ですよね。", "キャッシュレスレジなら混雑も減りそう。", "学食の席の混雑も原因ですね。"]
    llm = JudgeLLM(latency=1.0)
    s = session(monkeypatch, llm)
    s.play(chat_script(talk * 8, start=T0, gap=0.3, dur=2.0))
    s.run(T0 + 50)
    calls = [t for t, _trig, _a in llm.calls if t > T0]
    assert len(calls) <= 3 and not s.eng.redirect_log



# ---------- モック評価: ②積極的の時点は LLM 自身の見守りを確かめる ----------
class AlwaysRedirect:
    name = "always-redirect"

    def generate_json(self, system, user, schema, context):
        allowed = context["allowed_actions"]
        action = allowed[0] if len(allowed) == 1 else ("redirect_topic" if "redirect_topic" in allowed else "no_intervention")
        speech = "" if action == "no_intervention" else "テーマに戻りましょう。"
        return {"analysis": ANALYSIS, "decision": {"action": action, "reasoning": "r", "speech": speech, "urgency": 0.8, "target_speaker": None}}


@pytest.mark.parametrize("name, at", [("01_smooth", 95.5), ("01_smooth", 125.0), ("03_off_topic", 204.5), ("07_monologue", 270.0)])
def test_high_checkpoints_fail_when_llm_redirects_on_topic(name, at):
    """発話中・短い間の時点で LLM が本題の議論を脱線と誤って修正すれば不合格(ルール層の見送りで合格にならない)。
    ②標準では同じ時点はルール層が見送るので、同じ LLM でも合格する(従来どおり)。"""
    from config import ROOT
    from facilitator.scenario import load_scenario, run_scenario

    sc = load_scenario(ROOT / "scenarios" / f"{name}.yaml")
    for drift, passed in (("high", False), ("standard", True)):
        res = run_scenario(sc, AlwaysRedirect(), load_settings(llm_provider="mock"), policy=InterventionPolicy(drift=Level(drift)))
        c = next(c for c in res.checks if c.at == at)
        assert c.passed is passed, (drift, c.record.gate, c.fail_reason)
