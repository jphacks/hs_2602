"""脱線の軌道修正(redirect_topic)を短い区切りで話す仕組みのテスト。LLM はスタブで、API は呼ばない。

疑似時計(tests/sim.py)でエンジンを時系列に回す。設定は本番どおり(クールダウン45秒、分析間隔20秒、待機上限15秒)。
LLM の応答待ちの間に起きる発言は、スタブが呼ばれた時刻を起点に予定する。実スレッドでの確認は最後の1件だけ。
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import facilitator.engine as engine_mod  # noqa: E402
from config import ROOT, load_settings  # noqa: E402
from facilitator.agent.agent import FacilitatorAgent  # noqa: E402
from facilitator.agent.topic_return import detect_topic_return  # noqa: E402
from facilitator.engine import Engine, SessionConfig  # noqa: E402
from facilitator.llm.anthropic_llm import AnthropicLLM  # noqa: E402
from facilitator.llm.mock_llm import MockLLM  # noqa: E402
from facilitator.models import Analysis  # noqa: E402
from facilitator.scenario import load_scenario  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402
from tests.sim import THEME, SimSession  # noqa: E402

ANALYSIS = {
    "phase": "solution_ideation", "phase_reason": "r", "health": "off_topic",
    "issues": ["混雑の原因"], "opinions": ["キャッシュレスレジ", "空き教室の開放"], "conflicts": [],
    "agreements": [], "missing_perspectives": [], "recent_summary": "",
}
REDIRECT = "盛り上がっているところすみません。少しテーマに戻りましょう。キャッシュレスレジの話はどうでしたか?"
CHAT = ["駅前のラーメン屋、替え玉無料なんですよ。", "バイト先の近くですよね。", "まかないも出るらしいです。"]
T_CHAT = 50.0  # 開始の説明のクールダウン(45秒)明けに雑談を始める


class StubLLM:
    """1回目の呼び出しで指定の行動を返し(latency 秒かかる)、以降は介入しない。
    on_call(t) で、呼ばれた時刻 t を起点に応答待ち中の出来事を予定できる。"""

    name = "stub"

    def __init__(self, action="redirect_topic", speech=REDIRECT, latency=1.0, on_call=None) -> None:
        self.action, self.speech, self.latency, self.on_call = action, speech, latency, on_call
        self.calls, self.call_times, self.clock, self.last_usage = 0, [], time, None
        self.now = lambda: 0.0  # エンジン時刻(SimSession が差し替える)

    def generate_json(self, system, user, schema, context):
        allowed = context["allowed_actions"]
        self.calls += 1
        first = self.calls == 1
        if first and self.on_call:
            self.on_call(self.now())
        self.call_times.append(self.now())
        if first:
            self.clock.sleep(self.latency)
        if len(allowed) == 1:  # 強制イベント
            action, speech = allowed[0], "時間になりました。"
        elif first:
            action, speech = self.action, self.speech
        else:
            action, speech = "no_intervention", ""
        self.last_usage = {"input_tokens": 1000, "output_tokens": 200, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        return {"analysis": ANALYSIS, "decision": {"action": action, "reasoning": "r", "speech": speech, "urgency": 0.8, "target_speaker": None}}


def session(monkeypatch, llm, **overrides) -> SimSession:
    """雑談が1発言出たあとに2秒以上の間ができ、最初の分析が始まるまでの台本を流す。"""
    s = SimSession(monkeypatch, llm, **overrides)
    llm.now = s.now
    s.say(T_CHAT, "speaker_1", "学食のラーメン、一番並んでますよね。")
    s.say(T_CHAT + 1.5, "speaker_2", CHAT[0])
    return s


def run_until_redirect(s: SimSession, until: float = 120.0):
    s.run(until, stop=lambda: bool(s.eng.redirect_log))
    assert s.eng.redirect_log, "軌道修正の結果が記録されない"
    s.run(s.now() + 3.0)  # 発言の記録・読み上げまで進める
    return s.eng.redirect_log[0]


# ---------- 脱線への介入 ----------
def test_redirect_speaks_at_short_pause_while_chat_continues(monkeypatch):
    """応答待ちの間も雑談が続いていても取り消さず、0.8秒の区切りで話す。発言は2文まで。"""
    holder = {}
    llm = StubLLM(latency=1.0, on_call=lambda t: (holder["s"].say(t + 0.1, "speaker_3", CHAT[1], 0.3),
                                                   holder["s"].say(t + 0.5, "speaker_3", CHAT[2], 0.3)))
    s = holder["s"] = session(monkeypatch, llm)
    e = run_until_redirect(s)
    assert e["outcome"] == "spoken" and e["new_utterances"] == 2
    assert 0.0 <= e["wait_sec"] < 3.0
    assert [u.text for u in s.ai("テーマに戻")] == ["盛り上がっているところすみません。少しテーマに戻りましょう。"]  # 2文に制限
    start = s.ai("テーマに戻")[0].start_time  # エンジン時刻
    assert start - max(e for _, e in s.spans if e <= start) >= 0.8 - 1e-6  # 0.8秒の区切りで話した
    m = s.eng.metrics()
    assert m["redirect_spoken"] == 1 and m["redirect_chosen"] == 1


def test_normal_intervention_still_needs_two_seconds(monkeypatch):
    """通常の介入(ask_question)は従来どおり、応答後に2秒の沈黙がなければ取りやめる。"""
    holder = {}
    llm = StubLLM(action="ask_question", speech="原因はどう整理できそうですか?", latency=1.0,
                  on_call=lambda t: holder["s"].say(t + 0.5, "speaker_3", CHAT[1], 0.3))  # 応答の直前に発言
    s = holder["s"] = session(monkeypatch, llm)
    s.run(120, stop=lambda: any(r.llm_called for r in s.eng.agent.history))
    rec = next(r for r in s.eng.agent.history if r.llm_called)
    assert rec.llm_called and "発話取りやめ" in (rec.overridden or "")
    s.run(s.now() + 2)
    assert not s.ai("原因はどう整理") and not s.eng.redirect_log


# ---------- 判断の取り消し ----------
def test_redirect_canceled_when_participants_return_to_topic(monkeypatch):
    holder = {}
    llm = StubLLM(latency=1.0, on_call=lambda t: holder["s"].say(t + 0.2, "speaker_3", "話を戻すと、どの案がいいですかね。", 0.3))
    s = holder["s"] = session(monkeypatch, llm)
    e = run_until_redirect(s)
    assert e["outcome"] == "canceled" and e["reason"].startswith("本題に戻った")
    assert not s.ai("テーマに戻")
    assert "発話取りやめ" in (next(r for r in s.eng.agent.history if r.llm_called).overridden or "")


def test_redirect_canceled_on_multiple_topic_terms(monkeypatch):
    holder = {}
    llm = StubLLM(latency=1.0, on_call=lambda t: holder["s"].say(t + 0.2, "speaker_3", "キャッシュレスレジと空き教室なら、どっちが早いですかね。", 0.3))
    s = holder["s"] = session(monkeypatch, llm)
    e = run_until_redirect(s)
    assert e["outcome"] == "canceled" and "本題の語" in e["reason"]


def test_single_topic_term_does_not_cancel(monkeypatch):
    """本題の語が1つ出ただけ(「学食」)では本題復帰と断定しない。"""
    holder = {}
    llm = StubLLM(latency=1.0, on_call=lambda t: holder["s"].say(t + 0.2, "speaker_3", "学食のカレーも好きなんですよ。", 0.3))
    s = holder["s"] = session(monkeypatch, llm)
    assert run_until_redirect(s)["outcome"] == "spoken"


def test_no_pause_cancels_after_15s_without_interrupting(monkeypatch):
    """区切りがない(0.4秒間隔で発言が続く)まま本番の待機上限15秒に達したら、割り込まずに取り消す。"""
    holder = {}

    def keep_talking(t):
        for i in range(45):  # 約18秒
            holder["s"].say(t + 0.3 + i * 0.4, "speaker_3", CHAT[i % 3], 0.2)

    llm = StubLLM(latency=0.5, on_call=keep_talking)
    s = holder["s"] = session(monkeypatch, llm)
    e = run_until_redirect(s)
    assert e["outcome"] == "canceled" and e["reason"] == "区切りなし"
    assert 15.0 <= e["wait_sec"] < 15.3
    assert not s.ai("テーマに戻")


def test_never_speaks_while_vad_detects_speech(monkeypatch):
    """音声モード相当: VAD が発話中のあいだは話さず、声が止まってから0.8秒の区切りで話す。"""
    holder = {}
    llm = StubLLM(latency=0.5, on_call=lambda t: holder["s"].say(t + 0.1, "speaker_3", CHAT[1], 2.4))  # 2.4秒話し続ける
    s = holder["s"] = session(monkeypatch, llm)
    e = run_until_redirect(s)
    assert e["outcome"] == "spoken" and e["wait_sec"] >= 2.0 + 0.8 - 1e-6
    start = s.ai("テーマに戻")[0].start_time
    assert all(not (a <= start <= b) for a, b in s.spans)


# ---------- 既存機能との整合 ----------
def test_forced_event_preempts_waiting_redirect(monkeypatch):
    holder = {}

    def busy(t):
        holder["s"].say(t + 0.1, "speaker_3", CHAT[1], 10.0)  # 区切りがない状態で待機させる
        holder["s"].at(t + 0.8, lambda: setattr(holder["s"].eng.state, "duration_sec", holder["s"].now() + 1.0))  # 時間切れが迫る

    llm = StubLLM(latency=0.5, on_call=busy)
    s = holder["s"] = session(monkeypatch, llm)
    e = run_until_redirect(s)
    assert e["outcome"] == "canceled" and e["reason"] == "強制イベントを優先"
    s.run(s.now() + 30, stop=lambda: s.eng.status == "finished")
    assert any(iv.action.value in ("time_warning", "close_gd") for iv in s.eng.state.interventions)


@pytest.mark.parametrize("how, reason", [("pause", "一時停止"), ("stop", "セッション終了")])
def test_pause_or_stop_cancels_waiting_redirect(monkeypatch, how, reason):
    holder = {}

    def busy(t):
        holder["s"].say(t + 0.1, "speaker_3", CHAT[1], 10.0)
        holder["s"].at(t + 0.8, lambda: getattr(holder["s"].eng, how)())

    llm = StubLLM(latency=0.5, on_call=busy)
    s = holder["s"] = session(monkeypatch, llm)
    e = run_until_redirect(s)
    assert e["outcome"] == "canceled" and e["reason"] == reason
    assert not s.ai("テーマに戻")


# ---------- 計測 ----------
def test_metrics_record_llm_usage_and_latency(monkeypatch):
    s = session(monkeypatch, StubLLM(latency=0.5))
    run_until_redirect(s)
    m = s.eng.metrics()
    assert m["llm_calls"] >= 1 and m["llm_tokens"]["input_tokens"] >= 1000 and m["llm_latency_avg"] == pytest.approx(0.5)
    rec = next(r for r in s.eng.agent.history if r.llm_called)
    assert rec.llm_usage["output_tokens"] == 200 and rec.context_last_utt_id is not None
    assert "metrics" in s.eng.snapshot()


def test_anthropic_llm_records_usage_without_network():
    llm = AnthropicLLM.__new__(AnthropicLLM)
    llm.settings, llm.name, llm.last_usage = load_settings(), "anthropic:test", None
    usage = SimpleNamespace(input_tokens=1200, output_tokens=300, cache_creation_input_tokens=0, cache_read_input_tokens=900)
    resp = SimpleNamespace(usage=usage, stop_reason="end_turn", content=[SimpleNamespace(type="text", text='{"ok": 1}')])
    llm.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: resp)))
    assert llm.generate_json("s", "u", {}, {}) == {"ok": 1}
    assert llm.last_usage == {"input_tokens": 1200, "output_tokens": 300, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 900}


# ---------- 本題復帰の判定(単体) ----------
def test_topic_return_detection_rules():
    a = Analysis(**ANALYSIS)
    drift = ["学食のラーメンが一番並んでますよね。", "駅前のラーメン屋の方がおいしい。"]
    assert detect_topic_return(THEME, a, drift, ["そういえば本題ですが"])
    assert detect_topic_return(THEME, a, drift, ["空き教室とキャッシュレスレジ、どっちが早い?"])
    assert detect_topic_return(THEME, a, drift, ["学食のカレーが好き"]) is None  # 1語だけ
    assert detect_topic_return(THEME, a, drift, ["駅前のラーメン屋、替え玉無料"]) is None  # 脱線部分の語は数えない
    assert detect_topic_return(THEME, a, drift, ["じゃあ次に進みましょう"]) is None  # 限界: 本題の語がない復帰は検出できない


# ---------- 分析の限界(API なしで確認) ----------
def test_continuous_chat_leaves_no_analysis_window():
    """シナリオ03の脱線部分(発言の間隔が1秒程度)では、定期分析の時刻でも沈黙条件を満たさず LLM が呼ばれない。"""
    sc = load_scenario(ROOT / "scenarios" / "03_off_topic.yaml")
    s = load_settings()
    st = DiscussionState(settings=s, theme=sc["theme"], duration_sec=sc["duration_sec"])
    st.started, st.start_time = True, 0.0
    agent = FacilitatorAgent(st, MockLLM(), s)
    pending = sorted(sc["utterances"], key=lambda u: u["t"] + u["d"])
    skipped, ticks, t = 0, [], float(sc["start_elapsed"])
    while t <= 240:
        while pending and pending[0]["t"] + pending[0]["d"] <= t:
            u = pending.pop(0)
            st.add_utterance(u["s"], u["text"], u["t"], u["t"] + u["d"], u.get("c", 1.0))
        st.speaking_now = any(u["t"] <= t < u["t"] + u["d"] for u in pending)
        trig = agent.should_tick(t)
        skipped += agent.skipped_due is not None
        if trig:
            ticks.append(t)
            agent.tick(t, trig)
        t += 0.5
    assert not [x for x in ticks if 205 <= x <= 240]  # 脱線が続く間、一度も分析されない
    assert skipped * 0.5 >= 30  # 定期分析の時刻なのに見送った時間が30秒以上


# ---------- 実スレッドでの確認(少数) ----------
def test_real_threads_redirect_smoke(monkeypatch):
    """本番と同じスレッド構成(判断ループ・判断スレッド)で、軌道修正が区切りで話されること。"""
    llm = StubLLM(latency=0.3)
    monkeypatch.setattr(engine_mod, "make_llm", lambda settings: llm)
    eng = Engine(load_settings(), tts_chars_per_sec=1000.0, cooldown_sec=0.0, analysis_interval_sec=0.5)
    eng.start(SessionConfig(THEME, 5, ["A", "B"], mode="text", intro=False, tts=False))
    try:
        end = time.monotonic() + 3
        while not eng.state.started and time.monotonic() < end:
            time.sleep(0.02)
        eng.add_text("speaker_1", CHAT[0])
        end = time.monotonic() + 6
        while not eng.redirect_log and time.monotonic() < end:
            time.sleep(0.02)
        assert eng.redirect_log and eng.redirect_log[0]["outcome"] == "spoken"
        assert threading.active_count() >= 2
    finally:
        eng.stop()
