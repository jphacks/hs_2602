"""脱線への早期介入の時系列テスト。実際のエンジンに、参加者の発言を時刻どおりに流し込む(API は呼ばない)。

- 疑似時計(tests/sim.py)で回すので実時間は待たない。クールダウン45秒・待機上限15秒は本番どおり。
  脱線の台本では、決めた2.6秒の間で分析が始まるよう分析間隔だけ1秒にする(元の実時間版と同じ)
- 音声モード相当: 発話中は VAD の speaking_now を立て、話し終えた時刻に発言を確定し、声の終わりを記録する
- LLM は発話ログの中身で判断するモック(本題の語がない発言が3つ続いたら脱線とみなす)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.sim import SimSession, chat_script  # noqa: E402

THEME = "大学の学食の混雑を改善するには？"
TOPIC = ("学食", "混雑", "レジ", "席", "対策", "原因", "キャッシュレス", "並")
ON = ["学食の混雑って、やっぱりレジが原因ですよね。", "席が足りないのも原因だと思います。"]
CHAT = ["駅前のラーメン屋、替え玉無料なんですよ。", "え、知らなかった。今度行こうかな。", "あそこの近くでバイトしてるんです。",
        "バイト何してるんですか？", "居酒屋です。まかないが出ます。", "いいなあ、唐揚げとか？"]


T0 = 50.0  # 開始の説明のクールダウン(45秒)明けに台本を始める


class JudgeLLM:
    """発話ログの末尾で本題の語がない人間の発言が3つ続いていれば redirect_topic(1回だけ)。応答に latency 秒かかる。"""

    name = "mock-judge"

    def __init__(self, latency: float = 1.0) -> None:
        self.latency, self.redirected, self.calls, self.clock = latency, False, 0, time

    def generate_json(self, system, user, schema, context):
        self.calls += 1
        self.clock.sleep(self.latency)
        allowed = context["allowed_actions"]
        human = [line.split(": ", 1)[1] for line in context["transcript"] if "AI(" not in line]
        run = 0
        for text in reversed(human):
            if any(w in text for w in TOPIC):
                break
            run += 1
        action, speech = "no_intervention", ""
        if len(allowed) == 1:
            action, speech = allowed[0], "時間になりました。"
        elif run >= 3 and not self.redirected and "redirect_topic" in allowed:
            self.redirected = True
            action = "redirect_topic"
            speech = "盛り上がっているところすみません。学食の混雑の話に戻りましょう。レジの原因はどうでしたか?"
        analysis = {"phase": "cause_analysis", "phase_reason": "r", "health": "off_topic" if run >= 3 else "progressing",
                    "issues": ["混雑の原因"], "opinions": ["キャッシュレスレジ"], "conflicts": [], "agreements": [],
                    "missing_perspectives": [], "recent_summary": ""}
        return {"analysis": analysis, "decision": {"action": action, "reasoning": "r", "speech": speech, "urgency": 0.8, "target_speaker": None}}


def ai_turns(eng, part):
    return [u for u in eng.state.utterances if u.source == "ai" and part in u.text]


def drift_script(after):
    """本題2発言 → 雑談3発言(間0.6秒)→ 一度だけ2.6秒の間(ここで分析が始まる)→ after(t_gap) の台本。"""
    script = chat_script(ON, start=T0, gap=1.0) + chat_script(CHAT[:3], start=T0 + 4.0, gap=0.6)
    t_gap = script[-1][0] + script[-1][3] + 2.6  # 応答待ちの間に次の発言が始まる
    return script + after(t_gap)


def run_until_redirect(s, until):
    s.run(until, stop=lambda: bool(s.eng.redirect_log))
    s.run(s.now() + 3)
    return s.eng.redirect_log


# 1. 継続的な脱線: 判断後も雑談が続くが取り消さず、次の短い区切りで話す
def test_continuous_drift_redirects_at_short_pause_without_overlap(monkeypatch):
    s = SimSession(monkeypatch, JudgeLLM(latency=1.0), analysis_interval_sec=1.0)  # 2.6秒の間で分析が始まるように
    s.play(drift_script(lambda t: chat_script(CHAT[3:], start=t, gap=1.2)))  # 待ち中も雑談(間1.2秒)
    log = run_until_redirect(s, T0 + 40)
    assert log and log[0]["outcome"] == "spoken", log
    ai = ai_turns(s.eng, "戻りましょう")
    assert len(ai) == 1 and ai[0].text.count("。") + ai[0].text.count("?") <= 2  # 1〜2文
    start = ai[0].start_time
    assert all(not (a <= start <= b) for a, b in s.spans), "人の発言に重ねて話した"
    prev_end = max(b for a, b in s.spans if b <= start)
    assert start - prev_end >= 0.8 - 1e-6  # 0.8秒の区切りで話した
    assert 0 < log[0]["new_utterances"]  # 判断後に増えた雑談では取り消していない


# 2. 一時的な雑談・関連する経験談: 分析は走るが介入しない
def test_brief_chat_or_related_anecdote_is_left_alone(monkeypatch):
    llm = JudgeLLM(latency=0.5)
    s = SimSession(monkeypatch, llm)
    s.play(chat_script(ON + [CHAT[0], "そういえば、学食のレジで5分待ったことがあります。", "それ、まさに混雑の原因ですね。"], start=T0, gap=2.5))
    s.run(T0 + 30)
    assert llm.calls >= 2  # 2秒以上の間で分析は行われている
    assert not ai_turns(s.eng, "戻りましょう") and not s.eng.redirect_log


# 3. 判断後に本題へ戻った: 待機中に「話を戻すと〜」→ 古い軌道修正を取り消す
def test_stale_redirect_canceled_when_group_returns(monkeypatch):
    s = SimSession(monkeypatch, JudgeLLM(latency=1.5), analysis_interval_sec=1.0)  # 2.6秒の間で分析が始まるように
    s.play(drift_script(lambda t: [(t, "speaker_1", "話を戻すと、レジの対策を考えましょうか。", 1.0)]))
    log = run_until_redirect(s, T0 + 40)
    assert log and log[0]["outcome"] == "canceled" and log[0]["reason"].startswith("本題に戻った"), log
    assert not ai_turns(s.eng, "戻りましょう")


# 4. 区切りがない: 判断後も0.5秒の間で話し続ける → 重ねて話さず、15秒で取り消す
def test_no_pause_for_15s_cancels_without_overlap(monkeypatch):
    s = SimSession(monkeypatch, JudgeLLM(latency=1.0), analysis_interval_sec=1.0)  # 2.6秒の間で分析が始まるように
    s.play(drift_script(lambda t: chat_script((CHAT * 6)[:14], start=t, gap=0.5, dur=0.8)))  # 約18秒
    log = run_until_redirect(s, T0 + 60)
    assert log and log[0]["outcome"] == "canceled" and log[0]["reason"] == "区切りなし", log
    assert 15.0 <= log[0]["wait_sec"] < 15.3
    assert not ai_turns(s.eng, "戻りましょう")


# 通常の介入・時間告知・終了が変わっていないこと(短いGDを最後まで流す。設定は本番のまま)
def test_normal_flow_time_warning_and_close_unchanged(monkeypatch):
    s = SimSession(monkeypatch, JudgeLLM(latency=0.2), minutes=1.0)
    s.play(chat_script(ON, start=5.0, gap=1.0))
    s.run(80, stop=lambda: s.eng.status == "finished")
    acts = [iv.action.value for iv in s.eng.state.interventions]
    assert acts[0] == "start_gd" and "time_warning" in acts and acts[-1] == "close_gd", acts
    assert "redirect_topic" not in acts and not s.eng.redirect_log
