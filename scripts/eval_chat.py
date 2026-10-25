"""雑談の司会(ChatHostAgent)の会話シナリオ評価(第5段階)。

実際の ChatHostAgent(状態・ルール層・行動の検証・話題の管理)を疑似時刻で動かし、LLM だけを本物(既定は Claude)にする。
参加者は台本の人物で、AI の発言に反応して話す(responder)。LLM の生の判断(行動・発言・analysis)と、
ルール層・検証を通した後の実際の行動を両方記録する。

限界: 疑似時刻なので、LLM の応答時間(5〜13秒)・音声認識の遅れ・実際の会話のテンポは再現しない。
判断の間に参加者が話し始める状況(stale_reason による取りやめ)もここでは起きない。

使い方:
  .venv/Scripts/python.exe scripts/eval_chat.py                  # 全シナリオ(1回ずつ)
  .venv/Scripts/python.exe scripts/eval_chat.py 01 06 --repeat 2  # 一部を2回ずつ
  .venv/Scripts/python.exe scripts/eval_chat.py --llm mock        # 配線確認(API なし)
結果: logs/chat-eval-<時刻>.md と .json。API の呼び出しは全体で --max-calls 回まで(既定 80)。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_settings  # noqa: E402
from facilitator.chat.host import ChatHostAgent  # noqa: E402
from facilitator.llm.base import LLMError, generate  # noqa: E402
from facilitator.llm.factory import make_llm  # noqa: E402
from facilitator.models import Action, Decision  # noqa: E402
from facilitator.policy import preset_policy  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402

PRICE = {"input_tokens": 4.0, "output_tokens": 20.0, "cache_creation_input_tokens": 5.0, "cache_read_input_tokens": 0.20}
NAMES = {"speaker_1": "たろう", "speaker_2": "はなこ", "speaker_3": "けんた"}
Line = tuple[float, float, str, str]  # (AI が話し終えてから話し始めるまでの秒数, 長さ, 話者, 発言)


class BudgetExceeded(Exception):
    pass


class Recorder:
    """LLM の入出力と使用トークンを記録する。全体の呼び出し回数に上限をかける。"""

    def __init__(self, inner: Any, max_calls: int) -> None:
        self.inner, self.name, self.max_calls = inner, inner.name, max_calls
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, system, user, schema, context, on_partial=None):
        if len(self.calls) >= self.max_calls:
            raise BudgetExceeded(f"API の呼び出しが上限 {self.max_calls} 回に達した")
        call: dict[str, Any] = {"user": user}
        t0 = time.monotonic()
        try:
            out = generate(self.inner, system, user, schema, context, None)
            call["response"] = out
            return out
        except LLMError as e:
            call["error"] = str(e)
            raise
        finally:
            call["latency"] = round(time.monotonic() - t0, 2)
            call["usage"] = getattr(self.inner, "last_usage", None)
            self.calls.append(call)

    def __getattr__(self, name):
        return getattr(self.inner, name)


@dataclass
class Scenario:
    key: str
    title: str
    expect: str
    responder: Callable[[int, str, str, ChatHostAgent], list[Line]]
    until: float
    theme: str = "自由な雑談"
    setup: Callable[[DiscussionState, ChatHostAgent], float] | None = None  # 開始前の状態を作る。返り値は開始時刻
    opening: str | None = None  # 開始の話題を固定する(比較しやすくするため)


@dataclass
class Run:
    scenario: Scenario
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def simulate(sc: Scenario, llm: Recorder, settings) -> Run:
    st = DiscussionState(settings=settings, theme=sc.theme, duration_sec=600)
    for sid, n in NAMES.items():
        st.set_speaker_name(sid, n)
    ag = ChatHostAgent(st, llm, settings, purpose="general", discussion_type="casual",
                       policy=preset_policy("general_gw"), rng=random.Random(0))
    if sc.opening:
        ag.opening = sc.opening
    run = Run(sc)
    t = sc.setup(st, ag) if sc.setup else 0.0
    pending: list[tuple[float, float, str, str]] = []
    turn = 0
    seen_phase = 0
    try:
        while t <= sc.until:
            for item in sorted(pending, key=lambda x: x[1]):
                if item[1] <= t:
                    pending.remove(item)
                    u = st.add_utterance(item[2], item[3], item[0], item[1], 0.9)
                    run.events.append({"t": round(u.start_time, 1), "kind": "human", "who": NAMES[u.speaker_id], "text": u.text})
            st.speaking_now = any(a <= t < b for a, b, _s, _x in pending)
            if not any(u.source == "ai" and u.end_time > t for u in st.utterances):
                trig = ag.should_tick(t)
                if trig:
                    n_calls = len(llm.calls)
                    ts = ag.topic_status(t)  # ルール層が許可を決めるときに見た話題の状態
                    rec = ag.tick(t, trig)
                    raw = llm.calls[-1].get("response") if len(llm.calls) > n_calls else None
                    ev = {
                        "t": round(t, 1), "kind": "judge", "trigger": trig, "gate": rec.gate,
                        "allowed": rec.allowed_actions, "phase": ag.chat_phase,
                        "raw": raw and raw.get("decision"), "analysis": raw and raw.get("analysis"),
                        "action": rec.decision.action.value, "speech": rec.decision.speech,
                        "overridden": rec.overridden, "llm": rec.llm_called,
                        "topic": ts.topic and {"text": ts.topic.text, "by": ts.topic.source},
                        "reactions": ts.reactions, "topic_after": ag.topics.current and ag.topics.current.text,
                    }
                    run.events.append(ev)
                    if rec.decision.action != Action.no_intervention:
                        end = st.utterances[-1].end_time
                        for delay, dur, sid, text in sc.responder(turn, rec.decision.action.value, rec.decision.speech, ag) or []:
                            pending.append((end + delay, end + delay + dur, sid, text))
                        turn += 1
            for ph in ag.chat_log[seen_phase:]:
                run.events.append({"t": ph[0], "kind": "phase", "to": ph[1], "why": ph[2]})
            seen_phase = len(ag.chat_log)
            t = round(t + 0.25, 2)
    except BudgetExceeded as e:
        run.error = str(e)
    run.events.append({"t": round(t, 1), "kind": "end", "topics": [(h.text, h.source) for h in ag.topics.history]})
    return run


# ================= シナリオ =================
def chain(lines: list[tuple[str, str]], start: float = 1.0, gap: float = 1.0, dur: float = 3.0) -> list[Line]:
    """続けて話す発言(別の人が gap 秒の間で応じる = つながった発言)。"""
    return [(start + k * (dur + gap), dur, sid, text) for k, (sid, text) in enumerate(lines)]


def r01(i, action, speech, ag):  # 1人だけが答え、その後は呼びかけ方次第で少しだけ
    if i == 0:
        return [(1.5, 4, "speaker_1", "最近は家でパンを焼くのにハマってます。")]
    if i == 1 and action in ("invite_all", "invite_person"):
        return [(2, 3, "speaker_2", "パンいいですね、私は食べる専門です。")]
    if i == 1 and action == "ask_question":
        return [(1.5, 5, "speaker_1", "最初はホットケーキミックスで、だんだん本格的になってきました。")]
    return []


SHORT = [("speaker_1", "特にないです。"), ("speaker_2", "うーん、寝てるくらいです。"), ("speaker_3", "あんまり…。"),
         ("speaker_1", "普通です。"), ("speaker_2", "どうだろう…。"), ("speaker_3", "特には。")]


def r02(i, action, speech, ag):  # 毎回1人が短く答えるだけ
    sid, text = SHORT[i % len(SHORT)]
    return [(2, 1.5, sid, text)]


def r03(i, action, speech, ag):  # 1人が答え、深掘り・橋渡しのあと3人で広がる
    if i == 0:
        return [(1.5, 4, "speaker_1", "最近キャンプにハマってて、先週も行ってきました。")]
    if i == 1:
        return chain([("speaker_1", "湖のそばで焚き火をしたんですけど、すごくよかったです。"),
                      ("speaker_2", "焚き火いいですね!どこの湖ですか?"),
                      ("speaker_1", "本栖湖です。富士山も見えて。"),
                      ("speaker_3", "本栖湖、僕も行ったことあります。夜すごく寒くなかったですか?"),
                      ("speaker_1", "寒かったです!寝袋だけじゃ足りなくて。"),
                      ("speaker_2", "私もキャンプやってみたいんですけど、道具が大変そうで。"),
                      ("speaker_3", "最初はレンタルでもいいらしいですよ。")])
    return []


def r04(i, action, speech, ag):  # 開始直後から3人で盛り上がる
    if i == 0:
        return chain([("speaker_1", "最近、朝ドラにハマってます。"), ("speaker_2", "え、私も見てます!"),
                      ("speaker_3", "僕も!今週の展開すごくなかったですか?"), ("speaker_1", "わかります、泣きました。"),
                      ("speaker_2", "あの主人公のお父さん、いい人ですよね。"), ("speaker_3", "そうそう、最初は怖かったのに。"),
                      ("speaker_1", "来週どうなるんだろう。"), ("speaker_2", "予告だとまた波乱がありそうでしたね。"),
                      ("speaker_3", "楽しみですね。")])
    return []


def r05(i, action, speech, ag):  # 開始の話題で盛り上がり、自然に映画の話へ移って、止まる
    if i == 0:
        return chain([("speaker_1", "休みの日は家でゆっくりしてます。"), ("speaker_2", "わかります、私もです。"),
                      ("speaker_3", "家で何してるんですか?"), ("speaker_1", "配信で映画を見てることが多いです。"),
                      ("speaker_2", "最近何か見ました?"), ("speaker_1", "宇宙の映画を見ました、すごくよかったです。"),
                      ("speaker_3", "あ、僕もそれ映画館で見ました!音がすごかった。"),
                      ("speaker_2", "そんなにいいんですね、見てみようかな。")])
    return []  # その後、会話が止まる(再介入の内容を見る)


def r06(i, action, speech, ag):  # 誰も答えない
    return []


def r07(i, action, speech, ag):  # 話題で盛り上がり、一段落して止まる
    if i == 0:
        return chain([("speaker_1", "最近はラーメンの食べ歩きにハマってます。"), ("speaker_2", "いいですね、どこがおすすめですか?"),
                      ("speaker_1", "駅前の味噌ラーメンのお店がおいしいです。"), ("speaker_3", "あそこ並びますよね。"),
                      ("speaker_1", "平日の昼なら空いてますよ。"), ("speaker_2", "今度行ってみます。"),
                      ("speaker_3", "僕も行ってみます。"), ("speaker_1", "ぜひぜひ。")])
    return []


def r08(i, action, speech, ag):  # 1人が答えたあと、AI に話題を頼む
    if i == 0:
        return [(1.5, 3, "speaker_1", "うーん、特にないかもです。"), (6, 3, "speaker_2", "AIさん、何か話題ある？")]
    if i == 1:
        return [(2, 4, "speaker_3", "それなら、この前ちょっと遠出しました。")]
    return []


def r09(i, action, speech, ag):  # 答えた人は、自分への深掘り・個人指名には答えを避ける。全体への呼びかけには別の人が1回だけ答える
    st = ag.state
    last = next((u for u in reversed(st.utterances) if u.source == "human"), None)
    if i == 0:
        return [(1.5, 4, "speaker_1", "休みの日は実家に帰ることが多いです。")]
    if action in ("ask_question", "invite_person") and last is not None:
        return [(1.5, 4, last.speaker_id, "あー、そこはちょっと…あんまり話したくないかもです。")]
    if action in ("invite_all", "offer_topic", "change_topic") and not any(u.speaker_id == "speaker_2" for u in st.utterances):
        return [(2, 3, "speaker_2", "私は最近カフェ巡りしてます。")]
    return []


def setup10(st, ag) -> float:
    """以前に4つの話題を出して、どれも広がらなかった状態から始める(誰も答えない)。"""
    st.started, st.start_time = True, 0.0
    past = [(Action.start_gd, "こんにちは、今日はよろしくお願いします。まずは、最近ちょっとハマっていることはありますか?"),
            (Action.change_topic, "話は変わりますが、最近食べたもので、おいしかったものはありますか?"),
            (Action.change_topic, "話は変わりますが、最近行ってよかった場所はありますか?"),
            (Action.change_topic, "話は変わりますが、休みの日は、どんなふうに過ごすことが多いですか?")]
    ag.opening = "最近ちょっとハマっていることはありますか?"
    for k, (a, text) in enumerate(past):
        ag._act(Decision(action=a, speech=text), 25.0 * k)
        st.add_utterance("speaker_1" if k % 2 == 0 else "speaker_2", ["特にないです。", "うーん。"][k % 2],
                         25.0 * k + 6, 25.0 * k + 8, 0.9)
    ag._act(Decision(action=Action.invite_all, speech="どなたからでも大丈夫ですよ。"), 100.0)
    return 101.0


SCENARIOS = [
    Scenario("01", "開始の話題に1人だけが答える", "1人の答えを受けて自然に深掘りか橋渡し。すぐに話題を変えない", r01, 70, opening="最近ちょっとハマっていることはありますか?"),
    Scenario("02", "参加者が短く答えるだけ", "質問攻めにしない。同じ問いを重ねず、反応が薄い状態が続けば話題を変える", r02, 130, opening="休みの日は、どんなふうに過ごすことが多いですか?"),
    Scenario("03", "深掘りで会話が広がる", "広がったら話題を維持して見守る。話題を変えない", r03, 75, opening="最近ちょっとハマっていることはありますか?"),
    Scenario("04", "開始直後から盛り上がる", "開始の発言のあとは介入しない", r04, 60, opening="最近ちょっとハマっていることはありますか?"),
    Scenario("05", "自走期に参加者が話題を変え、その後止まる", "再介入では最新の話題(映画)を踏まえる。topic_by=participants", r05, 90, opening="休みの日は、どんなふうに過ごすことが多いですか?"),
    Scenario("06", "AI の話題に誰も答えない", "同じような質問を連発しない。間をあけ、言い方や話題を変える", r06, 150, opening="最近ちょっとハマっていることはありますか?"),
    Scenario("07", "会話が一段落する", "一段落して止まったら、自然なつなぎで次の話題を出す", r07, 90, opening="最近ちょっとハマっていることはありますか?"),
    Scenario("08", "AI に直接「何か話題ある？」と頼む", "依頼に応じて答えやすい話題を出す", r08, 50, opening="最近ちょっとハマっていることはありますか?"),
    Scenario("09", "個人的な質問を避ける参加者がいる", "深追いせず、受け止めて全体へ広げるか話題を変える", r09, 75, opening="休みの日は、どんなふうに過ごすことが多いですか?"),
    Scenario("10", "以前と似た話題を出しそうな状況", "最近の話題(趣味・食べ物・場所・休日)と重ならない話題を出す", r06, 160, setup=setup10),
]


# ================= 出力 =================
def cost(calls: list[dict[str, Any]]) -> tuple[dict[str, int], float]:
    tok: dict[str, int] = {}
    for c in calls:
        for k, v in (c.get("usage") or {}).items():
            tok[k] = tok.get(k, 0) + v
    return tok, sum(PRICE.get(k, 0) * v / 1e6 for k, v in tok.items())


def render(runs: list[Run], llm: Recorder) -> str:
    out = [f"# 雑談の司会の評価(LLM: {llm.name})", ""]
    tok, usd = cost(llm.calls)
    lat = [c["latency"] for c in llm.calls if "latency" in c]
    out += [f"- LLM 呼び出し: {len(llm.calls)} 回 / トークン: {tok} / 概算 ${usd:.3f}",
            f"- 応答時間: 平均 {sum(lat) / len(lat):.1f} 秒" if lat else "- 応答時間: -", ""]
    for r in runs:
        sc = r.scenario
        out += [f"## {sc.key} {sc.title}", f"期待: {sc.expect}", ""]
        for e in r.events:
            if e["kind"] == "human":
                out.append(f"- `{e['t']:6.1f}` 👤 {e['who']}: {e['text']}")
            elif e["kind"] == "phase":
                out.append(f"- `{e['t']:6.1f}` ── 時期 → {e['to']}({e['why']})")
            elif e["kind"] == "end":
                out.append(f"- 話題の記録: {e['topics']}")
            elif e["kind"] == "judge":
                head = f"- `{e['t']:6.1f}` 🤖 [{e['trigger']} / {e['gate']}]"
                if not e["llm"]:
                    out.append(f"{head} {e['action']}" + (f": {e['speech']}" if e["speech"] else ""))
                    continue
                raw, an = e["raw"] or {}, e["analysis"] or {}
                out.append(f"{head} 許可={','.join(e['allowed'])}")
                out.append(f"    - LLM: **{raw.get('action')}** 「{raw.get('speech', '')}」 理由: {raw.get('reasoning', '')}")
                out.append(f"    - analysis: health={an.get('health')} phase={an.get('phase')} topic=「{an.get('topic')}」"
                           f" topic_by={an.get('topic_by')}")
                if e["overridden"] or raw.get("action") != e["action"] or raw.get("speech", "") != e["speech"]:
                    out.append(f"    - 実行: **{e['action']}** 「{e['speech']}」 補正: {e['overridden']}")
                out.append(f"    - 判断時の話題: {e['topic']} 反応={e['reactions']}" +
                           (f" → 判断後: {e['topic_after']}" if e["topic"] and e["topic_after"] != e["topic"]["text"] else ""))
        if r.error:
            out.append(f"- ⚠ {r.error}")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", help="シナリオの番号(省略で全部)")
    ap.add_argument("--llm", default="anthropic")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--max-calls", type=int, default=80)
    args = ap.parse_args()
    settings = load_settings(llm_provider=args.llm, num_participants=len(NAMES))  # 台本の人間は3人
    llm = Recorder(make_llm(settings), args.max_calls)
    runs = []
    for sc in [s for s in SCENARIOS if not args.keys or s.key in args.keys]:
        for _ in range(args.repeat):
            runs.append(simulate(sc, llm, settings))
            print(f"{sc.key}: 呼び出し累計 {len(llm.calls)}", flush=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    md = settings.log_dir / f"chat-eval-{stamp}.md"
    md.write_text(render(runs, llm), encoding="utf-8")
    (settings.log_dir / f"chat-eval-{stamp}.json").write_text(
        json.dumps([{"key": r.scenario.key, "events": r.events, "error": r.error} for r in runs], ensure_ascii=False, indent=1),
        encoding="utf-8")
    tok, usd = cost(llm.calls)
    print(f"LLM 呼び出し {len(llm.calls)} 回 / 概算 ${usd:.3f} / {md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
