"""AI 参加者入りのシミュレーション(仮想時計)。

人間の発言は台本(YAML)どおりに流し、進行役の判断と AI 参加者の発言をアプリと同じロジックで動かす。
--llm handoff にすると、立場の生成・AI 参加者の発言・進行役の判断がすべてチャット経由になる。

使い方:
  python scripts/simulate.py scenarios/bots/b1_practice.yaml --llm handoff
  python scripts/simulate.py scenarios/bots/b2_reserved.yaml --llm mock
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from facilitator.agent.agent import FacilitatorAgent  # noqa: E402
from facilitator.llm.base import LLMError  # noqa: E402
from facilitator.llm.factory import make_llm  # noqa: E402
from facilitator.logging import TickLogger  # noqa: E402
from facilitator.participants import (  # noqa: E402
    bot_mode_for,
    generate_personas,
    generate_utterance,
    pick_bot,
    transcript_for_bot,
)
from facilitator.report import RecordingLLM, TestReport  # noqa: E402
from facilitator.scenario import load_scenario  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402

MODE_JA = {"active": "練習相手モード(間があれば発言)", "reserved": "控えめモード(名指し・長い沈黙のみ)"}


def mmss(sec: float) -> str:
    return f"{int(sec) // 60:02d}:{int(sec) % 60:02d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", type=Path)
    ap.add_argument("--llm", choices=["anthropic", "mock", "handoff"])
    ap.add_argument("--until", type=float, help="終了時刻(経過秒)。省略時は台本の until、なければ最後の発言+20秒")
    ap.add_argument("--step", type=float, default=0.5)
    args = ap.parse_args()

    sc = load_scenario(args.scenario)
    s = load_settings(**({"llm_provider": args.llm} if args.llm else {}))
    humans = sc.get("participants") or {}
    bots = int(sc.get("ai_count", 0))
    s.num_participants = len(humans) + bots
    st = DiscussionState(settings=s, theme=sc["theme"], duration_sec=sc["duration_sec"])
    for sid, name in humans.items():
        st.set_speaker_name(sid, name, confirmed=True)
    llm = RecordingLLM(make_llm(s))  # 入力と返事をまとめログに残す
    purpose = sc.get("purpose", "job")
    report = TestReport("AI 参加者入りシミュレーション(scripts/simulate.py)", sc["id"], {"LLM": llm.name})
    sec = report.section(sc["title"])
    sec.state = st

    print(f"=== {sc['title']} / テーマ: {sc['theme']} ===")
    m = llm.mark()
    personas = generate_personas(llm, sc["theme"], purpose, bots, list(humans.values()), len(humans) + 1)
    sec.personas, sec.persona_call = personas, llm.since(m)
    for p in personas:
        st.set_speaker_name(p.speaker_id, p.name, confirmed=True)
        st.bot_ids.add(p.speaker_id)
    mode = bot_mode_for(len(humans), bots)
    print(f"人間 {len(humans)}人 / AI参加者 {bots}人 → {MODE_JA.get(mode, 'AI参加者なし')}")
    sec.info = {"テーマ": sc["theme"], "人数": f"人間 {len(humans)}人 / AI参加者 {bots}人", "AI参加者のモード": MODE_JA.get(mode, "なし")}
    for p in personas:
        print(f"  🤖 {p.name}【{p.stance}】{p.opinion}")

    agent = FacilitatorAgent(st, llm, s, logger=TickLogger(s.log_dir, f"sim-{sc['id']}"), purpose=purpose)
    pending = sorted(sc["utterances"], key=lambda u: u["t"] + u["d"])
    end = args.until or sc.get("until") or max(u["t"] + u["d"] for u in pending) + 20
    t, busy_until, last_bot_end, answered = 0.0, -1.0, -1e9, -1

    while t <= end and not st.finished:
        while pending and pending[0]["t"] + pending[0]["d"] <= t:
            u = pending.pop(0)
            st.add_utterance(u["s"], u["text"], u["t"], u["t"] + u["d"], u.get("c", 1.0))
            print(f"[{mmss(u['t'])}] 👤 {humans.get(u['s'], u['s'])}: {u['text']}")
        st.speaking_now = any(u["t"] <= t < u["t"] + u["d"] for u in pending)
        if t >= busy_until:
            # 1. 進行役
            trig = agent.should_tick(t)
            if trig:
                m = llm.mark()
                rec = agent.tick(t, trig)
                sec.decision(rec, llm.since(m))
                d = rec.decision
                if d.action.value != "no_intervention":
                    dur = len(d.speech) / s.tts_chars_per_sec
                    busy_until = t + dur
                    print(f"[{mmss(st.elapsed(t))}] 🎙 進行役 {d.action.value}: {d.speech}\n        └ 理由: {d.reasoning}")
                elif rec.gate == "llm":
                    print(f"[{mmss(st.elapsed(t))}]    (進行役: 見守り — {d.reasoning})")
            # 2. AI 参加者(進行役が話していないときだけ)
            if t >= busy_until:
                picked = pick_bot(st, personas, mode, t, last_bot_end, answered, len(humans))
                if picked:
                    p, reason, addressed = picked
                    m = llm.mark()
                    try:
                        text = generate_utterance(llm, p, purpose, st.theme, transcript_for_bot(st), reason)
                    except LLMError as e:
                        text = ""
                        print(f"        (AI参加者 {p.name} の発言を生成できませんでした: {e})")
                    if text:
                        dur = len(text) / s.tts_chars_per_sec
                        st.add_utterance(p.speaker_id, text, t, t + dur, 1.0, source="bot")
                        sec.bot(p.speaker_id, t, reason, llm.since(m))
                        p.spoken += 1
                        busy_until = last_bot_end = t + dur
                        if addressed >= 0:
                            answered = addressed
                        print(f"[{mmss(st.elapsed(t))}] 🤖 {p.name}: {text}\n        └ 話した理由: {reason}")
        t += args.step

    print(f"=== 終了 経過 {mmss(st.elapsed(t))} / 進行役の介入 {len(st.interventions)}回 ===")
    t_end = max([t] + [u.end_time for u in st.utterances])  # 話している途中の発言も集計に含める
    for x in st.speaker_stats(t_end):
        tag = "AI参加者" if x.is_bot else "人間"
        print(f"  {x.label:<6}({tag}) 発言 {x.turns}回・{x.total_speech_sec:.0f}秒")
    print(f"\n📄 まとめログ: {report.write(s.log_dir)}")


if __name__ == "__main__":
    main()
