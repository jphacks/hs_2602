"""1つのシナリオを仮想時計で時系列再生し、実運用と同じトリガー(定期 + 沈黙 + 時間閾値)でエージェントを動かす。

使い方:
  python scripts/run_scenario.py scenarios/02_stalled.yaml
  python scripts/run_scenario.py scenarios/05_time_short.yaml --until 610 --llm mock
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from facilitator.agent.agent import FacilitatorAgent  # noqa: E402
from facilitator.llm.factory import make_llm  # noqa: E402
from facilitator.logging import TickLogger  # noqa: E402
from facilitator.report import RecordingLLM, TestReport  # noqa: E402
from facilitator.scenario import load_scenario  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402


def mmss(sec: float) -> str:
    return f"{int(sec) // 60:02d}:{int(sec) % 60:02d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", type=Path)
    ap.add_argument("--llm", choices=["anthropic", "mock", "handoff"])
    ap.add_argument("--until", type=float, help="再生終了時刻(経過秒)。省略時は最後の発話+30秒")
    ap.add_argument("--step", type=float, default=0.5, help="仮想時計の刻み(秒)")
    args = ap.parse_args()

    settings = load_settings(**({"llm_provider": args.llm} if args.llm else {}))
    sc = load_scenario(args.scenario)
    state = DiscussionState(settings=settings, theme=sc["theme"], duration_sec=sc["duration_sec"])
    for sid, name in (sc.get("participants") or {}).items():
        state.set_speaker_name(sid, name, confirmed=sc.get("names_confirmed", True))
    t = float(sc.get("start_elapsed", 0))
    if t > 0:
        state.started, state.start_time, state.phase = True, 0.0, sc.get("initial_phase", "ideation")

    llm = RecordingLLM(make_llm(settings))
    agent = FacilitatorAgent(state, llm, settings, logger=TickLogger(settings.log_dir, sc["id"]))
    report = TestReport("シナリオ再生(scripts/run_scenario.py)", sc["id"], {"LLM": llm.name})
    sec = report.section(sc["title"])
    sec.state, sec.info = state, {"テーマ": sc["theme"]}
    pending = sorted(sc["utterances"], key=lambda u: u["t"] + u["d"])
    end = args.until or max(u["t"] + u["d"] for u in pending) + 30

    print(f"=== {sc['title']} / テーマ: {sc['theme']} ===")
    while t <= end and not state.finished:
        while pending and pending[0]["t"] + pending[0]["d"] <= t:
            u = pending.pop(0)
            utt = state.add_utterance(u["s"], u["text"], u["t"], u["t"] + u["d"], u.get("c", 1.0))
            who = utt.speaker_label or utt.speaker_id
            print(f"[{mmss(u['t'])}] {who}: {u['text']}")
        state.speaking_now = any(u["t"] <= t < u["t"] + u["d"] for u in pending)
        trigger = agent.should_tick(t)
        if trigger:
            m = llm.mark()
            rec = agent.tick(t, trigger)
            sec.decision(rec, llm.since(m))
            d = rec.decision
            tag = f"{trigger}/{rec.gate}"
            if d.action.value == "no_intervention":
                print(f"   ・{mmss(t)} [{tag}] 見守り: {d.reasoning}")
            else:
                print(f"   ★{mmss(t)} [{tag}] {d.action.value}: {d.speech}\n      理由: {d.reasoning}")
                if rec.overridden:
                    print(f"      ルール層: {rec.overridden}")
        t += args.step
    print(f"=== 終了 phase={state.phase} 介入{len(state.interventions)}回 ===")
    print(f"📄 まとめログ: {report.write(settings.log_dir)}")


if __name__ == "__main__":
    main()
