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
from facilitator.scenario import load_scenario  # noqa: E402
from facilitator.state import DiscussionState  # noqa: E402


def mmss(sec: float) -> str:
    return f"{int(sec) // 60:02d}:{int(sec) % 60:02d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", type=Path)
    ap.add_argument("--llm", choices=["anthropic", "mock"])
    ap.add_argument("--until", type=float, help="再生終了時刻(経過秒)。省略時は最後の発話+30秒")
    ap.add_argument("--step", type=float, default=0.5, help="仮想時計の刻み(秒)")
    args = ap.parse_args()

    settings = load_settings(**({"llm_provider": args.llm} if args.llm else {}))
    sc = load_scenario(args.scenario)
    state = DiscussionState(settings=settings, theme=sc["theme"], duration_sec=sc["duration_sec"])
    for sid, name in (sc.get("participants") or {}).items():
        state.set_speaker_name(sid, name, confirmed=sc.get("names_confirmed", True))
    t = float(sc.get("start_elapsed", 0))
    agent = FacilitatorAgent(state, make_llm(settings), settings, logger=TickLogger(settings.log_dir, sc["id"]),
                             discussion_type=sc.get("discussion_type"))
    if t > 0:
        state.started, state.start_time = True, 0.0
        state.observed_phase = sc.get("initial_phase", "")
        state.phase_observed_at = 0.0 if state.observed_phase else None
    pending = sorted(sc["utterances"], key=lambda u: u["t"] + u["d"])
    end = args.until or max(u["t"] + u["d"] for u in pending) + 30

    print(f"=== {sc['title']} / テーマ: {sc['theme']} ===")
    skipped: dict[str, list[float]] = {}  # 定期分析の時刻なのに見送った時刻(理由別)
    while t <= end and not state.finished:
        while pending and pending[0]["t"] + pending[0]["d"] <= t:
            u = pending.pop(0)
            utt = state.add_utterance(u["s"], u["text"], u["t"], u["t"] + u["d"], u.get("c", 1.0))
            who = utt.speaker_label or utt.speaker_id
            print(f"[{mmss(u['t'])}] {who}: {u['text']}")
        state.speaking_now = any(u["t"] <= t < u["t"] + u["d"] for u in pending)
        trigger = agent.should_tick(t)
        if agent.skipped_due:
            skipped.setdefault(agent.skipped_due, []).append(t)
        if trigger:
            rec = agent.tick(t, trigger)
            d = rec.decision
            tag = f"{trigger}/{rec.gate}"
            if d.action.value == "no_intervention":
                print(f"   ・{mmss(t)} [{tag}] 見守り: {d.reasoning}")
            else:
                print(f"   ★{mmss(t)} [{tag}] {d.action.value}: {d.speech}\n      理由: {d.reasoning}")
                if rec.overridden:
                    print(f"      ルール層: {rec.overridden}")
        t += args.step
    print(f"=== 終了 phase={state.observed_phase or '-'} 介入{len(state.interventions)}回 "
          f"LLM呼び出し{sum(r.llm_called for r in agent.history)}回 ===")
    label = {"speaking": "発話中", "short_silence": "沈黙2秒未満"}
    for reason, ts in skipped.items():
        spans, start = [], ts[0]
        for a, b in zip(ts, ts[1:] + [None]):
            if b is None or b - a > args.step + 1e-6:
                spans.append(f"{mmss(start)}-{mmss(a)}")
                start = b
        print(f"定期分析の時刻だったが見送り({label.get(reason, reason)}): 計{len(ts) * args.step:.1f}秒 [{', '.join(spans)}]")


if __name__ == "__main__":
    main()
