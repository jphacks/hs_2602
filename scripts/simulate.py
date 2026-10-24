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
from facilitator.llm.factory import make_llm  # noqa: E402
from facilitator.report import RecordingLLM  # noqa: E402
from facilitator.scenario import load_scenario  # noqa: E402
from facilitator.virtual_session import VirtualSession  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", type=Path)
    ap.add_argument("--llm", choices=["anthropic", "mock", "handoff"])
    ap.add_argument("--until", type=float, help="終了時刻(経過秒)。省略時は台本の until、なければ最後の発言+20秒")
    ap.add_argument("--step", type=float, default=0.5)
    args = ap.parse_args()

    sc = load_scenario(args.scenario)
    s = load_settings(**({"llm_provider": args.llm} if args.llm else {}))
    print(f"=== {sc['title']} ===")
    vs = VirtualSession(
        s,
        RecordingLLM(make_llm(s)),
        theme=sc["theme"],
        humans=sc.get("participants") or {},
        ai_count=int(sc.get("ai_count", 0)),
        duration_sec=sc["duration_sec"],
        purpose=sc.get("purpose", "job"),
        title=sc["title"],
        report_kind="AI 参加者入りシミュレーション(scripts/simulate.py)",
        report_name=sc["id"],
    )
    pending = sorted(sc["utterances"], key=lambda u: u["t"] + u["d"])
    end = args.until or sc.get("until") or max(u["t"] + u["d"] for u in pending) + 20
    t = 0.0
    while t <= end and not vs.st.finished:
        while pending and pending[0]["t"] + pending[0]["d"] <= t:
            u = pending.pop(0)
            vs.human(u["s"], u["text"], u["t"], u["t"] + u["d"], u.get("c", 1.0))
        vs.step(t, speaking_now=any(u["t"] <= t < u["t"] + u["d"] for u in pending))
        t += args.step
    vs.t = t
    print()
    print(vs.summary())
    print(f"\n📄 まとめログ: {vs.report.write(s.log_dir)}")


if __name__ == "__main__":
    main()
