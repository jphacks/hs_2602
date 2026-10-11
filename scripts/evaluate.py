"""全シナリオを流し、期待判断との一致を評価する。

使い方:
  python scripts/evaluate.py                   # 全シナリオ(.env の LLM 設定)
  python scripts/evaluate.py --llm mock        # API なしで配線だけ確認
  python scripts/evaluate.py -s 02 -s 06 -v    # 一部シナリオ、詳細表示
  python scripts/evaluate.py --repeat 3        # 判断のブレを見る
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ROOT, load_settings  # noqa: E402
from facilitator.llm.factory import make_llm  # noqa: E402
from facilitator.logging import TickLogger  # noqa: E402
from facilitator.scenario import load_scenario, run_scenario  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", choices=["anthropic", "mock"])
    ap.add_argument("--model")
    ap.add_argument("-s", "--scenario", action="append", help="シナリオIDの前方一致(複数可)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    overrides = {}
    if args.llm:
        overrides["llm_provider"] = args.llm
    if args.model:
        overrides["llm_model"] = args.model
    settings = load_settings(**overrides)
    llm = make_llm(settings)

    paths = sorted((ROOT / "scenarios").glob("*.yaml"))
    if args.scenario:
        paths = [p for p in paths if any(p.stem.startswith(s) for s in args.scenario)]

    logger = TickLogger(settings.log_dir, "eval")
    rows, total, passed = [], 0, 0
    for rep in range(args.repeat):
        for p in paths:
            sc = load_scenario(p)
            res = run_scenario(sc, llm, settings, logger)
            for c in res.checks:
                total += 1
                passed += c.passed
                d = c.record.decision
                mark = "OK " if c.passed else "NG "
                print(
                    f"{mark}{res.id:<20} t={c.at:>6.1f} 期待={'/'.join(c.expect):<40} "
                    f"結果={d.action.value}{'(' + d.target_speaker + ')' if d.target_speaker else ''}"
                    f"  [{c.record.gate}, {c.record.latency_sec:.1f}s]"
                )
                if args.verbose or not c.passed:
                    if c.fail_reason:
                        print(f"     NG理由: {c.fail_reason} / 想定: {c.why}")
                    print(f"     状態: phase={c.record.phase} health={c.record.analysis.health}")
                    print(f"     理由: {d.reasoning}")
                    if d.speech:
                        print(f"     発話: {d.speech}")
                    if c.record.overridden:
                        print(f"     ルール層: {c.record.overridden}")
                rows.append(
                    {
                        "rep": rep,
                        "scenario": res.id,
                        "at": c.at,
                        "expect": c.expect,
                        "passed": c.passed,
                        "fail_reason": c.fail_reason,
                        "record": c.record.model_dump(mode="json"),
                    }
                )

    print(f"\n一致率: {passed}/{total} = {passed / max(1, total):.0%}  (LLM: {llm.name})")
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"eval-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"詳細: {out.relative_to(ROOT)}  /  判断ログ: {logger.path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
