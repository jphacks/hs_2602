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
from facilitator.report import RecordingLLM, TestReport  # noqa: E402
from facilitator.scenario import load_scenario, run_scenario  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", choices=["anthropic", "mock", "handoff"])
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
    llm = RecordingLLM(make_llm(settings))

    paths = sorted((ROOT / "scenarios").glob("*.yaml"))
    if args.scenario:
        paths = [p for p in paths if any(p.stem.startswith(s) for s in args.scenario)]

    logger = TickLogger(settings.log_dir, "eval")
    report = TestReport("シナリオ評価(scripts/evaluate.py)", "eval", {"LLM": llm.name})
    rows, total, passed = [], 0, 0
    phase_total, phase_passed = 0, 0  # フェーズ認識は行動の一致率とは別に集計する
    by_src: dict[tuple[str, bool], list[bool]] = {}  # (判断の出どころ, 介入不要が正解の時点か) → 合否
    for rep in range(args.repeat):
        for p in paths:
            sc = load_scenario(p)
            title = sc["id"] + (f" (#{rep + 1})" if args.repeat > 1 else "") + f" — {sc['title']}"
            res = run_scenario(sc, llm, settings, logger, section=report.section(title))
            for c in res.checks:
                total += 1
                passed += c.passed
                if c.phase_ok is not None:
                    phase_total += 1
                    phase_passed += c.phase_ok
                d = c.record.decision
                mark = "OK " if c.passed else "NG "
                pmark = {True: " フェーズOK", False: " フェーズNG", None: ""}[c.phase_ok]
                src = {"rule": "ルール", "llm": "LLM", "llm_overridden": "LLM→書換"}[c.judged_by]
                by_src.setdefault((c.judged_by, d.action.value == "no_intervention" and c.expect == ["no_intervention"]), []).append(c.passed)
                print(
                    f"{mark}{res.id:<20} t={c.at:>6.1f} 期待={'/'.join(c.expect):<40} "
                    f"結果={d.action.value}{'(' + d.target_speaker + ')' if d.target_speaker else ''}"
                    f"  [{src}: {c.record.gate}, {c.record.latency_sec:.1f}s]{pmark}"
                )
                if args.verbose or not c.passed or c.phase_ok is False:
                    if c.fail_reason:
                        print(f"     NG理由: {c.fail_reason} / 想定: {c.why}")
                    r = c.record
                    print(
                        f"     フェーズ: 実際={r.observed_phase or '-'} 目標={r.target_phase} "
                        f"差={r.phase_gap if r.phase_gap is not None else '-'}{' (遅れ)' if r.phase_behind else ''}"
                        + (f" 期待={'/'.join(c.expect_phase)}" if c.expect_phase else "")
                        + f" health={r.analysis.health}"
                    )
                    if c.record.overridden:
                        print(f"     書き換え: {c.record.overridden}")
                    if r.analysis.phase_reason:
                        print(f"     フェーズの根拠: {r.analysis.phase_reason}")
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
                        "expect_phase": c.expect_phase,
                        "phase_ok": c.phase_ok,
                        "judged_by": c.judged_by,
                        "record": c.record.model_dump(mode="json"),
                    }
                )

    print(f"\n一致率: {passed}/{total} = {passed / max(1, total):.0%}  (LLM: {llm.name})")
    if phase_total:
        print(f"フェーズ認識の一致率: {phase_passed}/{phase_total} = {phase_passed / phase_total:.0%}")
    quiet = {k[0]: v for k, v in by_src.items() if k[1]}
    if quiet:
        print("介入不要が正解の時点: " + " / ".join(f"{ {'rule': 'ルール層の見送り', 'llm': 'LLM の見守り判断', 'llm_overridden': 'LLM→書換'}[k]} {sum(v)}/{len(v)}" for k, v in sorted(quiet.items())))
    calls = [r["record"] for r in rows if r["record"]["llm_called"]]
    tokens: dict[str, int] = {}
    for rec in calls:
        for k, v in (rec["llm_usage"] or {}).items():
            tokens[k] = tokens.get(k, 0) + v
    avg = sum(rec["latency_sec"] for rec in calls) / len(calls) if calls else 0.0
    print(f"LLM呼び出し: {len(calls)}回 / 平均応答 {avg:.1f}秒 / トークン: " + (", ".join(f"{k}={v}" for k, v in tokens.items()) or "記録なし(モック)"))
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"eval-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    report.meta["一致率"] = f"{passed}/{total} = {passed / max(1, total):.0%}"
    print(f"詳細: {out.relative_to(ROOT)}  /  判断ログ: {logger.path.relative_to(ROOT)}")
    print(f"📄 まとめログ: {report.write(settings.log_dir).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
