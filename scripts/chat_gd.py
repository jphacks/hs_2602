"""チャット GD: アプリを使わず、チャットの中だけで模擬グループディスカッションを行う。

裏で起動しておき、人間の発言を受信箱(exchange/inbox.jsonl)で受け取る。発言のたびに仮想時計を進め、
進行役の判断と AI 参加者の発言をアプリと同じロジックで動かして、起きたことを exchange/transcript.txt に追記する。
--llm handoff(既定)なら、AI が考える場面はすべてチャットの AI が答える(API は使わない)。

起動:
  python scripts/chat_gd.py --theme "大学の学食の混雑を改善するには？" --humans たろう --ai 2 --minutes 5

受信箱に書く命令(1行 1 JSON):
  {"type": "say", "speaker": "たろう", "text": "…", "pause": 8}   発言(pause = 発言後に待つ沈黙の秒数)
  {"type": "wait", "sec": 15}                                     沈黙を続ける
  {"type": "end"}                                                 制限時間まで進めて終了し、結果を出す
状態は exchange/status.txt(idle = 次の命令を待っている / busy = 処理中 / done = 終了)。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ROOT, load_settings  # noqa: E402
from facilitator.llm.factory import make_llm  # noqa: E402
from facilitator.report import RecordingLLM  # noqa: E402
from facilitator.virtual_session import VirtualSession  # noqa: E402

EX = ROOT / "exchange"
INBOX, TRANSCRIPT, STATUS = EX / "inbox.jsonl", EX / "transcript.txt", EX / "status.txt"
MAX_FOLLOW_SEC = 60.0  # 1回の発言のあとに AI だけで進める時間の上限


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", required=True)
    ap.add_argument("--humans", required=True, help="人間の参加者名(カンマ区切り)")
    ap.add_argument("--ai", type=int, default=0, help="AI 参加者の人数(0〜3)")
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--purpose", choices=["job", "general"], default="job")
    ap.add_argument("--llm", choices=["handoff", "mock", "anthropic"], default="handoff")
    args = ap.parse_args()

    EX.mkdir(exist_ok=True)
    for f in (INBOX, TRANSCRIPT):
        f.write_text("", encoding="utf-8")

    def out(line: str) -> None:
        with TRANSCRIPT.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def status(x: str) -> None:
        STATUS.write_text(x, encoding="utf-8")

    status("busy")
    s = load_settings(llm_provider=args.llm)
    names = [n.strip() for n in args.humans.split(",") if n.strip()]
    humans = {f"speaker_{i + 1}": n for i, n in enumerate(names)}
    by_name = {n: sid for sid, n in humans.items()}
    vs = VirtualSession(
        s,
        RecordingLLM(make_llm(s)),
        theme=args.theme,
        humans=humans,
        ai_count=max(0, min(3, args.ai)),
        duration_sec=args.minutes * 60,
        purpose=args.purpose,
        title=args.theme,
        report_kind="チャット GD(scripts/chat_gd.py)",
        report_name="chat",
        out=out,
    )
    vs.step(0.0)  # 開始の説明
    vs.run_until(vs.busy_until + 0.5)
    status("idle")

    seen = 0
    while True:
        lines = INBOX.read_text(encoding="utf-8").splitlines()
        if len(lines) <= seen:
            time.sleep(0.3)
            continue
        cmd = json.loads(lines[seen])
        seen += 1
        status("busy")
        st = vs.st
        if cmd["type"] == "say":
            sid = by_name.get(cmd.get("speaker", ""), next(iter(humans)))
            text = cmd["text"]
            start = max(vs.t, vs.busy_until)  # AI が話し終わってから話す
            dur = float(cmd.get("duration") or max(1.5, len(text) / 6.0))
            vs.human(sid, text, start, start + dur)
            vs.t = start + dur
            pause = float(cmd.get("pause", 8))
            # 発言のあと pause 秒の沈黙(AI が話したら、その後からまた pause 秒)まで進める
            limit = vs.t + MAX_FOLLOW_SEC
            t = vs.t
            while not st.finished and t <= limit:
                last_end = max(u.end_time for u in st.utterances)
                if t > last_end + pause:
                    break
                vs.step(t)
                t += 0.5
            vs.t = t
        elif cmd["type"] == "wait":
            vs.run_until(vs.t + float(cmd.get("sec", 10)))
        elif cmd["type"] == "end":
            vs.run_until(st.start_time + st.duration_sec + 30)
        if cmd["type"] == "end" or st.finished:
            vs.run_until(vs.busy_until + 0.5)
            out("")
            out(vs.summary())
            path = vs.report.write(s.log_dir)
            out(f"📄 まとめログ: {path.relative_to(ROOT)}")
            status("done")
            return
        out(f"  … 経過 {vs.et(vs.t) // 60:02.0f}:{vs.et(vs.t) % 60:04.1f} / 残り {st.remaining(vs.t) // 60:02.0f}:{st.remaining(vs.t) % 60:02.0f}")
        status("idle")


if __name__ == "__main__":
    main()
