"""テキスト → TTS(macOS say)→ スピーカー出力の単体デモ。読み上げ速度を計測する。

使い方:
  python scripts/spike_tts.py
  python scripts/spike_tts.py --voice Eddy --rate 230 "任意のテキスト"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facilitator.tts.say_tts import SayTTS  # noqa: E402

SAMPLES = [
    "それでは、グループディスカッションを始めます。テーマは「大学の学食の混雑を改善するには？」です。",
    "導入コストが論点になっていますね。まずは混雑の原因を整理して、どの部分を改善すべきか考えてみませんか？",
    "みさきさんは、ここまでの議論をどう考えますか？",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--voice", default="Kyoko")
    ap.add_argument("--rate", type=int, default=210)
    args = ap.parse_args()

    tts = SayTTS(args.voice, args.rate)
    texts = [" ".join(args.text)] if args.text else SAMPLES
    total_chars, total_sec = 0, 0.0
    for t in texts:
        sec = tts.speak(t)
        total_chars += len(t)
        total_sec += sec
        print(f"{sec:5.1f}s / {len(t)}文字 ({len(t) / sec:.1f}文字/秒)  {t}")
    print(f"平均 {total_chars / total_sec:.1f} 文字/秒(config.tts_chars_per_sec の参考値)")


if __name__ == "__main__":
    main()
