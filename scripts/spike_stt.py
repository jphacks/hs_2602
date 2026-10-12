"""マイク → VAD → AssemblyAI リアルタイム文字起こし(+話者分離)の単体デモ。

使い方:
  python scripts/spike_stt.py                                  # 既定: universal-3-6-pro, 話者分離あり
  python scripts/spike_stt.py --model universal-3-5-pro
  python scripts/spike_stt.py --param language_codes=ja        # 任意の接続パラメータを追加(検証用)
  python scripts/spike_stt.py --seconds 60 --max-speakers 3
Ctrl+C で終了。音声は保存しない。
"""

from __future__ import annotations

import argparse
import queue
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402,F401  (.env の読み込み)
from facilitator.audio.mic import Mic  # noqa: E402
from facilitator.audio.vad import EnergyVAD  # noqa: E402
from facilitator.stt.assemblyai_stt import AssemblyAIStream  # noqa: E402
from facilitator.stt.base import TranscriptEvent  # noqa: E402


class WavPlayer:
    """WAV を 50ms ずつ実時間で流す(マイクの代役)。"""

    def __init__(self, path: Path, on_block) -> None:  # noqa: ANN001
        import numpy as np
        import wave

        with wave.open(str(path)) as w:
            assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        tail = np.zeros(16000 * 3, dtype=np.int16)  # 最後のターンを確定させるための無音
        self.data = np.concatenate([data, tail])
        self.on_block = on_block
        self._stop = threading.Event()
        self.done = threading.Event()

    def start(self) -> None:
        def run() -> None:
            n = 800
            t_start = time.monotonic()
            for k, i in enumerate(range(0, len(self.data), n)):
                if self._stop.is_set():
                    return
                self.on_block(self.data[i : i + n])
                time.sleep(max(0.0, t_start + (k + 1) * 0.05 - time.monotonic()))  # 誤差を累積させない
            print("\n(WAV の再生が終わりました)")
            self.done.set()

        threading.Thread(target=run, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="universal-3-6-pro")
    ap.add_argument("--max-speakers", type=int, default=4)
    ap.add_argument("--no-diarize", action="store_true")
    ap.add_argument("--param", action="append", default=[], help="key=value 形式の追加パラメータ")
    ap.add_argument("--seconds", type=float, default=0, help="0 なら Ctrl+C まで")
    ap.add_argument("--device", default=None, help="入力デバイス番号または名前")
    ap.add_argument("--debug", action="store_true", help="end_of_turn の生メッセージを表示")
    ap.add_argument("--wav", type=Path, help="マイクの代わりに 16kHz/mono/16bit の WAV を実時間で流す(検証用)")
    args = ap.parse_args()

    extra = dict(p.split("=", 1) for p in args.param)
    stt = AssemblyAIStream(
        speech_model=args.model,
        speaker_labels=not args.no_diarize,
        max_speakers=args.max_speakers,
        extra_params=extra,
    )
    print(f"接続パラメータ: { {k: v for k, v in stt.params.items()} }")

    t0 = [0.0]  # 最初の音声を送った時刻(ストリーム内時刻の原点)
    latencies: list[float] = []
    turns: list[TranscriptEvent] = []
    vad = EnergyVAD()
    q: queue.Queue = queue.Queue()
    lock = threading.Lock()

    def rel() -> float:
        return time.monotonic() - t0[0] if t0[0] else 0.0

    def on_event(e: TranscriptEvent) -> None:
        with lock:
            if args.debug and e.raw.get("end_of_turn"):
                r = {k: v for k, v in e.raw.items() if k != "words"}
                w = [(x.get("text"), x.get("speaker"), x.get("word_is_final")) for x in e.raw.get("words", [])[:4]]
                print(f"\n  RAW {r}\n  WORDS(先頭4) {w}")
            if e.kind == "partial":
                print(f"\r  … {e.text[-60:]:<60}", end="", flush=True)
            elif e.kind == "final":
                lat = rel() - e.end if e.end is not None else float("nan")
                if e.end is not None:
                    latencies.append(lat)
                turns.append(e)
                conf = f"{e.speaker_confidence:.2f}" if e.speaker_confidence is not None else "-"
                print(
                    f"\r[{e.start or 0:6.1f}-{e.end or 0:6.1f}s] 話者={e.speaker} (conf {conf}) "
                    f"遅延={lat:.2f}s  #{e.turn_order}\n    {e.text}"
                )
            elif e.kind == "revision":
                print(f"\n  ↺ 話者ラベル修正: {e.revisions}")
            elif e.kind == "begin":
                print(f"接続しました(session {e.text[:8]}…)。話してください。Ctrl+C で終了")
            else:
                print(f"\n  [{e.kind}] {e.text}")

    def on_block(block) -> None:  # noqa: ANN001
        q.put(block)

    stt.start(on_event)
    if args.wav:
        mic = WavPlayer(args.wav, on_block)
    else:
        mic = Mic(on_block, device=int(args.device) if args.device and args.device.isdigit() else args.device)
    mic.start()
    t_end = time.monotonic() + args.seconds if args.seconds else None
    n_blocks, levels = 0, []
    warned, t_begin = False, time.monotonic()
    try:
        while (t_end is None or time.monotonic() < t_end) and not getattr(mic, "done", threading.Event()).is_set():
            try:
                block = q.get(timeout=0.5)
            except queue.Empty:
                if not n_blocks and not warned and time.monotonic() - t_begin > 3:
                    warned = True
                    print("\n  ⚠ マイクから音声が届いていません。macOS の マイクの権限(システム設定 > プライバシーとセキュリティ > マイク)を確認してください")
                continue
            if not t0[0]:
                t0[0] = time.monotonic()
            stt.send_audio(block.tobytes())
            ev = vad.process(block)
            n_blocks += 1
            levels.append(vad.level_db)
            if n_blocks == 40:  # 最初の2秒で入力レベルを確認
                avg = sum(levels) / len(levels)
                with lock:
                    print(f"\r  入力レベル(最初の2秒の平均): {avg:.0f} dB", end="")
                    print("  ← ほぼ無音です。マイクの選択・権限・距離を確認してください" if avg < -60 else "  (OK)")
            if ev:
                with lock:
                    mark = "🎤 発話開始" if ev == "start" else "   発話終了"
                    print(f"\r  {mark} @ {rel():.1f}s (level {vad.level_db:.0f}dB / noise {vad.noise_db:.0f}dB)")
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()
        print("\n終了処理中(最終の話者ラベル修正を待ちます)…")
        stt.stop()

    print("\n=== まとめ ===")
    print(f"確定ターン数: {len(turns)}")
    if latencies:
        print(f"発話終了→確定テキストの遅延: 中央値 {statistics.median(latencies):.2f}s / 最大 {max(latencies):.2f}s")
    speakers = [t.speaker for t in turns]
    print(f"話者ラベル: {speakers}")


if __name__ == "__main__":
    main()
