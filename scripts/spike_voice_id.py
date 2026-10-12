"""自己紹介で声を登録 → リアルタイム文字起こし + 声の照合で「誰の発言か」を判定する検証スクリプト。

使い方(Mac の「ターミナル」アプリで実行):
  python scripts/spike_voice_id.py --names たろう,はなこ,けんた
  python scripts/spike_voice_id.py --names たろう,はなこ,けんた --order rr --seconds 90
    --order rr : 参加者が名前の順に1文ずつ交代で話す前提で、正答率を計算する
音声・声の特徴量は保存しない(メモリ上のみ)。
"""

from __future__ import annotations

import argparse
import queue
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402,F401  (.env の読み込み)
from facilitator.audio.mic import Mic  # noqa: E402
from facilitator.audio.vad import EnergyVAD  # noqa: E402
from facilitator.speaker.voice_id import AudioBuffer, VoiceIdentifier  # noqa: E402
from facilitator.stt.assemblyai_stt import AssemblyAIStream  # noqa: E402
from facilitator.stt.base import TranscriptEvent  # noqa: E402


def enroll_all(q: queue.Queue, names: list[str], sec: float, vid: VoiceIdentifier) -> None:
    vad = EnergyVAD()
    # 最初の1秒で雑音レベルを学習
    print("周囲の雑音を測っています(1秒、静かにしてください)…")
    t_end = time.monotonic() + 1.0
    while time.monotonic() < t_end:
        vad.process(q.get())
    for i, name in enumerate(names):
        input(f"\n▶ {name}さん(speaker_{i + 1})の番です。Enter を押したら {sec:.0f} 秒間、自己紹介をしてください。")
        while not q.empty():
            q.get_nowait()
        voiced = []
        t_end = time.monotonic() + sec
        while time.monotonic() < t_end:
            b = q.get()
            vad.process(b)
            if vad.level_db > vad.noise_db + 8:
                voiced.append(b)
            print(f"\r   残り {t_end - time.monotonic():4.1f} 秒  レベル {vad.level_db:5.0f} dB", end="", flush=True)
        audio = np.concatenate(voiced) if voiced else np.zeros(0, dtype=np.int16)
        print(f"\r   声が入っていた時間: {len(audio) / 16000:.1f} 秒" + ("" if len(audio) > 16000 * 3 else "  ⚠ 短すぎます"))
        if len(audio):
            vid.enroll(f"speaker_{i + 1}", audio)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True, help="カンマ区切りの名前(話す順)")
    ap.add_argument("--enroll-sec", type=float, default=10)
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--order", choices=["none", "rr"], default="none")
    ap.add_argument("--model", default="universal-3-6-pro")
    args = ap.parse_args()
    names = args.names.split(",")
    id2name = {f"speaker_{i + 1}": n for i, n in enumerate(names)}

    q: queue.Queue = queue.Queue()
    mic = Mic(q.put)
    mic.start()
    vid = VoiceIdentifier()
    enroll_all(q, names, args.enroll_sec, vid)

    # ---- 会話フェーズ ----
    buf = AudioBuffer()
    stt = AssemblyAIStream(speech_model=args.model, speaker_labels=False)
    lock = threading.Lock()
    results: list[tuple[str, str, float]] = []  # (予測, 正解, 確信度)
    id_ms: list[float] = []

    def on_event(e: TranscriptEvent) -> None:
        if e.kind != "final" or e.start is None:
            if e.kind == "partial":
                with lock:
                    print(f"\r  … {e.text[-50:]:<50}", end="", flush=True)
            return
        t0 = time.monotonic()
        r = vid.identify(buf.segment(e.start - 0.1, e.end + 0.1))
        id_ms.append((time.monotonic() - t0) * 1000)
        expected = f"speaker_{len(results) % len(names) + 1}" if args.order == "rr" else ""
        results.append((r.speaker_id, expected, r.confidence))
        who = id2name.get(r.speaker_id, "不明")
        mark = "" if not expected else (" ✓" if r.speaker_id == expected else f" ✗(正解 {id2name[expected]})")
        scores = " ".join(f"{id2name[k]}:{v:.2f}" for k, v in r.scores.items())
        with lock:
            print(f"\r[{e.start:6.1f}s] {who} (確信度 {r.confidence:.2f}){mark}  [{scores}]\n    {e.text}")

    stt.start(on_event)
    print(f"\n=== 会話を始めてください({args.seconds:.0f}秒){' 名前の順に1文ずつ交代で' if args.order == 'rr' else ''} ===")
    while not q.empty():
        q.get_nowait()
    t_end = time.monotonic() + args.seconds
    try:
        while time.monotonic() < t_end:
            try:
                b = q.get(timeout=0.5)
            except queue.Empty:
                continue
            buf.append(b)
            stt.send_audio(b.tobytes())
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()
        stt.stop()
        vid.clear()  # 声の特徴量を破棄

    print("\n=== まとめ ===")
    n = len(results)
    unknown = sum(1 for p, _, _ in results if p == "unknown")
    print(f"発話 {n} 件 / 不明 {unknown} 件 / 照合時間 中央値 {statistics.median(id_ms) if id_ms else 0:.0f}ms")
    if args.order == "rr" and n:
        correct = sum(1 for p, e, _ in results if p == e)
        known = [(p, e) for p, e, _ in results if p != "unknown"]
        print(f"正答率(全体): {correct}/{n} = {correct / n:.0%}")
        if known:
            print(f"正答率(不明を除く): {sum(p == e for p, e in known)}/{len(known)} = {sum(p == e for p, e in known) / len(known):.0%}")


if __name__ == "__main__":
    main()
