"""音声区間検出(VAD): 途中で部屋が騒がしくなっても「発話中」のまま固まらない。"""

from __future__ import annotations

import numpy as np

from facilitator.audio.vad import EnergyVAD


def block(db: float, n: int = 800) -> np.ndarray:
    """指定した音量(dBFS)の 50ms ブロック(16kHz)。"""
    amp = 10 ** (db / 20) * 32768 * np.sqrt(2)
    t = np.arange(n)
    return (amp * np.sin(2 * np.pi * 200 * t / 16000)).astype(np.int16)


def feed(vad: EnergyVAD, db: float, sec: float) -> list[str]:
    return [e for e in (vad.process(block(db)) for _ in range(int(sec * 20))) if e]


def test_speech_with_pauses_is_detected_normally():
    vad = EnergyVAD()
    feed(vad, -60, 2)
    assert feed(vad, -30, 1) == ["start"] and vad.speaking
    assert feed(vad, -60, 1) == ["end"] and not vad.speaking


def test_constant_noise_does_not_stay_speaking_forever():
    vad = EnergyVAD()
    feed(vad, -60, 2)  # 静かな部屋で雑音レベルを学習
    feed(vad, -40, 20)  # 途中から空調などで 20dB うるさくなる(切れ目なし)
    assert not vad.speaking  # 15 秒で雑音とみなして「発話中」を解除する
    feed(vad, -40, 3)
    assert not vad.speaking  # 新しい雑音レベルを覚えたので、再び発話中にはならない
    assert feed(vad, -15, 1) == ["start"]  # その上で話せば検出できる
