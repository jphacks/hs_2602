"""マイク入力(16kHz / mono / int16)。音声はメモリ上でのみ扱い、保存しない。"""

from __future__ import annotations

from typing import Callable

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
BLOCK_MS = 50


class Mic:
    def __init__(self, on_block: Callable[[np.ndarray], None], device: int | str | None = None) -> None:
        self.on_block = on_block
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=SAMPLE_RATE * BLOCK_MS // 1000,
            device=device,
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        self.on_block(indata[:, 0].copy())

    def start(self) -> None:
        self.stream.start()

    def stop(self) -> None:
        self.stream.stop()
        self.stream.close()
