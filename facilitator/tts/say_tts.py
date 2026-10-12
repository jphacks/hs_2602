"""macOS 標準の say コマンドによる TTS(API キー不要)。"""

from __future__ import annotations

import subprocess
import time


class SayTTS:
    def __init__(self, voice: str = "Kyoko", rate: int = 210) -> None:
        self.voice = voice
        self.rate = rate  # 1分あたりの語数の目安(日本語でも速さの調整に効く)
        self._proc: subprocess.Popen | None = None

    def speak(self, text: str) -> float:
        """読み上げが終わるまでブロックし、かかった秒数を返す。"""
        t0 = time.monotonic()
        self._proc = subprocess.Popen(["say", "-v", self.voice, "-r", str(self.rate), text])
        self._proc.wait()
        self._proc = None
        return time.monotonic() - t0

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
