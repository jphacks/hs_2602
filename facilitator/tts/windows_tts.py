"""Windows 標準の音声合成(System.Speech)による TTS(API キー不要)。SayTTS と同じインターフェース。"""

from __future__ import annotations

import base64
import subprocess
import time

# テキストは Base64(UTF-8)で渡す。引用符のエスケープや文字化けを避け、任意のコードとして解釈されないようにする
_SCRIPT = (
    "$t=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{b64}'));"
    "Add-Type -AssemblyName System.Speech;"
    "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
    "try{{$s.SelectVoice('{voice}')}}catch{{}};"  # 指定の声がなければ既定の声で読む
    "$s.Rate={rate};"
    "$s.Speak($t)"
)


class WindowsTTS:
    def __init__(self, voice: str = "Microsoft Haruka Desktop", rate: int = 1) -> None:
        self.voice = voice
        self.rate = rate  # -10(遅い)〜10(速い)
        self._proc: subprocess.Popen | None = None
        self._stopped = False

    def speak(self, text: str) -> float:
        """読み上げが終わるまでブロックし、かかった秒数を返す。"""
        t0 = time.monotonic()
        b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
        script = _SCRIPT.format(b64=b64, voice=self.voice.replace("'", "''"), rate=int(self.rate))
        self._stopped = False
        self._proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        _, err = self._proc.communicate()
        code = self._proc.returncode
        self._proc = None
        if code != 0 and not self._stopped:
            raise RuntimeError(f"Windows TTS failed (exit {code}): {err.decode(errors='replace').strip()[:200]}")
        return time.monotonic() - t0

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._stopped = True
            self._proc.terminate()
