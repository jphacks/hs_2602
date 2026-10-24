"""エネルギーベースの簡易 VAD(雑音レベルを追従)。

Silero VAD は torch/onnxruntime が必要で Python 3.14 での動作が未確認のため、MVP ではこの簡易版を使う。
用途は「今誰かが話しているか」の判定(AI が割り込まないため)で、文字起こしの区切りは STT 側のターン検出に任せる。
"""

from __future__ import annotations

import numpy as np


class EnergyVAD:
    def __init__(
        self,
        block_ms: int = 50,
        margin_db: float = 12.0,  # 雑音レベル + margin を超えたら音声とみなす
        min_db: float = -50.0,  # これ未満は常に無音
        onset_ms: int = 150,  # この長さ連続したら発話開始
        hangover_ms: int = 500,  # この長さ無音が続いたら発話終了
        max_speech_ms: int = 15000,  # 0.5秒の切れ目もなくこれ以上続く「発話」は雑音とみなす
    ) -> None:
        self.block_ms = block_ms
        self.margin_db = margin_db
        self.min_db = min_db
        self.onset_blocks = max(1, onset_ms // block_ms)
        self.hang_blocks = max(1, hangover_ms // block_ms)
        self.max_blocks = max(1, max_speech_ms // block_ms)
        self.noise_db = -60.0
        self.speaking = False
        self._on = 0
        self._off = 0
        self._run = 0  # 発話中と判定してから続いているブロック数
        self.level_db = -100.0

    def process(self, block: np.ndarray) -> str | None:
        """1ブロックを処理し、状態が変わったら "start" / "end" を返す。"""
        x = block.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(x * x)) + 1e-10)
        db = 20 * np.log10(rms)
        self.level_db = db
        # 雑音レベル: 下がるときは速く、上がるときはゆっくり追従
        if db < self.noise_db:
            self.noise_db = 0.7 * self.noise_db + 0.3 * db
        elif not self.speaking:
            self.noise_db = 0.995 * self.noise_db + 0.005 * db
        voiced = db > max(self.min_db, self.noise_db + self.margin_db)

        if voiced:
            self._on += 1
            self._off = 0
        else:
            self._off += 1
            self._on = 0
        if not self.speaking and self._on >= self.onset_blocks:
            self.speaking = True
            self._run = 0
            return "start"
        if self.speaking:
            self._run += 1
            # 雑音レベルは発話中に上げないため、部屋が途中で騒がしくなる(空調・ファンなど)と「発話中」のまま戻らなくなる。
            # 人の話は 0.5 秒程度の切れ目が必ず入るので、切れ目なく続く場合は雑音とみなし、今の音量を雑音レベルにして終了にする
            if self._run >= self.max_blocks:
                self.noise_db = db
                self.speaking = False
                self._on = self._off = 0
                return "end"
        if self.speaking and self._off >= self.hang_blocks:
            self.speaking = False
            return "end"
        return None
