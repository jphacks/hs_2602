"""自己紹介の声を登録し、発話ごとに「登録済みの誰の声に近いか」を照合する(方式C・軽量版)。

- 埋め込みモデル: sherpa-onnx + 3D-Speaker CAM++(models/campplus_zh_en_advanced.onnx)
- 声の特徴量はメモリ上だけで保持し、保存しない(セッション終了で破棄)
- 確信度 = 最良一致の類似度と、2位との差(マージン)から算出。閾値未満は unknown
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx

from config import ROOT

DEFAULT_MODEL = ROOT / "models" / "campplus_zh_en_advanced.onnx"
SAMPLE_RATE = 16000


@dataclass
class IdResult:
    speaker_id: str  # 登録済み ID または "unknown"
    confidence: float  # 0〜1
    best_sim: float
    margin: float
    scores: dict[str, float]


class VoiceIdentifier:
    def __init__(
        self,
        model: Path = DEFAULT_MODEL,
        min_sec: float = 1.0,  # これより短い発話は判定しない
        sim_threshold: float = 0.35,  # 最良一致の類似度の下限(実測で調整)
        margin_threshold: float = 0.08,  # 2位との差の下限(実測で調整)
    ) -> None:
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=2)
        if not cfg.validate():
            raise ValueError(f"話者埋め込みモデルの設定が不正です: {model}")
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self.min_sec = min_sec
        self.sim_threshold = sim_threshold
        self.margin_threshold = margin_threshold
        self.enrolled: dict[str, list[np.ndarray]] = {}

    # ---------- 埋め込み ----------
    def embed(self, pcm16: np.ndarray) -> np.ndarray:
        x = pcm16.astype(np.float32) / 32768.0
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=x)
        stream.input_finished()
        v = np.array(self.extractor.compute(stream), dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    # ---------- 登録 ----------
    def enroll(self, speaker_id: str, pcm16: np.ndarray) -> None:
        self.enrolled.setdefault(speaker_id, []).append(self.embed(pcm16))

    def clear(self) -> None:
        self.enrolled.clear()

    def _centroids(self) -> dict[str, np.ndarray]:
        out = {}
        for sid, vs in self.enrolled.items():
            c = np.mean(vs, axis=0)
            out[sid] = c / (np.linalg.norm(c) + 1e-9)
        return out

    # ---------- 照合 ----------
    def identify(self, pcm16: np.ndarray) -> IdResult:
        if not self.enrolled or len(pcm16) < self.min_sec * SAMPLE_RATE:
            return IdResult("unknown", 0.0, 0.0, 0.0, {})
        v = self.embed(pcm16)
        scores = {sid: float(np.dot(v, c)) for sid, c in self._centroids().items()}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_id, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -1.0
        margin = best - second
        # 類似度とマージンの両方が閾値を超えたときだけ確信度を高くする(0.7 以上で「特定できた」扱い)
        conf_sim = np.clip((best - self.sim_threshold) / 0.25 + 0.7, 0, 1) if best >= self.sim_threshold else 0.0
        conf_margin = (
            np.clip((margin - self.margin_threshold) / 0.15 + 0.7, 0, 1) if margin >= self.margin_threshold else 0.0
        )
        conf = float(min(conf_sim, conf_margin))
        sid = best_id if conf >= 0.7 else "unknown"
        return IdResult(sid, round(conf, 2), round(best, 3), round(margin, 3), {k: round(s, 3) for k, s in scores.items()})


class AudioBuffer:
    """ストリーム開始からの音声をメモリ上に保持し、[start, end] 秒の区間を切り出す。古い音声は捨てる。"""

    def __init__(self, keep_sec: float = 180.0) -> None:
        self.keep = int(keep_sec * SAMPLE_RATE)
        self.buf = np.zeros(0, dtype=np.int16)
        self.offset = 0  # buf[0] のストリーム内サンプル位置

    def append(self, block: np.ndarray) -> None:
        self.buf = np.concatenate([self.buf, block])
        if len(self.buf) > self.keep:
            cut = len(self.buf) - self.keep
            self.buf = self.buf[cut:]
            self.offset += cut

    def segment(self, start: float, end: float) -> np.ndarray:
        a = max(0, int(start * SAMPLE_RATE) - self.offset)
        b = max(0, int(end * SAMPLE_RATE) - self.offset)
        return self.buf[a:b].copy()
