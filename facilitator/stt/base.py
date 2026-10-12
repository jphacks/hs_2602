"""STT(+話者分離)の抽象インターフェース。方式B(参加者ごとのマイク)等への差し替えもこの境界で行う。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol


@dataclass
class TranscriptEvent:
    kind: Literal["begin", "partial", "final", "revision", "error", "end"]
    text: str = ""
    speaker: str | None = None  # API の生ラベル(例: "A")。"PENDING" は未確定
    speaker_confidence: float | None = None  # 相対比較のみに使う(較正されていない)
    start: float | None = None  # ストリーム開始からの秒
    end: float | None = None
    turn_order: int | None = None
    revisions: list[dict[str, Any]] = field(default_factory=list)  # kind=revision 時: [{turn_order, speaker}]
    raw: dict[str, Any] = field(default_factory=dict)


class STTStream(Protocol):
    def start(self, on_event: Callable[[TranscriptEvent], None]) -> None: ...

    def send_audio(self, pcm16: bytes) -> None:
        """16kHz / mono / int16 little-endian の PCM を送る。"""
        ...

    def stop(self) -> None: ...
