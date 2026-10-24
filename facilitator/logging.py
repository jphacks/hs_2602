"""構造化ログ(JSONL)。判断の再現・デバッグ用。API キーや発話本文(既定)は出力しない。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from facilitator.models import TickRecord


class TickLogger:
    def __init__(self, log_dir: Path, name: str, include_speech: bool = True) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{name}.jsonl"
        self.include_speech = include_speech

    def log_tick(self, record: TickRecord) -> None:
        data = record.model_dump(mode="json")
        if not self.include_speech:
            data["decision"]["speech"] = f"<{len(record.decision.speech)} chars>"
        self._write(data)

    def log_event(self, data: dict) -> None:
        """判断(tick)以外の出来事(軌道修正の発話・取り消しなど)を記録する。"""
        self._write({"event": data})

    def _write(self, data: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
