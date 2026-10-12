"""AssemblyAI Streaming v3(WebSocket)による STT + 話者分離。

仕様の出典: https://www.assemblyai.com/docs/streaming/streaming-diarization (2026-10 確認)
- エンドポイント: wss://streaming.assemblyai.com/v3/ws、API キーは Authorization ヘッダー
- speaker_labels=true / max_speakers(1〜10) / speaker_labels_revision_interval_ms(>=120000)
- Turn: transcript, end_of_turn, turn_is_formatted, speaker_label, speaker_confidence, words[{start,end(ms),speaker}]
- SpeakerRevision: revisions[{turn_order, speaker_label}]
未確認: 日本語に対応するモデル ID(universal-3-6-pro / universal-3-5-pro)、言語を固定するパラメータ名
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Callable
from urllib.parse import urlencode

import websocket

from facilitator.stt.base import TranscriptEvent

URL = "wss://streaming.assemblyai.com/v3/ws"


class AssemblyAIStream:
    def __init__(
        self,
        speech_model: str = "universal-3-6-pro",
        sample_rate: int = 16000,
        speaker_labels: bool = True,
        max_speakers: int | None = 4,
        revision_interval_ms: int | None = 120000,
        extra_params: dict[str, str] | None = None,
    ) -> None:
        params: dict[str, Any] = {"sample_rate": sample_rate, "speech_model": speech_model, "format_turns": "true"}
        if speaker_labels:
            params["speaker_labels"] = "true"  # urlencode に bool を渡すと "True" になるため文字列で
            if max_speakers:
                params["max_speakers"] = max_speakers
            if revision_interval_ms:
                params["speaker_labels_revision_interval_ms"] = revision_interval_ms
        params.update(extra_params or {})
        self.url = f"{URL}?{urlencode(params)}"
        self.params = params
        self._ws: websocket.WebSocketApp | None = None
        self._thread: threading.Thread | None = None
        self._open = threading.Event()
        self._on_event: Callable[[TranscriptEvent], None] = lambda e: None

    def start(self, on_event: Callable[[TranscriptEvent], None]) -> None:
        key = os.getenv("ASSEMBLYAI_API_KEY")
        if not key:
            raise RuntimeError("ASSEMBLYAI_API_KEY が .env に設定されていません")
        self._on_event = on_event
        self._ws = websocket.WebSocketApp(
            self.url,
            header={"Authorization": key},
            on_open=lambda ws: self._open.set(),
            on_message=self._handle,
            on_error=lambda ws, err: on_event(TranscriptEvent("error", text=str(err))),
            on_close=lambda ws, code, msg: on_event(TranscriptEvent("end", text=f"close {code} {msg or ''}")),
        )
        self._thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._thread.start()
        if not self._open.wait(timeout=10):
            raise RuntimeError("AssemblyAI への接続がタイムアウトしました")

    def send_audio(self, pcm16: bytes) -> None:
        if self._ws and self._open.is_set():
            try:
                self._ws.send(pcm16, opcode=websocket.ABNF.OPCODE_BINARY)
            except websocket.WebSocketConnectionClosedException:
                self._open.clear()

    def stop(self) -> None:
        if self._ws and self._open.is_set():
            try:
                self._ws.send(json.dumps({"type": "Terminate"}))
            except websocket.WebSocketConnectionClosedException:
                pass
        if self._thread:
            self._thread.join(timeout=5)
        if self._ws:
            self._ws.close()

    # ---------- 受信 ----------
    def _handle(self, ws: Any, message: str) -> None:
        msg = json.loads(message)
        t = msg.get("type")
        if t == "Begin":
            self._on_event(TranscriptEvent("begin", text=str(msg.get("id", "")), raw=msg))
        elif t == "Turn":
            words = msg.get("words") or []
            start = words[0]["start"] / 1000 if words else None
            end = words[-1]["end"] / 1000 if words else None
            final = bool(msg.get("end_of_turn")) and bool(msg.get("turn_is_formatted", True))
            self._on_event(
                TranscriptEvent(
                    "final" if final else "partial",
                    text=msg.get("transcript", ""),
                    speaker=msg.get("speaker_label"),
                    speaker_confidence=msg.get("speaker_confidence"),
                    start=start,
                    end=end,
                    turn_order=msg.get("turn_order"),
                    raw=msg,
                )
            )
        elif t == "SpeakerRevision":
            revs = [
                {"turn_order": r.get("turn_order"), "speaker": r.get("speaker_label")}
                for r in msg.get("revisions", [])
            ]
            self._on_event(TranscriptEvent("revision", revisions=revs, raw=msg))
        elif t == "Termination":
            self._on_event(TranscriptEvent("end", text="terminated", raw=msg))
        elif t == "Error" or "error" in msg:
            self._on_event(TranscriptEvent("error", text=json.dumps(msg, ensure_ascii=False), raw=msg))
