"""UI サーバー(FastAPI + WebSocket)。

起動(マイクを使うので macOS の「ターミナル」アプリで):
  .venv/bin/python -m uvicorn app.server:app --port 8000
ブラウザで http://localhost:8000 を開く。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CUSTOM_KEYS, PRESETS, load_settings  # noqa: E402
from facilitator.engine import Engine, SessionConfig  # noqa: E402

STATIC = Path(__file__).parent / "static"
app = FastAPI()
engine = Engine(load_settings())


class StartReq(BaseModel):
    theme: str
    minutes: float = 10
    names: list[str]
    mode: str = "voice"
    intro: bool = True
    tts: bool = True
    use_speaker_info: bool = True
    purpose: str = "job"
    preset: str = "A"
    custom: dict[str, float] = {}


class TextReq(BaseModel):
    speaker_id: str
    text: str


class RelabelReq(BaseModel):
    id: int
    speaker_id: str


class ToggleReq(BaseModel):
    use_speaker_info: bool


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.post("/api/start")
def start(req: StartReq) -> dict:
    names = [n.strip() for n in req.names if n.strip()][:4]
    if len(names) < 2:
        return {"ok": False, "error": "参加者は2〜4人を入力してください"}
    try:
        engine.start(
            SessionConfig(
                req.theme.strip(),
                req.minutes,
                names,
                req.mode,
                req.intro,
                req.tts,
                req.use_speaker_info,
                purpose=req.purpose if req.purpose in ("job", "general") else "job",
                preset=req.preset if req.preset in ("A", "B", "custom") else "A",
                custom=req.custom,
            )
        )
    except Exception as e:  # マイク・STT の起動失敗を UI に伝える(キー等の秘匿情報は含まれない)
        engine.stop()
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {"ok": True}


@app.get("/api/info")
def info() -> dict:
    """画面の初期値。モック版(mock-test ブランチ)ではテキスト入力を既定にする。"""
    base = load_settings()
    mock = base.llm_provider == "mock"
    defaults = {k: getattr(base, k) for k in sorted(CUSTOM_KEYS)}
    return {
        "llm": "mock" if mock else base.llm_model,
        "default_mode": "text" if mock else "voice",
        "presets": {name: {**defaults, **vals} for name, vals in PRESETS.items()},
    }


@app.post("/api/pause")
def pause() -> dict:
    engine.pause()
    return {"ok": True}


@app.post("/api/resume")
def resume() -> dict:
    engine.resume()
    return {"ok": True}


@app.post("/api/stop")
def stop() -> dict:
    engine.stop()
    return {"ok": True}


@app.post("/api/text")
def text(req: TextReq) -> dict:
    engine.add_text(req.speaker_id, req.text)
    return {"ok": True}


@app.post("/api/relabel")
def relabel(req: RelabelReq) -> dict:
    engine.relabel(req.id, req.speaker_id)
    return {"ok": True}


@app.post("/api/toggle")
def toggle(req: ToggleReq) -> dict:
    engine.set_use_speaker_info(req.use_speaker_info)
    return {"ok": True}


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    await sock.accept()
    try:
        while True:
            await sock.send_json(engine.snapshot())
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, RuntimeError):
        pass
