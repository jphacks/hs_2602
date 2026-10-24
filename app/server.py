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

import random  # noqa: E402

from config import CUSTOM_KEYS, PRESETS, STYLE_KNOBS, load_settings  # noqa: E402
from facilitator.llm.base import LLMError  # noqa: E402
from facilitator.llm.factory import make_llm  # noqa: E402
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
    ai_count: int = 0  # AI 参加者の人数(names は人間の参加者のみ)
    ai_level: int = 2  # AI 参加者のレベル(1〜3)
    knobs: dict[str, int] = {}  # 詳細の5項目(0=控えめ / 1=標準 / 2=積極的)


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
    ai_count = max(0, min(3, req.ai_count))
    if not names or len(names) + ai_count < 2:
        return {"ok": False, "error": "人間1人以上、合計2人以上にしてください"}
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
                ai_count=ai_count,
                ai_level=req.ai_level if req.ai_level in (1, 2, 3) else 2,
                knobs={k: v for k, v in req.knobs.items() if k in STYLE_KNOBS and v in (0, 1, 2)},
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
        "knobs": {k: v["label"] for k, v in STYLE_KNOBS.items()},
    }


# テーマの自動生成(AI が使えないときやモックのときは候補から選ぶ)
THEME_POOL = {
    "job": [
        "大学の学食の混雑を改善するには？", "新しい観光名所をつくるなら？", "10年後に残る仕事・なくなる仕事",
        "地方の若者流出を止めるには？", "優れているのは現金かキャッシュレスか", "コンビニの売上を1.5倍にするには？",
        "無人島に1つだけ持っていくなら何か", "社員の満足度を高める新しい福利厚生を考えよ",
    ],
    "general": [
        "次の学園祭の企画を決めよう", "チームの会議を短くするには？", "研究室の共有スペースの使い方",
        "新歓イベントのアイデア", "授業でスマホを使うのはありか", "地域のゴミの分別を徹底するには？",
    ],
}


class ThemeReq(BaseModel):
    purpose: str = "job"
    exclude: str = ""


@app.post("/api/theme")
def theme(req: ThemeReq) -> dict:
    purpose = req.purpose if req.purpose in THEME_POOL else "job"
    pool = [t for t in THEME_POOL[purpose] if t != req.exclude] or THEME_POOL[purpose]
    llm = make_llm(load_settings())
    if llm.name in ("mock", "handoff"):
        return {"theme": random.choice(pool), "source": "pool"}
    who = "就職活動の選考で行うグループディスカッション" if purpose == "job" else "授業・会議・ワークショップで行うグループディスカッション"
    system = (
        f"あなたは{who}のテーマを考えます。3〜4人が5〜15分で議論でき、正解が1つに決まらず、"
        "立場や観点によって意見が分かれるテーマを1つだけ作ってください。30字以内の疑問形か「〜せよ」の形にしてください。"
    )
    user = f"直前に使ったテーマ(これとは違うものにする): {req.exclude or 'なし'}"
    schema = {"type": "object", "properties": {"theme": {"type": "string"}}, "required": ["theme"], "additionalProperties": False}
    try:
        t = str(llm.generate_json(system, user, schema, {"kind": "テーマの自動生成"}).get("theme", "")).strip()
        if t:
            return {"theme": t[:60], "source": "llm"}
    except LLMError:
        pass
    return {"theme": random.choice(pool), "source": "pool"}


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
