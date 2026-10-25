"""チャット経由モード(LLM_PROVIDER=handoff): API を呼ばず、プロンプトをファイルに書き出して返事を待つ。

Claude Code などのチャットの AI(または人)が、書き出されたリクエストを読んで JSON の返事を書き込む。
API キーを使わずに、エージェントの判断の流れ(ルール層 → LLM の入力 → 判断)を確かめるためのもの。

- リクエスト: exchange/request_NNN.md (システムプロンプト・ユーザープロンプト・出力スキーマ)
- 返事     : exchange/response_NNN.json (スキーマどおりの JSON)
返事を待つ間は処理が止まるので、仮想時計で進むシナリオ再生(scripts/evaluate.py, run_scenario.py)向け。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from config import ROOT
from facilitator.llm.base import LLMError

DEFAULT_DIR = ROOT / "exchange"


class HandoffLLM:
    name = "handoff"

    def __init__(self, directory: Path | None = None, timeout_sec: float | None = None) -> None:
        self.dir = Path(directory or os.getenv("HANDOFF_DIR") or DEFAULT_DIR)
        self.timeout = float(timeout_sec or os.getenv("HANDOFF_TIMEOUT_SEC") or 900)
        self.dir.mkdir(parents=True, exist_ok=True)
        # 前回の実行の残りを片付ける(同じ番号の返事を誤って読まないため)
        for f in self.dir.glob("request_*.md"):
            f.unlink()
        for f in self.dir.glob("response_*.json"):
            f.unlink()
        self.n = 0

    def generate_json(
        self, system: str, user: str, schema: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        self.n += 1
        req = self.dir / f"request_{self.n:03d}.md"
        res = self.dir / f"response_{self.n:03d}.json"
        req.write_text(
            f"# リクエスト {self.n:03d}: {context.get('kind', '進行役の判断')}\n\n"
            f"返事は `{res.name}` に、下の出力スキーマどおりの JSON だけを書いてください。\n\n"
            f"## システムプロンプト\n\n{system}\n\n"
            f"## ユーザープロンプト\n\n{user}\n\n"
            f"## 出力スキーマ\n\n```json\n{json.dumps(schema, ensure_ascii=False, indent=1)}\n```\n",
            encoding="utf-8",
        )
        kind = context.get("kind", "進行役の判断")
        print(f"\n⏳ 返事待ち({kind}): {req.relative_to(ROOT) if req.is_relative_to(ROOT) else req}", flush=True)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if res.exists():
                time.sleep(0.2)  # 書き込みの途中を読まないよう少し待つ
                try:
                    return json.loads(res.read_text(encoding="utf-8"))
                except json.JSONDecodeError as e:
                    raise LLMError(f"{res.name} の JSON が不正です") from e
            time.sleep(0.5)
        raise LLMError(f"{res.name} の返事が {self.timeout:.0f} 秒以内に来ませんでした")
