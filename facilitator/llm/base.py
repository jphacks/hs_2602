"""LLM の抽象インターフェース。実装を差し替えてもエージェント側は変わらない。"""

from __future__ import annotations

from typing import Any, Protocol


class LLMError(Exception):
    pass


class LLMClient(Protocol):
    name: str

    def generate_json(
        self, system: str, user: str, schema: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """schema に従う JSON を返す。失敗時は LLMError を送出する。

        context はプロンプトの元になった構造化データ。モック実装がルールで応答するために使う。
        """
        ...
