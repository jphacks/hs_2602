"""LLM の抽象インターフェース。実装を差し替えてもエージェント側は変わらない。"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Protocol


class LLMError(Exception):
    pass


class LLMClient(Protocol):
    name: str

    def generate_json(
        self, system: str, user: str, schema: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """schema に従う JSON を返す。失敗時は LLMError を送出する。

        context はプロンプトの元になった構造化データ。モック実装がルールで応答するために使う。
        属性 streams が True の実装は、キーワード引数 on_partial(生成途中の JSON テキストを受け取る関数)も受け付ける。
        """
        ...


def supports_stream(llm: Any) -> bool:
    """on_partial を受け取れる LLM か(streams 属性があり、generate_json が on_partial を受け付ける)。"""
    if not getattr(llm, "streams", False):
        return False
    try:
        params = inspect.signature(llm.generate_json).parameters
    except (TypeError, ValueError):
        return False
    return "on_partial" in params


def generate(
    llm: LLMClient,
    system: str,
    user: str,
    schema: dict[str, Any],
    context: dict[str, Any],
    on_partial: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """on_partial はストリーミングに対応した LLM にだけ渡す(対応していなければ、返事を待ってから一度に処理する)。"""
    if on_partial is not None and supports_stream(llm):
        return llm.generate_json(system, user, schema, context, on_partial=on_partial)  # type: ignore[call-arg]
    return llm.generate_json(system, user, schema, context)
