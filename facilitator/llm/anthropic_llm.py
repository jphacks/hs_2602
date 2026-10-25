"""Claude API による実装(構造化出力 + 拒否時のサーバー側フォールバック)。"""

from __future__ import annotations

import json
from typing import Any, Callable

import anthropic

from config import Settings
from facilitator.llm.base import LLMError


class AnthropicLLM:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.name = f"anthropic:{settings.llm_model}"
        self.streams = True  # on_partial で生成途中の JSON を受け取れる
        # キーは環境変数 ANTHROPIC_API_KEY(.env)から SDK が読む。コード・ログには出さない。
        self.client = anthropic.Anthropic(timeout=settings.llm_timeout_sec, max_retries=1)
        self.last_usage: dict[str, int] | None = None  # 直前の呼び出しの使用トークン数(費用の計測用)

    def generate_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        context: dict[str, Any],
        on_partial: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """on_partial を渡すとストリーミングで受け取り、生成途中の JSON テキスト(先頭からの累積)を渡す。"""
        self.last_usage = None
        params: dict[str, Any] = dict(
            model=self.settings.llm_model,
            max_tokens=self.settings.llm_max_tokens,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": self.settings.llm_effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        try:
            if on_partial is None:
                response = self.client.beta.messages.create(**params)
            else:
                with self.client.beta.messages.stream(**params) as stream:
                    acc = ""
                    for delta in stream.text_stream:
                        acc += delta
                        on_partial(acc)
                    response = stream.get_final_message()
        except anthropic.APIError as e:
            raise LLMError(f"API error: {type(e).__name__}") from e

        usage = response.usage
        self.last_usage = {
            k: int(getattr(usage, k, 0) or 0)
            for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        }
        if response.stop_reason in ("refusal", "max_tokens"):
            raise LLMError(f"stop_reason={response.stop_reason}")
        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError("JSON parse failed") from e
