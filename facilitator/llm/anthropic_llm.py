"""Claude API による実装(構造化出力 + 拒否時のサーバー側フォールバック)。"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from config import Settings
from facilitator.llm.base import LLMError


class AnthropicLLM:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.name = f"anthropic:{settings.llm_model}"
        # キーは環境変数 ANTHROPIC_API_KEY(.env)から SDK が読む。コード・ログには出さない。
        self.client = anthropic.Anthropic(timeout=settings.llm_timeout_sec, max_retries=1)

    def generate_json(
        self, system: str, user: str, schema: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            response = self.client.beta.messages.create(
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
        except anthropic.APIError as e:
            raise LLMError(f"API error: {type(e).__name__}") from e

        if response.stop_reason in ("refusal", "max_tokens"):
            raise LLMError(f"stop_reason={response.stop_reason}")
        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError("JSON parse failed") from e
