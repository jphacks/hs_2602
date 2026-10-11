from __future__ import annotations

from config import Settings
from facilitator.llm.base import LLMClient


def make_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "mock":
        from facilitator.llm.mock_llm import MockLLM

        return MockLLM()
    from facilitator.llm.anthropic_llm import AnthropicLLM

    return AnthropicLLM(settings)
