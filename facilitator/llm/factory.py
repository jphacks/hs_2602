from __future__ import annotations

from config import Settings
from facilitator.llm.base import LLMClient


def make_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "mock":
        from facilitator.llm.mock_llm import MockLLM

        return MockLLM()
    if settings.llm_provider == "handoff":
        from facilitator.llm.handoff_llm import HandoffLLM

        return HandoffLLM()
    from facilitator.llm.anthropic_llm import AnthropicLLM

    return AnthropicLLM(settings)
