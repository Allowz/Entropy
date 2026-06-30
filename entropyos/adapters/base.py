from __future__ import annotations

from abc import ABC, abstractmethod

from entropyos.config import LLMConfig


class LLMAdapter(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int | None = None, temperature: float | None = None) -> str: ...

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError(f"{type(self).__name__} does not support embeddings")

    @abstractmethod
    def name(self) -> str: ...


def create_adapter(config: LLMConfig) -> LLMAdapter:
    provider = config.provider.lower()
    if provider == "openai":
        from entropyos.adapters.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(config)
    elif provider == "anthropic":
        from entropyos.adapters.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(config)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
