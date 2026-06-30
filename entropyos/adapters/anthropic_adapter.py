from __future__ import annotations

from entropyos.adapters.base import LLMAdapter


class AnthropicAdapter(LLMAdapter):
    def __init__(self, config):
        super().__init__(config)
        self._client = None

    def _lazy_client(self):
        if self._client is None:
            from anthropic import Anthropic
            kwargs = {"api_key": self.config.api_key} if self.config.api_key else {}
            self._client = Anthropic(**kwargs)
        return self._client

    def complete(self, prompt: str, max_tokens: int | None = None, temperature: float | None = None) -> str:
        client = self._lazy_client()
        resp = client.messages.create(
            model=self.config.model,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature if temperature is not None else self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text if resp.content else ""

    def name(self) -> str:
        return f"anthropic:{self.config.model}"
