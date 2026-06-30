from __future__ import annotations

from entropyos.adapters.base import LLMAdapter


class OpenAIAdapter(LLMAdapter):
    def __init__(self, config):
        super().__init__(config)
        self._client = None

    def _lazy_client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs = {"api_key": self.config.api_key} if self.config.api_key else {}
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def complete(self, prompt: str, max_tokens: int | None = None, temperature: float | None = None) -> str:
        client = self._lazy_client()
        resp = client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature if temperature is not None else self.config.temperature,
        )
        return resp.choices[0].message.content or ""

    def name(self) -> str:
        return f"openai:{self.config.model}"
