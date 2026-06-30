from __future__ import annotations

from typing import Any

import httpx


class EntropyClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=timeout,
        )

    def score(self, text: str, context: str | None = None) -> dict[str, Any]:
        resp = self._client.post("/score", json={"text": text, "context": context})
        resp.raise_for_status()
        return resp.json()

    def compress(self, text: str, level: float | None = None) -> dict[str, Any]:
        resp = self._client.post("/compress", json={"text": text, "level": level})
        resp.raise_for_status()
        return resp.json()

    def retrieve(self, query: str, top_k: int = 5) -> dict[str, Any]:
        resp = self._client.post("/retrieve", json={"query": query, "top_k": top_k})
        resp.raise_for_status()
        return resp.json()

    def optimize(self, prompt: str, documents: list[str] | None = None) -> dict[str, Any]:
        resp = self._client.post("/optimize", json={"prompt": prompt, "documents": documents or []})
        resp.raise_for_status()
        return resp.json()

    def evaluate(self, original_prompt: str, response: str, compressed_prompt: str | None = None) -> dict[str, Any]:
        resp = self._client.post("/evaluate", json={
            "original_prompt": original_prompt,
            "compressed_prompt": compressed_prompt,
            "response": response,
        })
        resp.raise_for_status()
        return resp.json()

    def pipeline(self, prompt: str, user_id: str = "default", session_id: str = "default",
                 documents: list[str] | None = None, agent_state: dict | None = None) -> dict[str, Any]:
        resp = self._client.post("/pipeline", json={
            "prompt": prompt,
            "user_id": user_id,
            "session_id": session_id,
            "documents": documents,
            "agent_state": agent_state,
        })
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._client.close()
