from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from entropyos.models import ScoreResult, CompressResult, EvalResult, Memory


class ScorerPlugin(ABC):
    @abstractmethod
    def score(self, text: str, context: str | None = None) -> ScoreResult: ...

    @abstractmethod
    def name(self) -> str: ...


class CompressorPlugin(ABC):
    @abstractmethod
    def compress(self, text: str) -> CompressResult: ...

    @abstractmethod
    def name(self) -> str: ...


class EvaluatorPlugin(ABC):
    @abstractmethod
    def evaluate(self, original: str, compressed: str | None, response: str) -> EvalResult: ...

    @abstractmethod
    def name(self) -> str: ...


class RetrievalPlugin(ABC):
    @abstractmethod
    def retrieve(self, query: str, memories: list[Memory], top_k: int) -> list[Memory]: ...

    @abstractmethod
    def name(self) -> str: ...


class MemoryPlugin(ABC):
    @abstractmethod
    def on_insert(self, memory: Memory) -> None: ...

    @abstractmethod
    def on_tick(self, memories: list[Memory]) -> list[Memory]: ...

    @abstractmethod
    def name(self) -> str: ...
