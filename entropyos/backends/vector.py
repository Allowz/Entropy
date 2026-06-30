from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any


class VectorBackend(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    def similarity(self, a: list[float], b: list[float]) -> float: ...

    @abstractmethod
    def name(self) -> str: ...


class NGramVectorBackend(VectorBackend):
    def __init__(self, n: int = 3):
        self.n = n

    def embed(self, text: str) -> list[float]:
        cleaned = re.sub(r'[^a-z0-9\s]', '', text.lower())
        tokens = cleaned.split()
        ngrams: Counter[str] = Counter()
        for token in tokens:
            token = f"^{token}$"
            for i in range(len(token) - self.n + 1):
                ngrams[token[i:i + self.n]] += 1
        if not ngrams:
            return []
        max_count = max(ngrams.values())
        return [ngrams[g] / max_count for g in sorted(ngrams)]

    def similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(ai * bi for ai, bi in zip(a, b))
        norm_a = math.sqrt(sum(ai * ai for ai in a))
        norm_b = math.sqrt(sum(bi * bi for bi in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def name(self) -> str:
        return f"ngram-{self.n}"


class EmbeddingVectorBackend(VectorBackend):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: Any = None

    def _lazy_load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def embed(self, text: str) -> list[float]:
        self._lazy_load()
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(ai * bi for ai, bi in zip(a, b))
        return max(0.0, min(1.0, dot))

    def name(self) -> str:
        return f"embed-{self.model_name}"


def create_vector_backend(engine: str = "ngram", model_name: str = "all-MiniLM-L6-v2") -> VectorBackend:
    if engine == "ngram":
        return NGramVectorBackend(n=3)
    elif engine == "embedding":
        return EmbeddingVectorBackend(model_name=model_name)
    else:
        return NGramVectorBackend(n=3)
