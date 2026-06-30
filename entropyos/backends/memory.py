from __future__ import annotations

import json
import os
import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from entropyos.models import Memory, MemoryLevel


class MemoryBackend(ABC):
    @abstractmethod
    def save(self, memories: list[Memory]) -> None: ...

    @abstractmethod
    def load(self) -> list[Memory]: ...

    @abstractmethod
    def name(self) -> str: ...


class LocalMemoryBackend(MemoryBackend):
    def save(self, memories: list[Memory]) -> None:
        pass

    def load(self) -> list[Memory]:
        return []

    def name(self) -> str:
        return "local"


class FileMemoryBackend(MemoryBackend):
    def __init__(self, path: str | Path = "entropy_memory.pkl"):
        self.path = Path(path)

    def save(self, memories: list[Memory]) -> None:
        data = []
        for m in memories:
            data.append({
                "id": m.id,
                "content": m.content,
                "level": m.level.value,
                "value": {
                    "novelty_score": m.value.novelty_score,
                    "relevance_score": m.value.relevance_score,
                    "usage_frequency": m.value.usage_frequency,
                    "age_seconds": m.value.age_seconds,
                    "dependency": m.value.dependency,
                    "compression_ratio": m.value.compression_ratio,
                    "combined_value": m.value.combined_value,
                },
                "created_at": m.created_at,
                "last_accessed": m.last_accessed,
                "access_count": m.access_count,
                "compressed": m.compressed,
                "compressed_content": m.compressed_content,
                "embedding": m.embedding,
                "user_id": m.user_id,
                "session_id": m.session_id,
                "metadata": m.metadata,
            })
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(data, f)

    def load(self) -> list[Memory]:
        if not self.path.exists():
            return []
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        result = []
        for d in data:
            m = Memory(
                id=d["id"],
                content=d["content"],
                level=MemoryLevel(d["level"]),
                created_at=d["created_at"],
                last_accessed=d["last_accessed"],
                access_count=d["access_count"],
                compressed=d["compressed"],
                compressed_content=d.get("compressed_content", ""),
                embedding=d.get("embedding"),
                user_id=d.get("user_id", "default"),
                session_id=d.get("session_id", "default"),
                metadata=d.get("metadata", {}),
            )
            m.value.novelty_score = d["value"]["novelty_score"]
            m.value.relevance_score = d["value"]["relevance_score"]
            m.value.usage_frequency = d["value"]["usage_frequency"]
            m.value.age_seconds = d["value"]["age_seconds"]
            m.value.dependency = d["value"]["dependency"]
            m.value.compression_ratio = d["value"]["compression_ratio"]
            m.value.combined_value = d["value"]["combined_value"]
            result.append(m)
        return result

    def name(self) -> str:
        return f"file:{self.path}"


def create_memory_backend(backend_type: str, **kwargs) -> MemoryBackend:
    if backend_type == "file":
        return FileMemoryBackend(**kwargs)
    return LocalMemoryBackend()
