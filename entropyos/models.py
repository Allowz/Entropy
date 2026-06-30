from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Protocol


class MemoryLevel(IntEnum):
    L0_CURRENT = 0
    L1_CONVERSATION = 1
    L2_SESSION = 2
    L3_LONG_TERM = 3
    L4_ARCHIVE = 4


@dataclass
class ScoreResult:
    entropy: float
    novelty: float
    redundancy: float
    importance: float
    dependency: float
    information_score: float
    method: str = "heuristic"


@dataclass
class MemoryValue:
    novelty_score: float = 0.0
    relevance_score: float = 0.0
    usage_frequency: int = 0
    age_seconds: float = 0.0
    dependency: float = 0.0
    compression_ratio: float = 1.0
    combined_value: float = 0.0


@dataclass
class Memory:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    level: MemoryLevel = MemoryLevel.L0_CURRENT
    value: MemoryValue = field(default_factory=MemoryValue)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    compressed: bool = False
    compressed_content: str = ""
    embedding: list[float] | None = None
    user_id: str = "default"
    session_id: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def age(self) -> float:
        return time.time() - self.created_at


@dataclass
class CompressResult:
    compressed: str
    compression_ratio: float
    info_preserved: float
    method: str = "heuristic"


@dataclass
class RetrievalResult:
    memories: list[Memory]
    information_gain: float


@dataclass
class EvalResult:
    accuracy: float
    completeness: float
    hallucination_score: float
    compression_effectiveness: float
    information_retained: float


@dataclass
class AuditEntry:
    timestamp: float = field(default_factory=time.time)
    action: str = ""
    user_id: str = ""
    resource: str = ""
    detail: str = ""
    ip: str = ""


class ModelAdapter(Protocol):
    def complete(self, prompt: str, **kwargs) -> str: ...
    def embed(self, text: str) -> list[float]: ...
    def cosine_similarity(self, a: list[float], b: list[float]) -> float: ...
