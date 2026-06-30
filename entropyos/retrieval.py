from __future__ import annotations

import math
import time

from entropyos.backends.vector import NGramVectorBackend, VectorBackend, create_vector_backend
from entropyos.config import RetrievalConfig
from entropyos.memory import HierarchicalMemory
from entropyos.models import Memory, MemoryLevel


class RetrievalEngine:
    def __init__(self, memory: HierarchicalMemory, config: RetrievalConfig | None = None):
        self.memory = memory
        self.config = config or RetrievalConfig()
        self._vector: VectorBackend = create_vector_backend(
            engine=self.config.engine,
            model_name=self.config.embedding_model,
        )

    def retrieve(self, query: str, top_k: int | None = None, min_score: float | None = None) -> tuple[list[Memory], float]:
        k = top_k or self.config.top_k
        min_s = min_score if min_score is not None else self.config.min_score
        query_vec = self._vector.embed(query)
        candidates = self.memory.all_memories()

        scored: list[tuple[Memory, float]] = []
        for mem in candidates:
            mem_emb = mem.embedding or self._vector.embed(mem.content)
            if mem.embedding is None:
                mem.embedding = mem_emb
            sim = self._vector.similarity(query_vec, mem_emb)
            info_gain = self._compute_info_gain(mem, sim)
            scored.append((mem, info_gain))

        if self.config.use_mmr:
            scored = self._mmr_rerank(scored, query_vec, k)

        scored.sort(key=lambda x: x[1], reverse=True)
        filtered = [(mem, s) for mem, s in scored if s >= min_s]
        top = filtered[:k]

        for mem, _ in top:
            self.memory.access(mem)

        total_gain = sum(s for _, s in top) if top else 0.0
        return [mem for mem, _ in top], total_gain

    def _compute_info_gain(self, mem: Memory, similarity: float) -> float:
        relevance = similarity
        dependency = mem.value.dependency
        freshness = self._freshness(mem)
        confidence = self._confidence(mem)

        return (
            relevance * 0.35 +
            dependency * 0.20 +
            freshness * 0.25 +
            confidence * 0.20
        )

    def _mmr_rerank(self, scored: list[tuple[Memory, float]], query_vec: list[float], k: int) -> list[tuple[Memory, float]]:
        if not scored:
            return scored
        selected: list[tuple[Memory, float]] = []
        candidates = scored[:]

        while len(selected) < k and candidates:
            best_idx = 0
            best_score = -1.0
            for i, (mem, rel) in enumerate(candidates):
                mmr = self.config.mmr_lambda * rel
                if selected:
                    max_sim = max(
                        self._vector.similarity(
                            query_vec,
                            mem.embedding or self._vector.embed(mem.content),
                        )
                        for _, s_mem in selected
                    )
                    mmr -= (1 - self.config.mmr_lambda) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i
            selected.append(candidates.pop(best_idx))

        return selected + candidates

    @staticmethod
    def _freshness(mem: Memory) -> float:
        age = time.time() - mem.created_at
        return max(0.0, 1.0 - age / (7 * 86400))

    @staticmethod
    def _confidence(mem: Memory) -> float:
        base = min(1.0, mem.access_count / 5.0)
        recency = max(0.0, 1.0 - (time.time() - mem.last_accessed) / (7 * 86400))
        return base * 0.5 + recency * 0.5
