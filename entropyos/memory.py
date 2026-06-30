from __future__ import annotations

import math
import time

from entropyos.backends.memory import FileMemoryBackend, LocalMemoryBackend, create_memory_backend
from entropyos.config import MemoryConfig
from entropyos.models import Memory, MemoryLevel, MemoryValue
from entropyos.security.audit import AuditLogger
from entropyos.security.encrypt import MemoryEncryptor
from entropyos.security.pii import PIIDetector


class HierarchicalMemory:
    def __init__(
        self,
        config: MemoryConfig | None = None,
        encryptor: MemoryEncryptor | None = None,
        audit: AuditLogger | None = None,
        pii: PIIDetector | None = None,
    ):
        self.config = config or MemoryConfig()
        self.encryptor = encryptor or MemoryEncryptor()
        self.audit = audit or AuditLogger()
        self.pii = pii or PIIDetector()
        self._backend = create_memory_backend(self.config.backend)
        self._stores: dict[MemoryLevel, list[Memory]] = {
            level: [] for level in MemoryLevel
        }
        self._load_from_backend()

    def insert(self, content: str, metadata: dict | None = None) -> Memory:
        if self.pii and self.pii.has_pii(content):
            content = self.pii.redact(content)
        mem = Memory(
            content=content,
            level=MemoryLevel.L0_CURRENT,
            metadata=metadata or {},
        )
        if self.encryptor.key:
            mem = self.encryptor.encrypt(mem)
        self._stores[MemoryLevel.L0_CURRENT].append(mem)
        self._enforce_budget()
        self.audit.log("memory_insert", resource=mem.id[:8], detail=f"len={len(content)}")
        return mem

    def get_level(self, level: MemoryLevel) -> list[Memory]:
        return self._stores[level][:]

    def promote(self, mem: Memory) -> None:
        current = mem.level
        if current >= MemoryLevel.L4_ARCHIVE:
            return
        next_level = MemoryLevel(current + 1)
        mem.level = next_level
        self._stores[next_level].append(mem)
        self._stores[current] = [m for m in self._stores[current] if m.id != mem.id]
        self.audit.log("memory_promote", resource=mem.id[:8], detail=f"L{current}->L{next_level}")

    def demote(self, mem: Memory) -> None:
        current = mem.level
        if current <= MemoryLevel.L0_CURRENT:
            return
        prev_level = MemoryLevel(current - 1)
        mem.level = prev_level
        self._stores[prev_level].append(mem)
        self._stores[current] = [m for m in self._stores[current] if m.id != mem.id]
        self.audit.log("memory_demote", resource=mem.id[:8], detail=f"L{current}->L{prev_level}")

    def access(self, mem: Memory) -> None:
        mem.last_accessed = time.time()
        mem.access_count += 1

    def compute_value(self, mem: Memory) -> MemoryValue:
        age = mem.age
        n_score = self._decay(age, self.config.novelty_decay_half_life_hours * 3600)
        r_score = self._decay(age, self.config.relevance_decay_half_life_hours * 3600)
        freq = min(1.0, mem.access_count / 10.0)
        dep = mem.value.dependency
        compression_ratio = mem.value.compression_ratio

        combined = (
            n_score * 0.25 +
            r_score * 0.30 +
            freq * 0.20 +
            (1.0 - min(1.0, age / (365 * 86400))) * 0.10 +
            dep * 0.10 +
            (1.0 - compression_ratio) * 0.05
        )
        combined = max(0.0, min(1.0, combined))

        return MemoryValue(
            novelty_score=n_score,
            relevance_score=r_score,
            usage_frequency=mem.access_count,
            age_seconds=age,
            dependency=dep,
            compression_ratio=compression_ratio,
            combined_value=combined,
        )

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> list[Memory]:
        candidates = self.all_memories()
        scored = []
        q_words = set(query.lower().split())
        for mem in candidates:
            m_words = set(mem.content.lower().split())
            if not q_words or not m_words:
                score = 0.0
            else:
                overlap = len(q_words & m_words)
                score = overlap / max(len(q_words), len(m_words))
            if score >= min_score:
                scored.append((score, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def tick(self) -> list[Memory]:
        promoted: list[Memory] = []
        now = time.time()
        ttls = {
            MemoryLevel.L0_CURRENT: self.config.l0_ttl_seconds,
            MemoryLevel.L1_CONVERSATION: self.config.l1_ttl_seconds,
            MemoryLevel.L2_SESSION: self.config.l2_ttl_seconds,
            MemoryLevel.L3_LONG_TERM: self.config.l3_ttl_seconds,
            MemoryLevel.L4_ARCHIVE: self.config.l4_ttl_seconds,
        }

        for level in reversed(list(MemoryLevel)):
            ttl = ttls[level]
            for mem in self._stores[level][:]:
                if ttl > 0 and (now - mem.last_accessed) > ttl * 3:
                    self._stores[level].remove(mem)
                    self.audit.log("memory_delete", resource=mem.id[:8], detail="ttl_expired")
                    continue
                val = self.compute_value(mem)
                mem.value = val
                if level < MemoryLevel.L4_ARCHIVE and val.combined_value > 0.7:
                    self.promote(mem)
                    promoted.append(mem)
                elif level > MemoryLevel.L0_CURRENT and val.combined_value < 0.2:
                    self.demote(mem)
                if level < MemoryLevel.L4_ARCHIVE and val.combined_value < 0.05:
                    self._stores[level].remove(mem)
                    self.audit.log("memory_delete", resource=mem.id[:8], detail="low_value")

        self._save_to_backend()
        return promoted

    def all_memories(self) -> list[Memory]:
        result: list[Memory] = []
        for level in MemoryLevel:
            result.extend(self._stores[level])
        return result

    def clear(self) -> None:
        for level in MemoryLevel:
            self._stores[level].clear()

    def _enforce_budget(self) -> None:
        total = sum(len(v) for v in self._stores.values())
        if total > 100_000:
            self._stores[MemoryLevel.L4_ARCHIVE] = sorted(
                self._stores[MemoryLevel.L4_ARCHIVE],
                key=lambda m: self.compute_value(m).combined_value,
            )[:10_000]

    def _save_to_backend(self) -> None:
        if isinstance(self._backend, FileMemoryBackend):
            self._backend.save(self.all_memories())

    def _load_from_backend(self) -> None:
        if isinstance(self._backend, FileMemoryBackend):
            memories = self._backend.load()
            for m in memories:
                self._stores[m.level].append(m)

    @staticmethod
    def _decay(age: float, half_life: float) -> float:
        if half_life <= 0:
            return 0.0
        return math.exp(-math.log(2) * age / half_life)
