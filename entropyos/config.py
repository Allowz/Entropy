from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MemoryConfig:
    l0_ttl_seconds: int = 300
    l1_ttl_seconds: int = 3600
    l2_ttl_seconds: int = 86400
    l3_ttl_seconds: int = 604800
    l4_ttl_seconds: int = 0
    novelty_decay_half_life_hours: float = 24.0
    relevance_decay_half_life_hours: float = 48.0
    backend: str = "local"


@dataclass
class RetrievalConfig:
    engine: str = "vector"
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 5
    min_score: float = 0.3
    use_mmr: bool = True
    mmr_lambda: float = 0.5


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 1024
    temperature: float = 0.0
    timeout_seconds: float = 30.0


@dataclass
class SecurityConfig:
    encrypt_memory: bool = False
    encryption_key: str = ""
    pii_detection: bool = True
    audit_log: bool = True
    audit_log_path: str = ""
    memory_expiration_days: int = 90


@dataclass
class PluginConfig:
    enabled: bool = True
    scan_paths: list[str] = field(default_factory=list)
    auto_discover: bool = True


@dataclass
class EntropyConfig:
    max_token_budget: int = 4096
    max_memory_budget: int = 100_000
    compression_level: float = 0.5
    recall_threshold: float = 0.3
    hallucination_threshold: float = 0.15
    latency_budget_ms: float = 500.0
    cost_budget: float = 0.01
    log_level: str = "INFO"

    memory: MemoryConfig = field(default_factory=MemoryConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    plugins: PluginConfig = field(default_factory=PluginConfig)

    _source: str = "defaults"

    @classmethod
    def from_dict(cls, d: dict) -> EntropyConfig:
        cfg = cls()
        for key, val in d.items():
            if hasattr(cfg, key) and not isinstance(getattr(cfg, key), (MemoryConfig, RetrievalConfig, LLMConfig, SecurityConfig, PluginConfig)):
                setattr(cfg, key, val)
        if "memory" in d:
            for k, v in d["memory"].items():
                if hasattr(cfg.memory, k):
                    setattr(cfg.memory, k, v)
        if "retrieval" in d:
            for k, v in d["retrieval"].items():
                if hasattr(cfg.retrieval, k):
                    setattr(cfg.retrieval, k, v)
        if "llm" in d:
            for k, v in d["llm"].items():
                if hasattr(cfg.llm, k):
                    setattr(cfg.llm, k, v)
        if "security" in d:
            for k, v in d["security"].items():
                if hasattr(cfg.security, k):
                    setattr(cfg.security, k, v)
        if "plugins" in d:
            for k, v in d["plugins"].items():
                if hasattr(cfg.plugins, k):
                    setattr(cfg.plugins, k, v)
        cfg._apply_env_overrides()
        cfg._source = "dict"
        return cfg

    @classmethod
    def from_file(cls, path: str | Path) -> EntropyConfig:
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        cfg = cls.from_dict(d)
        cfg._source = str(path)
        return cfg

    @classmethod
    def from_env(cls) -> EntropyConfig:
        cfg = cls()
        cfg._apply_env_overrides()
        cfg._source = "env"
        return cfg

    def _apply_env_overrides(self) -> None:
        mapping = {
            "ENTROPY_MAX_TOKEN_BUDGET": ("max_token_budget", int),
            "ENTROPY_MAX_MEMORY_BUDGET": ("max_memory_budget", int),
            "ENTROPY_COMPRESSION_LEVEL": ("compression_level", float),
            "ENTROPY_RECALL_THRESHOLD": ("recall_threshold", float),
            "ENTROPY_LOG_LEVEL": ("log_level", str),
            "ENTROPY_LLM_PROVIDER": ("llm.provider", str),
            "ENTROPY_LLM_MODEL": ("llm.model", str),
            "ENTROPY_LLM_API_KEY": ("llm.api_key", str),
            "ENTROPY_LLM_BASE_URL": ("llm.base_url", str),
            "ENTROPY_SECURITY_ENCRYPT_MEMORY": ("security.encrypt_memory", bool),
            "ENTROPY_SECURITY_ENCRYPTION_KEY": ("security.encryption_key", str),
            "ENTROPY_SECURITY_PII_DETECTION": ("security.pii_detection", bool),
            "ENTROPY_MEMORY_BACKEND": ("memory.backend", str),
            "ENTROPY_RETRIEVAL_ENGINE": ("retrieval.engine", str),
        }
        for env_var, (attr_path, converter) in mapping.items():
            val = os.environ.get(env_var)
            if val is not None:
                parts = attr_path.split(".")
                if len(parts) == 1:
                    setattr(self, parts[0], converter(val))
                elif len(parts) == 2:
                    parent = getattr(self, parts[0])
                    setattr(parent, parts[1], converter(val))
