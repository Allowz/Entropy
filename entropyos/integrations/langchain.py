from __future__ import annotations

import json
import time
from typing import Any

from entropyos.config import EntropyConfig
from entropyos.pipeline import EntropyPipeline

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.messages import BaseMessage, HumanMessage
    from langchain_core.outputs import LLMResult
except ImportError:
    BaseCallbackHandler = object
    BaseLanguageModel = object
    BaseMessage = object
    HumanMessage = object
    LLMResult = object


class EntropyCallback(BaseCallbackHandler):
    """LangChain callback that intercepts prompts, runs them through EntropyOS,
    and injects the optimized prompt before the LLM call."""

    def __init__(self, config: EntropyConfig | None = None):
        self.pipeline = EntropyPipeline(config=config)
        self.metrics: list[dict[str, Any]] = []

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        for i, prompt in enumerate(prompts):
            result = self.pipeline.run(prompt)
            prompts[i] = result["compiled_prompt"]
            self.metrics.append({
                "original_len": len(prompt),
                "compressed_len": len(result["compiled_prompt"]),
                "compression_ratio": result["optimization"]["compression_ratio"],
                "info_preserved": result["optimization"]["info_preserved"],
                "info_score": result["score"]["information_score"],
                "latency_ms": result["total_time_ms"],
            })

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        for msg_list in messages:
            for msg in msg_list:
                if isinstance(msg, HumanMessage):
                    result = self.pipeline.run(msg.content)
                    msg.content = result["compiled_prompt"]
                    self.metrics.append({
                        "original_len": len(msg.content),
                        "compressed_len": len(result["compiled_prompt"]),
                        "compression_ratio": result["optimization"]["compression_ratio"],
                        "info_preserved": result["optimization"]["info_preserved"],
                        "info_score": result["score"]["information_score"],
                        "latency_ms": result["total_time_ms"],
                    })

    def summary(self) -> dict[str, Any]:
        if not self.metrics:
            return {"calls": 0}
        total_original = sum(m["original_len"] for m in self.metrics)
        total_compressed = sum(m["compressed_len"] for m in self.metrics)
        saved_pct = round(max(0, total_original - total_compressed) / total_original * 100, 1) if total_original else 0
        return {
            "calls": len(self.metrics),
            "total_original_chars": total_original,
            "total_compressed_chars": total_compressed,
            "avg_compression_ratio": sum(m["compression_ratio"] for m in self.metrics) / len(self.metrics),
            "avg_info_preserved": sum(m["info_preserved"] for m in self.metrics) / len(self.metrics),
            "avg_info_score": sum(m["info_score"] for m in self.metrics) / len(self.metrics),
            "avg_latency_ms": sum(m["latency_ms"] for m in self.metrics) / len(self.metrics),
            "total_tokens_saved_pct": saved_pct,
        }


class EntropyLangchain:
    """Wraps a LangChain LLM to transparently optimize all prompts."""

    def __init__(self, llm: BaseLanguageModel, config: EntropyConfig | None = None):
        self.llm = llm
        self.callback = EntropyCallback(config=config)
        if hasattr(llm, "callbacks"):
            existing = llm.callbacks or []
            self.llm.callbacks = existing + [self.callback]

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        result = self.callback.pipeline.run(prompt)
        optimized = result["compiled_prompt"]
        self.callback.metrics.append({
            "original_len": len(prompt),
            "compressed_len": len(optimized),
            "compression_ratio": result["optimization"]["compression_ratio"],
            "info_preserved": result["optimization"]["info_preserved"],
            "info_score": result["score"]["information_score"],
            "latency_ms": result["total_time_ms"],
        })
        response = self.llm.invoke(optimized, **kwargs)
        content = response.content if hasattr(response, "content") else str(response)
        return content

    def summary(self) -> dict[str, Any]:
        return self.callback.summary()
