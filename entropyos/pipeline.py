from __future__ import annotations

import time

from entropyos.compressor import SemanticCompressor
from entropyos.config import EntropyConfig
from entropyos.evaluator import ResponseEvaluator
from entropyos.memory import HierarchicalMemory
from entropyos.optimizer import ContextOptimizer
from entropyos.prompt_compiler import PromptCompiler
from entropyos.retrieval import RetrievalEngine
from entropyos.scorer import InformationScorer
from entropyos.security.audit import AuditLogger
from entropyos.security.encrypt import MemoryEncryptor
from entropyos.security.pii import PIIDetector


class EntropyPipeline:
    def __init__(self, config: EntropyConfig | None = None):
        self.config = config or EntropyConfig()
        self.audit = AuditLogger(path=self.config.security.audit_log_path)
        self.encryptor = MemoryEncryptor(key=self.config.security.encryption_key)
        self.pii = PIIDetector()

        self.scorer = InformationScorer(config=self.config)
        self.compressor = SemanticCompressor(level=self.config.compression_level, config=self.config)
        self.memory = HierarchicalMemory(
            config=self.config.memory,
            encryptor=self.encryptor,
            audit=self.audit,
            pii=self.pii,
        )
        self.retrieval = RetrievalEngine(self.memory, config=self.config.retrieval)
        self.optimizer = ContextOptimizer(self.scorer, self.compressor)
        self.compiler = PromptCompiler(self.scorer)
        self.evaluator = ResponseEvaluator()

    def run(
        self,
        prompt: str,
        user_id: str = "default",
        session_id: str = "default",
        documents: list | None = None,
        agent_state: dict | None = None,
        use_llm: bool = False,
    ) -> dict:
        timeline: dict[str, float] = {}
        t0 = time.perf_counter()

        if self.pii.has_pii(prompt):
            prompt = self.pii.redact(prompt)

        score_result = self.scorer.score_llm(prompt) if use_llm else self.scorer.score(prompt)
        timeline["score"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        retrieved, info_gain = self.retrieval.retrieve(
            query=prompt,
            top_k=self.config.retrieval.top_k,
            min_score=self.config.recall_threshold,
        )
        timeline["retrieve"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        assembled = self.compiler.compile(
            user_prompt=prompt,
            memories=retrieved,
            documents=documents or [],
            agent_state=agent_state,
        )
        timeline["assemble"] = time.perf_counter() - t2

        t3 = time.perf_counter()
        optimized, opt_report = self.optimizer.optimize(assembled, [], [])
        timeline["optimize"] = time.perf_counter() - t3

        total_time = time.perf_counter() - t0

        pipeline_result: dict = {
            "compiled_prompt": optimized,
            "score": {
                "entropy": round(score_result.entropy, 4),
                "novelty": round(score_result.novelty, 4),
                "redundancy": round(score_result.redundancy, 4),
                "importance": round(score_result.importance, 4),
                "dependency": round(score_result.dependency, 4),
                "information_score": round(score_result.information_score, 4),
                "method": score_result.method,
            },
            "optimization": {
                "deduped_lines": opt_report.get("deduped_lines", 0),
                "compression_ratio": round(opt_report.get("compression_ratio", 1.0), 4),
                "conflicts_found": opt_report.get("conflicts_found", 0),
                "info_preserved": round(opt_report.get("info_preserved", 1.0), 4),
            },
            "retrieved_memories": [
                {
                    "content": m.content[:100],
                    "level": m.level.name,
                    "value": round(m.value.combined_value, 4),
                }
                for m in retrieved
            ],
            "information_gain": round(info_gain, 4),
            "timing": {k: round(v, 4) for k, v in timeline.items()},
            "total_time_ms": round(total_time * 1000, 1),
        }

        t4 = time.perf_counter()
        eval_result = self.evaluator.evaluate(assembled, optimized, optimized)
        timeline["evaluate"] = time.perf_counter() - t4

        pipeline_result["evaluation"] = {
            "accuracy": round(eval_result.accuracy, 4),
            "completeness": round(eval_result.completeness, 4),
            "hallucination_risk": round(eval_result.hallucination_score, 4),
            "compression_effectiveness": round(eval_result.compression_effectiveness, 4),
            "information_retained": round(eval_result.information_retained, 4),
        }
        pipeline_result["total_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        self.memory.insert(
            content=prompt,
            metadata={
                "user_id": user_id,
                "session_id": session_id,
                "info_score": score_result.information_score,
            },
        )
        self.memory.tick()

        self.audit.log(
            "pipeline_run",
            user_id=user_id,
            resource=user_id,
            detail=f"score={score_result.information_score:.3f} time={pipeline_result['total_time_ms']:.1f}ms",
        )

        return pipeline_result
