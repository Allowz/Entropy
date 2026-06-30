from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from entropyos.compressor import SemanticCompressor
from entropyos.config import EntropyConfig
from entropyos.evaluator import ResponseEvaluator
from entropyos.pipeline import EntropyPipeline
from entropyos.scorer import InformationScorer

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    raise ImportError("Install with `pip install entropyos[api]` to use the REST API")


def create_app(config: EntropyConfig | None = None) -> FastAPI:
    cfg = config or _load_config()
    pipeline = EntropyPipeline(config=cfg)
    scorer = InformationScorer(config=cfg)
    compressor = SemanticCompressor(config=cfg)
    evaluator = ResponseEvaluator()

    app = FastAPI(title="EntropyOS", version="0.9.0")
    app.state.pipeline = pipeline
    app.state.scorer = scorer
    app.state.compressor = compressor
    app.state.evaluator = evaluator

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class ScoreRequest(BaseModel):
        text: str
        context: str | None = None
        use_llm: bool = False

    class CompressRequest(BaseModel):
        text: str
        level: float | None = None
        use_llm: bool = False

    class RetrieveRequest(BaseModel):
        query: str
        top_k: int = 5

    class OptimizeRequest(BaseModel):
        prompt: str
        documents: list[str] = []

    class EvaluateRequest(BaseModel):
        original_prompt: str
        compressed_prompt: str | None = None
        response: str

    class PipelineRequest(BaseModel):
        prompt: str
        user_id: str = "default"
        session_id: str = "default"
        documents: list[str] | None = None
        agent_state: dict[str, Any] | None = None
        use_llm: bool = False

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "0.9.0"}

    @app.post("/score")
    def score_endpoint(req: ScoreRequest):
        try:
            if req.use_llm:
                result = app.state.scorer.score_llm(req.text, req.context)
            else:
                result = app.state.scorer.score(req.text, req.context)
            return {
                "entropy": result.entropy,
                "information_score": result.information_score,
                "redundancy": result.redundancy,
                "importance": result.importance,
                "novelty": result.novelty,
                "dependency": result.dependency,
                "method": result.method,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/compress")
    def compress_endpoint(req: CompressRequest):
        try:
            c = app.state.compressor
            if req.level is not None:
                c = SemanticCompressor(level=req.level, config=cfg)
            if req.use_llm:
                result = c.compress_llm(req.text)
            else:
                result = c.compress(req.text)
            return {
                "compressed": result.compressed,
                "compression_ratio": result.compression_ratio,
                "info_preserved": result.info_preserved,
                "method": result.method,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/retrieve")
    def retrieve_endpoint(req: RetrieveRequest):
        try:
            memories, info_gain = app.state.pipeline.retrieval.retrieve(
                req.query, top_k=req.top_k,
            )
            return {
                "memories": [
                    {
                        "content": m.content[:200],
                        "level": m.level.name,
                        "value": round(m.value.combined_value, 4),
                    }
                    for m in memories
                ],
                "information_gain": round(info_gain, 4),
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/optimize")
    def optimize_endpoint(req: OptimizeRequest):
        try:
            optimized, report = app.state.pipeline.optimizer.optimize(
                req.prompt, [], req.documents,
            )
            return {"optimized_prompt": optimized, "report": report}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/evaluate")
    def evaluate_endpoint(req: EvaluateRequest):
        try:
            result = app.state.evaluator.evaluate(
                req.original_prompt, req.compressed_prompt, req.response,
            )
            return {
                "accuracy": result.accuracy,
                "completeness": result.completeness,
                "hallucination_risk": result.hallucination_score,
                "compression_effectiveness": result.compression_effectiveness,
                "information_retained": result.information_retained,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/pipeline")
    def pipeline_endpoint(req: PipelineRequest, request: Request):
        try:
            result = app.state.pipeline.run(
                prompt=req.prompt,
                user_id=req.user_id,
                session_id=req.session_id,
                documents=req.documents,
                agent_state=req.agent_state,
                use_llm=req.use_llm,
            )
            result["client_ip"] = request.client.host if request.client else ""
            return result
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/memory/clear")
    def clear_memory():
        app.state.pipeline.memory.clear()
        return {"status": "cleared"}

    @app.get("/memory/stats")
    def memory_stats():
        mems = app.state.pipeline.memory.all_memories()
        from entropyos.models import MemoryLevel
        counts = {l.name: 0 for l in MemoryLevel}
        for m in mems:
            counts[m.level.name] = counts.get(m.level.name, 0) + 1
        return {"total": len(mems), "by_level": counts}

    return app


def _load_config() -> EntropyConfig:
    paths = [
        Path(os.environ.get("ENTROPY_CONFIG", "")),
        Path("entropyos.yaml"),
        Path("entropyos.yml"),
        Path("entropyos.json"),
        Path.home() / ".entropyos" / "config.yaml",
    ]
    for p in paths:
        if p and p.exists():
            return EntropyConfig.from_file(p)
    return EntropyConfig.from_env()


app = create_app()
