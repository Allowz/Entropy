from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cli() -> None:
    parser = argparse.ArgumentParser(prog="entropyos", description="EntropyOS CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="Score information density")
    p_score.add_argument("text", help="Text to score")
    p_score.add_argument("--context", help="Optional context for novelty")

    p_compress = sub.add_parser("compress", help="Compress text")
    p_compress.add_argument("text", help="Text to compress")
    p_compress.add_argument("--level", type=float, default=0.5, help="Compression level (0-1)")

    p_pipeline = sub.add_parser("pipeline", help="Run the full pipeline")
    p_pipeline.add_argument("prompt", help="User prompt")
    p_pipeline.add_argument("--user", default="default")
    p_pipeline.add_argument("--session", default="default")
    p_pipeline.add_argument("--doc", action="append", default=[], help="Reference documents")
    p_pipeline.add_argument("--state", help="Agent state JSON")
    p_pipeline.add_argument("--server", help="Remote server URL")
    p_pipeline.add_argument("--api-key", help="API key for remote server")

    p_serve = sub.add_parser("serve", help="Start the API server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--config", help="Path to config file")

    p_config = sub.add_parser("init", help="Generate a default config file")
    p_config.add_argument("--path", default="entropyos.yaml")

    args = parser.parse_args()

    if args.command == "init":
        _cmd_init(args.path)
    elif args.command == "serve":
        _cmd_serve(args)
    elif args.command == "score":
        _cmd_score(args)
    elif args.command == "compress":
        _cmd_compress(args)
    elif args.command == "pipeline":
        _cmd_pipeline(args)


def _cmd_init(path: str) -> None:
    config = {
        "max_token_budget": 4096,
        "compression_level": 0.5,
        "log_level": "INFO",
        "memory": {"backend": "local"},
        "retrieval": {"engine": "ngram", "top_k": 5},
        "llm": {"provider": "openai", "model": "gpt-4o-mini", "api_key": ""},
        "security": {"pii_detection": True, "audit_log": True},
        "plugins": {"enabled": True, "scan_paths": []},
    }
    with open(path, "w") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config written to {path}")


def _cmd_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError:
        print("Install with: pip install entropyos[api]", file=sys.stderr)
        sys.exit(1)
    from entropyos.api import app
    uvicorn.run(app, host=args.host, port=args.port)


def _cmd_score(args: argparse.Namespace) -> None:
    from entropyos.scorer import InformationScorer
    scorer = InformationScorer()
    result = scorer.score(args.text, args.context)
    print(json.dumps({
        "entropy": round(result.entropy, 4),
        "novelty": round(result.novelty, 4),
        "redundancy": round(result.redundancy, 4),
        "importance": round(result.importance, 4),
        "dependency": round(result.dependency, 4),
        "information_score": round(result.information_score, 4),
    }, indent=2))


def _cmd_compress(args: argparse.Namespace) -> None:
    from entropyos.compressor import SemanticCompressor
    compressor = SemanticCompressor(level=args.level)
    result = compressor.compress(args.text)
    print(json.dumps({
        "compressed": result.compressed,
        "compression_ratio": round(result.compression_ratio, 4),
        "info_preserved": round(result.info_preserved, 4),
    }, indent=2))


def _cmd_pipeline(args: argparse.Namespace) -> None:
    if args.server:
        from entropyos.sdk import EntropyClient
        client = EntropyClient(base_url=args.server, api_key=args.api_key or "")
        agent_state = json.loads(args.state) if args.state else None
        result = client.pipeline(
            prompt=args.prompt,
            user_id=args.user,
            session_id=args.session,
            documents=args.doc,
            agent_state=agent_state,
        )
    else:
        from entropyos.pipeline import EntropyPipeline
        pipe = EntropyPipeline()
        agent_state = json.loads(args.state) if args.state else None
        result = pipe.run(
            prompt=args.prompt,
            user_id=args.user,
            session_id=args.session,
            documents=args.doc,
            agent_state=agent_state,
        )
    print(json.dumps(_serialize(result), indent=2, default=str))


def _serialize(obj):
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 4)
    return obj


if __name__ == "__main__":
    cli()
