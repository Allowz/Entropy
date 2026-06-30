from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from entropyos.compressor import SemanticCompressor
from entropyos.pipeline import EntropyPipeline


@dataclass
class BenchmarkResult:
    scenario: str = ""
    baseline_chars: int = 0
    entropyos_chars: int = 0
    avg_info_preserved: float = 0.0
    avg_latency_ms: float = 0.0

    @property
    def token_reduction_pct(self) -> float:
        return (1 - self.entropyos_chars / self.baseline_chars) * 100 if self.baseline_chars else 0.0

    @property
    def cost_savings_pct(self) -> float:
        return self.token_reduction_pct


def estimate_cost(chars: int, model: str = "gpt-4o-mini") -> float:
    tokens = chars / 4
    if "mini" in model:
        return tokens * 0.15 / 1_000_000
    return tokens * 2.50 / 1_000_000


def build_full_context(
    system_prompt: str,
    conversation_turns: list[tuple[str, str]],
    docs: list[str],
    query: str,
) -> str:
    parts = [
        "System: " + system_prompt,
    ]
    if conversation_turns:
        parts.append("\nConversation:")
        for u, a in conversation_turns:
            parts.append(f"User: {u}\nAssistant: {a}")
    if docs:
        parts.append("\nDocuments:")
        for d in docs:
            parts.append(f"- {d}")
    parts.append(f"\nUser: {query}")
    return "\n".join(parts)


SCENARIOS = {
    "simple_chat": {
        "system": "You are a helpful assistant.",
        "conversation": [
            ("What is Python?", "Python is a programming language."),
        ],
        "docs": [],
        "query": "What is Python used for?",
    },
    "multi_turn": {
        "system": (
            "You are a helpful programming assistant. "
            "You must use the provided context to answer accurately. "
            "You must cite sources when available. "
            "You must be concise and clear."
        ),
        "conversation": [
            ("What is Python?", "Python is a high-level programming language."),
            ("Who created it?", "Guido van Rossum created Python in 1991."),
            ("What are the features?", "Python has dynamic typing, garbage collection, OOP."),
            ("Is it good for beginners?", "Yes, Python is great for beginners."),
        ],
        "docs": [
            "Python is a high-level programming language created by Guido van Rossum.",
            "Python supports OOP, functional, and procedural programming.",
            "Python is dynamically typed and garbage-collected.",
            "Python is widely used in web development, data science, and AI.",
            "Python has a large standard library and active community.",
        ],
        "query": "What is Python and who created it?",
    },
    "rag_verbose": {
        "system": (
            "You are a research assistant that helps answer questions. "
            "You must use the provided context to answer accurately. "
            "You must cite sources when available. "
            "You must be concise and clear. "
            "You must use the provided context to answer accurately. "
            "You must be concise and clear."
        ),
        "conversation": [
            ("What is machine learning?", "ML is a subset of AI that learns from data."),
            ("What is deep learning?", "Deep learning uses neural networks with many layers."),
            ("What is the difference?", "Deep learning is a subset of machine learning."),
        ],
        "docs": [
            "Machine learning is a subset of artificial intelligence.",
            "Machine learning enables systems to learn from experience without explicit programming.",
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Deep learning is a subset of machine learning.",
            "Both ML and DL are used in data science, computer vision, and NLP.",
        ],
        "query": (
            "What is machine learning and how is it different from deep learning? "
            "I think it might be about AI but I am not sure. "
            "Basically I need to understand the difference between ML and DL. "
            "It should be noted that I am a beginner who is just starting to learn."
        ),
    },
    "code_review": {
        "system": (
            "You are a code review assistant. "
            "You must check for bugs, security issues, and style problems. "
            "You must provide specific line numbers for issues. "
            "You must suggest fixes for each problem found. "
            "You must check for bugs, security issues, and style problems."
        ),
        "conversation": [
            ("Review this function:", "I'll review the code you provide."),
        ],
        "docs": [],
        "query": (
            "Review this Python function:\n"
            "```python\ndef factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n-1)\n"
            "```\n"
            "I think this function looks correct but I am not sure about edge cases. "
            "It should be noted that n should be a positive integer."
        ),
    },
    "long_context": {
        "system": (
            "You are a helpful assistant. " * 5
        ),
        "conversation": [
            ("Hello", "Hi there! How can I help?") for _ in range(3)
        ],
        "docs": [
            "The sky is blue because of Rayleigh scattering." for _ in range(5)
        ],
        "query": "What causes the sky to appear blue? I think it has something to do with light scattering.",
    },
}


def run_benchmark(scenario_name: str, cfg: dict) -> BenchmarkResult:
    result = BenchmarkResult(scenario=scenario_name)
    pipeline = EntropyPipeline()
    compressor = SemanticCompressor(level=0.5)

    full_context = build_full_context(
        system_prompt=cfg["system"],
        conversation_turns=cfg.get("conversation", []),
        docs=cfg.get("docs", []),
        query=cfg["query"],
    )

    result.baseline_chars = len(full_context)

    pr = pipeline.run(full_context)
    result.entropyos_chars = len(pr["compiled_prompt"])
    result.avg_info_preserved = pr["optimization"]["info_preserved"]
    result.avg_latency_ms = pr["total_time_ms"]

    return result


def print_report(results: list[BenchmarkResult]) -> None:
    print("=" * 80)
    print("EntropyOS Benchmark — Real-World Token Reduction")
    print("=" * 80)

    total_baseline = 0
    total_entropy = 0
    total_latency = 0.0

    print(f"\n{'Scenario':<20} {'Before':>8} {'After':>8} {'Reduction':>10} {'Saved/Yr':>10} {'Info%':>8} {'Latency':>9}")
    print("-" * 80)

    for r in results:
        saved = total_baseline - total_entropy
        annual_savings_dollars = estimate_cost(saved) * 100_000
        print(
            f"{r.scenario:<20} "
            f"{r.baseline_chars:>8} "
            f"{r.entropyos_chars:>8} "
            f"{r.token_reduction_pct:>9.1f}% "
            f"${annual_savings_dollars:>7.2f} "
            f"{r.avg_info_preserved:>7.1%} "
            f"{r.avg_latency_ms:>8.2f}ms"
        )
        total_baseline += r.baseline_chars
        total_entropy += r.entropyos_chars
        total_latency += r.avg_latency_ms

    total_saved = total_baseline - total_entropy
    overall_reduction = (1 - total_entropy / total_baseline) * 100 if total_baseline else 0
    annual_cost = estimate_cost(total_saved) * 100_000

    print("=" * 80)
    print(f"\nTotals ({len(results)} scenarios):")
    print(f"  Baseline chars:     {total_baseline:,}")
    print(f"  EntropyOS chars:    {total_entropy:,}")
    print(f"  Chars saved/call:   {total_saved:,}")
    print(f"  Overall reduction:  {overall_reduction:.1f}%")
    print(f"  Annual savings*:    ${annual_cost:.2f}")
    print(f"  Avg latency:        {total_latency/len(results):.2f}ms/call")
    print(f"  (*) 100K calls/yr @ GPT-4o-mini")


def main():
    results: list[BenchmarkResult] = []
    for name, cfg in SCENARIOS.items():
        r = run_benchmark(name, cfg)
        results.append(r)
    print_report(results)


if __name__ == "__main__":
    main()
