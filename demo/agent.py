from __future__ import annotations

import random
from typing import Any

from entropyos.pipeline import EntropyPipeline


class MockLLM:
    """A mock LLM for testing — echoes context from a knowledge base."""

    def __init__(self, knowledge_base: dict[str, str]):
        self.kb = knowledge_base

    def invoke(self, prompt: str) -> str:
        for question, answer in self.kb.items():
            if question.lower() in prompt.lower():
                return answer
        return "I don't have information about that."

    def generate(self, prompt: str) -> str:
        return self.invoke(prompt)


KB: dict[str, str] = {
    "what is python": (
        "Python is a high-level, general-purpose programming language. "
        "Its design philosophy emphasizes code readability with the use of significant indentation. "
        "Python is dynamically typed and garbage-collected. "
        "It supports multiple programming paradigms, including structured, object-oriented, and functional programming."
    ),
    "what is the capital of france": (
        "The capital of France is Paris. "
        "Paris is located in the north-central part of France on the Seine River. "
        "It is one of the world's major centers of finance, diplomacy, commerce, culture, fashion, and gastronomy."
    ),
    "what is machine learning": (
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience "
        "without being explicitly programmed. It focuses on developing computer programs that can access data and use it "
        "to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, "
        "or instruction, in order to look for patterns in data and make better decisions in the future."
    ),
    "what is docker": (
        "Docker is a platform for developing, shipping, and running applications in containers. "
        "Containers allow a developer to package up an application with all of the parts it needs, "
        "such as libraries and other dependencies, and ship it all out as one package. "
        "Docker containers are lightweight, portable, and ensure consistency across development, staging, and production environments."
    ),
    "what is an api": (
        "An API (Application Programming Interface) is a set of defined rules that enable different applications to communicate with each other. "
        "It acts as an intermediary layer that processes data transfer between systems. "
        "APIs are used to integrate new capabilities into existing systems, enabling businesses to build on the functions of other applications."
    ),
}


class DemoAgent:
    """A simple RAG-style agent that answers questions from a knowledge base,
    optionally using EntropyOS to optimize prompts before looking up answers."""

    def __init__(self, use_entropy: bool = False):
        self.kb = KB
        self.llm = MockLLM(self.kb)
        self.use_entropy = use_entropy
        self.pipeline = EntropyPipeline() if use_entropy else None
        self.total_tokens_saved = 0
        self.total_calls = 0
        self.total_latency_ms = 0.0

    def ask(self, question: str) -> dict[str, Any]:
        import time
        self.total_calls += 1

        if self.use_entropy:
            t0 = time.perf_counter()
            result = self.pipeline.run(question)
            elapsed = (time.perf_counter() - t0) * 1000
            self.total_latency_ms += elapsed
            optimized = result["compiled_prompt"]
            original_chars = len(question)
            compressed_chars = len(optimized)

            answer = self.llm.invoke(optimized)

            saved = max(0, original_chars - compressed_chars)
            self.total_tokens_saved += saved

            return {
                "question": question,
                "answer": answer,
                "original_len": original_chars,
                "compressed_len": compressed_chars,
                "compression_ratio": result["optimization"]["compression_ratio"],
                "info_preserved": result["optimization"]["info_preserved"],
                "info_score": result["score"]["information_score"],
                "latency_ms": round(elapsed, 2),
            }
        else:
            answer = self.llm.invoke(question)
            return {
                "question": question,
                "answer": answer,
                "original_len": len(question),
                "compressed_len": len(question),
                "compression_ratio": 1.0,
                "info_preserved": 1.0,
                "info_score": 0.0,
                "latency_ms": 0.0,
            }

    def summary(self) -> dict[str, Any]:
        return {
            "mode": "entropyos" if self.use_entropy else "baseline",
            "total_calls": self.total_calls,
            "total_tokens_saved_chars": self.total_tokens_saved,
            "total_latency_ms": round(self.total_latency_ms, 2),
        }


def run_demo():
    print("=" * 60)
    print("EntropyOS Demo Agent")
    print("=" * 60)

    questions = [
        "What is Python? I think it might be a programming language but I am not sure. It is worth noting that I am a beginner.",
        "What is the capital of France? Basically I need to know this for my trip. The answer is probably important.",
        "What is machine learning? I believe it has something to do with AI and data. It should be noted that I am curious about this topic.",
    ]

    for use_entropy in [False, True]:
        label = "Baseline (no EntropyOS)" if not use_entropy else "With EntropyOS"
        agent = DemoAgent(use_entropy=use_entropy)
        results = []

        print(f"\n--- {label} ---")
        for q in questions:
            r = agent.ask(q)
            results.append(r)
            print(f"\n  Q: {r['question'][:60]}...")
            print(f"  A: {r['answer'][:80]}...")
            if use_entropy:
                print(f"  Compression: {1 - r['compression_ratio']:.1%} reduction")
                print(f"  Info preserved: {r['info_preserved']:.1%}")
                print(f"  Latency: {r['latency_ms']:.2f}ms")

        total_orig = sum(r["original_len"] for r in results)
        total_comp = sum(r["compressed_len"] for r in results)
        if total_orig > 0:
            reduction = (1 - total_comp / total_orig) * 100
            print(f"\n  Total: {reduction:.1f}% token reduction ({total_orig} → {total_comp} chars)")


if __name__ == "__main__":
    run_demo()
