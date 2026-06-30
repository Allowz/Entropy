import time

from entropyos.compressor import SemanticCompressor
from entropyos.memory import HierarchicalMemory
from entropyos.pipeline import EntropyPipeline
from entropyos.scorer import InformationScorer


def test_benchmark_scorer_speed():
    s = InformationScorer()
    texts = [
        "Short text.",
        "Medium " * 50 + "text.",
        ("Long " * 500) + "text.",
    ]
    for text in texts:
        t0 = time.perf_counter()
        for _ in range(100):
            s.score(text)
        elapsed = time.perf_counter() - t0
        ops = 100 / elapsed
        assert ops > 100


def test_benchmark_compressor_speed():
    c = SemanticCompressor(level=0.5)
    text = ("The quick brown fox jumps over the lazy dog. " * 100) + "The answer is 42."
    t0 = time.perf_counter()
    for _ in range(50):
        c.compress(text)
    elapsed = time.perf_counter() - t0
    ops = 50 / elapsed
    assert ops > 10


def test_benchmark_pipeline_latency():
    pipe = EntropyPipeline()
    t0 = time.perf_counter()
    for _ in range(20):
        pipe.run("What is the capital of France?")
    elapsed = time.perf_counter() - t0
    avg_ms = (elapsed / 20) * 1000
    assert avg_ms < 100


def test_benchmark_memory_scale():
    mem = HierarchicalMemory()
    t0 = time.perf_counter()
    for i in range(1000):
        mem.insert(f"Memory item number {i} with some additional content for realism.")
    insert_time = time.perf_counter() - t0
    assert insert_time < 5.0

    t0 = time.perf_counter()
    for _ in range(100):
        mem.search("number")
    search_time = time.perf_counter() - t0
    assert search_time < 2.0


def test_benchmark_compression_ratio():
    c = SemanticCompressor(level=0.5)
    verbose = (
        "It is worth noting that the answer is 42. "
        "I think it is very important to remember that John Smith must be notified. "
        "It should be noted that the deadline is 2025-01-01. "
        "Basically, the API key sk-abc must connect to https://example.com. "
    )
    r = c.compress(verbose)
    assert r.compression_ratio < 0.9
    assert r.info_preserved > 0.8


def test_benchmark_token_reduction():
    c = SemanticCompressor(level=0.7)
    original = "I think that the answer is probably 42. It is very important to note that this is basically the correct answer."
    r = c.compress(original)
    reduction = 1.0 - r.compression_ratio
    assert reduction > 0.1
