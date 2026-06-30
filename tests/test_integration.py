from entropyos.pipeline import EntropyPipeline
from entropyos.memory import HierarchicalMemory
from entropyos.retrieval import RetrievalEngine
from entropyos.compressor import SemanticCompressor
from entropyos.scorer import InformationScorer
from entropyos.evaluator import ResponseEvaluator


def test_pipeline_end_to_end():
    pipe = EntropyPipeline()
    for i in range(3):
        pipe.memory.insert(f"Memory item number {i} about France.")
    pipe.memory.tick()

    result = pipe.run(
        prompt="Tell me about France and its capital.",
        documents=["France is in Europe.", "Paris is the capital."],
        user_id="integration_test",
        session_id="e2e_test",
    )

    assert result["score"]["information_score"] >= 0
    assert result["optimization"]["compression_ratio"] <= 1.0
    assert "evaluation" in result

    memories = pipe.memory.all_memories()
    assert len(memories) >= 3


def test_memory_retrieval_with_context():
    mem = HierarchicalMemory()
    mem.insert("Einstein developed the theory of relativity.")
    mem.insert("Newton formulated the laws of motion.")
    mem.insert("Paris is the capital of France.")

    eng = RetrievalEngine(mem)
    results, gain = eng.retrieve("physics theories", top_k=2)
    assert len(results) >= 1
    assert gain >= 0


def test_compression_vs_preservation():
    c = SemanticCompressor(level=0.7)
    text = (
        "I think that the API endpoint at https://api.example.com/v1/users "
        "returns the user data. The function get_user(id) must be called with "
        "a valid integer. The date 2025-01-15 is the deadline. John Smith is the admin."
    )
    r = c.compress(text)
    assert "https://api.example.com/v1/users" in r.compressed
    assert "get_user" in r.compressed
    assert "2025-01-15" in r.compressed
    assert "John Smith" in r.compressed
    assert r.compression_ratio < 0.98


def test_evaluator_accuracy_tracking():
    e = ResponseEvaluator()
    r1 = e.evaluate("What is 2+2?", "What is 2+2?", "4")
    assert r1.accuracy > 0.8

    r2 = e.evaluate("What is 2+2?", None, "I'm not sure, maybe 5?")
    assert r2.completeness >= 0


def test_scorer_information_tradeoff():
    s = InformationScorer()
    dense = s.score("The API key sk-abc123DEF456 must connect to https://example.com by 2025-01-01.")
    sparse = s.score("um uh well i think maybe it could be something like um maybe not sure")
    assert dense.information_score > sparse.information_score


def test_memory_value_function():
    from entropyos.models import Memory, MemoryValue
    mem = Memory(content="test", access_count=10)
    mem.created_at -= 3600
    mem.value.dependency = 0.8
    mem.value.compression_ratio = 0.5
    hm = HierarchicalMemory()
    val = hm.compute_value(mem)
    assert val.combined_value > 0


def test_pipeline_multiple_runs():
    pipe = EntropyPipeline()
    queries = ["What is Python?", "Tell me about France.", "How does gravity work?"]
    for q in queries:
        result = pipe.run(q)
        assert result["total_time_ms"] >= 0
        assert result["score"]["information_score"] >= 0


def test_retrieval_mmr():
    mem = HierarchicalMemory()
    mem.insert("Python is a programming language.")
    mem.insert("Python is used for data science.")
    mem.insert("Python was created by Guido van Rossum.")
    mem.insert("Java is also a programming language.")
    eng = RetrievalEngine(mem)
    results, _ = eng.retrieve("Tell me about Python", top_k=3)
    contents = [m.content for m in results]
    python_count = sum(1 for c in contents if "Python" in c)
    assert python_count >= 2
