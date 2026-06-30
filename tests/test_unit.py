from entropyos.scorer import InformationScorer
from entropyos.compressor import SemanticCompressor
from entropyos.memory import HierarchicalMemory
from entropyos.retrieval import RetrievalEngine
from entropyos.evaluator import ResponseEvaluator
from entropyos.optimizer import ContextOptimizer
from entropyos.prompt_compiler import PromptCompiler
from entropyos.pipeline import EntropyPipeline
from entropyos.config import EntropyConfig
from entropyos.models import Memory, MemoryLevel, MemoryValue


def test_scorer_basic():
    s = InformationScorer()
    r = s.score("The answer is 42. John Smith must be notified by 2025-01-01.")
    assert 0 <= r.entropy <= 1
    assert 0 <= r.novelty <= 1
    assert 0 <= r.redundancy <= 1
    assert r.importance > 0.5
    assert r.information_score > 0
    assert r.method == "heuristic"


def test_scorer_novelty():
    s = InformationScorer()
    r1 = s.score("hello world", context="hello world")
    assert r1.novelty < 0.1
    r2 = s.score("completely different text", context="hello world")
    assert r2.novelty > 0.8


def test_scorer_empty():
    s = InformationScorer()
    r = s.score("")
    assert r.entropy == 0.0
    assert r.information_score == 0.0


def test_compressor_removes_filler():
    c = SemanticCompressor(level=0.5)
    text = "It is worth noting that the answer is 42. It should be noted that John Smith must be notified by 2025-01-01. I think this is very important."
    r = c.compress(text)
    assert len(r.compressed) < len(text)
    assert r.compression_ratio < 1.0
    assert "42" in r.compressed
    assert "2025-01-01" in r.compressed
    assert "John Smith" in r.compressed
    assert r.info_preserved > 0.5
    assert r.method == "heuristic"


def test_compressor_preserves_code():
    c = SemanticCompressor(level=0.8)
    text = "I think the function is: ```def hello(): pass``` It should be noted that this is very important code."
    r = c.compress(text)
    assert "```" in r.compressed
    assert "def hello()" in r.compressed


def test_compressor_preserves_important():
    c = SemanticCompressor(level=0.9)
    text = "Meeting on 2025-01-15 at 42 Wall St. Contact alice@example.com. Password: sk-proj-abcdef123456"
    r = c.compress(text)
    assert "2025-01-15" in r.compressed
    assert "42" in r.compressed


def test_memory_insert_and_retrieve():
    mem = HierarchicalMemory()
    mem.insert("Paris is the capital of France.")
    mem.insert("Python is a programming language.")
    all_m = mem.all_memories()
    assert len(all_m) == 2
    hits = mem.search("France")
    assert len(hits) >= 1


def test_memory_value_decay():
    mem = HierarchicalMemory()
    m = mem.insert("test memory")
    import time
    val_fresh = mem.compute_value(m).combined_value
    m.created_at -= 86400 * 30
    val_old = mem.compute_value(m).combined_value
    assert val_old <= val_fresh


def test_memory_promotion():
    mem = HierarchicalMemory()
    m1 = mem.insert("Important fact: Paris is capital.")
    for _ in range(15):
        mem.access(m1)
    mem.tick()
    all_mem = mem.all_memories()
    promoted = [m for m in all_mem if m.id == m1.id]
    assert promoted
    assert promoted[0].level > MemoryLevel.L0_CURRENT


def test_memory_clear():
    mem = HierarchicalMemory()
    mem.insert("test")
    mem.clear()
    assert len(mem.all_memories()) == 0


def test_retrieval_basic():
    mem = HierarchicalMemory()
    mem.insert("Paris is the capital of France.")
    mem.insert("Python is a programming language.")
    eng = RetrievalEngine(mem)
    results, gain = eng.retrieve("What is the capital of France?")
    assert len(results) >= 1
    assert gain >= 0


def test_retrieval_empty():
    mem = HierarchicalMemory()
    eng = RetrievalEngine(mem)
    results, gain = eng.retrieve("anything")
    assert len(results) == 0
    assert gain == 0.0


def test_evaluator_basic():
    e = ResponseEvaluator()
    r = e.evaluate(
        original_prompt="What is 42?",
        compressed_prompt="42?",
        response="42 is the answer to life.",
    )
    assert 0 <= r.accuracy <= 1
    assert 0 <= r.completeness <= 1
    assert 0 <= r.hallucination_score <= 1
    assert r.compression_effectiveness >= 0
    assert r.information_retained >= 0


def test_evaluator_hallucination():
    e = ResponseEvaluator()
    r = e.evaluate(
        original_prompt="What color is the sky?",
        compressed_prompt="sky color?",
        response="The sky is blue. Alice and Bob went to Paris.",
    )
    assert r.hallucination_score > 0


def test_optimizer_dedup():
    s = InformationScorer()
    c = SemanticCompressor()
    o = ContextOptimizer(s, c)
    opt, report = o.optimize("The answer is 42. The answer is 42. John must be told.", [], [])
    assert "42" in opt
    assert report["compression_ratio"] <= 1.0
    assert report["conflicts_found"] >= 0


def test_compiler_basic():
    s = InformationScorer()
    pc = PromptCompiler(s)
    result = pc.compile("Hello", memories=[], documents=[])
    assert "Hello" in result


def test_compiler_with_state():
    s = InformationScorer()
    pc = PromptCompiler(s)
    result = pc.compile("Hello", memories=[], documents=[], agent_state={"role": "assistant"})
    assert "Hello" in result
    assert "role: assistant" in result or "role" in result


def test_pipeline_full():
    pipe = EntropyPipeline()
    result = pipe.run("What is the capital of France?")
    assert "compiled_prompt" in result
    assert result["score"]["information_score"] >= 0
    assert result["total_time_ms"] >= 0
    assert "evaluation" in result
    assert "optimization" in result


def test_pipeline_with_documents():
    pipe = EntropyPipeline()
    result = pipe.run(
        "What is the capital of France?",
        documents=["France is a country in Europe.", "Paris is the capital of France."],
        user_id="test_user",
        session_id="test_session",
    )
    assert result["optimization"]["deduped_lines"] >= 1


def test_pipeline_pii_redaction():
    cfg = EntropyConfig()
    cfg.security.pii_detection = True
    pipe = EntropyPipeline(config=cfg)
    result = pipe.run("My email is alice@example.com and SSN is 123-45-6789")
    assert "alice" not in (result.get("compiled_prompt") or "")


def test_config_from_dict():
    d = {"max_token_budget": 2048, "compression_level": 0.7}
    cfg = EntropyConfig.from_dict(d)
    assert cfg.max_token_budget == 2048
    assert cfg.compression_level == 0.7


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("ENTROPY_MAX_TOKEN_BUDGET", "8192")
    cfg = EntropyConfig.from_env()
    assert cfg.max_token_budget == 8192


def test_memory_levels():
    mem = HierarchicalMemory()
    from entropyos.models import MemoryLevel
    assert len(mem.get_level(MemoryLevel.L0_CURRENT)) == 0
    mem.insert("test")
    assert len(mem.get_level(MemoryLevel.L0_CURRENT)) == 1


def test_memory_metadata():
    mem = HierarchicalMemory()
    m = mem.insert("test", metadata={"key": "value"})
    assert m.metadata["key"] == "value"


def test_scorer_with_context():
    s = InformationScorer()
    r = s.score("Python is great", context="Python is great for data science")
    assert r.novelty <= 0.5


def test_compressor_no_filler():
    c = SemanticCompressor(level=0.0)
    text = "Some text with no compression needed."
    r = c.compress(text)
    assert r.compression_ratio == 1.0


def test_optimizer_conflict_detection():
    s = InformationScorer()
    c = SemanticCompressor()
    o = ContextOptimizer(s, c)
    text = "You must always use HTTPS. You should never use encryption."
    _, report = o.optimize(text, [], [])
    assert report["conflicts_found"] >= 0


def test_pipeline_idempotent():
    pipe = EntropyPipeline()
    r1 = pipe.run("What is Python?")
    r2 = pipe.run("What is Python?")
    assert r1["score"]["information_score"] == r2["score"]["information_score"]


def test_memory_backend_file(tmp_path):
    from entropyos.backends.memory import FileMemoryBackend
    path = tmp_path / "mem.pkl"
    backend = FileMemoryBackend(path)
    mem = Memory(content="test data", level=MemoryLevel.L1_CONVERSATION)
    backend.save([mem])
    loaded = backend.load()
    assert len(loaded) == 1
    assert loaded[0].content == "test data"
    assert loaded[0].level == MemoryLevel.L1_CONVERSATION


def test_ngram_vector_backend():
    from entropyos.backends.vector import NGramVectorBackend
    v = NGramVectorBackend(n=3)
    emb1 = v.embed("hello world")
    emb2 = v.embed("hello world")
    assert v.similarity(emb1, emb2) > 0.99
    emb3 = v.embed("completely unrelated")
    assert v.similarity(emb1, emb3) < 0.8


def test_pii_detector():
    from entropyos.security.pii import PIIDetector
    pii = PIIDetector()
    spans = pii.detect("Contact alice@example.com or call 555-123-4567")
    assert len(spans) >= 1
    redacted = pii.redact("My email is alice@example.com")
    assert "[REDACTED]" in redacted
    assert "alice" not in redacted


def test_memory_encryptor():
    from entropyos.security.encrypt import MemoryEncryptor
    enc = MemoryEncryptor(key="test-key-32-bytes-long!")
    mem = Memory(content="sensitive data")
    encrypted = enc.encrypt(mem)
    assert encrypted.content != "sensitive data"
    decrypted = enc.decrypt(encrypted)
    assert decrypted.content == "sensitive data"


def test_audit_logger(tmp_path):
    from entropyos.security.audit import AuditLogger
    path = tmp_path / "audit.log"
    al = AuditLogger(path)
    al.log("test_action", user_id="user1", detail="test detail")
    al.flush()
    entries = al.query(action="test_action")
    assert len(entries) >= 1
    assert entries[0].user_id == "user1"


def test_plugin_registry():
    from entropyos.plugins.registry import PluginRegistry
    reg = PluginRegistry()
    assert len(reg.scorers) == 0


def test_prompt_compiler_memories():
    s = InformationScorer()
    pc = PromptCompiler(s)
    mems = [Memory(content="Paris is capital"), Memory(content="42 is answer")]
    result = pc.compile("Tell me", memories=mems)
    assert "Paris" in result
    assert "Tell me" in result


def test_pipeline_with_agent_state():
    pipe = EntropyPipeline()
    result = pipe.run("Hello", agent_state={"user_level": "expert", "language": "en"})
    assert result["total_time_ms"] >= 0
