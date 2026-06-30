from demo.agent import KB, DemoAgent, MockLLM


def test_knowledge_base_has_entries():
    assert len(KB) >= 3


def test_mock_llm_answers():
    llm = MockLLM(KB)
    answer = llm.invoke("What is python?")
    assert "programming" in answer.lower()


def test_mock_llm_unknown():
    llm = MockLLM(KB)
    answer = llm.invoke("something completely unknown")
    assert "don't have information" in answer


def test_demo_agent_baseline():
    agent = DemoAgent(use_entropy=False)
    result = agent.ask("What is Python?")
    assert result["compression_ratio"] == 1.0
    assert len(result["answer"]) > 0


def test_demo_agent_with_entropy():
    agent = DemoAgent(use_entropy=True)
    result = agent.ask("What is Python? I think it is a programming language.")
    assert result["compression_ratio"] <= 1.0
    assert result["info_preserved"] >= 0.5
    assert len(result["answer"]) > 0


def test_demo_agent_verbose_reduction():
    agent = DemoAgent(use_entropy=True)
    result = agent.ask(
        "What is the capital of France? I think it might be Paris but I am not sure. "
        "It should be noted that I am planning a trip there."
    )
    assert result["compression_ratio"] < 0.9


def test_demo_agent_summary():
    agent = DemoAgent(use_entropy=True)
    for _ in range(3):
        agent.ask("What is Python?")
    summary = agent.summary()
    assert summary["total_calls"] == 3
    assert summary["total_tokens_saved_chars"] >= 0
