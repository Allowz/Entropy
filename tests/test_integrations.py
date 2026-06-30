from entropyos.integrations.langchain import EntropyCallback, EntropyLangchain


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(self):
        self.callbacks = []

    def invoke(self, prompt: str, **kwargs):
        return FakeMessage(content=f"Answer for: {prompt[:50]}")


def test_callback_intercepts_prompts():
    callback = EntropyCallback()
    prompts = ["What is Python? I think it is a language."]
    callback.on_llm_start({}, prompts)
    assert len(callback.metrics) == 1
    assert callback.metrics[0]["compression_ratio"] <= 1.0
    assert callback.metrics[0]["original_len"] > 0


def test_callback_summary():
    callback = EntropyCallback()
    callback.on_llm_start({}, ["Hello world. I think this is a test."])
    summary = callback.summary()
    assert summary["calls"] == 1
    assert summary["total_original_chars"] > 0
    assert summary["total_tokens_saved_pct"] >= 0


def test_callback_empty():
    callback = EntropyCallback()
    summary = callback.summary()
    assert summary["calls"] == 0


def test_callback_multiple_calls():
    callback = EntropyCallback()
    for i in range(5):
        callback.on_llm_start({}, [f"Test prompt number {i}. I think it is worth noting that this is filler."])
    assert len(callback.metrics) == 5
    summary = callback.summary()
    assert summary["calls"] == 5
    assert summary["avg_compression_ratio"] <= 1.0


def test_langchain_wrapper():
    llm = FakeLLM()
    wrapper = EntropyLangchain(llm)
    response = wrapper.invoke("What is Python? I think it is a language.")
    assert len(response) > 0
    summary = wrapper.summary()
    assert summary["calls"] == 1


def test_callback_chat_model():
    class FakeChatMsg:
        def __init__(self, content: str):
            self.content = content

    callback = EntropyCallback()
    msgs = [[FakeChatMsg(content="Hello. I think this should be compressed.")]]
    callback.on_chat_model_start({}, msgs)
    assert len(callback.metrics) == 1
    assert callback.metrics[0]["compression_ratio"] <= 1.0


def test_langchain_multiple_invocations():
    llm = FakeLLM()
    wrapper = EntropyLangchain(llm)
    queries = [
        "What is Python? I think it is a language.",
        "What is the capital of France? I believe it might be Paris.",
        "Tell me about Docker. Basically I need to understand containers.",
    ]
    for q in queries:
        r = wrapper.invoke(q)
        assert len(r) > 0
    summary = wrapper.summary()
    assert summary["calls"] == 3
