from entropyos.pipeline import EntropyPipeline


def test_smoke_quick():
    pipe = EntropyPipeline()
    result = pipe.run("What is the capital of France?")
    assert result["total_time_ms"] < 5000
    assert len(result["compiled_prompt"]) > 0
