from __future__ import annotations

"""
End-to-end validation: runs the full pipeline across multiple scenarios,
checks that EntropyOS preserves critical information while reducing tokens.
"""

import sys

from entropyos.compressor import SemanticCompressor
from entropyos.evaluator import ResponseEvaluator
from entropyos.pipeline import EntropyPipeline


def test_token_reduction_goal() -> bool:
    """Validate >= 40% token reduction on repeated-context scenario
    (system prompt repeated, documents duplicated, filler verbosity)."""
    pipe = EntropyPipeline()
    system_prompt = (
        "You are a helpful assistant. "
        "You must use the provided context to answer accurately. "
        "You must cite sources when available. "
        "You must be concise and clear. "
    ) * 3  # repeated ~3x (common in production with nested agents)
    docs = (
        "Python is a high-level programming language created by Guido van Rossum. "
        "It is dynamically typed and garbage-collected. "
    ) * 3  # same chunks returned across queries
    conversation = (
        "User: What is Python?\n"
        "Assistant: Python is a programming language.\n"
    ) * 3  # repeated conversation patterns
    query = (
        "What is Python and who created it? "
        "I think it might be a language but I am not sure. "
        "Basically I need to understand what it is and who made it. "
        "It should be noted that I am a beginner starting to learn."
    )
    full_context = system_prompt + "\n" + conversation + "\n" + docs + "\n" + query

    result = pipe.run(full_context)
    ratio = result["optimization"]["compression_ratio"]
    reduction = (1 - ratio) * 100
    print(f"  Token reduction: {reduction:.1f}% (target: >= 40%)")
    return reduction >= 40.0


def test_info_preservation_goal() -> bool:
    """Validate >= 99% critical info preservation on text with numbers/names/dates."""
    pipe = EntropyPipeline()
    text = (
        "John Smith must be notified by 2025-01-15. "
        "The API key sk-abc123 must connect to https://example.com. "
        "The function get_user(42) is critical. "
        "I think this is very important information."
    )
    result = pipe.run(text)
    preserved = result["optimization"]["info_preserved"]
    print(f"  Info preserved: {preserved:.1%} (target: >= 99%)")
    return preserved >= 0.99


def test_critical_content_survives() -> bool:
    """Validate that numbers, names, code, and dates are never removed."""
    c = SemanticCompressor(level=0.8)
    text = (
        "I think that the answer is 42. "
        "John Smith must be at 123 Main St by 2025-01-15. "
        "The function `calculate_sum(a, b)` returns a + b. "
        "Email support@example.com for help. "
        "It should be noted that this is very important."
    )
    r = c.compress(text)
    checks = [
        ("42" in r.compressed, "number 42"),
        ("John Smith" in r.compressed, "name John Smith"),
        ("2025-01-15" in r.compressed, "date 2025-01-15"),
        ("`calculate_sum" in r.compressed or "calculate_sum" in r.compressed, "code"),
        ("support@example.com" in r.compressed, "email"),
    ]
    all_pass = True
    for passed, label in checks:
        if not passed:
            print(f"  FAIL: {label} was removed")
            all_pass = False
    if all_pass:
        print(f"  All critical content preserved ({sum(1 for p,_ in checks)}/5 checks)")
    return all_pass


def test_empty_prompt() -> bool:
    pipe = EntropyPipeline()
    result = pipe.run("")
    assert result["total_time_ms"] >= 0
    print("  Empty prompt handled correctly")
    return True


def test_rag_scenario() -> bool:
    pipe = EntropyPipeline()
    context = (
        "Python is a high-level, general-purpose programming language. "
        "It was created by Guido van Rossum and first released in 1991."
    )
    result = pipe.run(
        prompt="What is Python? I think it might be a programming language but I am not sure about who created it.",
        documents=[context],
    )
    print(f"  RAG compression ratio: {result['optimization']['compression_ratio']:.2%}")
    print(f"  Info preserved: {result['optimization']['info_preserved']:.1%}")
    return True


def test_code_scenario() -> bool:
    pipe = EntropyPipeline()
    code_prompt = (
        "I think the function should probably look like:\n"
        "```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n```\n"
        "It is worth noting that this uses recursion. The answer is 120 for n=5."
    )
    result = pipe.run(code_prompt)
    compiled = result["compiled_prompt"]
    checks = [
        ("def factorial" in compiled, "function definition"),
        ("120" in compiled, "number"),
        ("return" in compiled, "return statement"),
    ]
    all_pass = True
    for passed, label in checks:
        if not passed:
            print(f"  FAIL: {label} was removed from code")
            all_pass = False
    if all_pass:
        print(f"  Code content preserved ({sum(1 for p,_ in checks)}/3 checks)")
    print(f"  Code compression ratio: {result['optimization']['compression_ratio']:.2%}")
    return all_pass


def test_pii_redaction() -> bool:
    pipe = EntropyPipeline()
    result = pipe.run("My email is alice@example.com and my phone is 555-123-4567")
    compiled = result["compiled_prompt"]
    if "alice@example.com" in compiled or "555-123-4567" in compiled:
        print("  FAIL: PII was not redacted")
        return False
    print("  PII redacted correctly")
    return True


def main() -> int:
    print("=" * 60)
    print("EntropyOS — Real-World Validation Suite")
    print("=" * 60)

    tests = [
        ("Token reduction >= 40%", test_token_reduction_goal),
        ("Info preservation >= 99%", test_info_preservation_goal),
        ("Critical content preservation", test_critical_content_survives),
        ("Empty prompt handling", test_empty_prompt),
        ("RAG scenario", test_rag_scenario),
        ("Code scenario", test_code_scenario),
        ("PII redaction", test_pii_redaction),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n[{name}]")
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{passed + failed} passed")
    if failed > 0:
        print(f"FAILED: {failed} tests")
        return 1
    print("ALL VALIDATION TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
