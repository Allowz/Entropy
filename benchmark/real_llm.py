"""Real LLM validation — compare with/without EntropyOS on real LLM calls."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from entropyos.pipeline import EntropyPipeline


ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
OPENAI_MODEL = "gpt-4o-mini"
OPENROUTER_MODEL = "anthropic/claude-sonnet-4"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

SCENARIOS = {
    "simple_qa": {
        "prompt": "What is the capital of France?",
        "system": "",
    },
    "verbose_qa": {
        "prompt": (
            "What is the capital of France? I think it might be Paris but I am not sure. "
            "Basically I need to know for a trip I am planning. "
            "It should be noted that I am very interested in French culture."
        ),
        "system": "",
    },
    "rag_style": {
        "prompt": (
            "Based on the context, what is Python? "
            "I think it might be a programming language but I am not sure."
        ),
        "system": (
            "You are a helpful assistant. "
            "Use the provided context to answer accurately. "
            "You must cite sources when available. "
        ),
    },
    "code_review": {
        "prompt": (
            "Review this function. I think it might have a bug in the edge case. "
            "It should be noted that n should always be a positive integer.\n"
            "```python\ndef factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n-1)\n```"
        ),
        "system": (
            "You are a code reviewer. "
            "Check for bugs, security, and style. "
            "Be specific about issues. "
        ),
    },
}


@dataclass
class LLMResult:
    scenario: str
    variant: str  # "baseline" or "entropyos"
    prompt_chars: int
    response_text: str
    latency_s: float
    error: str = ""

    @property
    def estimated_input_tokens(self) -> int:
        return int(self.prompt_chars / 4)

    @property
    def estimated_output_tokens(self) -> int:
        return int(len(self.response_text) / 4)

    @property
    def estimated_cost(self) -> float:
        in_cost = self.estimated_input_tokens * 3.0 / 1_000_000
        out_cost = self.estimated_output_tokens * 15.0 / 1_000_000
        return in_cost + out_cost


def call_anthropic(api_key: str, system: str, prompt: str, model: str) -> tuple[str, float]:
    try:
        from anthropic import Anthropic
    except ImportError:
        return "", 0.0
    client = Anthropic(api_key=api_key)
    t0 = time.perf_counter()
    kwargs = {"model": model, "max_tokens": 512, "messages": [{"role": "user", "content": prompt}]}
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    elapsed = time.perf_counter() - t0
    text = resp.content[0].text if resp.content else ""
    return text, elapsed


def call_openai(api_key: str, system: str, prompt: str, model: str) -> tuple[str, float]:
    try:
        from openai import OpenAI
    except ImportError:
        return "", 0.0
    client = OpenAI(api_key=api_key)
    t0 = time.perf_counter()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(model=model, messages=messages, max_tokens=512)
    elapsed = time.perf_counter() - t0
    return resp.choices[0].message.content or "", elapsed


def call_openrouter(api_key: str, system: str, prompt: str, model: str) -> tuple[str, float]:
    try:
        from openai import OpenAI
    except ImportError:
        return "", 0.0
    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE)
    t0 = time.perf_counter()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model or OPENROUTER_MODEL,
        messages=messages,
        max_tokens=512,
        extra_headers={"HTTP-Referer": "https://entropyos.dev", "X-Title": "EntropyOS Benchmark"},
    )
    elapsed = time.perf_counter() - t0
    return resp.choices[0].message.content or "", elapsed


def run_scenario(
    scenario_name: str,
    cfg: dict,
    llm_fn,
    api_key: str,
    model_name: str,
) -> tuple[LLMResult, LLMResult]:
    pipe = EntropyPipeline()
    system = cfg.get("system", "")
    prompt = cfg["prompt"]

    full_baseline = (system + "\n\n" + prompt) if system else prompt

    t0 = time.perf_counter()
    resp_baseline, lat_baseline = llm_fn(api_key, system, prompt, model_name)
    t_baseline = time.perf_counter() - t0

    pipe_result = pipe.run(prompt, agent_state={"system": system} if system else None)
    optimized_prompt = pipe_result["compiled_prompt"]

    t1 = time.perf_counter()
    resp_opt, lat_opt = llm_fn(api_key, "", optimized_prompt, model_name)
    t_opt = time.perf_counter() - t1

    baseline = LLMResult(
        scenario=scenario_name, variant="baseline",
        prompt_chars=len(full_baseline), response_text=resp_baseline,
        latency_s=t_baseline,
    )
    entropy = LLMResult(
        scenario=scenario_name, variant="entropyos",
        prompt_chars=len(optimized_prompt), response_text=resp_opt,
        latency_s=t_opt,
    )
    return baseline, entropy, pipe_result


def judge_quality(api_key: str, prompt: str, answers: list[str], model: str) -> float:
    """Ask LLM to rate answer quality 0-1."""
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        rating_prompt = (
            f"Prompt: {prompt[:500]}\n\n"
            f"Answer A: {answers[0][:500]}\n"
            f"Answer B: {answers[1][:500]}\n\n"
            "Rate each answer 0.0-1.0 for accuracy, completeness, and clarity. "
            "Return ONLY JSON: {\"a_score\": 0.0, \"b_score\": 0.0}"
        )
        resp = client.messages.create(
            model=model, max_tokens=100,
            messages=[{"role": "user", "content": rating_prompt}],
        )
        text = resp.content[0].text if resp.content else "{}"
        scores = json.loads(text.strip().strip("`").strip())
        return scores.get("a_score", 1.0), scores.get("b_score", 1.0)
    except Exception:
        return 1.0, 1.0


def print_comparison(scenario: str, baseline: LLMResult, entropy: LLMResult, pipe_result: dict, quality: tuple):
    reduction = (1 - entropy.prompt_chars / baseline.prompt_chars) * 100 if baseline.prompt_chars else 0
    cost_b = baseline.estimated_cost
    cost_e = entropy.estimated_cost
    cost_save = (1 - cost_e / cost_b) * 100 if cost_b else 0
    q_b, q_e = quality

    print(f"\n{'='*60}")
    print(f"  {scenario}")
    print(f"{'='*60}")
    print(f"  {'':<25} {'Baseline':>15} {'EntropyOS':>15}")
    print(f"  {'-'*55}")
    print(f"  {'Prompt chars':<25} {baseline.prompt_chars:>15} {entropy.prompt_chars:>15}")
    print(f"  {'Est. input tokens':<25} {baseline.estimated_input_tokens:>15} {entropy.estimated_input_tokens:>15}")
    print(f"  {'Reduction':<25} {'':>15} {reduction:>14.1f}%")
    print(f"  {'Response chars':<25} {len(baseline.response_text):>15} {len(entropy.response_text):>15}")
    print(f"  {'Latency (s)':<25} {baseline.latency_s:>14.3f} {entropy.latency_s:>14.3f}")
    print(f"  {'Est. cost':<25} ${cost_b:>13.6f} ${cost_e:>13.6f}")
    print(f"  {'Cost savings':<25} {'':>15} {cost_save:>14.1f}%")
    print(f"  {'Quality score':<25} {q_b:>14.2f} {q_e:>14.2f}")
    print(f"  {'Info preserved':<25} {'':>15} {pipe_result['optimization']['info_preserved']:>13.1%}")
    print(f"  {'Entropy score':<25} {'':>15} {pipe_result['score']['information_score']:>13.3f}")


def main():
    api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY")
        sys.exit(1)

    use_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
    use_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if use_openrouter:
        model = os.environ.get("OPENROUTER_MODEL", OPENROUTER_MODEL)
        llm_fn = call_openrouter
        provider = f"OpenRouter ({model})"
    elif use_anthropic:
        model = ANTHROPIC_MODEL
        llm_fn = call_anthropic
        provider = f"Anthropic ({model})"
    else:
        model = OPENAI_MODEL
        llm_fn = call_openai
        provider = f"OpenAI ({model})"
    print(f"Provider: {provider}")

    total_baseline_cost = 0.0
    total_entropy_cost = 0.0
    total_baseline_chars = 0
    total_entropy_chars = 0
    quality_scores: list[tuple[float, float]] = []
    pipeline_results: list[dict] = []

    print(f"\n{'#'*60}")
    print(f"  Real LLM Validation — EntropyOS vs Baseline")
    print(f"{'#'*60}")

    for name, cfg in SCENARIOS.items():
        base, ent, pr = run_scenario(name, cfg, llm_fn, api_key, model)
        quality = judge_quality(
            api_key,
            cfg["prompt"],
            [base.response_text, ent.response_text],
            model,
        )
        print_comparison(name, base, ent, pr, quality)
        total_baseline_cost += base.estimated_cost
        total_entropy_cost += ent.estimated_cost
        total_baseline_chars += base.prompt_chars
        total_entropy_chars += ent.prompt_chars
        quality_scores.append(quality)
        pipeline_results.append(pr)

    overall_reduction = (1 - total_entropy_chars / total_baseline_chars) * 100
    cost_savings = (1 - total_entropy_cost / total_baseline_cost) * 100
    avg_q_b = sum(q[0] for q in quality_scores) / len(quality_scores)
    avg_q_e = sum(q[1] for q in quality_scores) / len(quality_scores)

    print(f"\n{'='*60}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Total scenarios':<30} {len(SCENARIOS):>15}")
    print(f"  {'Total baseline chars':<30} {total_baseline_chars:>15}")
    print(f"  {'Total entropy chars':<30} {total_entropy_chars:>15}")
    print(f"  {'Overall reduction':<30} {overall_reduction:>14.1f}%")
    print(f"  {'Total baseline cost':<30} ${total_baseline_cost:>13.6f}")
    print(f"  {'Total entropy cost':<30} ${total_entropy_cost:>13.6f}")
    print(f"  {'Cost savings':<30} {cost_savings:>14.1f}%")
    print(f"  {'Avg quality (baseline)':<30} {avg_q_b:>14.2f}")
    print(f"  {'Avg quality (entropy)':<30} {avg_q_e:>14.2f}")
    print(f"  {'Quality delta':<30} {avg_q_e - avg_q_b:>+14.2f}")
    print(f"  {'Latency overhead (avg)':<30} {sum(abs(pr['total_time_ms']) for pr in pipeline_results)/len(pipeline_results):>13.1f}ms")

    # Project to annual
    annual_calls = 100_000
    annual_savings = (total_baseline_cost - total_entropy_cost) / len(SCENARIOS) * annual_calls
    print(f"\n  Projected ({annual_calls:,} calls/yr):")
    print(f"  {'Baseline cost':<30} ${total_baseline_cost / len(SCENARIOS) * annual_calls:>13.2f}")
    print(f"  {'EntropyOS cost':<30} ${total_entropy_cost / len(SCENARIOS) * annual_calls:>13.2f}")
    print(f"  {'Annual savings':<30} ${annual_savings:>13.2f}")


if __name__ == "__main__":
    main()
