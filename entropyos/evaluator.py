from __future__ import annotations

import re
from entropyos.models import EvalResult


class ResponseEvaluator:
    def evaluate(
        self,
        original_prompt: str,
        compressed_prompt: str | None,
        response: str,
    ) -> EvalResult:
        accuracy = self._accuracy(response)
        completeness = self._completeness(original_prompt, response)
        hallucination_score = self._hallucination(response, original_prompt)
        compression_effectiveness = self._compression_effectiveness(
            original_prompt, compressed_prompt
        ) if compressed_prompt else 0.5
        info_retained = self._info_retained(original_prompt, response)

        return EvalResult(
            accuracy=accuracy,
            completeness=completeness,
            hallucination_score=hallucination_score,
            compression_effectiveness=compression_effectiveness,
            information_retained=info_retained,
        )

    @staticmethod
    def _accuracy(response: str) -> float:
        contradictions = len(re.findall(
            r'\b(?:on the other hand|however|but actually|wait|contradicts|incorrect|wrong)\b',
            response, re.IGNORECASE,
        ))
        score = 1.0 - (contradictions * 0.1)
        return max(0.0, min(1.0, score))

    @staticmethod
    def _completeness(prompt: str, response: str) -> float:
        questions = re.findall(r'\b(?:what|why|how|when|where|who|which)\b.*?\?', prompt, re.IGNORECASE)
        if not questions:
            return 1.0
        answered = 0
        for q in questions:
            for indicator in re.findall(r'\b(?:is|are|was|were|will|can|does|do|has|have)\b', q, re.IGNORECASE):
                if indicator.lower() in response.lower():
                    answered += 1
                    break
        return min(1.0, answered / len(questions))

    @staticmethod
    def _hallucination(response: str, prompt: str) -> float:
        prompt_facts = set(re.findall(r'\b[A-Z][a-z]{2,}\b', prompt))
        response_facts = set(re.findall(r'\b[A-Z][a-z]{2,}\b', response))
        if not response_facts:
            return 0.0
        novel_facts = response_facts - prompt_facts
        return min(1.0, len(novel_facts) / max(1, len(response_facts)))

    @staticmethod
    def _compression_effectiveness(original: str, compressed: str) -> float:
        if not original:
            return 0.0
        ratio = len(compressed) / len(original)
        if ratio >= 1.0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - ratio))

    @staticmethod
    def _info_retained(original: str, response: str) -> float:
        numbers = re.findall(r'\b\d+\b', original)
        names = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', original)
        key_terms = numbers + names
        if not key_terms:
            return 1.0
        found = sum(1 for t in key_terms if t in response)
        return found / len(key_terms)
