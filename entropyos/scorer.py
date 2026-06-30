from __future__ import annotations

import math
import re
from collections import Counter

from entropyos.config import EntropyConfig
from entropyos.models import ScoreResult


class InformationScorer:
    def __init__(self, config: EntropyConfig | None = None):
        self.config = config
        self._llm_scorer = None

    def score(self, text: str, context: str | None = None) -> ScoreResult:
        if not text or not text.strip():
            return ScoreResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "heuristic")
        entropy = self._shannon_entropy(text)
        novelty = self._novelty(text, context)
        redundancy = self._redundancy(text)
        importance = self._importance(text)
        dependency = self._dependency(text)

        info_score = novelty * 0.3 + importance * 0.35 + dependency * 0.2 - redundancy * 0.15
        info_score = max(0.0, min(1.0, info_score))

        return ScoreResult(
            entropy=entropy,
            novelty=novelty,
            redundancy=redundancy,
            importance=importance,
            dependency=dependency,
            information_score=info_score,
            method="heuristic",
        )

    def score_llm(self, text: str, context: str | None = None) -> ScoreResult:
        heuristic = self.score(text, context)
        if not self.config or not self.config.llm.api_key:
            return heuristic
        try:
            from entropyos.adapters import create_adapter
            adapter = create_adapter(self.config.llm)
            prompt = (
                "Rate this text on these dimensions from 0.0 to 1.0.\n"
                "Return ONLY a JSON object with keys: novelty, redundancy, importance, dependency\n"
                "Text: " + text[:1000] + "\n"
                "JSON:"
            )
            resp = adapter.complete(prompt, max_tokens=200, temperature=0.0)
            import json
            try:
                llm_scores = json.loads(resp.strip().strip("`").strip())
                heuristic.novelty = llm_scores.get("novelty", heuristic.novelty)
                heuristic.redundancy = llm_scores.get("redundancy", heuristic.redundancy)
                heuristic.importance = llm_scores.get("importance", heuristic.importance)
                heuristic.dependency = llm_scores.get("dependency", heuristic.dependency)
                heuristic.information_score = (
                    heuristic.novelty * 0.3 + heuristic.importance * 0.35 +
                    heuristic.dependency * 0.2 - heuristic.redundancy * 0.15
                )
                heuristic.information_score = max(0.0, min(1.0, heuristic.information_score))
                heuristic.method = "llm"
            except json.JSONDecodeError:
                pass
        except Exception:
            pass
        return heuristic

    @staticmethod
    def _shannon_entropy(text: str) -> float:
        if not text:
            return 0.0
        cleaned = text.strip()
        if not cleaned:
            return 0.0
        freq: Counter[str] = Counter(cleaned)
        total = len(cleaned)
        ent = 0.0
        for count in freq.values():
            p = count / total
            if p > 0:
                ent -= p * math.log2(p)
        max_ent = math.log2(min(total, 128))
        return ent / max_ent if max_ent > 0 else 0.0

    @staticmethod
    def _novelty(text: str, context: str | None) -> float:
        if not context:
            return 1.0
        words_t = set(text.lower().split())
        words_c = set(context.lower().split())
        if not words_t:
            return 0.0
        overlap = len(words_t & words_c)
        denom = len(words_t | words_c)
        jaccard = overlap / denom if denom > 0 else 0.0
        return 1.0 - jaccard

    @staticmethod
    def _redundancy(text: str) -> float:
        words = text.lower().split()
        if len(words) < 3:
            return 0.0
        total = len(words)
        unique = len(set(words))
        ratio = total / unique if unique > 0 else 1.0
        return min(1.0, (ratio - 1.0) / 5.0)

    @staticmethod
    def _importance(text: str) -> float:
        score = 0.0
        if re.search(r'\b\d+\b', text):
            score += 0.2
        if re.search(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', text):
            score += 0.2
        if re.search(r'```[\s\S]*?```|`[^`]+`', text):
            score += 0.25
        if re.search(r'\b(?:must|shall|required|critical|important|warning|note)\b', text, re.IGNORECASE):
            score += 0.15
        if re.search(r'\b(?:what|why|how|when|where|who|which|explain|define|describe|list|find|show|tell)\b.*\?', text, re.IGNORECASE):
            score += 0.1
        if re.search(r'\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})\b', text):
            score += 0.1
        if re.search(r'\b(?:https?://|www\.)\S+', text):
            score += 0.1
        return min(1.0, score)

    @staticmethod
    def _dependency(text: str) -> float:
        references = len(re.findall(
            r'\b(?:see|as mentioned|above|below|section|equation|figure|table)\b',
            text, re.IGNORECASE,
        ))
        refs = len(re.findall(r'\[\d+\]|\(\w+\)', text))
        total = references + refs
        return min(1.0, total / 10.0)
