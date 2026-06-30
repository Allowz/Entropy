from __future__ import annotations

import re
from entropyos.compressor import SemanticCompressor
from entropyos.scorer import InformationScorer


class ContextOptimizer:
    def __init__(self, scorer: InformationScorer, compressor: SemanticCompressor):
        self.scorer = scorer
        self.compressor = compressor

    def optimize(self, prompt: str, memories: list, documents: list) -> tuple[str, dict]:
        deduped = self._deduplicate(prompt, memories, documents)
        compressed, c_result = self._compress_context(deduped)
        preserved = self._preserve_facts(compressed, prompt)
        conflicts = self._detect_conflicts(preserved)
        ranked = self._priority_rank(preserved)

        report = {
            "deduped_lines": len(deduped.splitlines()),
            "compression_ratio": c_result.compression_ratio,
            "conflicts_found": len(conflicts),
            "info_preserved": c_result.info_preserved,
        }
        return ranked, report

    def _deduplicate(self, prompt: str, memories: list, documents: list) -> str:
        all_text = [prompt]
        seen: set[str] = set()
        for mem in memories + documents:
            content = getattr(mem, "content", None) or (mem if isinstance(mem, str) else "")
            for line in content.splitlines():
                stripped = line.strip().lower()
                if stripped and stripped not in seen:
                    seen.add(stripped)
                    all_text.append(line)
        return "\n".join(all_text)

    def _compress_context(self, text: str) -> tuple[str, object]:
        result = self.compressor.compress(text)
        return result.compressed, result

    @staticmethod
    def _preserve_facts(compressed: str, original: str) -> str:
        numbers = re.findall(r'\b\d+\b', original)
        names = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', original)
        missing = []
        for n in numbers:
            if n not in compressed:
                missing.append(n)
        for name in names:
            if name not in compressed:
                missing.append(name)
        if missing:
            compressed += "\n# preserved: " + ", ".join(missing)
        return compressed

    @staticmethod
    def _detect_conflicts(text: str) -> list[str]:
        conflicts = []
        statements = re.findall(r'(?:do|must|shall|always|never)\s+.+?[.\n]', text, re.IGNORECASE)
        for i, a in enumerate(statements):
            for b in statements[i + 1:]:
                if _contradicts(a, b):
                    conflicts.append(f"Conflict: {a.strip()} vs {b.strip()}")
        return conflicts

    @staticmethod
    def _priority_rank(text: str) -> str:
        lines = text.splitlines()
        important = []
        other = []
        for line in lines:
            if re.search(r'\b(critical|important|warning|must|shall|required)\b', line, re.IGNORECASE):
                important.append(line)
            else:
                other.append(line)
        return "\n".join(important + other)


def _contradicts(a: str, b: str) -> bool:
    a_lower = a.lower()
    b_lower = b.lower()
    if ("always" in a_lower or "must" in a_lower) and ("never" in b_lower or "avoid" in b_lower):
        return True
    if ("never" in a_lower or "avoid" in a_lower) and ("always" in b_lower or "must" in b_lower):
        return True
    return False
