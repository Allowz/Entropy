from __future__ import annotations

import json
import re

from entropyos.config import EntropyConfig
from entropyos.models import CompressResult


PROTECTED_PATTERNS = [
    re.compile(r'\b\d+\b'),
    re.compile(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+'),
    re.compile(r'```[\s\S]*?```'),
    re.compile(r'`[^`]+`'),
    re.compile(r'https?://\S+'),
    re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+'),
    re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'),
    re.compile(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'),
    re.compile(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b'),
    re.compile(r'(?:^|\n)(?:\s*[-*]\s+|\s*\d+\.\s+)'),
]

FILLER_PATTERNS = [
    re.compile(
        r'\b(in other words|that is to say|it is worth noting that|it should be noted that|'
        r'importantly|notably|as a matter of fact|basically|essentially|actually|'
        r'simply put|generally speaking|needless to say|it goes without saying|'
        r'as you know|as we all know|as previously mentioned|as stated earlier)\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(I think|I believe|I feel|I would say|it seems that|it appears that|'
        r'arguably|presumably|I would argue|I contend|I submit|one could say)\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(very|really|quite|extremely|highly|incredibly|absolutely|totally|completely|'
        r'utterly|entirely|somewhat|rather|fairly|pretty|probably|maybe|perhaps|possibly|'
        r'basically|essentially|literally|practically|virtually|just|simply|merely)\b',
        re.IGNORECASE,
    ),
    re.compile(r'\b(just|actually|basically|seriously|honestly|frankly|mostly|mostly|largely)\b', re.IGNORECASE),
]

HEDGE_PATTERNS = re.compile(
    r'\b(it is possible that|it could be that|it may be that|it might be that|'
    r'one might say that|you could argue that|the point is that|'
    r'the fact of the matter is|the truth is that)\b',
    re.IGNORECASE,
)

HEDGE_SUFFIXES = re.compile(
    r'\b(but I am not sure|but I\'m not sure|but I don\'t know|but I do not know|'
    r'I am not certain|I\'m not certain|I am not sure|I\'m not sure|'
    r'I don\'t know|I do not know|honestly|to be honest|to tell the truth)\b',
    re.IGNORECASE,
)

INSTRUCTION_FILLER = re.compile(
    r'\b(You should|You must|You need to|You can|You could|You will|Please|Always|Never)\s+'
    r'(?:always|never|always )?(try to|remember to|make sure to|be sure to|endeavor to)?'
    r'\s*(?:use the provided context|be concise|be accurate|provide examples|'
    r'cite sources|be helpful|be thorough|be detailed)\b',
    re.IGNORECASE,
)


class SemanticCompressor:
    def __init__(self, level: float = 0.5, config: EntropyConfig | None = None):
        self.level = level
        self.config = config

    def compress(self, text: str) -> CompressResult:
        if not text.strip():
            return CompressResult(compressed="", compression_ratio=1.0, info_preserved=1.0, method="heuristic")

        original_len = len(text)
        lines = text.splitlines(keepends=True)
        protected_spans = self._locate_protected(lines)

        processed = self._remove_filler(lines, protected_spans)

        if self.level >= 0.3:
            processed = self._deduplicate_sentences(processed, protected_spans)

        processed = self._deduplicate_lines(processed, protected_spans)

        compressed = "".join(processed)
        compressed = re.sub(r'\s{2,}', ' ', compressed)
        compressed = re.sub(r'\s+\.', '.', compressed)
        compressed = re.sub(r'\.{2,}', '.', compressed)
        compressed_len = len(compressed)
        ratio = compressed_len / original_len if original_len else 1.0
        preserved = self._estimate_info_preserved(compressed, original_len, text)

        return CompressResult(
            compressed=compressed,
            compression_ratio=ratio,
            info_preserved=preserved,
            method="heuristic",
        )

    def compress_llm(self, text: str) -> CompressResult:
        heuristic = self.compress(text)
        if not self.config or not self.config.llm.api_key:
            return heuristic
        try:
            from entropyos.adapters import create_adapter
            adapter = create_adapter(self.config.llm)
            prompt = (
                "Compress this text to its minimal essential form.\n"
                "NEVER remove: numbers, names, code, dates, constraints.\n"
                "You may remove: filler, repeated ideas, unnecessary adjectives, duplicated instructions.\n"
                "Return ONLY a JSON object with keys: compressed (string), compression_ratio (float), info_preserved (float 0-1)\n"
                "Text: " + text[:2000] + "\n"
                "JSON:"
            )
            resp = adapter.complete(prompt, max_tokens=1000, temperature=0.0)
            try:
                result = json.loads(resp.strip().strip("`").strip())
                return CompressResult(
                    compressed=result.get("compressed", heuristic.compressed),
                    compression_ratio=result.get("compression_ratio", heuristic.compression_ratio),
                    info_preserved=result.get("info_preserved", heuristic.info_preserved),
                    method="llm",
                )
            except json.JSONDecodeError:
                pass
        except Exception:
            pass
        return heuristic

    def _locate_protected(self, lines: list[str]) -> set[tuple[int, int, int]]:
        spans: set[tuple[int, int, int]] = set()
        for i, line in enumerate(lines):
            for pat in PROTECTED_PATTERNS:
                for m in pat.finditer(line):
                    spans.add((i, m.start(), m.end()))
        return spans

    def _remove_filler(self, lines: list[str], spans: set) -> list[str]:
        if self.level < 0.05:
            return lines[:]
        result = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                result.append(line)
                continue
            protected_chars: set[int] = set()
            for s in spans:
                if s[0] == i:
                    for c in range(s[1], s[2]):
                        protected_chars.add(c)
            modified = self._strip_filler_from_line(line, protected_chars)
            result.append(modified)
        return result

    def _strip_filler_from_line(self, line: str, protected_chars: set[int]) -> str:
        for pat in FILLER_PATTERNS:
            for m in reversed(list(pat.finditer(line))):
                if any(c in protected_chars for c in range(m.start(), m.end())):
                    continue
                line = line[:m.start()] + line[m.end():]
                if not line.strip():
                    return "\n"
        if self.level >= 0.5:
            for pat in (HEDGE_PATTERNS, HEDGE_SUFFIXES):
                for m in reversed(list(pat.finditer(line))):
                    if any(c in protected_chars for c in range(m.start(), m.end())):
                        continue
                    line = line[:m.start()] + line[m.end():]
        if self.level >= 0.7:
            for m in reversed(list(INSTRUCTION_FILLER.finditer(line))):
                if any(c in protected_chars for c in range(m.start(), m.end())):
                    continue
                line = line[:m.start()] + line[m.end():]
        return line

    def _deduplicate_sentences(self, lines: list[str], spans: set) -> list[str]:
        SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
        PROTECTED_CONTENT = re.compile(
            r'\b\d+\b|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|```|`[^`]+`|https?://\S+|[\w.+-]+@[\w-]+\.[\w.-]+|\d{4}[-/]\d{1,2}[-/]\d{1,2}'
        )
        result: list[str] = []
        seen_sentences: set[str] = set()

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                result.append(line)
                continue
            sentences = SENTENCE_SPLIT.split(stripped)
            kept: list[str] = []
            for sent in sentences:
                sent_stripped = sent.strip()
                key = sent_stripped.rstrip('.').strip().lower()
                if not key:
                    continue
                is_protected = bool(PROTECTED_CONTENT.search(sent_stripped))
                if not is_protected and key in seen_sentences:
                    continue
                seen_sentences.add(key)
                kept.append(sent_stripped)
            if kept:
                ending = "\n" if line.endswith("\n") else " "
                joined_sentences: list[str] = []
                for s in kept:
                    if s and not s.endswith((".", "!", "?")):
                        s += "."
                    joined_sentences.append(s)
                result.append(" ".join(joined_sentences) + ending.rstrip(" "))
            else:
                result.append("\n")
        return result

    @staticmethod
    def _deduplicate_lines(lines: list[str], spans: set) -> list[str]:
        seen: set[str] = set()
        result = []
        for i, line in enumerate(lines):
            stripped = line.strip().lower()
            if stripped and stripped in seen:
                if not any(s[0] == i for s in spans):
                    continue
            if stripped:
                seen.add(stripped)
            result.append(line)
        return result

    @staticmethod
    def _estimate_info_preserved(compressed: str, original_len: int, original: str) -> float:
        if not original.strip():
            return 1.0
        key_terms = re.findall(r'\b\d+\b|[A-Z][a-z]{2,}', original)
        if not key_terms:
            return 1.0
        found = sum(1 for t in key_terms if t in compressed)
        return found / len(key_terms)
