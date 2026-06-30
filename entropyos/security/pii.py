from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class PIISpan:
    type: str
    text: str
    start: int
    end: int


PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email", re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')),
    ("phone", re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')),
    ("ssn", re.compile(r'\b\d{3}-\d{2}-\d{4}\b')),
    ("credit_card", re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')),
    ("ip_address", re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')),
    ("api_key", re.compile(r'\b(sk-[a-zA-Z0-9]{20,}|sk-[a-zA-Z0-9-_]{20,})\b')),
    ("token", re.compile(r'\b[ghp|gho|ghu|ghs|ghr]_[a-zA-Z0-9]{36,}\b')),
]


class PIIDetector:
    def __init__(self, patterns: list[tuple[str, re.Pattern]] | None = None):
        self.patterns = patterns or PII_PATTERNS

    def detect(self, text: str) -> list[PIISpan]:
        spans: list[PIISpan] = []
        for pii_type, pat in self.patterns:
            for m in pat.finditer(text):
                spans.append(PIISpan(
                    type=pii_type,
                    text=m.group(),
                    start=m.start(),
                    end=m.end(),
                ))
        spans.sort(key=lambda s: s.start)
        return spans

    def redact(self, text: str, replacement: str = "[REDACTED]") -> str:
        spans = self.detect(text)
        if not spans:
            return text
        merged = self._merge_spans(spans)
        result = []
        pos = 0
        for s in merged:
            result.append(text[pos:s.start])
            result.append(replacement)
            pos = s.end
        result.append(text[pos:])
        return "".join(result)

    def has_pii(self, text: str) -> bool:
        return len(self.detect(text)) > 0

    @staticmethod
    def _merge_spans(spans: list[PIISpan]) -> list[PIISpan]:
        if not spans:
            return []
        merged = [spans[0]]
        for s in spans[1:]:
            if s.start <= merged[-1].end:
                merged[-1] = PIISpan(
                    type=f"{merged[-1].type}|{s.type}",
                    text=merged[-1].text + text[merged[-1].end:s.end],
                    start=merged[-1].start,
                    end=max(merged[-1].end, s.end),
                )
            else:
                merged.append(s)
        return merged
