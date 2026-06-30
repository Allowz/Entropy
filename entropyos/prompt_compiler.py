from __future__ import annotations

from entropyos.scorer import InformationScorer


class PromptCompiler:
    def __init__(self, scorer: InformationScorer):
        self.scorer = scorer

    def compile(
        self,
        user_prompt: str,
        memories: list,
        documents: list | None = None,
        agent_state: dict | None = None,
    ) -> str:
        parts: list[str] = []

        if agent_state:
            state_block = self._render_state(agent_state)
            if state_block:
                parts.append(state_block)

        if memories:
            mem_block = self._render_memories(memories)
            if mem_block:
                parts.append(mem_block)

        if documents:
            doc_block = self._render_documents(documents)
            if doc_block:
                parts.append(doc_block)

        parts.append(user_prompt)
        result = "\n\n".join(parts)
        return self._trim_to_budget(result)

    @staticmethod
    def _render_state(state: dict) -> str:
        lines = []
        for k, v in state.items():
            if isinstance(v, str) and len(v) < 500:
                lines.append(f"{k}: {v}")
            elif isinstance(v, (int, float, bool)):
                lines.append(f"{k}: {v}")
        return "\n".join(lines) if lines else ""

    @staticmethod
    def _render_memories(memories: list) -> str:
        lines = []
        seen: set[str] = set()
        for mem in memories:
            content = getattr(mem, "content", None) or str(mem)
            key = content.strip().lower()
            if key and key not in seen:
                seen.add(key)
                lines.append(f"- {content.strip()}")
        return "Context:\n" + "\n".join(lines) if lines else ""

    @staticmethod
    def _render_documents(documents: list) -> str:
        lines = []
        seen: set[str] = set()
        for doc in documents:
            content = getattr(doc, "content", None) or getattr(doc, "page_content", None) or str(doc)
            key = content.strip().lower()
            if key and key not in seen:
                seen.add(key)
                lines.append(content.strip())
        return "References:\n" + "\n\n".join(lines[:5]) if lines else ""

    @staticmethod
    def _trim_to_budget(text: str, budget: int = 4096) -> str:
        words = text.split()
        if len(words) <= budget:
            return text
        return " ".join(words[:budget])
