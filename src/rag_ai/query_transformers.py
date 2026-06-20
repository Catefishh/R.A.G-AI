from __future__ import annotations

from rag_ai.interfaces import LLM


class StaticQueryTransformer:
    def __init__(self, queries: list[str]) -> None:
        self.queries = queries

    def transform(self, question: str) -> list[str]:
        return [query.format(question=question) for query in self.queries]


class MultiQueryTransformer:
    def __init__(self, llm: LLM, *, variants: int = 3) -> None:
        self.llm = llm
        self.variants = variants

    def transform(self, question: str) -> list[str]:
        prompt = (
            f"Write {self.variants} alternative search queries for this question. "
            "Return one query per line.\nQuestion: "
            f"{question}"
        )
        return _parse_lines(self.llm.generate(prompt))[: self.variants]


class DecompositionTransformer:
    def __init__(self, llm: LLM, *, max_parts: int = 4) -> None:
        self.llm = llm
        self.max_parts = max_parts

    def transform(self, question: str) -> list[str]:
        prompt = f"Break this question into focused sub-questions. Return one per line.\nQuestion: {question}"
        return _parse_lines(self.llm.generate(prompt))[: self.max_parts]


class StepBackTransformer:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def transform(self, question: str) -> list[str]:
        prompt = f"Write one broader step-back question that helps answer this question:\n{question}"
        return _parse_lines(self.llm.generate(prompt))[:1]


class HyDETransformer:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def transform(self, question: str) -> list[str]:
        prompt = f"Write a concise hypothetical answer that would contain terms useful for retrieval:\n{question}"
        generated = self.llm.generate(prompt).strip()
        return [generated] if generated else []


def _parse_lines(text: str) -> list[str]:
    lines: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-*0123456789. )\t").strip()
        if cleaned:
            lines.append(cleaned)
    return lines
