from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from rag_ai.interfaces import RAGSystem


@dataclass(slots=True)
class EvalExample:
    question: str
    expected_source: str | None = None
    expected_answer_contains: str | None = None


@dataclass(slots=True)
class EvalResult:
    total: int
    source_hits: int
    answer_hits: int

    @property
    def source_hit_rate(self) -> float:
        return self.source_hits / self.total if self.total else 0.0

    @property
    def answer_hit_rate(self) -> float:
        return self.answer_hits / self.total if self.total else 0.0


def load_eval_csv(path: str | Path) -> list[EvalExample]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            EvalExample(
                question=row["question"],
                expected_source=row.get("expected_source") or None,
                expected_answer_contains=row.get("expected_answer_contains") or None,
            )
            for row in reader
        ]


def evaluate(system: RAGSystem, examples: list[EvalExample], *, k: int = 4) -> EvalResult:
    source_hits = 0
    answer_hits = 0
    for example in examples:
        answer = system.ask(example.question, k=k)
        if example.expected_source is None or any(
            example.expected_source in str(source.chunk.metadata.get("source", "")) for source in answer.sources
        ):
            source_hits += 1
        if example.expected_answer_contains is None or example.expected_answer_contains.lower() in answer.answer.lower():
            answer_hits += 1
    return EvalResult(total=len(examples), source_hits=source_hits, answer_hits=answer_hits)
