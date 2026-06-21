from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


Metadata = dict[str, Any]


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


@dataclass(slots=True)
class Document:
    text: str
    metadata: Metadata = field(default_factory=dict)
    id: str = field(default_factory=lambda: _id("doc"))


@dataclass(slots=True)
class Chunk:
    text: str
    document_id: str
    metadata: Metadata = field(default_factory=dict)
    id: str = field(default_factory=lambda: _id("chunk"))


@dataclass(slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(slots=True)
class RAGAnswer:
    question: str
    answer: str
    sources: list[RetrievedChunk]
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class GenerationMetrics:
    elapsed_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(slots=True)
class GenerationResult:
    text: str
    metrics: GenerationMetrics


@dataclass(slots=True)
class ComparisonAnswer:
    answer: str
    metrics: GenerationMetrics
    sources: list[RetrievedChunk] = field(default_factory=list)
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class ComparisonResult:
    question: str
    without_rag: ComparisonAnswer
    with_rag: ComparisonAnswer
