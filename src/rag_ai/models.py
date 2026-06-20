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
