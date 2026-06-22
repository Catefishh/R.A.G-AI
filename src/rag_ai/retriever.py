"""Retriever: ties an Embedder and a VectorStore together.

Given a natural-language query it returns the most relevant chunks, filtered by
a minimum similarity threshold so the pipeline can tell the difference between
"found relevant context" and "found nothing useful".
"""

from __future__ import annotations

from dataclasses import dataclass

from .documents import Chunk
from .embeddings import Embedder
from .vector_store import VectorStore


@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval, with its similarity score."""

    chunk: Chunk
    score: float

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def source(self) -> str:
        return self.chunk.source


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        top_k: int = 4,
        min_score: float = 0.15,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k if top_k is not None else self.top_k
        query_vector = self.embedder.embed_one(query)
        hits = self.store.search(query_vector, top_k=k)
        return [
            RetrievedChunk(chunk=chunk, score=score)
            for chunk, score in hits
            if score >= self.min_score
        ]
