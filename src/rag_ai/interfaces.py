from __future__ import annotations

from pathlib import Path
from typing import Protocol

from rag_ai.models import Chunk, Document, RAGAnswer, RetrievedChunk


class DocumentLoader(Protocol):
    def load(self, path: str | Path) -> list[Document]:
        ...


class TextSplitter(Protocol):
    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        ...


class EmbeddingModel(Protocol):
    def embed_query(self, text: str) -> list[float]:
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        ...

    def similarity_search(
        self,
        query_embedding: list[float],
        *,
        k: int = 4,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        ...


class Retriever(Protocol):
    def retrieve(
        self,
        question: str,
        *,
        k: int = 4,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        ...


class Reranker(Protocol):
    def rerank(self, question: str, chunks: list[RetrievedChunk], *, k: int | None = None) -> list[RetrievedChunk]:
        ...


class LLM(Protocol):
    def generate(self, prompt: str, *, context: list[RetrievedChunk] | None = None) -> str:
        ...


class QueryTransformer(Protocol):
    def transform(self, question: str) -> list[str]:
        ...


class RAGSystem(Protocol):
    def ask(self, question: str, *, k: int = 4, filters: dict[str, object] | None = None) -> RAGAnswer:
        ...
