from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_ai.embeddings import cosine_similarity
from rag_ai.models import Chunk, RetrievedChunk


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._items: list[tuple[Chunk, list[float]]] = []

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self._items.extend(zip(chunks, embeddings))

    def similarity_search(
        self,
        query_embedding: list[float],
        *,
        k: int = 4,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        results: list[RetrievedChunk] = []
        for chunk, embedding in self._items:
            if _matches_filters(chunk.metadata, filters):
                results.append(RetrievedChunk(chunk=chunk, score=cosine_similarity(query_embedding, embedding)))
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:k]

    def __len__(self) -> int:
        return len(self._items)


class ChromaVectorStore:
    def __init__(self, *, collection_name: str, persist_directory: str | Path = ".rag_ai/chroma") -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Chroma requires the optional 'chroma' extra: pip install -e .[chroma]") from exc

        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[_metadata_for_chroma(chunk) for chunk in chunks],
        )

    def similarity_search(
        self,
        query_embedding: list[float],
        *,
        k: int = 4,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        query: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            query["where"] = filters
        response = self.collection.query(**query)

        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        ids = response.get("ids", [[]])[0]

        results: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            clean_metadata = dict(metadata or {})
            document_id = str(clean_metadata.pop("document_id", ""))
            chunk = Chunk(id=chunk_id, text=text, document_id=document_id, metadata=clean_metadata)
            results.append(RetrievedChunk(chunk=chunk, score=1.0 / (1.0 + float(distance))))
        return results


def _matches_filters(metadata: dict[str, object], filters: dict[str, object] | None) -> bool:
    if not filters:
        return True
    return all(metadata.get(key) == value for key, value in filters.items())


def _metadata_for_chroma(chunk: Chunk) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {"document_id": chunk.document_id}
    for key, value in chunk.metadata.items():
        if isinstance(value, str | int | float | bool):
            metadata[key] = value
        elif value is not None:
            metadata[key] = str(value)
    return metadata
