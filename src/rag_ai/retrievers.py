from __future__ import annotations

from collections import defaultdict

from rag_ai.interfaces import EmbeddingModel, QueryTransformer, Retriever, VectorStore
from rag_ai.models import RetrievedChunk


class VectorRetriever:
    def __init__(self, store: VectorStore, embeddings: EmbeddingModel) -> None:
        self.store = store
        self.embeddings = embeddings

    def retrieve(
        self,
        question: str,
        *,
        k: int = 4,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        return self.store.similarity_search(self.embeddings.embed_query(question), k=k, filters=filters)


class RAGFusionRetriever:
    def __init__(
        self,
        retriever: Retriever,
        transformer: QueryTransformer,
        *,
        reciprocal_rank_k: int = 60,
    ) -> None:
        self.retriever = retriever
        self.transformer = transformer
        self.reciprocal_rank_k = reciprocal_rank_k

    def retrieve(
        self,
        question: str,
        *,
        k: int = 4,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        scores: dict[str, float] = defaultdict(float)
        chunks: dict[str, RetrievedChunk] = {}
        queries = [question, *self.transformer.transform(question)]
        for query in dict.fromkeys(queries):
            for rank, result in enumerate(self.retriever.retrieve(query, k=k, filters=filters), start=1):
                chunks[result.chunk.id] = result
                scores[result.chunk.id] += 1.0 / (self.reciprocal_rank_k + rank)

        fused = [
            RetrievedChunk(chunk=chunks[chunk_id].chunk, score=score)
            for chunk_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ]
        return fused[:k]
