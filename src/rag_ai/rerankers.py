from __future__ import annotations

from rag_ai.embeddings import TOKEN_RE
from rag_ai.models import RetrievedChunk


class NoOpReranker:
    def rerank(self, question: str, chunks: list[RetrievedChunk], *, k: int | None = None) -> list[RetrievedChunk]:
        return chunks[:k] if k is not None else chunks


class KeywordOverlapReranker:
    def rerank(self, question: str, chunks: list[RetrievedChunk], *, k: int | None = None) -> list[RetrievedChunk]:
        question_terms = {token.lower() for token in TOKEN_RE.findall(question)}
        reranked: list[RetrievedChunk] = []
        for item in chunks:
            chunk_terms = {token.lower() for token in TOKEN_RE.findall(item.chunk.text)}
            overlap = len(question_terms & chunk_terms)
            reranked.append(RetrievedChunk(chunk=item.chunk, score=item.score + overlap))
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:k] if k is not None else reranked
