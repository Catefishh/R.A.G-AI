from __future__ import annotations

from rag_ai.interfaces import LLM, Reranker, Retriever
from rag_ai.llms import generate_with_metrics
from rag_ai.models import GenerationMetrics, RAGAnswer
from rag_ai.prompts import INSUFFICIENT_CONTEXT, build_grounded_prompt
from rag_ai.rerankers import NoOpReranker


class RAGPipeline:
    def __init__(
        self,
        *,
        retriever: Retriever,
        llm: LLM,
        reranker: Reranker | None = None,
        min_score: float | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker or NoOpReranker()
        self.min_score = min_score

    def ask(self, question: str, *, k: int = 4, filters: dict[str, object] | None = None) -> RAGAnswer:
        retrieved = self.retriever.retrieve(question, k=k, filters=filters)
        if self.min_score is not None:
            retrieved = [item for item in retrieved if item.score >= self.min_score]
        retrieved = self.reranker.rerank(question, retrieved, k=k)

        if not retrieved:
            return RAGAnswer(
                question=question,
                answer=INSUFFICIENT_CONTEXT,
                sources=[],
                metadata={
                    "reason": "no_context",
                    "generation_metrics": GenerationMetrics(elapsed_ms=0.0),
                },
            )

        prompt = build_grounded_prompt(question, retrieved)
        generation = generate_with_metrics(self.llm, prompt, context=retrieved)
        return RAGAnswer(
            question=question,
            answer=generation.text,
            sources=retrieved,
            metadata={
                "prompt": prompt,
                "source_count": len(retrieved),
                "generation_metrics": generation.metrics,
            },
        )
