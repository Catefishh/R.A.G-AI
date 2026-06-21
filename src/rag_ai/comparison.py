from __future__ import annotations

from rag_ai.llms import generate_with_metrics
from rag_ai.models import ComparisonAnswer, ComparisonResult, GenerationMetrics
from rag_ai.pipeline import RAGPipeline
from rag_ai.prompts import build_baseline_prompt


class ComparisonRunner:
    """Run a fair same-model comparison with and without retrieved context."""

    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline

    def compare(
        self,
        question: str,
        *,
        k: int = 4,
        filters: dict[str, object] | None = None,
    ) -> ComparisonResult:
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")

        baseline = generate_with_metrics(
            self.pipeline.llm,
            build_baseline_prompt(question),
            context=None,
        )
        rag = self.pipeline.ask(question, k=k, filters=filters)
        metrics = rag.metadata.get("generation_metrics")
        if not isinstance(metrics, GenerationMetrics):
            metrics = GenerationMetrics(elapsed_ms=0.0)

        return ComparisonResult(
            question=question,
            without_rag=ComparisonAnswer(
                answer=baseline.text,
                metrics=baseline.metrics,
                metadata={"mode": "without_rag"},
            ),
            with_rag=ComparisonAnswer(
                answer=rag.answer,
                metrics=metrics,
                sources=rag.sources,
                metadata={**rag.metadata, "mode": "with_rag"},
            ),
        )
