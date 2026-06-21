from rag_ai import (
    ComparisonRunner,
    Document,
    GenerationMetrics,
    GenerationResult,
    HashEmbeddingModel,
    InMemoryVectorStore,
    RAGPipeline,
    RecursiveCharacterTextSplitter,
    VectorRetriever,
)


class RecordingLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def generate(self, prompt: str, *, context=None) -> str:
        return self.generate_detailed(prompt, context=context).text

    def generate_detailed(self, prompt: str, *, context=None) -> GenerationResult:
        self.calls.append((prompt, context))
        return GenerationResult(
            text="grounded [1]" if context else "baseline",
            metrics=GenerationMetrics(elapsed_ms=12.5, input_tokens=10, output_tokens=2),
        )


def _runner(llm: RecordingLLM) -> ComparisonRunner:
    embeddings = HashEmbeddingModel(dimensions=64)
    chunks = RecursiveCharacterTextSplitter(chunk_size=200).split_documents(
        [Document(text="The private launch code is ORBIT-42.", metadata={"source": "secret.md"})]
    )
    store = InMemoryVectorStore()
    store.add(chunks, embeddings.embed_documents([chunk.text for chunk in chunks]))
    return ComparisonRunner(RAGPipeline(retriever=VectorRetriever(store, embeddings), llm=llm))


def test_comparison_uses_same_model_but_baseline_has_no_context() -> None:
    llm = RecordingLLM()

    result = _runner(llm).compare("What is the private launch code?", k=1)

    assert len(llm.calls) == 2
    baseline_prompt, baseline_context = llm.calls[0]
    rag_prompt, rag_context = llm.calls[1]
    assert baseline_context is None
    assert "ORBIT-42" not in baseline_prompt
    assert rag_context and "ORBIT-42" in rag_prompt
    assert result.without_rag.answer == "baseline"
    assert result.with_rag.answer == "grounded [1]"
    assert result.with_rag.sources[0].chunk.metadata["source"] == "secret.md"
    assert result.with_rag.metrics.input_tokens == 10


def test_comparison_rejects_empty_question() -> None:
    try:
        _runner(RecordingLLM()).compare("   ")
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected an empty question to fail")
