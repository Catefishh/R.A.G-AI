from rag_ai import Document, HashEmbeddingModel, InMemoryVectorStore, RecursiveCharacterTextSplitter, VectorRetriever
from rag_ai.eval import EvalExample, evaluate
from rag_ai.llms import StaticLLM
from rag_ai.query_transformers import (
    DecompositionTransformer,
    HyDETransformer,
    MultiQueryTransformer,
    StaticQueryTransformer,
    StepBackTransformer,
)
from rag_ai.retrievers import RAGFusionRetriever
from rag_ai.routing import KeywordRouter, Route, RoutedRetriever


def _retriever_for(text: str, *, topic: str = "general") -> VectorRetriever:
    chunks = RecursiveCharacterTextSplitter(chunk_size=200).split_documents(
        [Document(text=text, metadata={"topic": topic, "source": f"{topic}.txt"})]
    )
    embeddings = HashEmbeddingModel(dimensions=64)
    store = InMemoryVectorStore()
    store.add(chunks, embeddings.embed_documents([chunk.text for chunk in chunks]))
    return VectorRetriever(store, embeddings)


def test_query_transformers_parse_llm_lines() -> None:
    llm = StaticLLM("1. alpha query\n- beta query\n")

    assert MultiQueryTransformer(llm).transform("x") == ["alpha query", "beta query"]
    assert DecompositionTransformer(llm).transform("x") == ["alpha query", "beta query"]
    assert StepBackTransformer(StaticLLM("broader question")).transform("x") == ["broader question"]
    assert HyDETransformer(StaticLLM("hypothetical answer")).transform("x") == ["hypothetical answer"]


def test_rag_fusion_combines_transformed_queries() -> None:
    retriever = _retriever_for("Apples are fruit. Bananas are yellow fruit.")
    fusion = RAGFusionRetriever(
        retriever,
        StaticQueryTransformer(["banana fruit", "apple fruit"]),
    )

    results = fusion.retrieve("fruit", k=2)

    assert results
    assert results[0].score > 0


def test_keyword_router_selects_matching_retriever() -> None:
    python = _retriever_for("Python uses dataclasses.", topic="python")
    storage = _retriever_for("Chroma persists embeddings.", topic="storage")
    routed = RoutedRetriever(
        KeywordRouter(
            [Route(name="storage", retriever=storage, keywords={"chroma", "embeddings"})],
            default=python,
        )
    )

    result = routed.retrieve("How does Chroma persist embeddings?", k=1)[0]

    assert result.chunk.metadata["topic"] == "storage"


def test_evaluate_counts_source_and_answer_hits() -> None:
    class System:
        def ask(self, question: str, *, k: int = 4, filters: dict[str, object] | None = None):
            from rag_ai.models import Chunk, RAGAnswer, RetrievedChunk

            chunk = Chunk(text="RAG answer", document_id="doc", metadata={"source": "guide.md"})
            return RAGAnswer(question=question, answer="The answer mentions RAG.", sources=[RetrievedChunk(chunk, 1.0)])

    result = evaluate(
        System(),
        [EvalExample(question="What?", expected_source="guide", expected_answer_contains="RAG")],
    )

    assert result.source_hit_rate == 1.0
    assert result.answer_hit_rate == 1.0
