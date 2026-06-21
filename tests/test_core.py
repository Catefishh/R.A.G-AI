from pathlib import Path

from rag_ai import (
    Document,
    HashEmbeddingModel,
    InMemoryVectorStore,
    RAGPipeline,
    RecursiveCharacterTextSplitter,
    VectorRetriever,
    load_path,
)
from rag_ai.loaders import load_bytes
from rag_ai.llms import ExtractiveLLM
from rag_ai.prompts import INSUFFICIENT_CONTEXT
from rag_ai.rerankers import KeywordOverlapReranker


def test_text_loader_adds_metadata(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("# RAG\nRetrieval augmented generation.", encoding="utf-8")

    documents = load_path(path)

    assert len(documents) == 1
    assert documents[0].text.startswith("# RAG")
    assert documents[0].metadata["file_type"] == "md"
    assert documents[0].metadata["source"] == str(path)


def test_directory_loader_skips_unsupported_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.bin").write_bytes(b"\x00")

    documents = load_path(tmp_path)

    assert [document.text for document in documents] == ["alpha"]


def test_byte_loader_rejects_unsupported_and_empty_files() -> None:
    assert load_bytes("empty.txt", b"  \n") == []
    try:
        load_bytes("data.csv", b"a,b")
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:
        raise AssertionError("Expected unsupported upload to fail")


def test_splitter_overlaps_and_propagates_metadata() -> None:
    document = Document(text="alpha beta gamma delta epsilon zeta", metadata={"source": "memory"})
    splitter = RecursiveCharacterTextSplitter(chunk_size=18, chunk_overlap=5)

    chunks = splitter.split_documents([document])

    assert len(chunks) > 1
    assert all(chunk.document_id == document.id for chunk in chunks)
    assert chunks[0].metadata["source"] == "memory"
    assert chunks[0].metadata["chunk_index"] == 0


def test_in_memory_retrieval_with_metadata_filter() -> None:
    docs = [
        Document(text="Python has dataclasses.", metadata={"topic": "python"}),
        Document(text="Chroma stores vectors.", metadata={"topic": "storage"}),
    ]
    chunks = RecursiveCharacterTextSplitter(chunk_size=100).split_documents(docs)
    embeddings = HashEmbeddingModel(dimensions=64)
    store = InMemoryVectorStore()
    store.add(chunks, embeddings.embed_documents([chunk.text for chunk in chunks]))

    retriever = VectorRetriever(store, embeddings)
    results = retriever.retrieve("vectors", filters={"topic": "storage"})

    assert len(results) == 1
    assert results[0].chunk.metadata["topic"] == "storage"


def test_pipeline_returns_cited_answer() -> None:
    document = Document(text="RAG combines retrieval with generation.", metadata={"source": "doc.txt"})
    splitter = RecursiveCharacterTextSplitter(chunk_size=100)
    chunks = splitter.split_documents([document])
    embeddings = HashEmbeddingModel(dimensions=64)
    store = InMemoryVectorStore()
    store.add(chunks, embeddings.embed_documents([chunk.text for chunk in chunks]))
    pipeline = RAGPipeline(
        retriever=VectorRetriever(store, embeddings),
        llm=ExtractiveLLM(),
        reranker=KeywordOverlapReranker(),
    )

    answer = pipeline.ask("What does RAG combine?")

    assert "retrieval" in answer.answer.lower()
    assert "[1]" in answer.answer
    assert answer.sources
    assert answer.metadata["generation_metrics"].elapsed_ms >= 0


def test_pipeline_handles_no_context() -> None:
    pipeline = RAGPipeline(
        retriever=VectorRetriever(InMemoryVectorStore(), HashEmbeddingModel()),
        llm=ExtractiveLLM(),
    )

    answer = pipeline.ask("Anything?")

    assert answer.answer == INSUFFICIENT_CONTEXT
    assert answer.metadata["reason"] == "no_context"
