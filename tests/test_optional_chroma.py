from pathlib import Path

import pytest

from rag_ai import ChromaVectorStore, Document, HashEmbeddingModel, RecursiveCharacterTextSplitter


def test_chroma_persists_collection(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")

    chunks = RecursiveCharacterTextSplitter(chunk_size=200).split_documents(
        [Document(text="Chroma persists local RAG embeddings.", metadata={"source": "chroma.md"})]
    )
    embeddings = HashEmbeddingModel(dimensions=64)
    vectors = embeddings.embed_documents([chunk.text for chunk in chunks])

    first = ChromaVectorStore(collection_name="test", persist_directory=tmp_path)
    first.add(chunks, vectors)

    second = ChromaVectorStore(collection_name="test", persist_directory=tmp_path)
    results = second.similarity_search(embeddings.embed_query("local embeddings"), k=1)

    assert results
    assert results[0].chunk.metadata["source"] == "chroma.md"
