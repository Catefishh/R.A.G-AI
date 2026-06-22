"""Tests for the RAG framework's deterministic, model-free components.

These tests deliberately avoid downloading any ML models: chunking, the vector
store's cosine search, and the retriever are exercised with a tiny fake embedder
so the suite runs in milliseconds and works offline.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_ai.documents import Document, chunk_documents, _split_text  # noqa: E402
from rag_ai.vector_store import VectorStore  # noqa: E402
from rag_ai.retriever import Retriever  # noqa: E402


# --- Chunking ---------------------------------------------------------------
def test_split_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = list(_split_text(text, chunk_size=100, overlap=20))
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    # No chunk should be empty.
    assert all(c.strip() for c in chunks)


def test_split_does_not_cut_words():
    text = "alpha beta gamma delta epsilon zeta eta theta"
    for chunk in _split_text(text, chunk_size=20, overlap=5):
        # Each emitted chunk should consist of whole words from the source.
        for token in chunk.split():
            assert token in text.split()


def test_chunk_documents_sets_metadata():
    docs = [Document(source="a.md", text="x " * 500)]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
    assert chunks
    assert chunks[0].source == "a.md"
    assert chunks[0].index == 0
    assert chunks[0].metadata["source"] == "a.md"


def test_empty_text_produces_no_chunks():
    assert list(_split_text("", 100, 10)) == []


# --- Vector store -----------------------------------------------------------
class FakeEmbedder:
    """Deterministic 3-d embedder: maps known words to fixed axes."""

    dimension = 3

    _vocab = {"cat": [1, 0, 0], "dog": [0, 1, 0], "fish": [0, 0, 1]}

    def _vec(self, text):
        v = np.zeros(3, dtype=np.float32)
        for word, axis in self._vocab.items():
            if word in text.lower():
                v += np.array(axis, dtype=np.float32)
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed(self, texts):
        return np.array([self._vec(t) for t in texts], dtype=np.float32)

    def embed_one(self, text):
        return self._vec(text)


def _make_store():
    from rag_ai.documents import Chunk

    embedder = FakeEmbedder()
    chunks = [
        Chunk(text="the cat sat", source="s", index=0),
        Chunk(text="a dog barked", source="s", index=1),
        Chunk(text="one fish two fish", source="s", index=2),
    ]
    store = VectorStore()
    store.add(chunks, embedder.embed([c.text for c in chunks]))
    return embedder, store


def test_search_returns_most_similar_first():
    embedder, store = _make_store()
    results = store.search(embedder.embed_one("cat"), top_k=3)
    assert results[0][0].text == "the cat sat"
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_search_on_empty_store_returns_empty():
    assert VectorStore().search(np.zeros(3), top_k=3) == []


def test_save_and_load_roundtrip(tmp_path):
    _, store = _make_store()
    path = tmp_path / "index"
    store.save(path)
    loaded = VectorStore.load(path)
    assert len(loaded) == len(store)
    assert loaded._chunks[0].text == store._chunks[0].text


# --- Retriever --------------------------------------------------------------
def test_retriever_filters_below_min_score():
    embedder, store = _make_store()
    # "cat" matches one chunk strongly; a high threshold drops the rest.
    retriever = Retriever(embedder, store, top_k=3, min_score=0.5)
    results = retriever.retrieve("cat")
    assert len(results) == 1
    assert results[0].source == "s"
    assert results[0].score >= 0.5


def test_retriever_respects_top_k():
    embedder, store = _make_store()
    retriever = Retriever(embedder, store, top_k=2, min_score=-1.0)
    assert len(retriever.retrieve("cat dog fish")) == 2
