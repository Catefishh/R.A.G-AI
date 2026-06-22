"""A minimal in-memory vector store backed by NumPy.

There is deliberately no FAISS/Chroma dependency: for a demo knowledge base of a
few hundred chunks, a brute-force cosine search over a NumPy matrix is instant
and completely transparent — you can read exactly how retrieval works. The store
can be saved to / loaded from a single ``.npz`` file plus a JSON sidecar.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .documents import Chunk


class VectorStore:
    """Stores chunk embeddings and supports cosine-similarity search.

    Embeddings are assumed to be L2-normalised (see :class:`Embedder`), so a
    dot product is the cosine similarity. Search therefore reduces to a single
    matrix-vector product followed by a top-k selection.
    """

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None  # (n, dim), normalised
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def is_empty(self) -> bool:
        return len(self._chunks) == 0

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if len(chunks) == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        """Return the top_k (chunk, score) pairs by cosine similarity."""
        if self.is_empty or self._vectors is None:
            return []
        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        scores = self._vectors @ query_vector  # cosine sim (vectors normalised)
        top_k = min(top_k, len(self._chunks))
        # argpartition for the top_k, then sort just those for correct ordering.
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self._chunks[i], float(scores[i])) for i in idx]

    # --- Persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path.with_suffix(".npz"), vectors=self._vectors)
        meta = [
            {
                "text": c.text,
                "source": c.source,
                "index": c.index,
                "metadata": c.metadata,
            }
            for c in self._chunks
        ]
        path.with_suffix(".json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        path = Path(path)
        store = cls()
        data = np.load(path.with_suffix(".npz"))
        store._vectors = data["vectors"]
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        store._chunks = [
            Chunk(
                text=m["text"],
                source=m["source"],
                index=m["index"],
                metadata=m.get("metadata", {}),
            )
            for m in meta
        ]
        return store
