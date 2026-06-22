"""Text embeddings via sentence-transformers.

Wrapped in a thin class so the model is loaded once and reused, and so the rest
of the pipeline depends on a tiny interface (``embed`` / ``embed_one``) rather
than on sentence-transformers directly.
"""

from __future__ import annotations

import numpy as np


class Embedder:
    """Lazily-loaded sentence-transformers embedder.

    The model is downloaded on first use and cached by sentence-transformers in
    the usual HuggingFace cache. Embeddings are L2-normalised so that a dot
    product equals cosine similarity downstream.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None  # loaded on first use

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into a (n, dim) float32 matrix."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text into a (dim,) float32 vector."""
        return self.embed([text])[0]
