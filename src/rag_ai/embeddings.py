from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from os import getenv


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class HashEmbeddingModel:
    """Deterministic local bag-of-words hashing embeddings for tests and demos."""

    def __init__(self, *, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        counts = Counter(token.lower() for token in TOKEN_RE.findall(text))
        for token, count in counts.items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign * float(count)
        return _normalize(vector)


class SentenceTransformersEmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "SentenceTransformers requires the optional 'local' extra: pip install -e .[local]"
            ) from exc
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return [embedding.tolist() for embedding in embeddings]


class OpenAIEmbeddingModel:
    def __init__(self, *, model: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI embeddings require the optional 'openai' extra: pip install -e .[openai]") from exc
        self.client = OpenAI(api_key=api_key or getenv("OPENAI_API_KEY"))
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimensions")
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    if denominator == 0:
        return 0.0
    return sum(x * y for x, y in zip(left, right)) / denominator


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
