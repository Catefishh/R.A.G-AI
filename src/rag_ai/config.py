"""Central configuration for the RAG framework.

All tunable knobs live here so the demo, tests, and library share one source of
truth. Everything is overridable via environment variables to make the Streamlit
app and CI easy to configure without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    """Runtime configuration.

    Defaults target a fully-local, CPU-only setup: a small sentence-transformers
    embedder and a small instruct LLM from HuggingFace. No external API keys are
    required.
    """

    # --- Embeddings ---
    embedding_model: str = field(
        default_factory=lambda: _env("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")
    )

    # --- Generation (local transformers model) ---
    llm_model: str = field(
        default_factory=lambda: _env("RAG_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    )
    max_new_tokens: int = field(
        default_factory=lambda: _env_int("RAG_MAX_NEW_TOKENS", 320)
    )
    temperature: float = field(
        default_factory=lambda: _env_float("RAG_TEMPERATURE", 0.2)
    )

    # --- Chunking ---
    chunk_size: int = field(default_factory=lambda: _env_int("RAG_CHUNK_SIZE", 600))
    chunk_overlap: int = field(
        default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 100)
    )

    # --- Retrieval ---
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 4))
    # Chunks below this cosine similarity are treated as irrelevant and dropped.
    min_score: float = field(default_factory=lambda: _env_float("RAG_MIN_SCORE", 0.15))

    # --- Paths ---
    docs_dir: Path = field(
        default_factory=lambda: Path(
            _env("RAG_DOCS_DIR", "examples/demo_docs")
        )
    )

    def __post_init__(self) -> None:
        # Normalise to a Path even if a string was passed in directly.
        self.docs_dir = Path(self.docs_dir)
