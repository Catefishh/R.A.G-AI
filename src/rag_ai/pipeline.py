"""The high-level RAG pipeline and the side-by-side comparison entry point.

This module is what the Streamlit app and the CLI talk to. It builds the
knowledge base once, then can answer any question two ways:

* :meth:`RAGPipeline.answer_plain` — the model alone (no retrieval).
* :meth:`RAGPipeline.answer_rag`   — retrieval-augmented.

:meth:`RAGPipeline.compare` runs both and returns them together so the UI can
render them next to each other.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import Config
from .documents import chunk_documents, load_documents
from .embeddings import Embedder
from .llm import LocalLLM
from .prompts import build_plain_messages, build_rag_messages
from .retriever import RetrievedChunk, Retriever
from .vector_store import VectorStore


@dataclass
class Answer:
    """One model response, with the metadata the UI needs to explain it."""

    question: str
    text: str
    used_rag: bool
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    elapsed_s: float = 0.0


@dataclass
class Comparison:
    plain: Answer
    rag: Answer


class RAGPipeline:
    def __init__(self, config: Config | None = None, llm: LocalLLM | None = None) -> None:
        self.config = config or Config()
        self.embedder = Embedder(self.config.embedding_model)
        self.store = VectorStore()
        self.retriever = Retriever(
            self.embedder,
            self.store,
            top_k=self.config.top_k,
            min_score=self.config.min_score,
        )
        # Shared by both pipelines so the comparison isolates the effect of RAG.
        self.llm = llm or LocalLLM(
            model_name=self.config.llm_model,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        self._indexed = False

    # --- Index building ----------------------------------------------------
    def build_index(self) -> int:
        """Load, chunk, embed and store the demo documents. Returns chunk count."""
        documents = load_documents(self.config.docs_dir)
        chunks = chunk_documents(
            documents, self.config.chunk_size, self.config.chunk_overlap
        )
        if not chunks:
            raise RuntimeError(
                f"No chunks produced from {self.config.docs_dir}. Add documents first."
            )
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, vectors)
        self._indexed = True
        return len(chunks)

    @property
    def is_indexed(self) -> bool:
        return self._indexed and not self.store.is_empty

    # --- Answering ---------------------------------------------------------
    def answer_plain(self, question: str) -> Answer:
        start = time.perf_counter()
        text = self.llm.generate(build_plain_messages(question))
        return Answer(
            question=question,
            text=text,
            used_rag=False,
            elapsed_s=time.perf_counter() - start,
        )

    def answer_rag(self, question: str) -> Answer:
        start = time.perf_counter()
        retrieved = self.retriever.retrieve(question)
        text = self.llm.generate(build_rag_messages(question, retrieved))
        return Answer(
            question=question,
            text=text,
            used_rag=True,
            retrieved=retrieved,
            elapsed_s=time.perf_counter() - start,
        )

    def compare(self, question: str) -> Comparison:
        """Answer the same question with and without RAG."""
        return Comparison(
            plain=self.answer_plain(question),
            rag=self.answer_rag(question),
        )
