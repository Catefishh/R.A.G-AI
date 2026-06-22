"""rag_ai — a small, transparent Retrieval-Augmented Generation framework.

The package is intentionally dependency-light and readable so it can double as a
teaching tool: every stage of the RAG pipeline (load -> chunk -> embed -> store
-> retrieve -> generate) lives in its own module with no hidden magic.
"""

from .config import Config
from .documents import Document, Chunk, load_documents, chunk_documents
from .embeddings import Embedder
from .vector_store import VectorStore
from .retriever import Retriever, RetrievedChunk
from .llm import LocalLLM
from .pipeline import RAGPipeline, Answer

__all__ = [
    "Config",
    "Document",
    "Chunk",
    "load_documents",
    "chunk_documents",
    "Embedder",
    "VectorStore",
    "Retriever",
    "RetrievedChunk",
    "LocalLLM",
    "RAGPipeline",
    "Answer",
]

__version__ = "0.1.0"
