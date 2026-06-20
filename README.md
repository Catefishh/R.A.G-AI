# R.A.G AI

A learning-oriented Retrieval-Augmented Generation framework in Python. It follows the shape of
the RAG From Scratch curriculum while exposing reusable modules, a CLI, and tests.

## What is included

- Local `.txt`, `.md`, and optional `.pdf` ingestion
- Configurable text chunking with metadata propagation
- Provider-agnostic embedding, vector-store, reranking, and LLM interfaces
- In-memory vector store for tests and examples
- Optional Chroma, OpenAI, and SentenceTransformers adapters
- Core RAG pipeline with grounded prompts, citations, and no-context behavior
- Optional query transformations: multi-query, RAG-fusion, decomposition, step-back, and HyDE
- Simple routing, reranking, and retrieval evaluation utilities

## Quick start

```powershell
python -m pip install -e .[dev,chroma]
rag-ai ingest .\docs --collection demo
rag-ai ask "What is this project?" --collection demo
```

The default CLI path uses deterministic local hash embeddings and an extractive local LLM, so it
does not require API keys. Use `--store memory` for tests and one-off commands that do not need
persistence across processes.

## Optional adapters

```powershell
python -m pip install -e .[chroma,pdf,openai,local]
```

Set `OPENAI_API_KEY` before using the OpenAI adapters.

## Development

```powershell
python -m pytest
```
