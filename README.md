# R.A.G AI

A learning-oriented Retrieval-Augmented Generation framework in Python. It follows the shape of
the RAG From Scratch curriculum while exposing reusable modules, a CLI, and tests.

It also includes a local comparison dashboard that runs the same Transformers model with and without
retrieved document context, making the effect of RAG directly inspectable.

## What is included

- Local `.txt`, `.md`, and optional `.pdf` ingestion
- Configurable text chunking with metadata propagation
- Provider-agnostic embedding, vector-store, reranking, and LLM interfaces
- In-memory vector store for tests and examples
- Optional Chroma, OpenAI, and SentenceTransformers adapters
- Core RAG pipeline with grounded prompts, citations, and no-context behavior
- Optional query transformations: multi-query, RAG-fusion, decomposition, step-back, and HyDE
- Simple routing, reranking, and retrieval evaluation utilities
- Streamlit comparison dashboard with local Hugging Face generation and embeddings
- Side-by-side latency, token usage, citations, and retrieved-passage evidence

## Quick start

```powershell
python -m pip install -e .[dev,chroma]
rag-ai ingest .\docs --collection demo
rag-ai ask "What is this project?" --collection demo
```

The default CLI path uses deterministic local hash embeddings and an extractive local LLM, so it
does not require API keys. Use `--store memory` for tests and one-off commands that do not need
persistence across processes.

## Local RAG comparison dashboard

Install the dashboard dependencies and launch Streamlit:

```powershell
python -m pip install -e .[demo]
streamlit run streamlit_app.py
```

Use the bundled fictional Project Aurora corpus for an immediate demonstration, or upload `.txt`,
`.md`, `.markdown`, and `.pdf` files. Uploaded content and its in-memory vector index exist only in
the current Streamlit session. The dashboard does not write uploads to disk.

The first run downloads the default public generation and embedding models from Hugging Face.
Later runs reuse the local Hugging Face cache. The model fields also accept a local
Transformers-compatible directory, including a model produced by a future fine-tuning workflow.
Inference runs locally; larger models may require a CUDA GPU and more memory.

The comparison is controlled: both sides use the same Transformers model and settings. The baseline sees
only the question, while the RAG answer receives retrieved passages and a citation requirement.
Latency and token counts are diagnostic rather than quality scores. RAG is most useful for
corpus-specific questions and can perform worse when retrieval is irrelevant or incomplete.

## Optional adapters

```powershell
python -m pip install -e .[chroma,pdf,openai,local]
```

Set `OPENAI_API_KEY` before using the OpenAI adapters.

OpenAI remains an optional framework adapter, but it is not used by the local dashboard and no API
key is required for public Hugging Face models.

## Development

```powershell
python -m pytest
```
