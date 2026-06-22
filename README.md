# RAG vs No-RAG — a visual demonstration

A small, **fully local** Retrieval-Augmented Generation (RAG) framework built to
show — visually and side by side — how retrieval changes what an AI model can
answer.

The same model answers each question twice:

| Without RAG | With RAG |
|-------------|----------|
| The model answers from its training only. For knowledge it never saw, it guesses, hallucinates, or admits it doesn't know. | The same model is given relevant passages retrieved from a knowledge base, and answers correctly — with citations. |

The knowledge base describes a **fictional company, "Nimbus Forge"**, that exists
nowhere in any training data. That is the whole trick: it makes the comparison
honest. Any correct, specific answer on the right can *only* have come from
retrieval.

## Why fully local?

No OpenAI key, no API calls, no data leaving your machine. Generation runs on a
small HuggingFace instruct model via `transformers`; embeddings use
`sentence-transformers`; the vector store is a transparent NumPy cosine search
you can read end to end. The first run downloads the models (~3 GB for the
default LLM); everything after is offline.

## Quick start

```bash
pip install -r requirements.txt

# Option A — the visual web app (recommended)
streamlit run app.py

# Option B — terminal comparison
python compare_cli.py "How much does the Aurelia Team edition cost?"
python compare_cli.py            # interactive loop
```

The first launch downloads the embedding model and the LLM, then builds the
knowledge base. Subsequent runs are fast and offline.

## What you'll see

Ask something like *"What does the AUR-429 error mean and how do I fix it?"*:

- **Without RAG** the model has never heard of Aurelia or its error codes, so it
  makes something up or declines.
- **With RAG** it retrieves the relevant FAQ chunk, explains that AUR-429 is a
  rate-limit error, gives the fix, and cites `03_support_faq.md`.

The web app also shows the **retrieved chunks and their similarity scores**, so
you can see exactly what the model was given.

## How it works

```
documents → chunk → embed → vector store
                                  │
question ─────────────────────────┤ (RAG only) retrieve top-k relevant chunks
                                  ▼
              ┌─────────────────────────────────────┐
              │  same local LLM, same settings       │
              │  • plain prompt   → "Without RAG"     │
              │  • prompt + context → "With RAG"      │
              └─────────────────────────────────────┘
```

Every stage is a small, separate module under `src/rag_ai/`:

| Module | Responsibility |
|--------|----------------|
| `config.py` | All tunable settings (env-overridable) |
| `documents.py` | Load `.txt`/`.md`/`.pdf`, chunk with overlap |
| `embeddings.py` | sentence-transformers embedder |
| `vector_store.py` | NumPy cosine-similarity store (save/load) |
| `retriever.py` | Embed query, search, threshold by score |
| `llm.py` | Local transformers generation |
| `prompts.py` | The two prompts being compared |
| `pipeline.py` | Build index, answer plain vs RAG, `compare()` |

## Configuration

Override any default with an environment variable:

| Variable | Default | Meaning |
|----------|---------|---------|
| `RAG_LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Generation model |
| `RAG_EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `RAG_CHUNK_SIZE` | `600` | Chunk size (chars) |
| `RAG_CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `RAG_TOP_K` | `4` | Chunks retrieved per query |
| `RAG_MIN_SCORE` | `0.15` | Minimum cosine similarity to keep |
| `RAG_DOCS_DIR` | `examples/demo_docs` | Knowledge base directory |

On a low-memory machine, try a smaller model:

```bash
RAG_LLM_MODEL="Qwen/Qwen2.5-0.5B-Instruct" streamlit run app.py
```

## Use your own knowledge base

Drop `.txt`, `.md`, or `.pdf` files into `examples/demo_docs/` (or point
`RAG_DOCS_DIR` elsewhere) and restart. The index rebuilds automatically.

## Tests

```bash
pytest tests/
```

The suite covers chunking, the vector store, and retrieval with a tiny fake
embedder, so it runs in milliseconds and needs no model downloads.

## License

See [LICENSE](LICENSE).
