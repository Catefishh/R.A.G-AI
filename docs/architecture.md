# Architecture

The framework is intentionally small and explicit. The core pipeline is:

1. Load documents into normalized `Document` objects.
2. Split documents into `Chunk` objects.
3. Embed chunks and write them to a `VectorStore`.
4. Retrieve top-k chunks for a question.
5. Assemble a grounded prompt from retrieved context.
6. Generate a structured `RAGAnswer` with citations.

Each stage is behind a protocol in `rag_ai.interfaces`, so new loaders, embedding models,
vector stores, rerankers, and LLMs can be added without changing the pipeline.

The default implementation is local and deterministic:

- `HashEmbeddingModel` for embeddings
- `InMemoryVectorStore` for storage
- `ExtractiveLLM` for answer generation

Optional adapters wrap Chroma, OpenAI, and SentenceTransformers when those packages are installed.

## Comparison dashboard

The local dashboard adds a thin Streamlit layer over reusable framework services:

1. `SentenceTransformersEmbeddingModel` embeds uploaded or bundled documents locally.
2. `InMemoryVectorStore` holds the session index; uploaded bytes are never persisted by the app.
3. `ComparisonRunner` sends a neutral question-only prompt to the pipeline's LLM for the baseline.
4. The same LLM instance then runs `RAGPipeline.ask`, which retrieves context and builds the
   grounded prompt.
5. `ComparisonResult` returns both answer runs with timing, available token counts, citations,
   retrieval scores, and source metadata.

`LLM.generate()` remains the compatibility contract. Adapters can optionally implement
`generate_detailed()` to return provider-specific token metrics; otherwise the shared wrapper still
records wall-clock latency. This keeps comparison logic independent of Streamlit and allows another
UI or API to reuse it.

The dashboard defaults to `Qwen/Qwen2.5-0.5B-Instruct` for generation and
`sentence-transformers/all-MiniLM-L6-v2` for embeddings. Both fields accept either a Hugging Face
Hub model ID or a compatible local directory. `TransformersLLM` disables arbitrary remote model
code, chooses CUDA when available, and otherwise falls back to CPU. Training and fine-tuning are
intentionally outside v1, but their saved model directories can be loaded without application
changes.

### Failure behavior

- Model loading reports invalid IDs or paths and insufficient-memory failures without replacing a
  working session index.
- Empty and unsupported uploads are rejected before indexing.
- A failed replacement index does not overwrite the last working index in session state.
- Generation failures are shown without deleting indexed documents.

The side-by-side result is evidence, not an automatic quality judgment. A stronger RAG answer on a
private corpus demonstrates grounding, while general-knowledge questions may favor the baseline.
