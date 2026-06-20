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
