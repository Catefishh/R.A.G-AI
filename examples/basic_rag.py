from rag_ai import (
    Document,
    HashEmbeddingModel,
    InMemoryVectorStore,
    RAGPipeline,
    RecursiveCharacterTextSplitter,
    VectorRetriever,
)
from rag_ai.llms import ExtractiveLLM


documents = [
    Document(
        text="RAG combines retrieval with generation. Documents are chunked, embedded, retrieved, and used as context.",
        metadata={"source": "example"},
    )
]

splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
chunks = splitter.split_documents(documents)

embeddings = HashEmbeddingModel()
store = InMemoryVectorStore()
store.add(chunks, embeddings.embed_documents([chunk.text for chunk in chunks]))

pipeline = RAGPipeline(
    retriever=VectorRetriever(store, embeddings),
    llm=ExtractiveLLM(),
)

answer = pipeline.ask("What does RAG combine?")
print(answer.answer)
print([source.chunk.metadata for source in answer.sources])
