from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_ai.embeddings import HashEmbeddingModel
from rag_ai.eval import evaluate, load_eval_csv
from rag_ai.llms import ExtractiveLLM
from rag_ai.loaders import load_path
from rag_ai.pipeline import RAGPipeline
from rag_ai.retrievers import VectorRetriever
from rag_ai.splitters import RecursiveCharacterTextSplitter
from rag_ai.stores import ChromaVectorStore, InMemoryVectorStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-ai")
    parser.add_argument("--persist-directory", default=".rag_ai/chroma")
    parser.add_argument("--store", choices=["memory", "chroma"], default="chroma")
    subcommands = parser.add_subparsers(dest="command", required=True)

    ingest = subcommands.add_parser("ingest", help="Load local documents into a collection")
    ingest.add_argument("path")
    ingest.add_argument("--collection", default="default")
    ingest.add_argument("--chunk-size", type=int, default=1000)
    ingest.add_argument("--chunk-overlap", type=int, default=150)

    ask = subcommands.add_parser("ask", help="Ask a question against a collection")
    ask.add_argument("question")
    ask.add_argument("--collection", default="default")
    ask.add_argument("--k", type=int, default=4)
    ask.add_argument("--filter", action="append", default=[], help="Metadata filter as key=value")

    eval_command = subcommands.add_parser("eval", help="Evaluate retrieval/answer behavior from a CSV")
    eval_command.add_argument("dataset")
    eval_command.add_argument("--collection", default="default")
    eval_command.add_argument("--k", type=int, default=4)

    args = parser.parse_args(argv)
    embeddings = HashEmbeddingModel()

    if args.command == "ingest":
        store = _store(args.store, args.collection, args.persist_directory)
        documents = load_path(args.path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        store.add(chunks, embeddings.embed_documents([chunk.text for chunk in chunks]))
        print(json.dumps({"documents": len(documents), "chunks": len(chunks), "collection": args.collection}))
        return 0

    store = _store(args.store, args.collection, args.persist_directory)
    pipeline = RAGPipeline(retriever=VectorRetriever(store, embeddings), llm=ExtractiveLLM())

    if args.command == "ask":
        answer = pipeline.ask(args.question, k=args.k, filters=_parse_filters(args.filter))
        print(answer.answer)
        if answer.sources:
            print("\nSources:")
            for index, source in enumerate(answer.sources, start=1):
                source_name = source.chunk.metadata.get("source", source.chunk.id)
                print(f"[{index}] {source_name} score={source.score:.4f}")
        return 0

    if args.command == "eval":
        result = evaluate(pipeline, load_eval_csv(args.dataset), k=args.k)
        print(
            json.dumps(
                {
                    "total": result.total,
                    "source_hit_rate": result.source_hit_rate,
                    "answer_hit_rate": result.answer_hit_rate,
                }
            )
        )
        return 0

    return 1


def _store(kind: str, collection: str, persist_directory: str) -> InMemoryVectorStore | ChromaVectorStore:
    if kind == "memory":
        return InMemoryVectorStore()
    return ChromaVectorStore(collection_name=collection, persist_directory=Path(persist_directory))


def _parse_filters(items: list[str]) -> dict[str, object] | None:
    filters: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid filter {item!r}. Expected key=value.")
        key, value = item.split("=", 1)
        filters[key] = value
    return filters or None


if __name__ == "__main__":
    raise SystemExit(main())
