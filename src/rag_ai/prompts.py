"""Prompt templates for the two pipelines being compared.

Keeping these side by side in one file makes the *only* difference between the
RAG and non-RAG runs explicit: same model, same question, same generation
settings — the RAG prompt simply has retrieved context prepended and is
instructed to ground its answer in that context.
"""

from __future__ import annotations

from .retriever import RetrievedChunk

# Plain (no-RAG) system prompt: the model answers from parametric memory only.
PLAIN_SYSTEM = (
    "You are a helpful assistant. Answer the user's question as accurately as "
    "you can. If you are not certain of the answer, say so honestly."
)

# RAG system prompt: the model must ground its answer in supplied context.
RAG_SYSTEM = (
    "You are a helpful assistant that answers questions using ONLY the provided "
    "context. The context is drawn from a trusted knowledge base. Follow these "
    "rules:\n"
    "- Base your answer strictly on the context below.\n"
    "- If the context does not contain the answer, say you don't have enough "
    "information rather than guessing.\n"
    "- Cite the source filename(s) you used in square brackets, e.g. [handbook.md]."
)


def build_plain_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": PLAIN_SYSTEM},
        {"role": "user", "content": question},
    ]


def format_context(retrieved: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into a numbered, source-labelled context block."""
    blocks = []
    for i, r in enumerate(retrieved, start=1):
        blocks.append(f"[{i}] (source: {r.source})\n{r.text}")
    return "\n\n".join(blocks)


def build_rag_messages(question: str, retrieved: list[RetrievedChunk]) -> list[dict]:
    if retrieved:
        context = format_context(retrieved)
    else:
        context = "(no relevant context was found in the knowledge base)"
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )
    return [
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": user},
    ]
