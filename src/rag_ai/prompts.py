from __future__ import annotations

from rag_ai.models import RetrievedChunk


INSUFFICIENT_CONTEXT = "I do not have enough context to answer that question."


def build_grounded_prompt(question: str, context: list[RetrievedChunk]) -> str:
    context_block = "\n\n".join(
        f"[{index}] {item.chunk.text}\nSource: {item.chunk.metadata.get('source', item.chunk.id)}"
        for index, item in enumerate(context, start=1)
    )
    return (
        "Answer the question using only the context below. "
        "If the context is insufficient, say you do not have enough context. "
        "Cite sources with bracketed numbers like [1].\n\n"
        f"Context:\n{context_block or '(none)'}\n\n"
        f"Question: {question}\nAnswer:"
    )
