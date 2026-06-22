"""Streamlit app: a side-by-side visual comparison of RAG vs no-RAG.

Run with:

    streamlit run app.py

The app builds a small local knowledge base (the fictional "Nimbus Forge"
company), then lets you ask a question and see two answers next to each other:

* Left  — the model alone, answering from its training only.
* Right — the same model, augmented with retrieved context.

Because the knowledge base describes a company that does not exist in any
training data, the left column will typically hallucinate or admit ignorance,
while the right column answers correctly and cites its sources.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make the local package importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_ai import Config, RAGPipeline  # noqa: E402

SAMPLE_QUESTIONS = [
    "Who founded Nimbus Forge and when?",
    "How much does the Aurelia Team edition cost and how many streams does it include?",
    "What does the AUR-429 error mean and how do I fix it?",
    "What is Tideway and how is it different from Kafka?",
    "How often does the Aurelia team deploy to production?",
    "What is the maximum UnifiedFrame message size?",
]

st.set_page_config(page_title="RAG vs No-RAG", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner="Loading models and building the knowledge base…")
def get_pipeline() -> RAGPipeline:
    """Build the pipeline once and cache it across reruns/sessions."""
    pipeline = RAGPipeline(Config())
    pipeline.build_index()
    return pipeline


def render_answer(answer, *, show_sources: bool) -> None:
    st.markdown(answer.text if answer.text else "_(empty response)_")
    st.caption(f"⏱ {answer.elapsed_s:.1f}s")
    if show_sources:
        if answer.retrieved:
            with st.expander(f"📚 Retrieved context ({len(answer.retrieved)} chunks)"):
                for i, r in enumerate(answer.retrieved, start=1):
                    st.markdown(f"**[{i}] {r.source}** · similarity `{r.score:.3f}`")
                    st.text(r.text)
                    st.divider()
        else:
            st.info(
                "No chunks scored above the relevance threshold — the model was "
                "told it has no supporting context."
            )


def main() -> None:
    st.title("🔍 RAG vs No-RAG — a side-by-side comparison")
    st.markdown(
        "Ask a question about the **fictional Nimbus Forge company**. The left "
        "column shows the model answering on its own; the right column shows the "
        "*same model* with Retrieval-Augmented Generation. Watch the difference."
    )

    with st.sidebar:
        st.header("Knowledge base")
        st.markdown(
            "The model has **no training knowledge** of Nimbus Forge — it only "
            "exists in the local documents below. That is what makes the "
            "comparison fair and the difference obvious."
        )
        pipeline = get_pipeline()
        st.success(f"Indexed {len(pipeline.store)} chunks")
        cfg = pipeline.config
        st.caption(f"LLM: `{cfg.llm_model}`")
        st.caption(f"Embeddings: `{cfg.embedding_model}`")
        st.caption(f"top_k={cfg.top_k} · min_score={cfg.min_score}")
        for doc in sorted({c.source for c in pipeline.store._chunks}):
            st.caption(f"• {doc}")

    preset = st.selectbox("Try a sample question", ["(write my own)"] + SAMPLE_QUESTIONS)
    default_q = "" if preset == "(write my own)" else preset
    question = st.text_input("Your question", value=default_q)

    if st.button("Compare answers", type="primary", disabled=not question.strip()):
        with st.spinner("Generating both answers with the local model…"):
            comparison = pipeline.compare(question.strip())

        left, right = st.columns(2)
        with left:
            st.subheader("❌ Without RAG")
            st.caption("Model only — no access to the knowledge base.")
            render_answer(comparison.plain, show_sources=False)
        with right:
            st.subheader("✅ With RAG")
            st.caption("Same model + retrieved context from the knowledge base.")
            render_answer(comparison.rag, show_sources=True)


if __name__ == "__main__":
    main()
