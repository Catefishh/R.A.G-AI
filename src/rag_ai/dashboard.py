from __future__ import annotations

from pathlib import Path
from typing import Iterable

from rag_ai.comparison import ComparisonRunner
from rag_ai.embeddings import SentenceTransformersEmbeddingModel
from rag_ai.interfaces import EmbeddingModel, LLM
from rag_ai.llms import TransformersLLM
from rag_ai.loaders import load_bytes, load_path
from rag_ai.models import ComparisonAnswer, ComparisonResult, Document
from rag_ai.pipeline import RAGPipeline
from rag_ai.retrievers import VectorRetriever
from rag_ai.splitters import RecursiveCharacterTextSplitter
from rag_ai.stores import InMemoryVectorStore


DEFAULT_GENERATION_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SAMPLE_PATH = Path(__file__).with_name("demo_docs")


def build_runner(
    documents: list[Document],
    *,
    embeddings: EmbeddingModel,
    llm: LLM,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> tuple[ComparisonRunner, int]:
    if not documents:
        raise ValueError("No readable document content was provided.")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ).split_documents(documents)
    if not chunks:
        raise ValueError("The selected documents did not contain readable text.")
    store = InMemoryVectorStore()
    store.add(chunks, embeddings.embed_documents([chunk.text for chunk in chunks]))
    pipeline = RAGPipeline(retriever=VectorRetriever(store, embeddings), llm=llm)
    return ComparisonRunner(pipeline), len(chunks)


def load_uploaded_documents(files: Iterable[object]) -> list[Document]:
    documents: list[Document] = []
    for uploaded in files:
        name = getattr(uploaded, "name", None)
        getvalue = getattr(uploaded, "getvalue", None)
        if not isinstance(name, str) or not callable(getvalue):
            raise ValueError("Invalid uploaded file.")
        documents.extend(load_bytes(name, getvalue()))
    return documents


def load_transformers_models(
    *,
    generation_model: str,
    embedding_model: str,
    device: str,
) -> tuple[SentenceTransformersEmbeddingModel, TransformersLLM]:
    requested_device = None if device == "auto" else device
    llm = TransformersLLM(
        model_name_or_path=generation_model,
        device=requested_device,
    )
    embeddings = SentenceTransformersEmbeddingModel(
        embedding_model,
        device=llm.device,
    )
    return embeddings, llm


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Dashboard requires: pip install -e .[demo]") from exc

    st.set_page_config(page_title="RAG Comparison Lab", page_icon="🔎", layout="wide")
    st.title("RAG Comparison Lab")
    st.caption(
        "Compare the same local Transformers model with and without retrieved document context. "
        "RAG is most useful for corpus-specific questions and is not guaranteed to improve "
        "every answer."
    )

    with st.sidebar:
        st.header("Local model settings")
        generation_model = st.text_input(
            "Generation model ID or local path",
            value=DEFAULT_GENERATION_MODEL,
        )
        embedding_model = st.text_input(
            "Embedding model ID or local path",
            value=DEFAULT_EMBEDDING_MODEL,
        )
        device = st.selectbox("Compute device", ["auto", "cpu", "cuda"])
        k = st.slider("Retrieved passages", min_value=1, max_value=10, value=4)
        with st.expander("Model loading notes"):
            st.write(
                "Public models download from Hugging Face on first use and are then reused "
                "from the local cache. You can also enter a Transformers-compatible local "
                "model directory. Larger models may require a CUDA GPU."
            )
            st.write("Models and documents stay on this machine.")

    cached_model_loader = st.cache_resource(show_spinner="Loading local models...")(
        load_transformers_models
    )

    st.subheader("1. Choose the knowledge base")
    source_mode = st.radio(
        "Document source",
        ["Bundled sample", "Upload documents"],
        horizontal=True,
        label_visibility="collapsed",
    )
    uploaded_files = []
    if source_mode == "Bundled sample":
        st.info(
            "The bundled corpus describes a fictional product with facts the base model "
            "should not know."
        )
    else:
        uploaded_files = st.file_uploader(
            "Upload TXT, Markdown, or PDF files",
            type=["txt", "md", "markdown", "pdf"],
            accept_multiple_files=True,
        )

    if st.button("Index documents", type="primary"):
        try:
            documents = (
                load_path(SAMPLE_PATH)
                if source_mode == "Bundled sample"
                else load_uploaded_documents(uploaded_files)
            )
            embeddings, llm = cached_model_loader(
                generation_model=generation_model,
                embedding_model=embedding_model,
                device=device,
            )
            runner, chunk_count = build_runner(
                documents,
                embeddings=embeddings,
                llm=llm,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.session_state["comparison_runner"] = runner
            st.session_state["index_summary"] = {
                "documents": len(documents),
                "chunks": chunk_count,
                "generation_model": generation_model,
                "embedding_model": embedding_model,
                "device": llm.device,
            }
            st.session_state.pop("comparison_result", None)

    if summary := st.session_state.get("index_summary"):
        st.success(
            f"Ready: {summary['documents']} document(s), {summary['chunks']} chunk(s), "
            f"model {summary['generation_model']} on {summary['device']}."
        )

    st.subheader("2. Ask one question")
    runner = st.session_state.get("comparison_runner")
    with st.form("question_form"):
        question = st.text_input(
            "Question",
            placeholder="What is Project Aurora's emergency rollback code?",
            disabled=runner is None,
        )
        submitted = st.form_submit_button("Compare answers", disabled=runner is None)
    if runner is None:
        st.caption("Index a knowledge base before asking a question.")
    elif submitted:
        try:
            with st.spinner("Running both answers through the same local model..."):
                st.session_state["comparison_result"] = runner.compare(question, k=k)
        except (RuntimeError, ValueError) as exc:
            st.error(str(exc))

    result = st.session_state.get("comparison_result")
    if isinstance(result, ComparisonResult):
        st.subheader("3. Compare the evidence")
        st.info(
            "Both answers use the same model. Only the RAG answer receives retrieved "
            "document context."
        )
        left, right = st.columns(2)
        with left:
            _render_answer(st, "Without RAG", result.without_rag)
        with right:
            _render_answer(st, "With RAG", result.with_rag)


def _render_answer(st: object, title: str, run: ComparisonAnswer) -> None:
    st.markdown(f"### {title}")
    st.write(run.answer)
    metric_columns = st.columns(3)
    metric_columns[0].metric("Latency", f"{run.metrics.elapsed_ms / 1000:.2f}s")
    metric_columns[1].metric("Input tokens", _metric_value(run.metrics.input_tokens))
    metric_columns[2].metric("Output tokens", _metric_value(run.metrics.output_tokens))
    if run.sources:
        with st.expander(f"What was retrieved? ({len(run.sources)} passages)"):
            for index, source in enumerate(run.sources, start=1):
                source_name = source.chunk.metadata.get(
                    "file_name"
                ) or source.chunk.metadata.get("source")
                st.markdown(f"**[{index}] {source_name} — score {source.score:.4f}**")
                if page := source.chunk.metadata.get("page"):
                    st.caption(f"Page {page}")
                st.write(source.chunk.text)


def _metric_value(value: int | None) -> str:
    return str(value) if value is not None else "N/A"


if __name__ == "__main__":
    main()
