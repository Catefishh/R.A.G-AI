from pathlib import Path

import pytest

from rag_ai import Document, HashEmbeddingModel
from rag_ai.dashboard import build_runner, load_uploaded_documents
from rag_ai.llms import StaticLLM


class Upload:
    name = "note.md"

    def getvalue(self) -> bytes:
        return b"# Private facts\nThe launch window is 04:20 UTC."


def test_dashboard_builds_session_runner_without_streamlit() -> None:
    runner, chunk_count = build_runner(
        [Document(text="The launch window is 04:20 UTC.", metadata={"source": "note.md"})],
        embeddings=HashEmbeddingModel(dimensions=64),
        llm=StaticLLM("answer"),
    )

    result = runner.compare("When is the launch window?", k=1)

    assert chunk_count == 1
    assert result.without_rag.answer == "answer"
    assert result.with_rag.sources


def test_uploaded_documents_are_loaded_without_disk_writes() -> None:
    documents = load_uploaded_documents([Upload()])

    assert documents[0].metadata["source"] == "note.md"
    assert "04:20" in documents[0].text


def test_dashboard_rejects_empty_documents() -> None:
    try:
        build_runner([], embeddings=HashEmbeddingModel(), llm=StaticLLM("answer"))
    except ValueError as exc:
        assert "No readable" in str(exc)
    else:
        raise AssertionError("Expected empty documents to fail")


def test_streamlit_app_smoke() -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).parents[1] / "streamlit_app.py"))
    app.run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "RAG Comparison Lab"
