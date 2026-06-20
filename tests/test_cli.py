from pathlib import Path

from rag_ai.cli import main


def test_cli_ingest_memory_outputs_counts(tmp_path: Path, capsys) -> None:
    path = tmp_path / "doc.txt"
    path.write_text("RAG combines retrieval and generation.", encoding="utf-8")

    exit_code = main(["--store", "memory", "ingest", str(path), "--collection", "test"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"documents": 1' in output
    assert '"chunks": 1' in output


def test_cli_invalid_filter_exits() -> None:
    try:
        main(["--store", "memory", "ask", "question", "--filter", "bad-filter"])
    except SystemExit as exc:
        assert "Invalid filter" in str(exc)
    else:
        raise AssertionError("Expected invalid filter to exit")
