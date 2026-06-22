"""Document loading and chunking.

Supports plain text, Markdown, and PDF source files. Chunking is a simple,
predictable sliding window over characters with overlap — easy to reason about
and good enough to demonstrate retrieval quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}


@dataclass
class Document:
    """A whole source document loaded from disk."""

    source: str  # filename / identifier
    text: str


@dataclass
class Chunk:
    """A retrievable slice of a document."""

    text: str
    source: str
    index: int  # position of this chunk within its source document
    metadata: dict = field(default_factory=dict)


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Reading PDF files requires 'pypdf'. Install it or use .txt/.md docs."
        ) from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_documents(docs_dir: str | Path) -> list[Document]:
    """Load every supported file in ``docs_dir`` into a Document.

    The directory is scanned recursively and results are sorted by path so the
    knowledge base is built deterministically (important for reproducible demos).
    """

    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    documents: list[Document] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if path.suffix.lower() == ".pdf":
            text = _read_pdf(path)
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
        text = text.strip()
        if text:
            documents.append(Document(source=path.name, text=text))
    return documents


def _split_text(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    """Sliding-window split on whitespace boundaries.

    We aim for ``chunk_size`` characters per chunk but snap the cut to the
    nearest whitespace so words are not sliced in half. ``overlap`` characters
    are carried into the next chunk to preserve context across boundaries.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size - 1))

    text = text.strip()
    if not text:
        return

    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # Snap to a whitespace boundary when we're not at the very end.
        if end < n:
            window = text.rfind(" ", start, end)
            if window > start:
                end = window
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= n:
            break
        next_start = max(end - overlap, start + 1)
        # Snap the overlap start back to the beginning of the word it lands in,
        # so chunks never begin mid-word (and no text is skipped).
        if 0 < next_start < n and not text[next_start - 1].isspace():
            ws = text.rfind(" ", start, next_start)
            if ws != -1:
                next_start = ws + 1
        start = next_start


def chunk_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Turn loaded documents into a flat list of retrievable chunks."""

    chunks: list[Chunk] = []
    for doc in documents:
        for i, piece in enumerate(_split_text(doc.text, chunk_size, chunk_overlap)):
            chunks.append(
                Chunk(
                    text=piece,
                    source=doc.source,
                    index=i,
                    metadata={"source": doc.source, "chunk": i},
                )
            )
    return chunks
