from __future__ import annotations

from pathlib import Path

from rag_ai.models import Document


SUPPORTED_TEXT_SUFFIXES = {".txt", ".md", ".markdown"}


class TextLoader:
    def __init__(self, *, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    def load(self, path: str | Path) -> list[Document]:
        file_path = Path(path)
        text = file_path.read_text(encoding=self.encoding)
        return [
            Document(
                text=text,
                metadata={
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_type": file_path.suffix.lower().lstrip("."),
                },
            )
        ]


class PdfLoader:
    def load(self, path: str | Path) -> list[Document]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF loading requires the optional 'pdf' extra: pip install -e .[pdf]") from exc

        file_path = Path(path)
        reader = PdfReader(str(file_path))
        documents: list[Document] = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "source": str(file_path),
                            "file_name": file_path.name,
                            "file_type": "pdf",
                            "page": index,
                        },
                    )
                )
        return documents


class DirectoryLoader:
    def __init__(self, *, recursive: bool = True) -> None:
        self.recursive = recursive

    def load(self, path: str | Path) -> list[Document]:
        root = Path(path)
        if not root.is_dir():
            raise ValueError(f"Expected a directory, got {root}")

        pattern = "**/*" if self.recursive else "*"
        documents: list[Document] = []
        for file_path in sorted(p for p in root.glob(pattern) if p.is_file()):
            if file_path.suffix.lower() in SUPPORTED_TEXT_SUFFIXES or file_path.suffix.lower() == ".pdf":
                documents.extend(load_path(file_path))
        return documents


def load_path(path: str | Path) -> list[Document]:
    file_path = Path(path)
    if file_path.is_dir():
        return DirectoryLoader().load(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    suffix = file_path.suffix.lower()
    if suffix in SUPPORTED_TEXT_SUFFIXES:
        return TextLoader().load(file_path)
    if suffix == ".pdf":
        return PdfLoader().load(file_path)
    raise ValueError(f"Unsupported document type: {file_path.suffix}")
