from __future__ import annotations

from rag_ai.models import Chunk, Document


class RecursiveCharacterTextSplitter:
    def __init__(self, *, chunk_size: int = 1000, chunk_overlap: int | None = None) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap is None:
            chunk_overlap = min(150, max(0, chunk_size // 6))
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            for index, text in enumerate(self.split_text(document.text)):
                metadata = dict(document.metadata)
                metadata["chunk_index"] = index
                chunks.append(Chunk(text=text, document_id=document.id, metadata=metadata))
        return chunks

    def split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                split_at = self._best_split(text, start, end)
                if split_at > start:
                    end = split_at
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(0, end - self.chunk_overlap)
            while start < len(text) and text[start].isspace():
                start += 1
        return chunks

    @staticmethod
    def _best_split(text: str, start: int, end: int) -> int:
        window = text[start:end]
        for separator in ("\n\n", "\n", ". ", " "):
            index = window.rfind(separator)
            if index >= max(1, len(window) // 3):
                return start + index + len(separator)
        return end
