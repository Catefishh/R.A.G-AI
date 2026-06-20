from __future__ import annotations

from os import getenv

from rag_ai.embeddings import TOKEN_RE
from rag_ai.models import RetrievedChunk
from rag_ai.prompts import INSUFFICIENT_CONTEXT


class StaticLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str, *, context: list[RetrievedChunk] | None = None) -> str:
        return self.response


class ExtractiveLLM:
    """Local fallback LLM that extracts relevant source snippets instead of calling a model."""

    def generate(self, prompt: str, *, context: list[RetrievedChunk] | None = None) -> str:
        if not context:
            return INSUFFICIENT_CONTEXT

        question_terms = {token.lower() for token in TOKEN_RE.findall(prompt.split("Question:")[-1])}
        best_index = 0
        best_overlap = -1
        for index, item in enumerate(context):
            terms = {token.lower() for token in TOKEN_RE.findall(item.chunk.text)}
            overlap = len(question_terms & terms)
            if overlap > best_overlap:
                best_index = index
                best_overlap = overlap

        if best_overlap <= 0:
            return INSUFFICIENT_CONTEXT

        source = context[best_index]
        snippet = " ".join(source.chunk.text.split())
        if len(snippet) > 500:
            snippet = snippet[:497].rstrip() + "..."
        return f"{snippet} [{best_index + 1}]"


class OpenAILLM:
    def __init__(self, *, model: str = "gpt-4.1-mini", api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI LLM requires the optional 'openai' extra: pip install -e .[openai]") from exc
        self.client = OpenAI(api_key=api_key or getenv("OPENAI_API_KEY"))
        self.model = model

    def generate(self, prompt: str, *, context: list[RetrievedChunk] | None = None) -> str:
        response = self.client.responses.create(model=self.model, input=prompt)
        return response.output_text
