from __future__ import annotations

from dataclasses import dataclass

from rag_ai.interfaces import Retriever
from rag_ai.models import RetrievedChunk


@dataclass(slots=True)
class Route:
    name: str
    retriever: Retriever
    keywords: set[str]


class KeywordRouter:
    def __init__(self, routes: list[Route], *, default: Retriever) -> None:
        self.routes = routes
        self.default = default

    def route(self, question: str) -> Retriever:
        lowered = question.lower()
        for route in self.routes:
            if any(keyword.lower() in lowered for keyword in route.keywords):
                return route.retriever
        return self.default


class RoutedRetriever:
    def __init__(self, router: KeywordRouter) -> None:
        self.router = router

    def retrieve(
        self,
        question: str,
        *,
        k: int = 4,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        return self.router.route(question).retrieve(question, k=k, filters=filters)
