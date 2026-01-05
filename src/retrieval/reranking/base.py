"""
Enterprise RAG System - Reranking Base Classes
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from src.core.logging import LoggerMixin
from src.core.types import SearchResult


@dataclass
class RerankResult:
    """Result from reranking."""
    results: list[SearchResult]
    original_count: int
    reranked_count: int
    latency_ms: float


class Reranker(ABC, LoggerMixin):
    """Abstract base class for rerankers."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        """
        Rerank search results based on query relevance.

        Args:
            query: The original search query
            results: List of search results to rerank
            top_k: Number of results to return after reranking

        Returns:
            RerankResult with reranked results
        """
        pass
