"""
Enterprise RAG System - Embeddings Base Classes
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from src.core.logging import LoggerMixin


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    embeddings: list[list[float]]
    model: str
    dimensions: int
    tokens_used: int


class EmbeddingProvider(ABC, LoggerMixin):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        pass

    @abstractmethod
    async def embed_texts(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
    ) -> EmbeddingResult:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Optional batch size for processing

        Returns:
            EmbeddingResult with embeddings
        """
        pass

    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        result = await self.embed_texts([text])
        return result.embeddings[0]

    async def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a search query.

        Some providers use different models for queries vs documents.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        return await self.embed_text(query)
