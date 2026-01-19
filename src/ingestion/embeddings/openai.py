"""
Enterprise RAG System - OpenAI Embeddings Provider
"""

from __future__ import annotations

import asyncio
from typing import Optional

from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.core.config import settings
from src.core.exceptions import EmbeddingError
from src.ingestion.embeddings.base import EmbeddingProvider, EmbeddingResult


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embeddings provider."""

    # Model dimension mappings
    MODEL_DIMENSIONS = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        self._api_key = api_key or settings.OPENAI_API_KEY
        self._model = model or settings.EMBEDDING_MODEL
        self._dimensions = dimensions or settings.EMBEDDING_DIMENSIONS
        self._client: Optional[AsyncOpenAI] = None

    def _get_client(self) -> AsyncOpenAI:
        """Get or create the OpenAI client."""
        if self._client is None:
            if not self._api_key:
                raise EmbeddingError("OpenAI API key not configured")
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        # Use configured dimensions, or fall back to model default
        if self._dimensions:
            return self._dimensions
        return self.MODEL_DIMENSIONS.get(self._model, 3072)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
    )
    async def embed_texts(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
    ) -> EmbeddingResult:
        """Generate embeddings using OpenAI API."""
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model=self._model,
                dimensions=self.dimensions,
                tokens_used=0,
            )

        client = self._get_client()
        batch_size = batch_size or 100  # OpenAI recommends batches of ~100

        all_embeddings = []
        total_tokens = 0

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                # Use dimensions parameter for text-embedding-3 models
                kwargs = {"model": self._model, "input": batch}
                if self._model.startswith("text-embedding-3"):
                    kwargs["dimensions"] = self.dimensions

                response = await client.embeddings.create(**kwargs)

                # Extract embeddings in order
                batch_embeddings = [None] * len(batch)
                for item in response.data:
                    batch_embeddings[item.index] = item.embedding

                all_embeddings.extend(batch_embeddings)
                total_tokens += response.usage.total_tokens

                self.logger.debug(
                    "Generated embeddings batch",
                    batch_num=i // batch_size + 1,
                    batch_size=len(batch),
                    tokens=response.usage.total_tokens,
                )

            except Exception as e:
                self.logger.error(
                    "Embedding generation failed",
                    error=str(e),
                    batch_start=i,
                )
                raise EmbeddingError(f"Failed to generate embeddings: {e}")

        self.logger.info(
            "Generated embeddings",
            num_texts=len(texts),
            total_tokens=total_tokens,
            model=self._model,
        )

        return EmbeddingResult(
            embeddings=all_embeddings,
            model=self._model,
            dimensions=self.dimensions,
            tokens_used=total_tokens,
        )

    async def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a query.

        OpenAI uses the same model for queries and documents,
        but we add a prefix for better retrieval quality.
        """
        # For retrieval, some research suggests prefixing helps
        # But OpenAI's models don't require it
        return await self.embed_text(query)


# Singleton instance
_embedding_provider: Optional[OpenAIEmbeddings] = None


def get_embedding_provider() -> OpenAIEmbeddings:
    """Get the global embedding provider instance."""
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = OpenAIEmbeddings()
    return _embedding_provider
