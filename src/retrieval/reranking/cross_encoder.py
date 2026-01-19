"""
Enterprise RAG System - Cross-Encoder Reranker

Uses a cross-encoder model to score query-document pairs.
More accurate than bi-encoder (embedding) similarity but slower.
"""

from __future__ import annotations

import time
from typing import Optional

from src.core.config import settings
from src.core.logging import LoggerMixin
from src.core.types import SearchResult
from src.retrieval.reranking.base import Reranker, RerankResult


class CrossEncoderReranker(Reranker):
    """
    Cross-encoder reranker using sentence-transformers.

    Cross-encoders process query and document together, allowing
    for more accurate relevance scoring than bi-encoders.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: int = 32,
        score_threshold: float = 0.0,
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: Cross-encoder model name
            batch_size: Batch size for scoring
            score_threshold: Minimum score threshold
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.batch_size = batch_size
        self.score_threshold = score_threshold
        self._model = None

    def _get_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                self.logger.info(
                    "Loaded cross-encoder model",
                    model=self.model_name,
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for cross-encoder reranking. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        """Rerank results using cross-encoder scoring."""
        start_time = time.time()

        if not results:
            return RerankResult(
                results=[],
                original_count=0,
                reranked_count=0,
                latency_ms=0.0,
            )

        top_k = top_k or settings.RERANK_TOP_K
        original_count = len(results)

        # Prepare query-document pairs
        pairs = [(query, r.content) for r in results]

        # Score with cross-encoder
        try:
            model = self._get_model()
            scores = model.predict(pairs, batch_size=self.batch_size)
        except Exception as e:
            self.logger.error("Cross-encoder scoring failed", error=str(e))
            # Return original results on failure
            return RerankResult(
                results=results[:top_k],
                original_count=original_count,
                reranked_count=min(top_k, original_count),
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Apply scores and filter
        scored_results = []
        for result, score in zip(results, scores):
            if score >= self.score_threshold:
                scored_results.append(SearchResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    content=result.content,
                    score=float(score),
                    metadata={
                        **result.metadata,
                        "original_score": result.score,
                        "rerank_score": float(score),
                    },
                    source=result.source,
                ))

        # Sort by cross-encoder score
        scored_results.sort(key=lambda x: x.score, reverse=True)
        final_results = scored_results[:top_k]

        latency_ms = (time.time() - start_time) * 1000

        self.logger.info(
            "Cross-encoder reranking complete",
            original_count=original_count,
            reranked_count=len(final_results),
            latency_ms=round(latency_ms, 2),
        )

        return RerankResult(
            results=final_results,
            original_count=original_count,
            reranked_count=len(final_results),
            latency_ms=latency_ms,
        )


class CohereReranker(Reranker):
    """
    Reranker using Cohere's rerank API.

    Requires COHERE_API_KEY environment variable.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "rerank-english-v3.0",
    ):
        self.model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        """Get Cohere client."""
        if self._client is None:
            import os
            try:
                import cohere
                api_key = self._api_key or os.getenv("COHERE_API_KEY")
                if not api_key:
                    raise ValueError("COHERE_API_KEY not set")
                self._client = cohere.Client(api_key)
            except ImportError:
                raise ImportError(
                    "cohere is required for Cohere reranking. "
                    "Install with: pip install cohere"
                )
        return self._client

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        """Rerank using Cohere API."""
        start_time = time.time()

        if not results:
            return RerankResult(
                results=[],
                original_count=0,
                reranked_count=0,
                latency_ms=0.0,
            )

        top_k = top_k or settings.RERANK_TOP_K
        original_count = len(results)

        try:
            client = self._get_client()

            # Prepare documents
            documents = [r.content for r in results]

            # Call Cohere rerank
            response = client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_n=top_k,
            )

            # Build reranked results
            reranked_results = []
            for item in response.results:
                original_result = results[item.index]
                reranked_results.append(SearchResult(
                    chunk_id=original_result.chunk_id,
                    document_id=original_result.document_id,
                    content=original_result.content,
                    score=item.relevance_score,
                    metadata={
                        **original_result.metadata,
                        "original_score": original_result.score,
                        "rerank_score": item.relevance_score,
                    },
                    source=original_result.source,
                ))

            latency_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "Cohere reranking complete",
                original_count=original_count,
                reranked_count=len(reranked_results),
                latency_ms=round(latency_ms, 2),
            )

            return RerankResult(
                results=reranked_results,
                original_count=original_count,
                reranked_count=len(reranked_results),
                latency_ms=latency_ms,
            )

        except Exception as e:
            self.logger.error("Cohere reranking failed", error=str(e))
            return RerankResult(
                results=results[:top_k],
                original_count=original_count,
                reranked_count=min(top_k, original_count),
                latency_ms=(time.time() - start_time) * 1000,
            )
