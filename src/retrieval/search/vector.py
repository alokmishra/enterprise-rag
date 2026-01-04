"""
Enterprise RAG System - Vector Search Implementation
"""

from typing import Any, Optional

from src.core.config import settings
from src.core.exceptions import SearchError
from src.core.logging import LoggerMixin
from src.core.types import SearchResult, RetrievalResult, RetrievalStrategy
from src.ingestion.embeddings import get_embedding_provider
from src.storage.vector import get_vector_store


class VectorSearcher(LoggerMixin):
    """Vector similarity search using embeddings."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ):
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.top_k = top_k or settings.DEFAULT_TOP_K

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """
        Search for documents similar to the query.

        Args:
            query: The search query
            top_k: Number of results to return
            filters: Optional metadata filters
            score_threshold: Minimum similarity score

        Returns:
            RetrievalResult with search results
        """
        import time
        start_time = time.time()

        top_k = top_k or self.top_k

        try:
            # Generate query embedding
            embedding_provider = get_embedding_provider()
            query_embedding = await embedding_provider.embed_query(query)

            # Search vector store
            vector_store = get_vector_store()
            raw_results = await vector_store.search(
                collection=self.collection_name,
                query_vector=query_embedding,
                top_k=top_k,
                filters=filters,
            )

            # Convert to SearchResult objects
            results = []
            for r in raw_results:
                # Apply score threshold if specified
                if score_threshold and r.score < score_threshold:
                    continue

                results.append(SearchResult(
                    chunk_id=r.id,
                    document_id=r.payload.get("document_id", ""),
                    content=r.payload.get("content", ""),
                    score=r.score,
                    metadata=r.payload.get("metadata", {}),
                    source=r.payload.get("source"),
                ))

            latency_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "Vector search completed",
                query_length=len(query),
                top_k=top_k,
                results_found=len(results),
                latency_ms=round(latency_ms, 2),
            )

            return RetrievalResult(
                query=query,
                strategy=RetrievalStrategy.VECTOR,
                results=results,
                total_found=len(results),
                latency_ms=latency_ms,
            )

        except Exception as e:
            self.logger.error("Vector search failed", error=str(e))
            raise SearchError(f"Vector search failed: {e}")

    async def search_by_embedding(
        self,
        embedding: list[float],
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """
        Search using a pre-computed embedding.

        Useful when you already have the embedding or want to reuse it.
        """
        top_k = top_k or self.top_k

        try:
            vector_store = get_vector_store()
            raw_results = await vector_store.search(
                collection=self.collection_name,
                query_vector=embedding,
                top_k=top_k,
                filters=filters,
            )

            return [
                SearchResult(
                    chunk_id=r.id,
                    document_id=r.payload.get("document_id", ""),
                    content=r.payload.get("content", ""),
                    score=r.score,
                    metadata=r.payload.get("metadata", {}),
                    source=r.payload.get("source"),
                )
                for r in raw_results
            ]

        except Exception as e:
            raise SearchError(f"Vector search by embedding failed: {e}")


class MultiQuerySearcher(LoggerMixin):
    """
    Generate multiple query variations and search with each.

    This improves recall by finding documents that might match
    different phrasings of the same question.
    """

    def __init__(
        self,
        vector_searcher: Optional[VectorSearcher] = None,
        num_variations: int = 3,
    ):
        self.vector_searcher = vector_searcher or VectorSearcher()
        self.num_variations = num_variations

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> RetrievalResult:
        """
        Search with multiple query variations.

        Note: Query generation requires LLM, will be implemented in generation module.
        For now, uses the original query only.
        """
        import time
        start_time = time.time()

        # TODO: Generate query variations using LLM
        # For now, just use the original query
        queries = [query]

        # Search with each query
        all_results: dict[str, SearchResult] = {}

        for q in queries:
            result = await self.vector_searcher.search(
                query=q,
                top_k=top_k,
                filters=filters,
            )

            # Deduplicate by chunk_id, keeping highest score
            for r in result.results:
                if r.chunk_id not in all_results or r.score > all_results[r.chunk_id].score:
                    all_results[r.chunk_id] = r

        # Sort by score and limit
        final_results = sorted(
            all_results.values(),
            key=lambda x: x.score,
            reverse=True,
        )[:top_k or self.vector_searcher.top_k]

        latency_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            query=query,
            strategy=RetrievalStrategy.MULTI_QUERY,
            results=final_results,
            total_found=len(final_results),
            latency_ms=latency_ms,
        )


# Singleton instances
_vector_searcher: Optional[VectorSearcher] = None


def get_vector_searcher() -> VectorSearcher:
    """Get the global vector searcher instance."""
    global _vector_searcher
    if _vector_searcher is None:
        _vector_searcher = VectorSearcher()
    return _vector_searcher
