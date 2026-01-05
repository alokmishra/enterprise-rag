"""
Enterprise RAG System - Hybrid Search

Combines dense (vector) and sparse (BM25) retrieval for improved results.
Uses Reciprocal Rank Fusion (RRF) to merge rankings.
"""

import asyncio
from typing import Any, Optional

from src.core.config import settings
from src.core.logging import LoggerMixin
from src.core.types import SearchResult, RetrievalResult, RetrievalStrategy
from src.retrieval.search.vector import VectorSearcher
from src.retrieval.search.sparse import BM25Index, get_bm25_index


class HybridSearcher(LoggerMixin):
    """
    Hybrid search combining vector and BM25 retrieval.

    Uses Reciprocal Rank Fusion (RRF) to combine rankings:
    RRF(d) = Σ 1/(k + rank(d))

    where k is a constant (default 60) and rank is the position
    in each result list.
    """

    def __init__(
        self,
        vector_searcher: Optional[VectorSearcher] = None,
        bm25_index: Optional[BM25Index] = None,
        alpha: Optional[float] = None,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid searcher.

        Args:
            vector_searcher: Vector search instance
            bm25_index: BM25 index instance
            alpha: Weight for vector search (0-1). 1.0 = all vector, 0.0 = all BM25
            rrf_k: RRF constant (higher = less emphasis on top ranks)
        """
        self.vector_searcher = vector_searcher or VectorSearcher()
        self.bm25_index = bm25_index or get_bm25_index()
        self.alpha = alpha if alpha is not None else settings.HYBRID_SEARCH_ALPHA
        self.rrf_k = rrf_k

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
        alpha: Optional[float] = None,
        fusion_method: str = "rrf",  # "rrf" or "weighted"
    ) -> RetrievalResult:
        """
        Execute hybrid search.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters (applied to vector search)
            alpha: Override weight for vector search
            fusion_method: "rrf" for Reciprocal Rank Fusion, "weighted" for score weighting

        Returns:
            RetrievalResult with fused results
        """
        import time
        start_time = time.time()

        top_k = top_k or settings.DEFAULT_TOP_K
        alpha = alpha if alpha is not None else self.alpha

        # Retrieve more candidates for fusion
        candidate_k = min(top_k * 3, 50)

        # Execute searches in parallel
        vector_task = self.vector_searcher.search(
            query=query,
            top_k=candidate_k,
            filters=filters,
        )

        # BM25 search (synchronous but fast)
        sparse_results = self.bm25_index.search(query, top_k=candidate_k)

        # Wait for vector search
        vector_result = await vector_task

        # Fuse results
        if fusion_method == "rrf":
            fused = self._rrf_fusion(
                vector_results=vector_result.results,
                sparse_results=sparse_results,
                top_k=top_k,
            )
        else:
            fused = self._weighted_fusion(
                vector_results=vector_result.results,
                sparse_results=sparse_results,
                alpha=alpha,
                top_k=top_k,
            )

        latency_ms = (time.time() - start_time) * 1000

        self.logger.info(
            "Hybrid search completed",
            query_length=len(query),
            vector_results=len(vector_result.results),
            sparse_results=len(sparse_results),
            fused_results=len(fused),
            latency_ms=round(latency_ms, 2),
        )

        return RetrievalResult(
            query=query,
            strategy=RetrievalStrategy.HYBRID,
            results=fused,
            total_found=len(fused),
            latency_ms=latency_ms,
        )

    def _rrf_fusion(
        self,
        vector_results: list[SearchResult],
        sparse_results: list,
        top_k: int,
    ) -> list[SearchResult]:
        """
        Fuse results using Reciprocal Rank Fusion.

        RRF score = Σ 1/(k + rank)
        """
        # Build RRF scores
        rrf_scores: dict[str, float] = {}
        result_map: dict[str, SearchResult] = {}

        # Process vector results
        for rank, result in enumerate(vector_results):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + \
                1.0 / (self.rrf_k + rank + 1)
            result_map[result.chunk_id] = result

        # Process sparse results
        for rank, result in enumerate(sparse_results):
            rrf_scores[result.id] = rrf_scores.get(result.id, 0) + \
                1.0 / (self.rrf_k + rank + 1)

            # Add to result map if not already present
            if result.id not in result_map:
                result_map[result.id] = SearchResult(
                    chunk_id=result.id,
                    document_id=result.metadata.get("document_id", ""),
                    content=result.content,
                    score=0.0,  # Will be updated
                    metadata=result.metadata,
                    source=result.metadata.get("source"),
                )

        # Sort by RRF score
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True,
        )[:top_k]

        # Build final results with RRF scores
        results = []
        for chunk_id in sorted_ids:
            result = result_map[chunk_id]
            # Update score to RRF score (normalized)
            result.score = rrf_scores[chunk_id]
            results.append(result)

        return results

    def _weighted_fusion(
        self,
        vector_results: list[SearchResult],
        sparse_results: list,
        alpha: float,
        top_k: int,
    ) -> list[SearchResult]:
        """
        Fuse results using weighted score combination.

        Combined score = alpha * vector_score + (1-alpha) * sparse_score

        Scores are normalized to [0, 1] before combining.
        """
        # Normalize vector scores
        vector_scores: dict[str, float] = {}
        result_map: dict[str, SearchResult] = {}

        if vector_results:
            max_vector = max(r.score for r in vector_results)
            min_vector = min(r.score for r in vector_results)
            range_vector = max_vector - min_vector if max_vector != min_vector else 1.0

            for result in vector_results:
                normalized = (result.score - min_vector) / range_vector
                vector_scores[result.chunk_id] = normalized
                result_map[result.chunk_id] = result

        # Normalize sparse scores
        sparse_scores: dict[str, float] = {}
        if sparse_results:
            max_sparse = max(r.score for r in sparse_results)
            min_sparse = min(r.score for r in sparse_results)
            range_sparse = max_sparse - min_sparse if max_sparse != min_sparse else 1.0

            for result in sparse_results:
                normalized = (result.score - min_sparse) / range_sparse
                sparse_scores[result.id] = normalized

                if result.id not in result_map:
                    result_map[result.id] = SearchResult(
                        chunk_id=result.id,
                        document_id=result.metadata.get("document_id", ""),
                        content=result.content,
                        score=0.0,
                        metadata=result.metadata,
                        source=result.metadata.get("source"),
                    )

        # Combine scores
        combined_scores: dict[str, float] = {}
        all_ids = set(vector_scores.keys()) | set(sparse_scores.keys())

        for chunk_id in all_ids:
            v_score = vector_scores.get(chunk_id, 0.0)
            s_score = sparse_scores.get(chunk_id, 0.0)
            combined_scores[chunk_id] = alpha * v_score + (1 - alpha) * s_score

        # Sort by combined score
        sorted_ids = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x],
            reverse=True,
        )[:top_k]

        # Build final results
        results = []
        for chunk_id in sorted_ids:
            result = result_map[chunk_id]
            result.score = combined_scores[chunk_id]
            results.append(result)

        return results


# Singleton instance
_hybrid_searcher: Optional[HybridSearcher] = None


def get_hybrid_searcher() -> HybridSearcher:
    """Get the global hybrid searcher instance."""
    global _hybrid_searcher
    if _hybrid_searcher is None:
        _hybrid_searcher = HybridSearcher()
    return _hybrid_searcher
