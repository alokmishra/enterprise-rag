"""
Enterprise RAG System - Advanced Retrieval Service

Combines all retrieval features into a unified service:
- Query expansion
- HyDE
- Hybrid search
- Reranking
- Context assembly
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from src.core.config import settings
from src.core.logging import LoggerMixin
from src.core.types import (
    ContextItem,
    RetrievalResult,
    RetrievalStrategy,
    SearchResult,
)
from src.retrieval import (
    VectorSearcher,
    HybridSearcher,
    QueryExpander,
    HyDEGenerator,
    LLMReranker,
    CrossEncoderReranker,
    ContextAssembler,
    MetadataFilter,
    get_vector_searcher,
    get_hybrid_searcher,
    get_context_assembler,
)


class RerankerType(str, Enum):
    """Available reranker types."""
    NONE = "none"
    LLM = "llm"
    CROSS_ENCODER = "cross_encoder"


@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k: int = 10
    rerank_top_k: int = 5
    use_query_expansion: bool = False
    use_hyde: bool = False
    reranker: RerankerType = RerankerType.NONE
    hybrid_alpha: float = 0.5
    context_max_tokens: int = 8000


@dataclass
class RetrievalResponse:
    """Response from advanced retrieval."""
    context_items: list[ContextItem]
    search_results: list[SearchResult]
    query_variations: list[str]
    strategy_used: RetrievalStrategy
    total_candidates: int
    final_count: int
    latency_ms: float


class AdvancedRetrievalService(LoggerMixin):
    """
    Advanced retrieval service with all Phase 3 features.
    """

    def __init__(self):
        self._vector_searcher = get_vector_searcher()
        self._hybrid_searcher = get_hybrid_searcher()
        self._query_expander = QueryExpander()
        self._hyde_generator = HyDEGenerator()
        self._llm_reranker = LLMReranker()
        self._context_assembler = get_context_assembler()
        self._cross_encoder: Optional[CrossEncoderReranker] = None

    def _get_cross_encoder(self) -> CrossEncoderReranker:
        """Lazy load cross-encoder reranker."""
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoderReranker()
        return self._cross_encoder

    async def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None,
        filters: Optional[MetadataFilter] = None,
    ) -> RetrievalResponse:
        """
        Execute advanced retrieval pipeline.

        Args:
            query: Search query
            config: Retrieval configuration
            filters: Optional metadata filters

        Returns:
            RetrievalResponse with context and results
        """
        start_time = time.time()
        config = config or RetrievalConfig()

        self.logger.info(
            "Starting advanced retrieval",
            query_length=len(query),
            strategy=config.strategy.value,
            use_expansion=config.use_query_expansion,
            use_hyde=config.use_hyde,
            reranker=config.reranker.value,
        )

        # Step 1: Query processing
        query_variations = [query]

        if config.use_query_expansion:
            expanded = await self._query_expander.expand(query)
            query_variations = expanded

        # Step 2: Retrieval
        all_results: list[SearchResult] = []

        for q in query_variations:
            if config.use_hyde and config.strategy == RetrievalStrategy.VECTOR:
                # Use HyDE embedding
                hyde_embedding = await self._hyde_generator.generate_hyde_embedding(q)
                results = await self._vector_searcher.search_by_embedding(
                    embedding=hyde_embedding,
                    top_k=config.top_k,
                    filters=filters.to_qdrant_filter() if filters else None,
                )
            elif config.strategy == RetrievalStrategy.HYBRID:
                # Hybrid search
                result = await self._hybrid_searcher.search(
                    query=q,
                    top_k=config.top_k,
                    filters=filters.to_qdrant_filter() if filters else None,
                    alpha=config.hybrid_alpha,
                )
                results = result.results
            else:
                # Vector search
                result = await self._vector_searcher.search(
                    query=q,
                    top_k=config.top_k,
                    filters=filters.to_qdrant_filter() if filters else None,
                )
                results = result.results

            all_results.extend(results)

        # Deduplicate by chunk_id, keeping highest score
        unique_results = self._deduplicate_results(all_results)
        total_candidates = len(unique_results)

        # Step 3: Reranking
        if config.reranker != RerankerType.NONE and unique_results:
            if config.reranker == RerankerType.LLM:
                rerank_result = await self._llm_reranker.rerank(
                    query=query,
                    results=unique_results,
                    top_k=config.rerank_top_k,
                )
                unique_results = rerank_result.results
            elif config.reranker == RerankerType.CROSS_ENCODER:
                cross_encoder = self._get_cross_encoder()
                rerank_result = await cross_encoder.rerank(
                    query=query,
                    results=unique_results,
                    top_k=config.rerank_top_k,
                )
                unique_results = rerank_result.results

        # Step 4: Context assembly
        assembled = self._context_assembler.assemble(
            results=unique_results,
            strategy="relevance",
        )

        latency_ms = (time.time() - start_time) * 1000

        self.logger.info(
            "Advanced retrieval complete",
            total_candidates=total_candidates,
            final_count=len(assembled.items),
            latency_ms=round(latency_ms, 2),
        )

        return RetrievalResponse(
            context_items=assembled.items,
            search_results=unique_results,
            query_variations=query_variations,
            strategy_used=config.strategy,
            total_candidates=total_candidates,
            final_count=len(assembled.items),
            latency_ms=latency_ms,
        )

    def _deduplicate_results(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Deduplicate results by chunk_id, keeping highest score."""
        seen: dict[str, SearchResult] = {}

        for result in results:
            if result.chunk_id not in seen or result.score > seen[result.chunk_id].score:
                seen[result.chunk_id] = result

        # Sort by score descending
        deduped = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        return deduped


# Singleton instance
_retrieval_service: Optional[AdvancedRetrievalService] = None


def get_retrieval_service() -> AdvancedRetrievalService:
    """Get the global retrieval service instance."""
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = AdvancedRetrievalService()
    return _retrieval_service
