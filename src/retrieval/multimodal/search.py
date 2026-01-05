"""Multi-modal search for RAG."""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union
from pathlib import Path

from src.ingestion.multimodal.base import ModalityType, MultiModalContent
from src.ingestion.multimodal.image import ImageEmbedder
from src.ingestion.multimodal.audio import AudioEmbedder

logger = logging.getLogger(__name__)


class SearchModality(str, Enum):
    """Search modality types."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    HYBRID = "hybrid"  # Combined text + other modality


@dataclass
class MultiModalQuery:
    """A multi-modal search query."""
    text: Optional[str] = None
    image: Optional[Union[bytes, str, Path]] = None
    audio: Optional[Union[bytes, str, Path]] = None

    # Query configuration
    modality: SearchModality = SearchModality.TEXT
    top_k: int = 10
    min_score: float = 0.0

    # Weights for hybrid search
    text_weight: float = 0.5
    visual_weight: float = 0.3
    audio_weight: float = 0.2

    # Filters
    filters: dict[str, Any] = field(default_factory=dict)
    tenant_id: Optional[str] = None

    # Search options
    rerank: bool = True
    expand_query: bool = False


@dataclass
class MultiModalSearchResult:
    """A single search result."""
    id: str
    score: float
    content: MultiModalContent
    modality_scores: dict[str, float] = field(default_factory=dict)
    highlights: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiModalSearchResponse:
    """Response from multi-modal search."""
    results: list[MultiModalSearchResult]
    total_count: int
    query_time_ms: float
    modalities_searched: list[SearchModality]
    metadata: dict[str, Any] = field(default_factory=dict)


class MultiModalSearcher:
    """Search across text, image, and audio embeddings."""

    def __init__(
        self,
        vector_store=None,
        text_embedder=None,
        image_embedder: Optional[ImageEmbedder] = None,
        audio_embedder: Optional[AudioEmbedder] = None,
        config: Optional[dict] = None,
    ):
        self.vector_store = vector_store
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder or ImageEmbedder()
        self.audio_embedder = audio_embedder or AudioEmbedder()
        self.config = config or {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize searcher components."""
        await asyncio.gather(
            self.image_embedder.initialize(),
            self.audio_embedder.initialize(),
        )
        self._initialized = True

    async def search(
        self,
        query: MultiModalQuery,
        **kwargs,
    ) -> MultiModalSearchResponse:
        """Perform multi-modal search."""
        import time
        start_time = time.time()

        if not self._initialized:
            await self.initialize()

        # Determine which modalities to search
        modalities = self._determine_modalities(query)

        # Generate query embeddings for each modality
        query_embeddings = await self._generate_query_embeddings(query, modalities)

        # Search each modality
        all_results = []
        for modality, embedding in query_embeddings.items():
            if embedding is not None:
                results = await self._search_modality(
                    modality=modality,
                    embedding=embedding,
                    query=query,
                )
                all_results.extend(results)

        # Fuse results from different modalities
        fused_results = self._fuse_results(all_results, query)

        # Apply filters
        filtered_results = self._apply_filters(fused_results, query.filters)

        # Rerank if requested
        if query.rerank and query.text:
            filtered_results = await self._rerank_results(
                results=filtered_results,
                query_text=query.text,
            )

        # Limit to top_k
        final_results = filtered_results[:query.top_k]

        query_time = (time.time() - start_time) * 1000

        return MultiModalSearchResponse(
            results=final_results,
            total_count=len(filtered_results),
            query_time_ms=query_time,
            modalities_searched=modalities,
            metadata={
                "embeddings_generated": list(query_embeddings.keys()),
                "pre_fusion_count": len(all_results),
            },
        )

    def _determine_modalities(self, query: MultiModalQuery) -> list[SearchModality]:
        """Determine which modalities to search based on query."""
        modalities = []

        if query.modality == SearchModality.HYBRID:
            if query.text:
                modalities.append(SearchModality.TEXT)
            if query.image:
                modalities.append(SearchModality.IMAGE)
            if query.audio:
                modalities.append(SearchModality.AUDIO)
        else:
            modalities.append(query.modality)

        return modalities if modalities else [SearchModality.TEXT]

    async def _generate_query_embeddings(
        self,
        query: MultiModalQuery,
        modalities: list[SearchModality],
    ) -> dict[SearchModality, Optional[list[float]]]:
        """Generate embeddings for query content."""
        embeddings = {}

        tasks = []
        modality_names = []

        for modality in modalities:
            if modality == SearchModality.TEXT and query.text:
                tasks.append(self._embed_text(query.text))
                modality_names.append(SearchModality.TEXT)

            elif modality == SearchModality.IMAGE and query.image:
                tasks.append(self.image_embedder.embed(query.image))
                modality_names.append(SearchModality.IMAGE)

            elif modality == SearchModality.AUDIO and query.audio:
                tasks.append(self.audio_embedder.embed(query.audio))
                modality_names.append(SearchModality.AUDIO)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for modality, result in zip(modality_names, results):
                if isinstance(result, Exception):
                    logger.error(f"Embedding generation failed for {modality}: {result}")
                    embeddings[modality] = None
                else:
                    embeddings[modality] = result

        return embeddings

    async def _embed_text(self, text: str) -> list[float]:
        """Generate text embedding."""
        if self.text_embedder:
            return await self.text_embedder.embed(text)

        # Fallback: use simple embedding
        from src.ingestion.embeddings import get_embedder
        embedder = await get_embedder()
        return await embedder.embed(text)

    async def _search_modality(
        self,
        modality: SearchModality,
        embedding: list[float],
        query: MultiModalQuery,
    ) -> list[MultiModalSearchResult]:
        """Search a specific modality."""
        if not self.vector_store:
            return []

        # Determine collection name based on modality
        collection_map = {
            SearchModality.TEXT: "text_embeddings",
            SearchModality.IMAGE: "image_embeddings",
            SearchModality.AUDIO: "audio_embeddings",
        }
        collection = collection_map.get(modality, "text_embeddings")

        # Build filter
        search_filter = {}
        if query.tenant_id:
            search_filter["tenant_id"] = query.tenant_id

        # Search vector store
        try:
            results = await self.vector_store.search(
                collection=collection,
                query_vector=embedding,
                limit=query.top_k * 2,  # Get more for fusion
                filter=search_filter,
            )

            return [
                MultiModalSearchResult(
                    id=r.id,
                    score=r.score,
                    content=self._result_to_content(r, modality),
                    modality_scores={modality.value: r.score},
                    metadata=r.payload or {},
                )
                for r in results
            ]
        except Exception as e:
            logger.error(f"Search failed for {modality}: {e}")
            return []

    def _result_to_content(self, result: Any, modality: SearchModality) -> MultiModalContent:
        """Convert vector store result to MultiModalContent."""
        payload = result.payload or {}

        modality_map = {
            SearchModality.TEXT: ModalityType.TEXT,
            SearchModality.IMAGE: ModalityType.IMAGE,
            SearchModality.AUDIO: ModalityType.AUDIO,
        }

        return MultiModalContent(
            id=result.id,
            modality=modality_map.get(modality, ModalityType.TEXT),
            text_content=payload.get("text", ""),
            source_path=payload.get("source_path"),
            metadata=payload,
        )

    def _fuse_results(
        self,
        results: list[MultiModalSearchResult],
        query: MultiModalQuery,
    ) -> list[MultiModalSearchResult]:
        """Fuse results from multiple modalities."""
        if not results:
            return []

        # Group by document ID
        doc_results: dict[str, MultiModalSearchResult] = {}

        for result in results:
            doc_id = result.id

            if doc_id not in doc_results:
                doc_results[doc_id] = result
            else:
                # Merge modality scores
                existing = doc_results[doc_id]
                existing.modality_scores.update(result.modality_scores)

        # Calculate fused scores
        for doc_id, result in doc_results.items():
            scores = result.modality_scores
            fused_score = 0.0
            total_weight = 0.0

            if SearchModality.TEXT.value in scores:
                fused_score += scores[SearchModality.TEXT.value] * query.text_weight
                total_weight += query.text_weight

            if SearchModality.IMAGE.value in scores:
                fused_score += scores[SearchModality.IMAGE.value] * query.visual_weight
                total_weight += query.visual_weight

            if SearchModality.AUDIO.value in scores:
                fused_score += scores[SearchModality.AUDIO.value] * query.audio_weight
                total_weight += query.audio_weight

            # Normalize
            if total_weight > 0:
                result.score = fused_score / total_weight

        # Sort by fused score
        sorted_results = sorted(
            doc_results.values(),
            key=lambda x: x.score,
            reverse=True,
        )

        return sorted_results

    def _apply_filters(
        self,
        results: list[MultiModalSearchResult],
        filters: dict[str, Any],
    ) -> list[MultiModalSearchResult]:
        """Apply metadata filters to results."""
        if not filters:
            return results

        filtered = []
        for result in results:
            matches = True
            for key, value in filters.items():
                result_value = result.metadata.get(key)
                if isinstance(value, list):
                    if result_value not in value:
                        matches = False
                        break
                elif result_value != value:
                    matches = False
                    break

            if matches:
                filtered.append(result)

        return filtered

    async def _rerank_results(
        self,
        results: list[MultiModalSearchResult],
        query_text: str,
    ) -> list[MultiModalSearchResult]:
        """Rerank results using cross-encoder."""
        try:
            from src.retrieval.reranking import get_reranker

            reranker = await get_reranker()

            # Prepare pairs
            pairs = [
                (query_text, r.content.text_content)
                for r in results
                if r.content.text_content
            ]

            if not pairs:
                return results

            # Get reranking scores
            scores = await reranker.rerank(pairs)

            # Update result scores
            for i, score in enumerate(scores):
                if i < len(results):
                    results[i].score = score
                    results[i].metadata["rerank_score"] = score

            # Sort by new scores
            return sorted(results, key=lambda x: x.score, reverse=True)

        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return results

    async def search_similar_images(
        self,
        image: Union[bytes, str, Path],
        top_k: int = 10,
        tenant_id: Optional[str] = None,
    ) -> list[MultiModalSearchResult]:
        """Search for similar images."""
        query = MultiModalQuery(
            image=image,
            modality=SearchModality.IMAGE,
            top_k=top_k,
            tenant_id=tenant_id,
            rerank=False,
        )
        response = await self.search(query)
        return response.results

    async def search_by_audio(
        self,
        audio: Union[bytes, str, Path],
        top_k: int = 10,
        tenant_id: Optional[str] = None,
    ) -> list[MultiModalSearchResult]:
        """Search for content similar to audio."""
        query = MultiModalQuery(
            audio=audio,
            modality=SearchModality.AUDIO,
            top_k=top_k,
            tenant_id=tenant_id,
            rerank=False,
        )
        response = await self.search(query)
        return response.results

    async def hybrid_search(
        self,
        text: str,
        image: Optional[Union[bytes, str, Path]] = None,
        audio: Optional[Union[bytes, str, Path]] = None,
        top_k: int = 10,
        text_weight: float = 0.5,
        visual_weight: float = 0.3,
        audio_weight: float = 0.2,
        tenant_id: Optional[str] = None,
    ) -> list[MultiModalSearchResult]:
        """Perform hybrid multi-modal search."""
        query = MultiModalQuery(
            text=text,
            image=image,
            audio=audio,
            modality=SearchModality.HYBRID,
            top_k=top_k,
            text_weight=text_weight,
            visual_weight=visual_weight,
            audio_weight=audio_weight,
            tenant_id=tenant_id,
        )
        response = await self.search(query)
        return response.results
