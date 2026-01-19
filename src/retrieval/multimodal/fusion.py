"""Multi-modal result fusion strategies."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from src.retrieval.multimodal.search import MultiModalSearchResult, SearchModality

logger = logging.getLogger(__name__)


class FusionStrategy(str, Enum):
    """Fusion strategies for multi-modal results."""
    WEIGHTED_SUM = "weighted_sum"
    RECIPROCAL_RANK = "reciprocal_rank"
    MAX_SCORE = "max_score"
    LEARNED = "learned"  # ML-based fusion


@dataclass
class FusionConfig:
    """Configuration for result fusion."""
    strategy: FusionStrategy = FusionStrategy.WEIGHTED_SUM
    text_weight: float = 0.5
    visual_weight: float = 0.3
    audio_weight: float = 0.2
    rrf_k: int = 60  # Reciprocal rank fusion constant
    normalize_scores: bool = True


class MultiModalFusion:
    """Fuse results from multiple modalities."""

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()

    def fuse(
        self,
        results_by_modality: dict[SearchModality, list[MultiModalSearchResult]],
        **kwargs,
    ) -> list[MultiModalSearchResult]:
        """Fuse results from multiple modalities."""
        if self.config.strategy == FusionStrategy.WEIGHTED_SUM:
            return self._weighted_sum_fusion(results_by_modality)
        elif self.config.strategy == FusionStrategy.RECIPROCAL_RANK:
            return self._reciprocal_rank_fusion(results_by_modality)
        elif self.config.strategy == FusionStrategy.MAX_SCORE:
            return self._max_score_fusion(results_by_modality)
        elif self.config.strategy == FusionStrategy.LEARNED:
            return self._learned_fusion(results_by_modality)
        else:
            return self._weighted_sum_fusion(results_by_modality)

    def _weighted_sum_fusion(
        self,
        results_by_modality: dict[SearchModality, list[MultiModalSearchResult]],
    ) -> list[MultiModalSearchResult]:
        """Fuse using weighted sum of scores."""
        doc_scores: dict[str, dict] = {}

        # Normalize scores per modality if configured
        if self.config.normalize_scores:
            results_by_modality = self._normalize_scores(results_by_modality)

        # Get weight for each modality
        weights = {
            SearchModality.TEXT: self.config.text_weight,
            SearchModality.IMAGE: self.config.visual_weight,
            SearchModality.AUDIO: self.config.audio_weight,
        }

        # Aggregate scores
        for modality, results in results_by_modality.items():
            weight = weights.get(modality, 0.3)

            for result in results:
                doc_id = result.id

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "result": result,
                        "weighted_score": 0.0,
                        "modality_scores": {},
                        "total_weight": 0.0,
                    }

                doc_scores[doc_id]["weighted_score"] += result.score * weight
                doc_scores[doc_id]["total_weight"] += weight
                doc_scores[doc_id]["modality_scores"][modality.value] = result.score

        # Create final results
        final_results = []
        for doc_id, data in doc_scores.items():
            result = data["result"]

            # Normalize by total weight
            if data["total_weight"] > 0:
                result.score = data["weighted_score"] / data["total_weight"]
            else:
                result.score = data["weighted_score"]

            result.modality_scores = data["modality_scores"]
            final_results.append(result)

        # Sort by fused score
        final_results.sort(key=lambda x: x.score, reverse=True)

        return final_results

    def _reciprocal_rank_fusion(
        self,
        results_by_modality: dict[SearchModality, list[MultiModalSearchResult]],
    ) -> list[MultiModalSearchResult]:
        """Fuse using Reciprocal Rank Fusion (RRF)."""
        k = self.config.rrf_k
        doc_scores: dict[str, dict] = {}

        for modality, results in results_by_modality.items():
            for rank, result in enumerate(results, start=1):
                doc_id = result.id

                # RRF score: 1 / (k + rank)
                rrf_score = 1 / (k + rank)

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "result": result,
                        "rrf_score": 0.0,
                        "modality_scores": {},
                        "ranks": {},
                    }

                doc_scores[doc_id]["rrf_score"] += rrf_score
                doc_scores[doc_id]["modality_scores"][modality.value] = result.score
                doc_scores[doc_id]["ranks"][modality.value] = rank

        # Create final results
        final_results = []
        for doc_id, data in doc_scores.items():
            result = data["result"]
            result.score = data["rrf_score"]
            result.modality_scores = data["modality_scores"]
            result.metadata["ranks"] = data["ranks"]
            final_results.append(result)

        final_results.sort(key=lambda x: x.score, reverse=True)

        return final_results

    def _max_score_fusion(
        self,
        results_by_modality: dict[SearchModality, list[MultiModalSearchResult]],
    ) -> list[MultiModalSearchResult]:
        """Fuse using max score across modalities."""
        doc_scores: dict[str, dict] = {}

        if self.config.normalize_scores:
            results_by_modality = self._normalize_scores(results_by_modality)

        for modality, results in results_by_modality.items():
            for result in results:
                doc_id = result.id

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "result": result,
                        "max_score": result.score,
                        "modality_scores": {},
                    }
                else:
                    if result.score > doc_scores[doc_id]["max_score"]:
                        doc_scores[doc_id]["max_score"] = result.score

                doc_scores[doc_id]["modality_scores"][modality.value] = result.score

        # Create final results
        final_results = []
        for doc_id, data in doc_scores.items():
            result = data["result"]
            result.score = data["max_score"]
            result.modality_scores = data["modality_scores"]
            final_results.append(result)

        final_results.sort(key=lambda x: x.score, reverse=True)

        return final_results

    def _learned_fusion(
        self,
        results_by_modality: dict[SearchModality, list[MultiModalSearchResult]],
    ) -> list[MultiModalSearchResult]:
        """Fuse using a learned fusion model."""
        # This would use a trained model to predict optimal fusion
        # For now, fall back to weighted sum
        logger.warning("Learned fusion not implemented, using weighted sum")
        return self._weighted_sum_fusion(results_by_modality)

    def _normalize_scores(
        self,
        results_by_modality: dict[SearchModality, list[MultiModalSearchResult]],
    ) -> dict[SearchModality, list[MultiModalSearchResult]]:
        """Normalize scores within each modality to [0, 1]."""
        normalized = {}

        for modality, results in results_by_modality.items():
            if not results:
                normalized[modality] = results
                continue

            # Find min and max scores
            scores = [r.score for r in results]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score

            # Normalize
            if score_range > 0:
                for result in results:
                    result.score = (result.score - min_score) / score_range
            else:
                for result in results:
                    result.score = 1.0

            normalized[modality] = results

        return normalized


class CrossModalReranker:
    """Rerank results using cross-modal understanding."""

    def __init__(
        self,
        model: str = "clip",
        config: Optional[dict] = None,
    ):
        self.model = model
        self.config = config or {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize cross-modal reranker."""
        self._initialized = True

    async def rerank(
        self,
        query_text: str,
        query_image: Optional[bytes] = None,
        results: list[MultiModalSearchResult] = None,
    ) -> list[MultiModalSearchResult]:
        """Rerank results using cross-modal similarity."""
        if not results:
            return []

        if not self._initialized:
            await self.initialize()

        if self.model == "clip":
            return await self._rerank_with_clip(query_text, query_image, results)

        return results

    async def _rerank_with_clip(
        self,
        query_text: str,
        query_image: Optional[bytes],
        results: list[MultiModalSearchResult],
    ) -> list[MultiModalSearchResult]:
        """Rerank using CLIP model."""
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel

            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            # Encode query text
            text_inputs = processor(text=[query_text], return_tensors="pt", padding=True)
            with torch.no_grad():
                text_features = model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Score each result
            for result in results:
                cross_modal_score = 0.0

                # If result has visual embedding, compute similarity
                if result.content.visual_embedding:
                    visual_tensor = torch.tensor([result.content.visual_embedding])
                    visual_tensor = visual_tensor / visual_tensor.norm(dim=-1, keepdim=True)

                    similarity = (text_features @ visual_tensor.T).item()
                    cross_modal_score = max(cross_modal_score, similarity)

                # Update score with cross-modal component
                result.score = 0.7 * result.score + 0.3 * cross_modal_score
                result.metadata["cross_modal_score"] = cross_modal_score

            # Sort by updated score
            results.sort(key=lambda x: x.score, reverse=True)

            return results

        except ImportError:
            logger.warning("CLIP not available for cross-modal reranking")
            return results
        except Exception as e:
            logger.error(f"Cross-modal reranking failed: {e}")
            return results
