"""Tests for multi-modal retrieval."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from src.retrieval.multimodal.search import (
    MultiModalSearcher,
    MultiModalQuery,
    MultiModalSearchResult,
    MultiModalSearchResponse,
    SearchModality,
)
from src.retrieval.multimodal.fusion import (
    MultiModalFusion,
    FusionStrategy,
    FusionConfig,
)
from src.ingestion.multimodal.base import ModalityType, MultiModalContent


class TestMultiModalQuery:
    """Tests for MultiModalQuery."""

    def test_create_text_query(self):
        """Test creating text query."""
        query = MultiModalQuery(
            text="What is machine learning?",
            modality=SearchModality.TEXT,
            top_k=10,
        )

        assert query.text == "What is machine learning?"
        assert query.modality == SearchModality.TEXT
        assert query.top_k == 10

    def test_create_image_query(self):
        """Test creating image query."""
        query = MultiModalQuery(
            image=b"image_bytes",
            modality=SearchModality.IMAGE,
            top_k=5,
        )

        assert query.image == b"image_bytes"
        assert query.modality == SearchModality.IMAGE

    def test_create_hybrid_query(self):
        """Test creating hybrid query."""
        query = MultiModalQuery(
            text="Find similar images",
            image=b"image_bytes",
            modality=SearchModality.HYBRID,
            text_weight=0.4,
            visual_weight=0.6,
        )

        assert query.text is not None
        assert query.image is not None
        assert query.modality == SearchModality.HYBRID
        assert query.text_weight == 0.4

    def test_query_with_filters(self):
        """Test query with filters."""
        query = MultiModalQuery(
            text="Search query",
            filters={"category": "science", "year": 2024},
            tenant_id="tenant-123",
        )

        assert query.filters["category"] == "science"
        assert query.tenant_id == "tenant-123"


class TestMultiModalSearchResult:
    """Tests for MultiModalSearchResult."""

    def test_create_result(self):
        """Test creating search result."""
        content = MultiModalContent(
            id="doc-123",
            modality=ModalityType.TEXT,
            text_content="Result content",
        )

        result = MultiModalSearchResult(
            id="doc-123",
            score=0.95,
            content=content,
            modality_scores={"text": 0.95},
            highlights=["result content"],
        )

        assert result.id == "doc-123"
        assert result.score == 0.95
        assert result.content.text_content == "Result content"


class TestMultiModalSearcher:
    """Tests for MultiModalSearcher."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = AsyncMock()
        return store

    @pytest.fixture
    def searcher(self, mock_vector_store):
        """Create searcher."""
        return MultiModalSearcher(
            vector_store=mock_vector_store,
        )

    @pytest.mark.asyncio
    async def test_initialize(self, searcher):
        """Test searcher initialization."""
        await searcher.initialize()
        assert searcher._initialized

    def test_determine_modalities_text(self, searcher):
        """Test modality determination for text query."""
        query = MultiModalQuery(text="test", modality=SearchModality.TEXT)
        modalities = searcher._determine_modalities(query)

        assert SearchModality.TEXT in modalities

    def test_determine_modalities_hybrid(self, searcher):
        """Test modality determination for hybrid query."""
        query = MultiModalQuery(
            text="test",
            image=b"img",
            modality=SearchModality.HYBRID,
        )
        modalities = searcher._determine_modalities(query)

        assert SearchModality.TEXT in modalities
        assert SearchModality.IMAGE in modalities

    @pytest.mark.asyncio
    async def test_search_text(self, searcher, mock_vector_store):
        """Test text search."""
        mock_vector_store.search.return_value = [
            MagicMock(id="doc1", score=0.9, payload={"text": "Result 1"}),
            MagicMock(id="doc2", score=0.8, payload={"text": "Result 2"}),
        ]

        with patch.object(searcher, "_embed_text") as mock_embed:
            mock_embed.return_value = [0.1] * 512
            searcher._initialized = True

            query = MultiModalQuery(text="test query", top_k=10)
            response = await searcher.search(query)

            assert isinstance(response, MultiModalSearchResponse)
            assert len(response.results) <= 10

    @pytest.mark.asyncio
    async def test_search_image(self, searcher, mock_vector_store):
        """Test image search."""
        mock_vector_store.search.return_value = [
            MagicMock(id="img1", score=0.85, payload={}),
        ]

        with patch.object(searcher.image_embedder, "embed") as mock_embed:
            mock_embed.return_value = [0.1] * 512
            searcher._initialized = True

            query = MultiModalQuery(
                image=b"image_bytes",
                modality=SearchModality.IMAGE,
            )
            response = await searcher.search(query)

            assert SearchModality.IMAGE in response.modalities_searched

    @pytest.mark.asyncio
    async def test_search_hybrid(self, searcher, mock_vector_store):
        """Test hybrid search."""
        mock_vector_store.search.return_value = [
            MagicMock(id="doc1", score=0.9, payload={"text": "Result"}),
        ]

        with patch.object(searcher, "_embed_text") as mock_text:
            with patch.object(searcher.image_embedder, "embed") as mock_img:
                mock_text.return_value = [0.1] * 512
                mock_img.return_value = [0.2] * 512
                searcher._initialized = True

                query = MultiModalQuery(
                    text="test",
                    image=b"image",
                    modality=SearchModality.HYBRID,
                )
                response = await searcher.search(query)

                assert len(response.modalities_searched) >= 1

    def test_fuse_results(self, searcher):
        """Test result fusion."""
        results = [
            MultiModalSearchResult(
                id="doc1",
                score=0.9,
                content=MultiModalContent(id="doc1", modality=ModalityType.TEXT),
                modality_scores={"text": 0.9},
            ),
            MultiModalSearchResult(
                id="doc1",
                score=0.8,
                content=MultiModalContent(id="doc1", modality=ModalityType.IMAGE),
                modality_scores={"image": 0.8},
            ),
            MultiModalSearchResult(
                id="doc2",
                score=0.7,
                content=MultiModalContent(id="doc2", modality=ModalityType.TEXT),
                modality_scores={"text": 0.7},
            ),
        ]

        query = MultiModalQuery(text="test", text_weight=0.5, visual_weight=0.5)
        fused = searcher._fuse_results(results, query)

        # doc1 should have combined score from both modalities
        assert len(fused) == 2
        doc1 = next(r for r in fused if r.id == "doc1")
        assert "text" in doc1.modality_scores
        assert "image" in doc1.modality_scores

    def test_apply_filters(self, searcher):
        """Test filter application."""
        results = [
            MultiModalSearchResult(
                id="doc1",
                score=0.9,
                content=MultiModalContent(id="doc1", modality=ModalityType.TEXT),
                metadata={"category": "science"},
            ),
            MultiModalSearchResult(
                id="doc2",
                score=0.8,
                content=MultiModalContent(id="doc2", modality=ModalityType.TEXT),
                metadata={"category": "art"},
            ),
        ]

        filtered = searcher._apply_filters(results, {"category": "science"})

        assert len(filtered) == 1
        assert filtered[0].id == "doc1"

    @pytest.mark.asyncio
    async def test_search_similar_images(self, searcher, mock_vector_store):
        """Test similar image search."""
        mock_vector_store.search.return_value = []

        with patch.object(searcher.image_embedder, "embed") as mock_embed:
            mock_embed.return_value = [0.1] * 512
            searcher._initialized = True

            results = await searcher.search_similar_images(b"image", top_k=5)

            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hybrid_search_method(self, searcher, mock_vector_store):
        """Test hybrid_search convenience method."""
        mock_vector_store.search.return_value = []

        with patch.object(searcher, "_embed_text") as mock_text:
            mock_text.return_value = [0.1] * 512
            searcher._initialized = True

            results = await searcher.hybrid_search(
                text="query",
                image=b"image",
                top_k=10,
            )

            assert isinstance(results, list)


class TestMultiModalFusion:
    """Tests for result fusion strategies."""

    @pytest.fixture
    def fusion(self):
        """Create fusion instance."""
        return MultiModalFusion(FusionConfig(
            strategy=FusionStrategy.WEIGHTED_SUM,
            text_weight=0.5,
            visual_weight=0.3,
            audio_weight=0.2,
        ))

    def test_weighted_sum_fusion(self, fusion):
        """Test weighted sum fusion."""
        results_by_modality = {
            SearchModality.TEXT: [
                MultiModalSearchResult(
                    id="doc1",
                    score=0.9,
                    content=MultiModalContent(id="doc1", modality=ModalityType.TEXT),
                ),
            ],
            SearchModality.IMAGE: [
                MultiModalSearchResult(
                    id="doc1",
                    score=0.8,
                    content=MultiModalContent(id="doc1", modality=ModalityType.IMAGE),
                ),
            ],
        }

        fused = fusion.fuse(results_by_modality)

        assert len(fused) == 1
        assert fused[0].id == "doc1"
        # Score should be weighted average
        assert 0 < fused[0].score <= 1

    def test_reciprocal_rank_fusion(self):
        """Test RRF fusion."""
        fusion = MultiModalFusion(FusionConfig(
            strategy=FusionStrategy.RECIPROCAL_RANK,
            rrf_k=60,
        ))

        results_by_modality = {
            SearchModality.TEXT: [
                MultiModalSearchResult(
                    id="doc1", score=0.9,
                    content=MultiModalContent(id="doc1", modality=ModalityType.TEXT),
                ),
                MultiModalSearchResult(
                    id="doc2", score=0.8,
                    content=MultiModalContent(id="doc2", modality=ModalityType.TEXT),
                ),
            ],
            SearchModality.IMAGE: [
                MultiModalSearchResult(
                    id="doc2", score=0.95,
                    content=MultiModalContent(id="doc2", modality=ModalityType.IMAGE),
                ),
                MultiModalSearchResult(
                    id="doc1", score=0.7,
                    content=MultiModalContent(id="doc1", modality=ModalityType.IMAGE),
                ),
            ],
        }

        fused = fusion.fuse(results_by_modality)

        # doc2 should rank higher (rank 1 in image, rank 2 in text)
        assert len(fused) == 2
        assert "ranks" in fused[0].metadata

    def test_max_score_fusion(self):
        """Test max score fusion."""
        fusion = MultiModalFusion(FusionConfig(
            strategy=FusionStrategy.MAX_SCORE,
        ))

        results_by_modality = {
            SearchModality.TEXT: [
                MultiModalSearchResult(
                    id="doc1", score=0.7,
                    content=MultiModalContent(id="doc1", modality=ModalityType.TEXT),
                ),
            ],
            SearchModality.IMAGE: [
                MultiModalSearchResult(
                    id="doc1", score=0.9,
                    content=MultiModalContent(id="doc1", modality=ModalityType.IMAGE),
                ),
            ],
        }

        fused = fusion.fuse(results_by_modality)

        # Should use max score (0.9 from image)
        assert fused[0].score == 0.9

    def test_normalize_scores(self, fusion):
        """Test score normalization."""
        results_by_modality = {
            SearchModality.TEXT: [
                MultiModalSearchResult(
                    id="doc1", score=100,
                    content=MultiModalContent(id="doc1", modality=ModalityType.TEXT),
                ),
                MultiModalSearchResult(
                    id="doc2", score=50,
                    content=MultiModalContent(id="doc2", modality=ModalityType.TEXT),
                ),
            ],
        }

        normalized = fusion._normalize_scores(results_by_modality)

        # Scores should be normalized to [0, 1]
        scores = [r.score for r in normalized[SearchModality.TEXT]]
        assert max(scores) == 1.0
        assert min(scores) == 0.0

    def test_empty_results(self, fusion):
        """Test fusion with empty results."""
        fused = fusion.fuse({})
        assert fused == []

    def test_single_modality(self, fusion):
        """Test fusion with single modality."""
        results_by_modality = {
            SearchModality.TEXT: [
                MultiModalSearchResult(
                    id="doc1", score=0.9,
                    content=MultiModalContent(id="doc1", modality=ModalityType.TEXT),
                ),
            ],
        }

        fused = fusion.fuse(results_by_modality)

        assert len(fused) == 1
        assert fused[0].score > 0
