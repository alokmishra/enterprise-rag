"""
Tests for src/retrieval/reranking/
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.types import SearchResult
from src.retrieval.reranking.base import RerankResult


class TestReranker:
    """Tests for the base Reranker class."""

    def test_reranker_is_abstract(self):
        """Test that Reranker is abstract."""
        from src.retrieval.reranking.base import Reranker
        from abc import ABC

        assert issubclass(Reranker, ABC)

    def test_reranker_requires_rerank_method(self):
        """Test that Reranker requires rerank method."""
        from src.retrieval.reranking.base import Reranker

        assert hasattr(Reranker, 'rerank')


class TestRerankResult:
    """Tests for the RerankResult dataclass."""

    def test_rerank_result_creation(self):
        """Test RerankResult can be created."""
        result = RerankResult(
            results=[],
            original_count=5,
            reranked_count=3,
            latency_ms=10.5,
        )
        assert result.original_count == 5
        assert result.reranked_count == 3
        assert result.latency_ms == 10.5

    def test_rerank_result_with_results(self):
        """Test RerankResult with search results."""
        search_results = [
            SearchResult(chunk_id="c1", document_id="d1", content="Content 1", score=0.95, metadata={}),
            SearchResult(chunk_id="c2", document_id="d1", content="Content 2", score=0.85, metadata={}),
        ]

        result = RerankResult(
            results=search_results,
            original_count=5,
            reranked_count=2,
            latency_ms=15.0,
        )
        assert len(result.results) == 2
        assert result.results[0].score == 0.95


class TestLLMReranker:
    """Tests for the LLMReranker class."""

    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLM response."""
        return MagicMock(
            content='[{"id": "c1", "score": 0.9}, {"id": "c2", "score": 0.7}]',
        )

    @pytest.fixture
    def sample_results(self):
        """Create sample search results."""
        return [
            SearchResult(chunk_id="c1", document_id="d1", content="First document", score=0.8, metadata={}),
            SearchResult(chunk_id="c2", document_id="d1", content="Second document", score=0.7, metadata={}),
        ]

    def test_llm_reranker_creation(self):
        """Test LLMReranker can be created."""
        from src.retrieval.reranking.llm_reranker import LLMReranker

        reranker = LLMReranker()
        assert reranker is not None
        assert reranker.batch_size == 10
        assert reranker.score_threshold == 0.3

    def test_llm_reranker_custom_params(self):
        """Test LLMReranker with custom parameters."""
        from src.retrieval.reranking.llm_reranker import LLMReranker

        reranker = LLMReranker(batch_size=5, score_threshold=0.5)
        assert reranker.batch_size == 5
        assert reranker.score_threshold == 0.5

    @pytest.mark.asyncio
    async def test_llm_reranker_rerank(self, mock_llm_response, sample_results):
        """Test LLMReranker rerank method."""
        from src.retrieval.reranking.llm_reranker import LLMReranker

        with patch('src.retrieval.reranking.llm_reranker.get_default_llm_client') as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value=mock_llm_response)
            mock_get_llm.return_value = mock_llm

            reranker = LLMReranker()
            result = await reranker.rerank(
                query="test query",
                results=sample_results,
                top_k=2,
            )

            assert isinstance(result, RerankResult)
            assert result.original_count == 2

    @pytest.mark.asyncio
    async def test_llm_reranker_respects_top_k(self, mock_llm_response, sample_results):
        """Test LLMReranker respects top_k."""
        from src.retrieval.reranking.llm_reranker import LLMReranker

        with patch('src.retrieval.reranking.llm_reranker.get_default_llm_client') as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value=mock_llm_response)
            mock_get_llm.return_value = mock_llm

            reranker = LLMReranker()
            result = await reranker.rerank(
                query="test",
                results=sample_results + sample_results,
                top_k=2,
            )

            assert result.reranked_count <= 2

    @pytest.mark.asyncio
    async def test_llm_reranker_empty_results(self):
        """Test LLMReranker with empty results."""
        from src.retrieval.reranking.llm_reranker import LLMReranker

        reranker = LLMReranker()
        result = await reranker.rerank(
            query="test",
            results=[],
            top_k=5,
        )

        assert result.original_count == 0
        assert result.reranked_count == 0


class TestCrossEncoderReranker:
    """Tests for the CrossEncoderReranker class."""

    @pytest.fixture
    def sample_results(self):
        """Create sample search results."""
        return [
            SearchResult(chunk_id="c1", document_id="d1", content="First", score=0.8, metadata={}),
            SearchResult(chunk_id="c2", document_id="d1", content="Second", score=0.7, metadata={}),
            SearchResult(chunk_id="c3", document_id="d1", content="Third", score=0.6, metadata={}),
        ]

    def test_cross_encoder_reranker_creation(self):
        """Test CrossEncoderReranker can be created."""
        from src.retrieval.reranking.cross_encoder import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        assert reranker is not None
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_cross_encoder_reranker_default_model(self):
        """Test CrossEncoderReranker uses default model."""
        from src.retrieval.reranking.cross_encoder import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        assert reranker.model_name == CrossEncoderReranker.DEFAULT_MODEL

    @pytest.mark.asyncio
    async def test_cross_encoder_reranker_rerank(self, sample_results):
        """Test CrossEncoderReranker rerank method."""
        from src.retrieval.reranking.cross_encoder import CrossEncoderReranker

        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=[0.9, 0.7, 0.5])

        with patch.object(CrossEncoderReranker, '_get_model', return_value=mock_model):
            reranker = CrossEncoderReranker()
            result = await reranker.rerank(
                query="test query",
                results=sample_results,
                top_k=2,
            )

            assert isinstance(result, RerankResult)
            assert result.reranked_count <= 2

    @pytest.mark.asyncio
    async def test_cross_encoder_reranker_empty_results(self):
        """Test CrossEncoderReranker with empty results."""
        from src.retrieval.reranking.cross_encoder import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        result = await reranker.rerank(
            query="test",
            results=[],
            top_k=5,
        )

        assert result.original_count == 0
        assert result.reranked_count == 0


class TestCohereReranker:
    """Tests for the CohereReranker class."""

    def test_cohere_reranker_creation(self):
        """Test CohereReranker can be created."""
        from src.retrieval.reranking.cross_encoder import CohereReranker

        reranker = CohereReranker(api_key="test-key")
        assert reranker is not None
        assert reranker.model == "rerank-english-v3.0"

    def test_cohere_reranker_custom_model(self):
        """Test CohereReranker with custom model."""
        from src.retrieval.reranking.cross_encoder import CohereReranker

        reranker = CohereReranker(api_key="test-key", model="rerank-multilingual-v3.0")
        assert reranker.model == "rerank-multilingual-v3.0"

    @pytest.mark.asyncio
    async def test_cohere_reranker_empty_results(self):
        """Test CohereReranker with empty results."""
        from src.retrieval.reranking.cross_encoder import CohereReranker

        reranker = CohereReranker(api_key="test-key")
        result = await reranker.rerank(
            query="test",
            results=[],
            top_k=5,
        )

        assert result.original_count == 0
        assert result.reranked_count == 0

    @pytest.mark.asyncio
    async def test_cohere_reranker_rerank(self):
        """Test CohereReranker rerank method."""
        from src.retrieval.reranking.cross_encoder import CohereReranker

        sample_results = [
            SearchResult(chunk_id="c1", document_id="d1", content="First", score=0.8, metadata={}),
            SearchResult(chunk_id="c2", document_id="d1", content="Second", score=0.7, metadata={}),
        ]

        mock_response = MagicMock()
        mock_response.results = [
            MagicMock(index=0, relevance_score=0.95),
            MagicMock(index=1, relevance_score=0.85),
        ]

        mock_client = MagicMock()
        mock_client.rerank = MagicMock(return_value=mock_response)

        with patch.object(CohereReranker, '_get_client', return_value=mock_client):
            reranker = CohereReranker(api_key="test-key")
            result = await reranker.rerank(
                query="test",
                results=sample_results,
                top_k=2,
            )

            assert isinstance(result, RerankResult)
            assert result.reranked_count == 2
