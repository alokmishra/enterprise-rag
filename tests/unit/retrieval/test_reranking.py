"""
Tests for src/retrieval/reranking/
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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


class TestLLMReranker:
    """Tests for the LLMReranker class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = AsyncMock()
        client.generate = AsyncMock(return_value=MagicMock(
            content='[{"index": 0, "score": 0.9}, {"index": 1, "score": 0.7}]',
        ))
        return client

    def test_llm_reranker_creation(self, mock_llm_client):
        """Test LLMReranker can be created."""
        from src.retrieval.reranking.llm_reranker import LLMReranker

        reranker = LLMReranker(llm_client=mock_llm_client)
        assert reranker is not None

    @pytest.mark.asyncio
    async def test_llm_reranker_rerank(self, mock_llm_client):
        """Test LLMReranker rerank method."""
        from src.retrieval.reranking.llm_reranker import LLMReranker

        reranker = LLMReranker(llm_client=mock_llm_client)

        results = await reranker.rerank(
            query="test query",
            documents=[
                {"id": "doc-1", "content": "First document"},
                {"id": "doc-2", "content": "Second document"},
            ],
            top_k=2,
        )
        assert results is not None

    @pytest.mark.asyncio
    async def test_llm_reranker_respects_top_k(self, mock_llm_client):
        """Test LLMReranker respects top_k."""
        from src.retrieval.reranking.llm_reranker import LLMReranker

        reranker = LLMReranker(llm_client=mock_llm_client)

        results = await reranker.rerank(
            query="test",
            documents=[{"id": f"doc-{i}", "content": f"Doc {i}"} for i in range(10)],
            top_k=3,
        )
        assert len(results) <= 3


class TestCrossEncoderReranker:
    """Tests for the CrossEncoderReranker class."""

    def test_cross_encoder_reranker_creation(self):
        """Test CrossEncoderReranker can be created."""
        from src.retrieval.reranking.cross_encoder import CrossEncoderReranker

        with patch('src.retrieval.reranking.cross_encoder.CrossEncoder'):
            reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
            assert reranker is not None

    @pytest.mark.asyncio
    async def test_cross_encoder_reranker_rerank(self):
        """Test CrossEncoderReranker rerank method."""
        from src.retrieval.reranking.cross_encoder import CrossEncoderReranker

        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=[0.9, 0.7, 0.5])

        with patch('src.retrieval.reranking.cross_encoder.CrossEncoder', return_value=mock_model):
            reranker = CrossEncoderReranker()

            results = await reranker.rerank(
                query="test query",
                documents=[
                    {"id": "doc-1", "content": "First"},
                    {"id": "doc-2", "content": "Second"},
                    {"id": "doc-3", "content": "Third"},
                ],
                top_k=2,
            )
            assert len(results) <= 2


class TestCohereReranker:
    """Tests for the CohereReranker class."""

    def test_cohere_reranker_creation(self):
        """Test CohereReranker can be created."""
        from src.retrieval.reranking.cross_encoder import CohereReranker

        with patch.dict('os.environ', {'COHERE_API_KEY': 'test-key'}):
            reranker = CohereReranker()
            assert reranker is not None

    @pytest.mark.asyncio
    async def test_cohere_reranker_rerank(self):
        """Test CohereReranker rerank method."""
        from src.retrieval.reranking.cross_encoder import CohereReranker

        mock_response = MagicMock()
        mock_response.results = [
            MagicMock(index=0, relevance_score=0.95),
            MagicMock(index=1, relevance_score=0.85),
        ]

        with patch.dict('os.environ', {'COHERE_API_KEY': 'test-key'}):
            reranker = CohereReranker()

            with patch.object(reranker, 'client') as mock_client:
                mock_client.rerank = AsyncMock(return_value=mock_response)

                results = await reranker.rerank(
                    query="test",
                    documents=[
                        {"id": "doc-1", "content": "First"},
                        {"id": "doc-2", "content": "Second"},
                    ],
                    top_k=2,
                )


class TestRerankResult:
    """Tests for the RerankResult model."""

    def test_rerank_result_creation(self):
        """Test RerankResult can be created."""
        from src.retrieval.reranking.base import RerankResult

        result = RerankResult(
            id="doc-1",
            content="Document content",
            original_score=0.8,
            rerank_score=0.95,
        )
        assert result.id == "doc-1"
        assert result.rerank_score == 0.95

    def test_rerank_result_comparison(self):
        """Test RerankResult can be compared by score."""
        from src.retrieval.reranking.base import RerankResult

        result1 = RerankResult(id="doc-1", content="", original_score=0.8, rerank_score=0.95)
        result2 = RerankResult(id="doc-2", content="", original_score=0.9, rerank_score=0.85)

        # result1 should rank higher due to rerank_score
        assert result1.rerank_score > result2.rerank_score
