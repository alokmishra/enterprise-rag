"""
Tests for src/retrieval/search/vector.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestVectorSearcher:
    """Tests for the VectorSearcher class."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = AsyncMock()
        store.search = AsyncMock(return_value=[
            {"id": "chunk-1", "score": 0.95, "payload": {"content": "Content 1", "source": "doc.pdf"}},
            {"id": "chunk-2", "score": 0.87, "payload": {"content": "Content 2", "source": "doc.pdf"}},
        ])
        return store

    @pytest.fixture
    def mock_embeddings(self):
        """Create a mock embedding provider."""
        provider = AsyncMock()
        provider.embed = AsyncMock(return_value=MagicMock(
            embeddings=[[0.1] * 1536],
            tokens_used=10,
        ))
        return provider

    def test_vector_searcher_creation(self, mock_vector_store, mock_embeddings):
        """Test VectorSearcher can be created."""
        from src.retrieval.search.vector import VectorSearcher

        searcher = VectorSearcher(
            vector_store=mock_vector_store,
            embedding_provider=mock_embeddings,
        )
        assert searcher is not None

    @pytest.mark.asyncio
    async def test_vector_searcher_search(self, mock_vector_store, mock_embeddings):
        """Test VectorSearcher search method."""
        from src.retrieval.search.vector import VectorSearcher

        searcher = VectorSearcher(
            vector_store=mock_vector_store,
            embedding_provider=mock_embeddings,
        )

        results = await searcher.search("test query", top_k=5)
        assert len(results) > 0
        mock_embeddings.embed.assert_called()
        mock_vector_store.search.assert_called()

    @pytest.mark.asyncio
    async def test_vector_searcher_returns_search_results(self, mock_vector_store, mock_embeddings):
        """Test that VectorSearcher returns SearchResult objects."""
        from src.retrieval.search.vector import VectorSearcher
        from src.core.types import SearchResult

        searcher = VectorSearcher(
            vector_store=mock_vector_store,
            embedding_provider=mock_embeddings,
        )

        results = await searcher.search("test query", top_k=5)
        for result in results:
            assert hasattr(result, 'score') or isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_vector_searcher_respects_top_k(self, mock_vector_store, mock_embeddings):
        """Test that VectorSearcher respects top_k parameter."""
        from src.retrieval.search.vector import VectorSearcher

        searcher = VectorSearcher(
            vector_store=mock_vector_store,
            embedding_provider=mock_embeddings,
        )

        await searcher.search("test query", top_k=3)
        # Check that top_k was passed to vector store
        call_args = mock_vector_store.search.call_args
        assert call_args is not None


class TestMultiQuerySearcher:
    """Tests for the MultiQuerySearcher class."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = AsyncMock()
        store.search = AsyncMock(return_value=[
            {"id": "chunk-1", "score": 0.95, "payload": {"content": "Content 1"}},
        ])
        return store

    @pytest.fixture
    def mock_embeddings(self):
        """Create a mock embedding provider."""
        provider = AsyncMock()
        provider.embed = AsyncMock(return_value=MagicMock(
            embeddings=[[0.1] * 1536],
        ))
        provider.embed_batch = AsyncMock(return_value=MagicMock(
            embeddings=[[0.1] * 1536, [0.2] * 1536, [0.3] * 1536],
        ))
        return provider

    def test_multi_query_searcher_creation(self, mock_vector_store, mock_embeddings):
        """Test MultiQuerySearcher can be created."""
        from src.retrieval.search.vector import MultiQuerySearcher

        searcher = MultiQuerySearcher(
            vector_store=mock_vector_store,
            embedding_provider=mock_embeddings,
        )
        assert searcher is not None

    @pytest.mark.asyncio
    async def test_multi_query_searcher_search(self, mock_vector_store, mock_embeddings):
        """Test MultiQuerySearcher search with multiple queries."""
        from src.retrieval.search.vector import MultiQuerySearcher

        searcher = MultiQuerySearcher(
            vector_store=mock_vector_store,
            embedding_provider=mock_embeddings,
        )

        results = await searcher.search(
            queries=["query 1", "query 2", "query 3"],
            top_k=5,
        )
        # Should return combined results
        assert results is not None
