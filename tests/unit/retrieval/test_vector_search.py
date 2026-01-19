"""
Tests for src/retrieval/search/vector.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.types import SearchResult, RetrievalResult, RetrievalStrategy


class TestVectorSearcher:
    """Tests for the VectorSearcher class."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = AsyncMock()
        store.search = AsyncMock(return_value=[
            MagicMock(id="chunk-1", score=0.95, payload={"content": "Content 1", "document_id": "doc-1", "source": "doc.pdf", "metadata": {}}),
            MagicMock(id="chunk-2", score=0.87, payload={"content": "Content 2", "document_id": "doc-1", "source": "doc.pdf", "metadata": {}}),
        ])
        return store

    @pytest.fixture
    def mock_embedding_provider(self):
        """Create a mock embedding provider."""
        provider = AsyncMock()
        provider.embed_query = AsyncMock(return_value=[0.1] * 1536)
        return provider

    def test_vector_searcher_creation(self):
        """Test VectorSearcher can be created."""
        from src.retrieval.search.vector import VectorSearcher

        searcher = VectorSearcher()
        assert searcher is not None

    def test_vector_searcher_with_collection_name(self):
        """Test VectorSearcher with custom collection name."""
        from src.retrieval.search.vector import VectorSearcher

        searcher = VectorSearcher(collection_name="custom_collection", top_k=20)
        assert searcher.collection_name == "custom_collection"
        assert searcher.top_k == 20

    @pytest.mark.asyncio
    async def test_vector_searcher_search(self, mock_vector_store, mock_embedding_provider):
        """Test VectorSearcher search method."""
        from src.retrieval.search.vector import VectorSearcher

        with patch('src.retrieval.search.vector.get_embedding_provider', return_value=mock_embedding_provider), \
             patch('src.retrieval.search.vector.get_vector_store', return_value=mock_vector_store):
            searcher = VectorSearcher()
            result = await searcher.search("test query", top_k=5)

            assert isinstance(result, RetrievalResult)
            assert result.strategy == RetrievalStrategy.VECTOR
            assert len(result.results) > 0
            mock_embedding_provider.embed_query.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_vector_searcher_returns_search_results(self, mock_vector_store, mock_embedding_provider):
        """Test that VectorSearcher returns SearchResult objects."""
        from src.retrieval.search.vector import VectorSearcher

        with patch('src.retrieval.search.vector.get_embedding_provider', return_value=mock_embedding_provider), \
             patch('src.retrieval.search.vector.get_vector_store', return_value=mock_vector_store):
            searcher = VectorSearcher()
            result = await searcher.search("test query", top_k=5)

            for search_result in result.results:
                assert isinstance(search_result, SearchResult)
                assert hasattr(search_result, 'score')
                assert hasattr(search_result, 'chunk_id')
                assert hasattr(search_result, 'content')

    @pytest.mark.asyncio
    async def test_vector_searcher_respects_top_k(self, mock_vector_store, mock_embedding_provider):
        """Test that VectorSearcher respects top_k parameter."""
        from src.retrieval.search.vector import VectorSearcher

        with patch('src.retrieval.search.vector.get_embedding_provider', return_value=mock_embedding_provider), \
             patch('src.retrieval.search.vector.get_vector_store', return_value=mock_vector_store):
            searcher = VectorSearcher()
            await searcher.search("test query", top_k=3)

            call_args = mock_vector_store.search.call_args
            assert call_args is not None
            assert call_args.kwargs.get('top_k') == 3

    @pytest.mark.asyncio
    async def test_vector_searcher_with_filters(self, mock_vector_store, mock_embedding_provider):
        """Test VectorSearcher with filters."""
        from src.retrieval.search.vector import VectorSearcher

        with patch('src.retrieval.search.vector.get_embedding_provider', return_value=mock_embedding_provider), \
             patch('src.retrieval.search.vector.get_vector_store', return_value=mock_vector_store):
            searcher = VectorSearcher()
            filters = {"document_type": "pdf"}
            await searcher.search("test query", top_k=5, filters=filters)

            call_args = mock_vector_store.search.call_args
            assert call_args.kwargs.get('filters') == filters

    @pytest.mark.asyncio
    async def test_vector_searcher_with_score_threshold(self, mock_vector_store, mock_embedding_provider):
        """Test VectorSearcher with score threshold."""
        from src.retrieval.search.vector import VectorSearcher

        mock_vector_store.search = AsyncMock(return_value=[
            MagicMock(id="chunk-1", score=0.95, payload={"content": "Content 1", "document_id": "doc-1", "metadata": {}}),
            MagicMock(id="chunk-2", score=0.30, payload={"content": "Content 2", "document_id": "doc-1", "metadata": {}}),
        ])

        with patch('src.retrieval.search.vector.get_embedding_provider', return_value=mock_embedding_provider), \
             patch('src.retrieval.search.vector.get_vector_store', return_value=mock_vector_store):
            searcher = VectorSearcher()
            result = await searcher.search("test query", score_threshold=0.5)

            assert len(result.results) == 1
            assert result.results[0].score >= 0.5

    @pytest.mark.asyncio
    async def test_vector_searcher_search_by_embedding(self, mock_vector_store):
        """Test VectorSearcher search_by_embedding method."""
        from src.retrieval.search.vector import VectorSearcher

        with patch('src.retrieval.search.vector.get_vector_store', return_value=mock_vector_store):
            searcher = VectorSearcher()
            embedding = [0.1] * 1536
            results = await searcher.search_by_embedding(embedding, top_k=5)

            assert len(results) > 0
            for result in results:
                assert isinstance(result, SearchResult)


class TestMultiQuerySearcher:
    """Tests for the MultiQuerySearcher class."""

    @pytest.fixture
    def mock_vector_searcher(self):
        """Create a mock vector searcher."""
        searcher = MagicMock()
        searcher.search = AsyncMock(return_value=RetrievalResult(
            query="test query",
            strategy=RetrievalStrategy.VECTOR,
            results=[
                SearchResult(chunk_id="chunk-1", document_id="doc-1", content="Content 1", score=0.95, metadata={}),
            ],
            total_found=1,
            latency_ms=10.0,
        ))
        searcher.top_k = 10
        return searcher

    def test_multi_query_searcher_creation(self, mock_vector_searcher):
        """Test MultiQuerySearcher can be created."""
        from src.retrieval.search.vector import MultiQuerySearcher

        searcher = MultiQuerySearcher(vector_searcher=mock_vector_searcher)
        assert searcher is not None

    def test_multi_query_searcher_num_variations(self, mock_vector_searcher):
        """Test MultiQuerySearcher with custom num_variations."""
        from src.retrieval.search.vector import MultiQuerySearcher

        searcher = MultiQuerySearcher(vector_searcher=mock_vector_searcher, num_variations=5)
        assert searcher.num_variations == 5

    @pytest.mark.asyncio
    async def test_multi_query_searcher_search(self, mock_vector_searcher):
        """Test MultiQuerySearcher search method."""
        from src.retrieval.search.vector import MultiQuerySearcher

        searcher = MultiQuerySearcher(vector_searcher=mock_vector_searcher)
        result = await searcher.search(query="test query", top_k=5)

        assert isinstance(result, RetrievalResult)
        assert result.strategy == RetrievalStrategy.MULTI_QUERY

    @pytest.mark.asyncio
    async def test_multi_query_searcher_deduplicates(self, mock_vector_searcher):
        """Test MultiQuerySearcher deduplicates results."""
        from src.retrieval.search.vector import MultiQuerySearcher

        searcher = MultiQuerySearcher(vector_searcher=mock_vector_searcher)
        result = await searcher.search(query="test query", top_k=10)

        chunk_ids = [r.chunk_id for r in result.results]
        assert len(chunk_ids) == len(set(chunk_ids))

    @pytest.mark.asyncio
    async def test_multi_query_searcher_with_filters(self, mock_vector_searcher):
        """Test MultiQuerySearcher with filters."""
        from src.retrieval.search.vector import MultiQuerySearcher

        searcher = MultiQuerySearcher(vector_searcher=mock_vector_searcher)
        filters = {"document_type": "pdf"}
        await searcher.search(query="test query", top_k=5, filters=filters)

        call_args = mock_vector_searcher.search.call_args
        assert call_args.kwargs.get('filters') == filters


class TestGetVectorSearcher:
    """Tests for the get_vector_searcher singleton function."""

    def test_get_vector_searcher_returns_instance(self):
        """Test get_vector_searcher returns an instance."""
        from src.retrieval.search.vector import get_vector_searcher, VectorSearcher
        import src.retrieval.search.vector as vector_module

        vector_module._vector_searcher = None

        searcher = get_vector_searcher()
        assert isinstance(searcher, VectorSearcher)

    def test_get_vector_searcher_returns_same_instance(self):
        """Test get_vector_searcher returns the same instance."""
        from src.retrieval.search.vector import get_vector_searcher
        import src.retrieval.search.vector as vector_module

        vector_module._vector_searcher = None

        searcher1 = get_vector_searcher()
        searcher2 = get_vector_searcher()
        assert searcher1 is searcher2
