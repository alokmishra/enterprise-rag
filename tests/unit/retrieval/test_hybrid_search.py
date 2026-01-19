"""
Tests for src/retrieval/search/hybrid.py and src/retrieval/search/sparse.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.types import SearchResult, RetrievalResult, RetrievalStrategy
from src.retrieval.search.sparse import SparseSearchResult


class TestBM25Index:
    """Tests for the BM25Index class."""

    def test_bm25_index_creation(self):
        """Test BM25Index can be created."""
        from src.retrieval.search.sparse import BM25Index

        index = BM25Index()
        assert index is not None
        assert index.k1 == 1.5
        assert index.b == 0.75

    def test_bm25_index_add_document(self):
        """Test adding a single document to BM25 index."""
        from src.retrieval.search.sparse import BM25Index

        index = BM25Index()
        index.add_document(
            doc_id="doc-1",
            content="First document content",
            metadata={"source": "test"},
        )
        assert index._doc_count == 1

    def test_bm25_index_add_documents(self):
        """Test adding multiple documents to BM25 index."""
        from src.retrieval.search.sparse import BM25Index

        index = BM25Index()
        index.add_documents([
            {"id": "doc-1", "content": "First document content"},
            {"id": "doc-2", "content": "Second document content"},
        ])
        assert index._doc_count == 2

    def test_bm25_index_search(self):
        """Test BM25 search."""
        from src.retrieval.search.sparse import BM25Index

        index = BM25Index()
        index.add_documents([
            {"id": "doc-1", "content": "The quick brown fox jumps over the lazy dog"},
            {"id": "doc-2", "content": "A fast red fox leaps across the sleepy hound"},
            {"id": "doc-3", "content": "Python programming language tutorial"},
        ])

        results = index.search("fox jumps", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, SparseSearchResult) for r in results)

    def test_bm25_index_search_returns_scores(self):
        """Test that BM25 search returns scores."""
        from src.retrieval.search.sparse import BM25Index

        index = BM25Index()
        index.add_document(
            doc_id="doc-1",
            content="Test document content",
        )

        results = index.search("test", top_k=1)
        if results:
            assert hasattr(results[0], 'score')
            assert results[0].score > 0

    def test_bm25_index_empty_query(self):
        """Test BM25 with empty query."""
        from src.retrieval.search.sparse import BM25Index

        index = BM25Index()
        index.add_document(doc_id="doc-1", content="Content")

        results = index.search("", top_k=5)
        assert results == []

    def test_bm25_index_remove_document(self):
        """Test removing document from index."""
        from src.retrieval.search.sparse import BM25Index

        index = BM25Index()
        index.add_documents([
            {"id": "doc-1", "content": "Content 1"},
            {"id": "doc-2", "content": "Content 2"},
        ])

        result = index.remove_document("doc-1")
        assert result is True
        assert index._doc_count == 1

    def test_bm25_index_clear(self):
        """Test clearing the index."""
        from src.retrieval.search.sparse import BM25Index

        index = BM25Index()
        index.add_documents([
            {"id": "doc-1", "content": "Content 1"},
            {"id": "doc-2", "content": "Content 2"},
        ])
        index.clear()
        assert index._doc_count == 0


class TestHybridSearcher:
    """Tests for the HybridSearcher class."""

    @pytest.fixture
    def mock_vector_searcher(self):
        """Create a mock vector searcher."""
        searcher = MagicMock()
        searcher.search = AsyncMock(return_value=RetrievalResult(
            query="test query",
            strategy=RetrievalStrategy.VECTOR,
            results=[
                SearchResult(chunk_id="chunk-1", document_id="doc-1", content="Vector result 1", score=0.9, metadata={}),
                SearchResult(chunk_id="chunk-2", document_id="doc-1", content="Vector result 2", score=0.8, metadata={}),
            ],
            total_found=2,
            latency_ms=10.0,
        ))
        return searcher

    @pytest.fixture
    def mock_bm25_index(self):
        """Create a mock BM25 index."""
        index = MagicMock()
        index.search = MagicMock(return_value=[
            SparseSearchResult(id="chunk-1", score=0.85, content="Sparse result 1", metadata={"document_id": "doc-1"}),
            SparseSearchResult(id="chunk-3", score=0.75, content="Sparse result 3", metadata={"document_id": "doc-2"}),
        ])
        return index

    def test_hybrid_searcher_creation(self, mock_vector_searcher, mock_bm25_index):
        """Test HybridSearcher can be created."""
        from src.retrieval.search.hybrid import HybridSearcher

        searcher = HybridSearcher(
            vector_searcher=mock_vector_searcher,
            bm25_index=mock_bm25_index,
        )
        assert searcher is not None
        assert searcher.rrf_k == 60

    def test_hybrid_searcher_creation_with_alpha(self, mock_vector_searcher, mock_bm25_index):
        """Test HybridSearcher with custom alpha."""
        from src.retrieval.search.hybrid import HybridSearcher

        searcher = HybridSearcher(
            vector_searcher=mock_vector_searcher,
            bm25_index=mock_bm25_index,
            alpha=0.7,
            rrf_k=50,
        )
        assert searcher.alpha == 0.7
        assert searcher.rrf_k == 50

    @pytest.mark.asyncio
    async def test_hybrid_searcher_search(self, mock_vector_searcher, mock_bm25_index):
        """Test HybridSearcher search method."""
        from src.retrieval.search.hybrid import HybridSearcher

        searcher = HybridSearcher(
            vector_searcher=mock_vector_searcher,
            bm25_index=mock_bm25_index,
        )

        result = await searcher.search("test query", top_k=5)
        assert isinstance(result, RetrievalResult)
        assert result.strategy == RetrievalStrategy.HYBRID

    @pytest.mark.asyncio
    async def test_hybrid_searcher_rrf_fusion(self, mock_vector_searcher, mock_bm25_index):
        """Test HybridSearcher uses RRF fusion."""
        from src.retrieval.search.hybrid import HybridSearcher

        searcher = HybridSearcher(
            vector_searcher=mock_vector_searcher,
            bm25_index=mock_bm25_index,
        )

        result = await searcher.search("test query", top_k=5, fusion_method="rrf")
        assert len(result.results) > 0
        mock_vector_searcher.search.assert_called_once()
        mock_bm25_index.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_searcher_weighted_fusion(self, mock_vector_searcher, mock_bm25_index):
        """Test HybridSearcher with weighted fusion."""
        from src.retrieval.search.hybrid import HybridSearcher

        searcher = HybridSearcher(
            vector_searcher=mock_vector_searcher,
            bm25_index=mock_bm25_index,
            alpha=0.7,
        )

        result = await searcher.search("test query", top_k=5, fusion_method="weighted")
        assert len(result.results) > 0

    @pytest.mark.asyncio
    async def test_hybrid_searcher_deduplication(self, mock_vector_searcher, mock_bm25_index):
        """Test HybridSearcher deduplicates results."""
        from src.retrieval.search.hybrid import HybridSearcher

        searcher = HybridSearcher(
            vector_searcher=mock_vector_searcher,
            bm25_index=mock_bm25_index,
        )

        result = await searcher.search("test query", top_k=10)
        chunk_ids = [r.chunk_id for r in result.results]
        assert len(chunk_ids) == len(set(chunk_ids))


class TestRRFFusion:
    """Tests for Reciprocal Rank Fusion."""

    def test_rrf_fusion_method(self):
        """Test RRF fusion produces combined results."""
        from src.retrieval.search.hybrid import HybridSearcher

        searcher = HybridSearcher()

        vector_results = [
            SearchResult(chunk_id="doc-1", document_id="d1", content="Content 1", score=0.9, metadata={}),
            SearchResult(chunk_id="doc-2", document_id="d1", content="Content 2", score=0.8, metadata={}),
        ]
        sparse_results = [
            SparseSearchResult(id="doc-2", score=0.95, content="Content 2", metadata={"document_id": "d1"}),
            SparseSearchResult(id="doc-1", score=0.7, content="Content 1", metadata={"document_id": "d1"}),
        ]

        fused = searcher._rrf_fusion(vector_results, sparse_results, top_k=2)
        assert len(fused) == 2
        assert all(r.score > 0 for r in fused)

    def test_rrf_k_parameter(self):
        """Test RRF with different k parameters."""
        from src.retrieval.search.hybrid import HybridSearcher

        searcher = HybridSearcher(rrf_k=30)
        assert searcher.rrf_k == 30
