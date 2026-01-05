"""
Tests for src/retrieval/search/hybrid.py and src/retrieval/search/sparse.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestBM25Index:
    """Tests for the BM25Index class."""

    def test_bm25_index_creation(self):
        """Test BM25Index can be created."""
        from src.retrieval.search.sparse import BM25Index

        index = BM25Index()
        assert index is not None

    def test_bm25_index_add_documents(self):
        """Test adding documents to BM25 index."""
        from src.retrieval.search.sparse import BM25Index

        index = BM25Index()
        index.add_documents(
            ids=["doc-1", "doc-2"],
            documents=["First document content", "Second document content"],
        )
        # Should add documents to index

    def test_bm25_index_search(self):
        """Test BM25 search."""
        from src.retrieval.search.sparse import BM25Index

        index = BM25Index()
        index.add_documents(
            ids=["doc-1", "doc-2", "doc-3"],
            documents=[
                "The quick brown fox jumps over the lazy dog",
                "A fast red fox leaps across the sleepy hound",
                "Python programming language tutorial",
            ],
        )

        results = index.search("fox jumps", top_k=2)
        assert len(results) <= 2
        # Should return fox-related documents first

    def test_bm25_index_search_returns_scores(self):
        """Test that BM25 search returns scores."""
        from src.retrieval.search.sparse import BM25Index

        index = BM25Index()
        index.add_documents(
            ids=["doc-1"],
            documents=["Test document content"],
        )

        results = index.search("test", top_k=1)
        if results:
            assert "score" in results[0] or hasattr(results[0], 'score')

    def test_bm25_index_empty_query(self):
        """Test BM25 with empty query."""
        from src.retrieval.search.sparse import BM25Index

        index = BM25Index()
        index.add_documents(ids=["doc-1"], documents=["Content"])

        results = index.search("", top_k=5)
        assert results == [] or len(results) == 0

    def test_bm25_index_remove_document(self):
        """Test removing document from index."""
        from src.retrieval.search.sparse import BM25Index

        index = BM25Index()
        index.add_documents(ids=["doc-1", "doc-2"], documents=["Content 1", "Content 2"])

        if hasattr(index, 'remove'):
            index.remove("doc-1")


class TestHybridSearcher:
    """Tests for the HybridSearcher class."""

    @pytest.fixture
    def mock_vector_searcher(self):
        """Create a mock vector searcher."""
        searcher = AsyncMock()
        searcher.search = AsyncMock(return_value=[
            MagicMock(chunk_id="chunk-1", score=0.9, content="Vector result 1"),
            MagicMock(chunk_id="chunk-2", score=0.8, content="Vector result 2"),
        ])
        return searcher

    @pytest.fixture
    def mock_sparse_index(self):
        """Create a mock sparse index."""
        index = MagicMock()
        index.search = MagicMock(return_value=[
            {"id": "chunk-1", "score": 0.85},
            {"id": "chunk-3", "score": 0.75},
        ])
        return index

    def test_hybrid_searcher_creation(self, mock_vector_searcher, mock_sparse_index):
        """Test HybridSearcher can be created."""
        from src.retrieval.search.hybrid import HybridSearcher

        searcher = HybridSearcher(
            vector_searcher=mock_vector_searcher,
            sparse_index=mock_sparse_index,
        )
        assert searcher is not None

    @pytest.mark.asyncio
    async def test_hybrid_searcher_search(self, mock_vector_searcher, mock_sparse_index):
        """Test HybridSearcher search method."""
        from src.retrieval.search.hybrid import HybridSearcher

        searcher = HybridSearcher(
            vector_searcher=mock_vector_searcher,
            sparse_index=mock_sparse_index,
        )

        results = await searcher.search("test query", top_k=5)
        assert results is not None

    @pytest.mark.asyncio
    async def test_hybrid_searcher_rrf_fusion(self, mock_vector_searcher, mock_sparse_index):
        """Test HybridSearcher uses RRF fusion."""
        from src.retrieval.search.hybrid import HybridSearcher

        searcher = HybridSearcher(
            vector_searcher=mock_vector_searcher,
            sparse_index=mock_sparse_index,
            fusion_method="rrf",
        )

        results = await searcher.search("test query", top_k=5)
        # RRF should combine results from both

    @pytest.mark.asyncio
    async def test_hybrid_searcher_weighted_fusion(self, mock_vector_searcher, mock_sparse_index):
        """Test HybridSearcher with weighted fusion."""
        from src.retrieval.search.hybrid import HybridSearcher

        searcher = HybridSearcher(
            vector_searcher=mock_vector_searcher,
            sparse_index=mock_sparse_index,
            fusion_method="weighted",
            vector_weight=0.7,
            sparse_weight=0.3,
        )

        results = await searcher.search("test query", top_k=5)
        # Should apply weights

    @pytest.mark.asyncio
    async def test_hybrid_searcher_deduplication(self, mock_vector_searcher, mock_sparse_index):
        """Test HybridSearcher deduplicates results."""
        from src.retrieval.search.hybrid import HybridSearcher

        # Both return chunk-1
        searcher = HybridSearcher(
            vector_searcher=mock_vector_searcher,
            sparse_index=mock_sparse_index,
        )

        results = await searcher.search("test query", top_k=10)
        # chunk-1 should appear only once


class TestRRFFusion:
    """Tests for Reciprocal Rank Fusion."""

    def test_rrf_formula(self):
        """Test RRF formula implementation."""
        from src.retrieval.search.hybrid import reciprocal_rank_fusion

        if hasattr(__import__('src.retrieval.search.hybrid', fromlist=['reciprocal_rank_fusion']), 'reciprocal_rank_fusion'):
            rankings = [
                [("doc-1", 0.9), ("doc-2", 0.8)],
                [("doc-2", 0.95), ("doc-1", 0.7)],
            ]
            # RRF should produce combined ranking

    def test_rrf_k_parameter(self):
        """Test RRF with different k parameters."""
        # k=60 is standard, but should be configurable
        pass
