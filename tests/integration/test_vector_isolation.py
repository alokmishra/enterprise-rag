"""
Integration tests for vector store tenant isolation.

These tests verify that tenant A's searches don't return tenant B's vectors.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.storage.vector.qdrant import QdrantVectorStore


class TestVectorSearchTenantIsolation:
    """Tests for vector search tenant isolation."""

    @pytest.mark.asyncio
    async def test_search_only_returns_tenant_vectors(self):
        """Tenant A's search doesn't return Tenant B's vectors."""
        store = QdrantVectorStore()

        # Mock the client
        mock_client = AsyncMock()
        store._client = mock_client

        # Mock search results that include both tenant vectors
        # (simulating what would be in the DB if no filter was applied)
        mock_results = [
            MagicMock(
                id="vec-1",
                score=0.9,
                payload={"content": "Tenant A doc", "tenant_id": "tenant-A"},
            ),
        ]
        mock_client.search = AsyncMock(return_value=mock_results)

        # Search as tenant A
        results = await store.search(
            collection="test_collection",
            query_vector=[0.1, 0.2, 0.3],
            top_k=10,
            tenant_id="tenant-A",
        )

        # Verify filter was applied
        mock_client.search.assert_called_once()
        call_kwargs = mock_client.search.call_args[1]
        query_filter = call_kwargs["query_filter"]

        # Verify tenant filter is present
        assert query_filter is not None
        filter_conditions = query_filter.must
        tenant_filter = None
        for cond in filter_conditions:
            if cond.key == "tenant_id":
                tenant_filter = cond
                break

        assert tenant_filter is not None
        assert tenant_filter.match.value == "tenant-A"

    @pytest.mark.asyncio
    async def test_insert_tags_vectors_with_tenant(self):
        """Inserted vectors are tagged with tenant_id."""
        store = QdrantVectorStore()

        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock()
        store._client = mock_client

        # Insert vectors for tenant A
        await store.insert(
            collection="test_collection",
            ids=["vec-1", "vec-2"],
            vectors=[[0.1, 0.2], [0.3, 0.4]],
            payloads=[{"content": "doc1"}, {"content": "doc2"}],
            tenant_id="tenant-A",
        )

        # Verify tenant_id was added to payloads
        mock_client.upsert.assert_called_once()
        points = mock_client.upsert.call_args[1]["points"]

        for point in points:
            assert point.payload["tenant_id"] == "tenant-A"

    @pytest.mark.asyncio
    async def test_separate_tenant_vectors_isolated(self):
        """Vectors from different tenants are isolated by filtering."""
        store = QdrantVectorStore()

        mock_client = AsyncMock()
        store._client = mock_client

        # Simulate inserting for tenant A
        mock_client.upsert = AsyncMock()
        await store.insert(
            collection="test_collection",
            ids=["vec-a1"],
            vectors=[[0.1, 0.2]],
            payloads=[{"content": "Tenant A document"}],
            tenant_id="tenant-A",
        )

        # Simulate inserting for tenant B
        await store.insert(
            collection="test_collection",
            ids=["vec-b1"],
            vectors=[[0.3, 0.4]],
            payloads=[{"content": "Tenant B document"}],
            tenant_id="tenant-B",
        )

        # Verify each insert tagged correctly
        calls = mock_client.upsert.call_args_list
        assert len(calls) == 2

        # First call (tenant A)
        tenant_a_points = calls[0][1]["points"]
        assert tenant_a_points[0].payload["tenant_id"] == "tenant-A"

        # Second call (tenant B)
        tenant_b_points = calls[1][1]["points"]
        assert tenant_b_points[0].payload["tenant_id"] == "tenant-B"


class TestHybridSearchTenantIsolation:
    """Tests for hybrid search tenant isolation."""

    @pytest.mark.asyncio
    async def test_hybrid_search_respects_tenant_isolation(self):
        """Hybrid search passes tenant_id to vector searcher."""
        from src.retrieval.search.hybrid import HybridSearcher
        from src.retrieval.search.vector import VectorSearcher

        # Create hybrid searcher with tenant
        searcher = HybridSearcher(tenant_id="tenant-hybrid")

        # Verify internal VectorSearcher has tenant_id
        assert searcher.vector_searcher.tenant_id == "tenant-hybrid"
        assert searcher.tenant_id == "tenant-hybrid"

    @pytest.mark.asyncio
    async def test_hybrid_search_passes_tenant_to_vector_search(self):
        """Hybrid search method passes tenant_id to vector search."""
        from src.retrieval.search.hybrid import HybridSearcher
        from src.retrieval.search.vector import VectorSearcher

        # Create mock vector searcher
        mock_vector_result = MagicMock()
        mock_vector_result.results = []

        mock_vector_searcher = MagicMock(spec=VectorSearcher)
        mock_vector_searcher.tenant_id = "tenant-test"
        mock_vector_searcher.search = AsyncMock(return_value=mock_vector_result)

        # Create mock BM25 index
        mock_bm25 = MagicMock()
        mock_bm25.search = MagicMock(return_value=[])

        searcher = HybridSearcher(
            vector_searcher=mock_vector_searcher,
            bm25_index=mock_bm25,
            tenant_id="tenant-test",
        )

        # Execute search
        await searcher.search(
            query="test query",
            top_k=10,
        )

        # Verify vector search was called with tenant_id
        mock_vector_searcher.search.assert_called_once()
        call_kwargs = mock_vector_searcher.search.call_args[1]
        assert call_kwargs["tenant_id"] == "tenant-test"

    @pytest.mark.asyncio
    async def test_hybrid_search_method_tenant_override(self):
        """Hybrid search method can override instance tenant_id."""
        from src.retrieval.search.hybrid import HybridSearcher
        from src.retrieval.search.vector import VectorSearcher

        mock_vector_result = MagicMock()
        mock_vector_result.results = []

        mock_vector_searcher = MagicMock(spec=VectorSearcher)
        mock_vector_searcher.tenant_id = "tenant-instance"
        mock_vector_searcher.search = AsyncMock(return_value=mock_vector_result)

        mock_bm25 = MagicMock()
        mock_bm25.search = MagicMock(return_value=[])

        searcher = HybridSearcher(
            vector_searcher=mock_vector_searcher,
            bm25_index=mock_bm25,
            tenant_id="tenant-instance",
        )

        # Execute search with method-level override
        await searcher.search(
            query="test query",
            top_k=10,
            tenant_id="tenant-override",
        )

        # Verify vector search was called with override tenant_id
        mock_vector_searcher.search.assert_called_once()
        call_kwargs = mock_vector_searcher.search.call_args[1]
        assert call_kwargs["tenant_id"] == "tenant-override"


class TestVectorSearcherTenantIsolation:
    """Tests for VectorSearcher tenant isolation."""

    @pytest.mark.asyncio
    async def test_vector_searcher_passes_tenant_to_store(self):
        """VectorSearcher passes tenant_id to vector store search."""
        from src.retrieval.search.vector import VectorSearcher

        with patch("src.retrieval.search.vector.get_embedding_provider") as mock_embed:
            with patch("src.retrieval.search.vector.get_vector_store") as mock_store:
                # Mock embedding provider
                mock_embed_instance = AsyncMock()
                mock_embed_instance.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
                mock_embed.return_value = mock_embed_instance

                # Mock vector store
                mock_store_instance = AsyncMock()
                mock_store_instance.search = AsyncMock(return_value=[])
                mock_store.return_value = mock_store_instance

                # Create searcher with tenant
                searcher = VectorSearcher(tenant_id="tenant-vs")

                # Execute search
                await searcher.search(query="test query", top_k=10)

                # Verify store search was called with tenant_id
                mock_store_instance.search.assert_called_once()
                call_kwargs = mock_store_instance.search.call_args[1]
                assert call_kwargs["tenant_id"] == "tenant-vs"

    @pytest.mark.asyncio
    async def test_vector_searcher_method_tenant_override(self):
        """VectorSearcher search method can override instance tenant_id."""
        from src.retrieval.search.vector import VectorSearcher

        with patch("src.retrieval.search.vector.get_embedding_provider") as mock_embed:
            with patch("src.retrieval.search.vector.get_vector_store") as mock_store:
                mock_embed_instance = AsyncMock()
                mock_embed_instance.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
                mock_embed.return_value = mock_embed_instance

                mock_store_instance = AsyncMock()
                mock_store_instance.search = AsyncMock(return_value=[])
                mock_store.return_value = mock_store_instance

                # Create searcher with instance tenant
                searcher = VectorSearcher(tenant_id="tenant-instance")

                # Execute search with method override
                await searcher.search(
                    query="test query",
                    top_k=10,
                    tenant_id="tenant-method-override",
                )

                # Verify override was used
                call_kwargs = mock_store_instance.search.call_args[1]
                assert call_kwargs["tenant_id"] == "tenant-method-override"

    @pytest.mark.asyncio
    async def test_vector_searcher_no_tenant_no_filter(self):
        """VectorSearcher without tenant_id doesn't add tenant filter."""
        from src.retrieval.search.vector import VectorSearcher

        with patch("src.retrieval.search.vector.get_embedding_provider") as mock_embed:
            with patch("src.retrieval.search.vector.get_vector_store") as mock_store:
                mock_embed_instance = AsyncMock()
                mock_embed_instance.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
                mock_embed.return_value = mock_embed_instance

                mock_store_instance = AsyncMock()
                mock_store_instance.search = AsyncMock(return_value=[])
                mock_store.return_value = mock_store_instance

                # Create searcher without tenant
                searcher = VectorSearcher()

                # Execute search
                await searcher.search(query="test query", top_k=10)

                # Verify no tenant_id was passed
                call_kwargs = mock_store_instance.search.call_args[1]
                assert call_kwargs["tenant_id"] is None
