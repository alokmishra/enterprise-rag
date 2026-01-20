"""
Tests for Qdrant tenant filtering.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestQdrantInsertTenantId:
    """Tests for tenant_id in Qdrant insert."""

    @pytest.mark.asyncio
    async def test_insert_includes_tenant_id_in_payload(self):
        """Inserted vectors have tenant_id in payload."""
        from src.storage.vector.qdrant import QdrantVectorStore

        store = QdrantVectorStore()

        # Mock the client
        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock()
        store._client = mock_client

        await store.insert(
            collection="test_collection",
            ids=["id1", "id2"],
            vectors=[[0.1, 0.2], [0.3, 0.4]],
            payloads=[{"content": "doc1"}, {"content": "doc2"}],
            tenant_id="tenant-123",
        )

        mock_client.upsert.assert_called_once()
        call_kwargs = mock_client.upsert.call_args[1]
        points = call_kwargs["points"]

        # Verify tenant_id was added to each payload
        for point in points:
            assert point.payload["tenant_id"] == "tenant-123"

    @pytest.mark.asyncio
    async def test_insert_without_tenant_id(self):
        """Insert without tenant_id does not add it to payload."""
        from src.storage.vector.qdrant import QdrantVectorStore

        store = QdrantVectorStore()

        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock()
        store._client = mock_client

        await store.insert(
            collection="test_collection",
            ids=["id1"],
            vectors=[[0.1, 0.2]],
            payloads=[{"content": "doc1"}],
        )

        mock_client.upsert.assert_called_once()
        call_kwargs = mock_client.upsert.call_args[1]
        points = call_kwargs["points"]

        # Verify no tenant_id in payload
        assert "tenant_id" not in points[0].payload

    @pytest.mark.asyncio
    async def test_insert_preserves_existing_payload(self):
        """Insert with tenant_id preserves existing payload fields."""
        from src.storage.vector.qdrant import QdrantVectorStore

        store = QdrantVectorStore()

        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock()
        store._client = mock_client

        await store.insert(
            collection="test_collection",
            ids=["id1"],
            vectors=[[0.1, 0.2]],
            payloads=[{"content": "doc1", "document_id": "doc-abc"}],
            tenant_id="tenant-456",
        )

        mock_client.upsert.assert_called_once()
        points = mock_client.upsert.call_args[1]["points"]

        # Verify both original payload and tenant_id are present
        assert points[0].payload["content"] == "doc1"
        assert points[0].payload["document_id"] == "doc-abc"
        assert points[0].payload["tenant_id"] == "tenant-456"


class TestQdrantSearchTenantFilter:
    """Tests for tenant_id filtering in Qdrant search."""

    @pytest.mark.asyncio
    async def test_search_adds_tenant_filter(self):
        """Search includes tenant_id filter condition."""
        from src.storage.vector.qdrant import QdrantVectorStore

        store = QdrantVectorStore()

        mock_client = AsyncMock()
        mock_client.search = AsyncMock(return_value=[])
        store._client = mock_client

        await store.search(
            collection="test_collection",
            query_vector=[0.1, 0.2],
            top_k=10,
            tenant_id="tenant-search",
        )

        mock_client.search.assert_called_once()
        call_kwargs = mock_client.search.call_args[1]

        # Verify query_filter was set
        query_filter = call_kwargs["query_filter"]
        assert query_filter is not None

        # Verify filter includes tenant_id
        filter_conditions = query_filter.must
        tenant_filter = None
        for cond in filter_conditions:
            if cond.key == "tenant_id":
                tenant_filter = cond
                break

        assert tenant_filter is not None
        assert tenant_filter.match.value == "tenant-search"

    @pytest.mark.asyncio
    async def test_search_without_tenant_returns_all(self):
        """Search without tenant_id (admin) returns all results."""
        from src.storage.vector.qdrant import QdrantVectorStore

        store = QdrantVectorStore()

        mock_client = AsyncMock()
        mock_client.search = AsyncMock(return_value=[])
        store._client = mock_client

        await store.search(
            collection="test_collection",
            query_vector=[0.1, 0.2],
            top_k=10,
            # No tenant_id - admin access
        )

        mock_client.search.assert_called_once()
        call_kwargs = mock_client.search.call_args[1]

        # Verify no filter was set
        assert call_kwargs["query_filter"] is None

    @pytest.mark.asyncio
    async def test_search_combines_tenant_with_other_filters(self):
        """Search combines tenant_id with other filters."""
        from src.storage.vector.qdrant import QdrantVectorStore

        store = QdrantVectorStore()

        mock_client = AsyncMock()
        mock_client.search = AsyncMock(return_value=[])
        store._client = mock_client

        await store.search(
            collection="test_collection",
            query_vector=[0.1, 0.2],
            top_k=10,
            filters={"document_type": "pdf"},
            tenant_id="tenant-combo",
        )

        mock_client.search.assert_called_once()
        call_kwargs = mock_client.search.call_args[1]

        query_filter = call_kwargs["query_filter"]
        assert query_filter is not None

        # Verify both filters are present
        filter_keys = [cond.key for cond in query_filter.must]
        assert "tenant_id" in filter_keys
        assert "document_type" in filter_keys


class TestVectorSearcherTenantId:
    """Tests for VectorSearcher tenant support."""

    def test_vector_searcher_accepts_tenant_id(self):
        """VectorSearcher accepts tenant_id in constructor."""
        from src.retrieval.search.vector import VectorSearcher

        searcher = VectorSearcher(tenant_id="tenant-vs")
        assert searcher.tenant_id == "tenant-vs"

    def test_vector_searcher_default_no_tenant(self):
        """VectorSearcher without tenant_id has None."""
        from src.retrieval.search.vector import VectorSearcher

        searcher = VectorSearcher()
        assert searcher.tenant_id is None


class TestHybridSearcherTenantId:
    """Tests for HybridSearcher tenant support."""

    def test_hybrid_searcher_accepts_tenant_id(self):
        """HybridSearcher accepts tenant_id in constructor."""
        from src.retrieval.search.hybrid import HybridSearcher
        from src.retrieval.search.vector import VectorSearcher

        searcher = HybridSearcher(tenant_id="tenant-hs")
        assert searcher.tenant_id == "tenant-hs"
        # VectorSearcher should also have the tenant_id
        assert searcher.vector_searcher.tenant_id == "tenant-hs"

    def test_hybrid_searcher_default_no_tenant(self):
        """HybridSearcher without tenant_id has None."""
        from src.retrieval.search.hybrid import HybridSearcher

        searcher = HybridSearcher()
        assert searcher.tenant_id is None
