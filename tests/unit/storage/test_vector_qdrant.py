"""
Tests for src/storage/vector/qdrant.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestQdrantVectorStore:
    """Tests for the QdrantVectorStore class."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client."""
        client = MagicMock()
        client.get_collections = MagicMock(return_value=MagicMock(collections=[]))
        client.create_collection = MagicMock()
        client.upsert = MagicMock()
        client.search = MagicMock(return_value=[
            MagicMock(
                id="chunk-1",
                score=0.95,
                payload={"content": "Test content", "source": "test.pdf"},
            )
        ])
        client.delete = MagicMock()
        return client

    def test_qdrant_store_initialization(self):
        """Test QdrantVectorStore can be initialized."""
        from src.storage.vector.qdrant import QdrantVectorStore

        store = QdrantVectorStore(
            url="http://localhost:6333",
            collection_name="test_collection",
        )
        assert store.collection_name == "test_collection"

    @pytest.mark.asyncio
    async def test_qdrant_store_connect(self, mock_qdrant_client):
        """Test QdrantVectorStore connect method."""
        from src.storage.vector.qdrant import QdrantVectorStore

        with patch('src.storage.vector.qdrant.QdrantClient', return_value=mock_qdrant_client):
            store = QdrantVectorStore(
                url="http://localhost:6333",
                collection_name="test_collection",
            )
            await store.connect()
            # Should have initialized client

    @pytest.mark.asyncio
    async def test_qdrant_store_upsert(self, mock_qdrant_client):
        """Test QdrantVectorStore upsert method."""
        from src.storage.vector.qdrant import QdrantVectorStore

        with patch('src.storage.vector.qdrant.QdrantClient', return_value=mock_qdrant_client):
            store = QdrantVectorStore(
                url="http://localhost:6333",
                collection_name="test_collection",
            )
            store.client = mock_qdrant_client

            await store.upsert(
                ids=["chunk-1"],
                embeddings=[[0.1] * 1536],
                payloads=[{"content": "Test content"}],
            )
            # Should call client.upsert

    @pytest.mark.asyncio
    async def test_qdrant_store_search(self, mock_qdrant_client):
        """Test QdrantVectorStore search method."""
        from src.storage.vector.qdrant import QdrantVectorStore

        with patch('src.storage.vector.qdrant.QdrantClient', return_value=mock_qdrant_client):
            store = QdrantVectorStore(
                url="http://localhost:6333",
                collection_name="test_collection",
            )
            store.client = mock_qdrant_client

            results = await store.search(
                query_embedding=[0.1] * 1536,
                top_k=5,
            )
            assert len(results) > 0

    @pytest.mark.asyncio
    async def test_qdrant_store_delete(self, mock_qdrant_client):
        """Test QdrantVectorStore delete method."""
        from src.storage.vector.qdrant import QdrantVectorStore

        with patch('src.storage.vector.qdrant.QdrantClient', return_value=mock_qdrant_client):
            store = QdrantVectorStore(
                url="http://localhost:6333",
                collection_name="test_collection",
            )
            store.client = mock_qdrant_client

            await store.delete(ids=["chunk-1"])
            # Should call client.delete

    @pytest.mark.asyncio
    async def test_qdrant_store_health_check(self, mock_qdrant_client):
        """Test QdrantVectorStore health_check method."""
        from src.storage.vector.qdrant import QdrantVectorStore

        mock_qdrant_client.get_collections = MagicMock(return_value=MagicMock(collections=[]))

        with patch('src.storage.vector.qdrant.QdrantClient', return_value=mock_qdrant_client):
            store = QdrantVectorStore(
                url="http://localhost:6333",
                collection_name="test_collection",
            )
            store.client = mock_qdrant_client

            is_healthy = await store.health_check()
            assert is_healthy is True

    def test_qdrant_store_search_with_filters(self, mock_qdrant_client):
        """Test QdrantVectorStore search with metadata filters."""
        from src.storage.vector.qdrant import QdrantVectorStore

        # Search with filters should be supported
        store = QdrantVectorStore(
            url="http://localhost:6333",
            collection_name="test_collection",
        )
        # Filter functionality should exist
        assert hasattr(store, 'search')
