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
        client = AsyncMock()
        client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
        client.create_collection = AsyncMock()
        client.delete_collection = AsyncMock()
        client.upsert = AsyncMock()
        client.search = AsyncMock(return_value=[
            MagicMock(
                id="chunk-1",
                score=0.95,
                payload={"content": "Test content", "source": "test.pdf"},
            )
        ])
        client.retrieve = AsyncMock(return_value=[
            MagicMock(
                id="chunk-1",
                payload={"content": "Test content"},
                vector=[0.1] * 1536,
            )
        ])
        client.delete = AsyncMock()
        client.close = AsyncMock()
        return client

    def test_qdrant_store_initialization(self):
        """Test QdrantVectorStore can be initialized."""
        from src.storage.vector.qdrant import QdrantVectorStore

        store = QdrantVectorStore(url="http://localhost:6333")
        assert store is not None
        assert store._url == "http://localhost:6333"

    @pytest.mark.asyncio
    async def test_qdrant_store_connect(self, mock_qdrant_client):
        """Test QdrantVectorStore connect method."""
        from src.storage.vector.qdrant import QdrantVectorStore

        with patch('src.storage.vector.qdrant.AsyncQdrantClient', return_value=mock_qdrant_client):
            store = QdrantVectorStore(url="http://localhost:6333")
            await store.connect()

            assert store.is_connected
            mock_qdrant_client.get_collections.assert_called()

    @pytest.mark.asyncio
    async def test_qdrant_store_disconnect(self, mock_qdrant_client):
        """Test QdrantVectorStore disconnect method."""
        from src.storage.vector.qdrant import QdrantVectorStore

        store = QdrantVectorStore(url="http://localhost:6333")
        store._client = mock_qdrant_client

        await store.disconnect()

        mock_qdrant_client.close.assert_called()
        assert store._client is None

    @pytest.mark.asyncio
    async def test_qdrant_store_create_collection(self, mock_qdrant_client):
        """Test QdrantVectorStore create_collection method."""
        from src.storage.vector.qdrant import QdrantVectorStore

        store = QdrantVectorStore(url="http://localhost:6333")
        store._client = mock_qdrant_client

        await store.create_collection(
            name="test_collection",
            dimension=1536,
            distance_metric="cosine",
        )

        mock_qdrant_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_qdrant_store_delete_collection(self, mock_qdrant_client):
        """Test QdrantVectorStore delete_collection method."""
        from src.storage.vector.qdrant import QdrantVectorStore

        store = QdrantVectorStore(url="http://localhost:6333")
        store._client = mock_qdrant_client

        await store.delete_collection("test_collection")

        mock_qdrant_client.delete_collection.assert_called_once_with(collection_name="test_collection")

    @pytest.mark.asyncio
    async def test_qdrant_store_collection_exists_true(self, mock_qdrant_client):
        """Test QdrantVectorStore collection_exists returns True for existing collection."""
        from src.storage.vector.qdrant import QdrantVectorStore

        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_qdrant_client.get_collections = AsyncMock(
            return_value=MagicMock(collections=[mock_collection])
        )

        store = QdrantVectorStore(url="http://localhost:6333")
        store._client = mock_qdrant_client

        exists = await store.collection_exists("test_collection")
        assert exists is True

    @pytest.mark.asyncio
    async def test_qdrant_store_collection_exists_false(self, mock_qdrant_client):
        """Test QdrantVectorStore collection_exists returns False for nonexistent collection."""
        from src.storage.vector.qdrant import QdrantVectorStore

        mock_qdrant_client.get_collections = AsyncMock(
            return_value=MagicMock(collections=[])
        )

        store = QdrantVectorStore(url="http://localhost:6333")
        store._client = mock_qdrant_client

        exists = await store.collection_exists("nonexistent")
        assert exists is False

    @pytest.mark.asyncio
    async def test_qdrant_store_insert(self, mock_qdrant_client):
        """Test QdrantVectorStore insert method."""
        from src.storage.vector.qdrant import QdrantVectorStore

        store = QdrantVectorStore(url="http://localhost:6333")
        store._client = mock_qdrant_client

        await store.insert(
            collection="test_collection",
            ids=["chunk-1"],
            vectors=[[0.1] * 1536],
            payloads=[{"content": "Test content"}],
        )

        mock_qdrant_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_qdrant_store_search(self, mock_qdrant_client):
        """Test QdrantVectorStore search method."""
        from src.storage.vector.qdrant import QdrantVectorStore

        store = QdrantVectorStore(url="http://localhost:6333")
        store._client = mock_qdrant_client

        results = await store.search(
            collection="test_collection",
            query_vector=[0.1] * 1536,
            top_k=5,
        )

        assert len(results) == 1
        assert results[0].id == "chunk-1"
        assert results[0].score == 0.95

    @pytest.mark.asyncio
    async def test_qdrant_store_search_with_filters(self, mock_qdrant_client):
        """Test QdrantVectorStore search with metadata filters."""
        from src.storage.vector.qdrant import QdrantVectorStore

        store = QdrantVectorStore(url="http://localhost:6333")
        store._client = mock_qdrant_client

        results = await store.search(
            collection="test_collection",
            query_vector=[0.1] * 1536,
            top_k=5,
            filters={"source": "test.pdf"},
        )

        mock_qdrant_client.search.assert_called_once()
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_qdrant_store_delete(self, mock_qdrant_client):
        """Test QdrantVectorStore delete method."""
        from src.storage.vector.qdrant import QdrantVectorStore

        store = QdrantVectorStore(url="http://localhost:6333")
        store._client = mock_qdrant_client

        await store.delete(collection="test_collection", ids=["chunk-1"])

        mock_qdrant_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_qdrant_store_get(self, mock_qdrant_client):
        """Test QdrantVectorStore get method."""
        from src.storage.vector.qdrant import QdrantVectorStore

        store = QdrantVectorStore(url="http://localhost:6333")
        store._client = mock_qdrant_client

        results = await store.get(collection="test_collection", ids=["chunk-1"])

        assert len(results) == 1
        mock_qdrant_client.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_qdrant_store_health_check(self, mock_qdrant_client):
        """Test QdrantVectorStore health_check method."""
        from src.storage.vector.qdrant import QdrantVectorStore

        store = QdrantVectorStore(url="http://localhost:6333")
        store._client = mock_qdrant_client

        result = await store.health_check()

        assert result["status"] == "healthy"
        assert "latency_ms" in result

    @pytest.mark.asyncio
    async def test_qdrant_store_health_check_disconnected(self):
        """Test QdrantVectorStore health_check when disconnected."""
        from src.storage.vector.qdrant import QdrantVectorStore

        store = QdrantVectorStore(url="http://localhost:6333")

        result = await store.health_check()

        assert result["status"] == "disconnected"


class TestQdrantSearchResult:
    """Tests for QdrantSearchResult class."""

    def test_search_result_initialization(self):
        """Test QdrantSearchResult can be initialized."""
        from src.storage.vector.qdrant import QdrantSearchResult

        result = QdrantSearchResult(
            id="chunk-1",
            score=0.95,
            payload={"content": "Test"},
        )

        assert result.id == "chunk-1"
        assert result.score == 0.95
        assert result.payload == {"content": "Test"}
        assert result.vector is None

    def test_search_result_to_dict(self):
        """Test QdrantSearchResult to_dict method."""
        from src.storage.vector.qdrant import QdrantSearchResult

        result = QdrantSearchResult(
            id="chunk-1",
            score=0.95,
            payload={"content": "Test"},
            vector=[0.1, 0.2],
        )

        result_dict = result.to_dict()

        assert result_dict["id"] == "chunk-1"
        assert result_dict["score"] == 0.95
        assert result_dict["payload"] == {"content": "Test"}
        assert result_dict["vector"] == [0.1, 0.2]


class TestQdrantVectorStoreFactory:
    """Tests for Qdrant vector store factory functions."""

    def test_get_vector_store_returns_singleton(self):
        """Test that get_vector_store returns the same instance."""
        from src.storage.vector.qdrant import get_vector_store, QdrantVectorStore
        import src.storage.vector.qdrant as qdrant_module

        qdrant_module._vector_store = None

        store1 = get_vector_store()
        store2 = get_vector_store()

        assert store1 is store2
        assert isinstance(store1, QdrantVectorStore)

        qdrant_module._vector_store = None
