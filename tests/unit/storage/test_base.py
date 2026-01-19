"""
Tests for src/storage/base.py
"""

from abc import ABC
from unittest.mock import AsyncMock

import pytest


class TestStorageBackendInterface:
    """Tests for the StorageBackend abstract base class."""

    def test_storage_backend_is_abstract(self):
        """Test that StorageBackend is abstract."""
        from src.storage.base import StorageBackend

        assert issubclass(StorageBackend, ABC)

    def test_storage_backend_requires_connect(self):
        """Test that StorageBackend requires connect method."""
        from src.storage.base import StorageBackend

        assert hasattr(StorageBackend, 'connect')

    def test_storage_backend_requires_disconnect(self):
        """Test that StorageBackend requires disconnect method."""
        from src.storage.base import StorageBackend

        assert hasattr(StorageBackend, 'disconnect')

    def test_storage_backend_requires_health_check(self):
        """Test that StorageBackend requires health_check method."""
        from src.storage.base import StorageBackend

        assert hasattr(StorageBackend, 'health_check')

    def test_storage_backend_requires_is_connected(self):
        """Test that StorageBackend requires is_connected property."""
        from src.storage.base import StorageBackend

        assert hasattr(StorageBackend, 'is_connected')


class TestVectorStoreInterface:
    """Tests for the VectorStore abstract base class."""

    def test_vector_store_is_abstract(self):
        """Test that VectorStore is abstract."""
        from src.storage.base import VectorStore

        assert issubclass(VectorStore, ABC)

    def test_vector_store_requires_insert(self):
        """Test that VectorStore requires insert method."""
        from src.storage.base import VectorStore

        assert hasattr(VectorStore, 'insert')

    def test_vector_store_requires_search(self):
        """Test that VectorStore requires search method."""
        from src.storage.base import VectorStore

        assert hasattr(VectorStore, 'search')

    def test_vector_store_requires_delete(self):
        """Test that VectorStore requires delete method."""
        from src.storage.base import VectorStore

        assert hasattr(VectorStore, 'delete')

    def test_vector_store_requires_get(self):
        """Test that VectorStore requires get method."""
        from src.storage.base import VectorStore

        assert hasattr(VectorStore, 'get')

    def test_vector_store_requires_create_collection(self):
        """Test that VectorStore requires create_collection method."""
        from src.storage.base import VectorStore

        assert hasattr(VectorStore, 'create_collection')

    def test_vector_store_requires_delete_collection(self):
        """Test that VectorStore requires delete_collection method."""
        from src.storage.base import VectorStore

        assert hasattr(VectorStore, 'delete_collection')

    def test_vector_store_requires_collection_exists(self):
        """Test that VectorStore requires collection_exists method."""
        from src.storage.base import VectorStore

        assert hasattr(VectorStore, 'collection_exists')

    def test_vector_store_extends_storage_backend(self):
        """Test that VectorStore extends StorageBackend."""
        from src.storage.base import VectorStore, StorageBackend

        assert issubclass(VectorStore, StorageBackend)


class TestDocumentStoreInterface:
    """Tests for the DocumentStore abstract base class."""

    def test_document_store_is_abstract(self):
        """Test that DocumentStore is abstract."""
        from src.storage.base import DocumentStore

        assert issubclass(DocumentStore, ABC)

    def test_document_store_requires_save(self):
        """Test that DocumentStore requires save method."""
        from src.storage.base import DocumentStore

        assert hasattr(DocumentStore, 'save')

    def test_document_store_requires_get(self):
        """Test that DocumentStore requires get method."""
        from src.storage.base import DocumentStore

        assert hasattr(DocumentStore, 'get')

    def test_document_store_requires_delete(self):
        """Test that DocumentStore requires delete method."""
        from src.storage.base import DocumentStore

        assert hasattr(DocumentStore, 'delete')

    def test_document_store_requires_list(self):
        """Test that DocumentStore requires list method."""
        from src.storage.base import DocumentStore

        assert hasattr(DocumentStore, 'list')

    def test_document_store_requires_update(self):
        """Test that DocumentStore requires update method."""
        from src.storage.base import DocumentStore

        assert hasattr(DocumentStore, 'update')

    def test_document_store_extends_storage_backend(self):
        """Test that DocumentStore extends StorageBackend."""
        from src.storage.base import DocumentStore, StorageBackend

        assert issubclass(DocumentStore, StorageBackend)


class TestCacheStoreInterface:
    """Tests for the CacheStore abstract base class."""

    def test_cache_store_is_abstract(self):
        """Test that CacheStore is abstract."""
        from src.storage.base import CacheStore

        assert issubclass(CacheStore, ABC)

    def test_cache_store_requires_get(self):
        """Test that CacheStore requires get method."""
        from src.storage.base import CacheStore

        assert hasattr(CacheStore, 'get')

    def test_cache_store_requires_set(self):
        """Test that CacheStore requires set method."""
        from src.storage.base import CacheStore

        assert hasattr(CacheStore, 'set')

    def test_cache_store_requires_delete(self):
        """Test that CacheStore requires delete method."""
        from src.storage.base import CacheStore

        assert hasattr(CacheStore, 'delete')

    def test_cache_store_requires_exists(self):
        """Test that CacheStore requires exists method."""
        from src.storage.base import CacheStore

        assert hasattr(CacheStore, 'exists')

    def test_cache_store_requires_clear(self):
        """Test that CacheStore requires clear method."""
        from src.storage.base import CacheStore

        assert hasattr(CacheStore, 'clear')

    def test_cache_store_extends_storage_backend(self):
        """Test that CacheStore extends StorageBackend."""
        from src.storage.base import CacheStore, StorageBackend

        assert issubclass(CacheStore, StorageBackend)
