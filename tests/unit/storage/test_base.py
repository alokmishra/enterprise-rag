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

        # Should have abstract connect method
        assert hasattr(StorageBackend, 'connect')

    def test_storage_backend_requires_disconnect(self):
        """Test that StorageBackend requires disconnect method."""
        from src.storage.base import StorageBackend

        assert hasattr(StorageBackend, 'disconnect')

    def test_storage_backend_requires_health_check(self):
        """Test that StorageBackend requires health_check method."""
        from src.storage.base import StorageBackend

        assert hasattr(StorageBackend, 'health_check')


class TestVectorStoreInterface:
    """Tests for the VectorStore abstract base class."""

    def test_vector_store_is_abstract(self):
        """Test that VectorStore is abstract."""
        from src.storage.base import VectorStore

        assert issubclass(VectorStore, ABC)

    def test_vector_store_requires_upsert(self):
        """Test that VectorStore requires upsert method."""
        from src.storage.base import VectorStore

        assert hasattr(VectorStore, 'upsert')

    def test_vector_store_requires_search(self):
        """Test that VectorStore requires search method."""
        from src.storage.base import VectorStore

        assert hasattr(VectorStore, 'search')

    def test_vector_store_requires_delete(self):
        """Test that VectorStore requires delete method."""
        from src.storage.base import VectorStore

        assert hasattr(VectorStore, 'delete')


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
