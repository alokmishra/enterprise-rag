"""
Enterprise RAG System - Storage Base Classes
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

from src.core.logging import LoggerMixin

T = TypeVar("T")


class StorageBackend(ABC, LoggerMixin):
    """Abstract base class for all storage backends."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the storage backend."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the storage backend."""
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Check health of the storage backend."""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to storage backend."""
        pass


class VectorStore(StorageBackend, Generic[T]):
    """Abstract base class for vector storage backends."""

    @abstractmethod
    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance_metric: str = "cosine",
    ) -> None:
        """Create a new vector collection."""
        pass

    @abstractmethod
    async def delete_collection(self, name: str) -> None:
        """Delete a vector collection."""
        pass

    @abstractmethod
    async def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        pass

    @abstractmethod
    async def insert(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Insert vectors into a collection."""
        pass

    @abstractmethod
    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[T]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def delete(
        self,
        collection: str,
        ids: list[str],
    ) -> None:
        """Delete vectors by ID."""
        pass

    @abstractmethod
    async def get(
        self,
        collection: str,
        ids: list[str],
    ) -> list[T]:
        """Get vectors by ID."""
        pass


class DocumentStore(StorageBackend, Generic[T]):
    """Abstract base class for document storage backends."""

    @abstractmethod
    async def save(self, document: T) -> str:
        """Save a document, return its ID."""
        pass

    @abstractmethod
    async def get(self, document_id: str) -> Optional[T]:
        """Get a document by ID."""
        pass

    @abstractmethod
    async def delete(self, document_id: str) -> bool:
        """Delete a document by ID."""
        pass

    @abstractmethod
    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[T]:
        """List documents with optional filtering."""
        pass

    @abstractmethod
    async def update(self, document_id: str, updates: dict[str, Any]) -> bool:
        """Update a document."""
        pass


class CacheStore(StorageBackend):
    """Abstract base class for cache storage backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set a value in cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass

    @abstractmethod
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache, optionally by pattern."""
        pass
