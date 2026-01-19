"""
Enterprise RAG System - Storage Module

This module provides storage backends for:
- Vector storage (Qdrant)
- Document storage (PostgreSQL)
- Cache storage (Redis)
"""

from __future__ import annotations

from src.storage.base import (
    StorageBackend,
    VectorStore,
    DocumentStore,
    CacheStore,
)
from src.storage.vector import (
    QdrantVectorStore,
    QdrantSearchResult,
    get_vector_store,
    init_vector_store,
    close_vector_store,
)
from src.storage.cache import (
    RedisCache,
    CacheError,
    get_cache,
    init_cache,
    close_cache,
)
from src.storage.document import (
    Database,
    DatabaseError,
    get_database,
    init_database,
    close_database,
    get_session,
    DocumentRepository,
    ChunkRepository,
    QueryLogRepository,
)

__all__ = [
    # Base classes
    "StorageBackend",
    "VectorStore",
    "DocumentStore",
    "CacheStore",
    # Vector store
    "QdrantVectorStore",
    "QdrantSearchResult",
    "get_vector_store",
    "init_vector_store",
    "close_vector_store",
    # Cache
    "RedisCache",
    "CacheError",
    "get_cache",
    "init_cache",
    "close_cache",
    # Database
    "Database",
    "DatabaseError",
    "get_database",
    "init_database",
    "close_database",
    "get_session",
    "DocumentRepository",
    "ChunkRepository",
    "QueryLogRepository",
]
