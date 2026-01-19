"""
Enterprise RAG System - Document Storage Module
"""

from __future__ import annotations

from src.storage.document.database import (
    Database,
    DatabaseError,
    get_database,
    init_database,
    close_database,
    get_session,
)
from src.storage.document.models import (
    Base,
    DocumentModel,
    ChunkModel,
    QueryLogModel,
)
from src.storage.document.repository import (
    DocumentRepository,
    ChunkRepository,
    QueryLogRepository,
)

__all__ = [
    # Database
    "Database",
    "DatabaseError",
    "get_database",
    "init_database",
    "close_database",
    "get_session",
    # Models
    "Base",
    "DocumentModel",
    "ChunkModel",
    "QueryLogModel",
    # Repositories
    "DocumentRepository",
    "ChunkRepository",
    "QueryLogRepository",
]
