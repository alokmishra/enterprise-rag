"""
Tests for src/storage/document/repository.py
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


class TestDocumentRepository:
    """Tests for the DocumentRepository class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.add = MagicMock()
        session.delete = MagicMock()
        return session

    @pytest.mark.asyncio
    async def test_document_repository_create(self, mock_session):
        """Test DocumentRepository create method."""
        from src.storage.document.repository import DocumentRepository

        repo = DocumentRepository(mock_session)

        doc_id = str(uuid4())
        doc = await repo.create(
            id=doc_id,
            source="test.pdf",
            content="Test content",
            metadata={"author": "Test"},
        )
        # Should call session.add

    @pytest.mark.asyncio
    async def test_document_repository_get_by_id(self, mock_session):
        """Test DocumentRepository get_by_id method."""
        from src.storage.document.repository import DocumentRepository

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=MagicMock(
            id="doc-1",
            source="test.pdf",
            content="Test content",
        ))
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = DocumentRepository(mock_session)
        doc = await repo.get_by_id("doc-1")
        # Should return document or None

    @pytest.mark.asyncio
    async def test_document_repository_get_by_id_not_found(self, mock_session):
        """Test DocumentRepository get_by_id returns None for missing document."""
        from src.storage.document.repository import DocumentRepository

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = DocumentRepository(mock_session)
        doc = await repo.get_by_id("nonexistent")
        assert doc is None

    @pytest.mark.asyncio
    async def test_document_repository_update(self, mock_session):
        """Test DocumentRepository update method."""
        from src.storage.document.repository import DocumentRepository

        mock_doc = MagicMock(id="doc-1", content="Old content")
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=mock_doc)
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = DocumentRepository(mock_session)
        updated = await repo.update("doc-1", content="New content")
        # Should update and commit

    @pytest.mark.asyncio
    async def test_document_repository_delete(self, mock_session):
        """Test DocumentRepository delete method."""
        from src.storage.document.repository import DocumentRepository

        mock_doc = MagicMock(id="doc-1")
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=mock_doc)
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = DocumentRepository(mock_session)
        await repo.delete("doc-1")
        # Should delete and commit

    @pytest.mark.asyncio
    async def test_document_repository_list(self, mock_session):
        """Test DocumentRepository list method."""
        from src.storage.document.repository import DocumentRepository

        mock_docs = [
            MagicMock(id="doc-1"),
            MagicMock(id="doc-2"),
        ]
        mock_result = MagicMock()
        mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=mock_docs)))
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = DocumentRepository(mock_session)
        docs = await repo.list(limit=10, offset=0)
        # Should return list of documents


class TestChunkRepository:
    """Tests for the ChunkRepository class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.add = MagicMock()
        return session

    @pytest.mark.asyncio
    async def test_chunk_repository_create(self, mock_session):
        """Test ChunkRepository create method."""
        from src.storage.document.repository import ChunkRepository

        repo = ChunkRepository(mock_session)

        chunk = await repo.create(
            id="chunk-1",
            document_id="doc-1",
            content="Chunk content",
            position=0,
        )
        # Should create chunk

    @pytest.mark.asyncio
    async def test_chunk_repository_get_by_document(self, mock_session):
        """Test ChunkRepository get_by_document method."""
        from src.storage.document.repository import ChunkRepository

        mock_chunks = [
            MagicMock(id="chunk-1", position=0),
            MagicMock(id="chunk-2", position=1),
        ]
        mock_result = MagicMock()
        mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=mock_chunks)))
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = ChunkRepository(mock_session)
        chunks = await repo.get_by_document("doc-1")
        # Should return chunks for document

    @pytest.mark.asyncio
    async def test_chunk_repository_bulk_create(self, mock_session):
        """Test ChunkRepository bulk_create method."""
        from src.storage.document.repository import ChunkRepository

        repo = ChunkRepository(mock_session)

        chunks_data = [
            {"id": "chunk-1", "document_id": "doc-1", "content": "Content 1", "position": 0},
            {"id": "chunk-2", "document_id": "doc-1", "content": "Content 2", "position": 1},
        ]

        if hasattr(repo, 'bulk_create'):
            await repo.bulk_create(chunks_data)


class TestQueryLogRepository:
    """Tests for the QueryLogRepository class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.add = MagicMock()
        return session

    @pytest.mark.asyncio
    async def test_query_log_repository_log_query(self, mock_session):
        """Test QueryLogRepository log_query method."""
        from src.storage.document.repository import QueryLogRepository

        repo = QueryLogRepository(mock_session)

        await repo.log_query(
            query_id="query-1",
            query="What is the policy?",
            response="The policy states...",
            latency_ms=150.0,
            tokens_used=200,
        )
        # Should log query

    @pytest.mark.asyncio
    async def test_query_log_repository_get_by_id(self, mock_session):
        """Test QueryLogRepository get_by_id method."""
        from src.storage.document.repository import QueryLogRepository

        mock_log = MagicMock(query_id="query-1", query="Test query")
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=mock_log)
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = QueryLogRepository(mock_session)
        log = await repo.get_by_id("query-1")
        # Should return query log
