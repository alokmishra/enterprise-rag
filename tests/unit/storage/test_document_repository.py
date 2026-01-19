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
        session.flush = AsyncMock()
        session.rollback = AsyncMock()
        session.add = MagicMock()
        session.add_all = MagicMock()
        return session

    @pytest.fixture
    def sample_document(self):
        """Create a sample Document for testing."""
        from src.core.types import Document, DocumentMetadata, DocumentStatus

        return Document(
            id=str(uuid4()),
            content="Test content",
            status=DocumentStatus.PENDING,
            metadata=DocumentMetadata(
                source="test",
                source_id="test-123",
                title="Test Document",
                author="Test Author",
                file_type="pdf",
                file_size=1024,
                language="en",
            ),
            chunks=[],
        )

    @pytest.mark.asyncio
    async def test_document_repository_create(self, mock_session, sample_document):
        """Test DocumentRepository create method."""
        from src.storage.document.repository import DocumentRepository

        repo = DocumentRepository(mock_session)
        doc_id = await repo.create(sample_document)

        assert doc_id == sample_document.id
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_document_repository_get(self, mock_session):
        """Test DocumentRepository get method."""
        from src.storage.document.repository import DocumentRepository
        from src.storage.document.models import DocumentModel

        mock_doc = MagicMock(spec=DocumentModel)
        mock_doc.id = "doc-1"
        mock_doc.content = "Test content"
        mock_doc.status = "pending"
        mock_doc.source = "test"
        mock_doc.source_id = "test-123"
        mock_doc.title = "Test"
        mock_doc.author = None
        mock_doc.file_type = "pdf"
        mock_doc.file_size = 1024
        mock_doc.language = "en"
        mock_doc.custom_metadata = {}
        mock_doc.document_created_at = None
        mock_doc.document_modified_at = None
        mock_doc.chunks = []

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=mock_doc)
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = DocumentRepository(mock_session)
        doc = await repo.get("doc-1")

        assert doc is not None
        assert doc.id == "doc-1"

    @pytest.mark.asyncio
    async def test_document_repository_get_not_found(self, mock_session):
        """Test DocumentRepository get returns None for missing document."""
        from src.storage.document.repository import DocumentRepository

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = DocumentRepository(mock_session)
        doc = await repo.get("nonexistent")
        assert doc is None

    @pytest.mark.asyncio
    async def test_document_repository_update(self, mock_session):
        """Test DocumentRepository update method."""
        from src.storage.document.repository import DocumentRepository

        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = DocumentRepository(mock_session)
        updated = await repo.update("doc-1", {"content": "New content"})

        assert updated is True

    @pytest.mark.asyncio
    async def test_document_repository_update_status(self, mock_session):
        """Test DocumentRepository update_status method."""
        from src.storage.document.repository import DocumentRepository
        from src.core.types import DocumentStatus

        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = DocumentRepository(mock_session)
        updated = await repo.update_status("doc-1", DocumentStatus.COMPLETED)

        assert updated is True

    @pytest.mark.asyncio
    async def test_document_repository_delete(self, mock_session):
        """Test DocumentRepository delete method."""
        from src.storage.document.repository import DocumentRepository

        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = DocumentRepository(mock_session)
        deleted = await repo.delete("doc-1")

        assert deleted is True

    @pytest.mark.asyncio
    async def test_document_repository_list(self, mock_session):
        """Test DocumentRepository list method."""
        from src.storage.document.repository import DocumentRepository
        from src.storage.document.models import DocumentModel

        mock_doc1 = MagicMock(spec=DocumentModel)
        mock_doc1.id = "doc-1"
        mock_doc1.content = "Content 1"
        mock_doc1.status = "pending"
        mock_doc1.source = "test"
        mock_doc1.source_id = "test-1"
        mock_doc1.title = "Doc 1"
        mock_doc1.author = None
        mock_doc1.file_type = "pdf"
        mock_doc1.file_size = 1024
        mock_doc1.language = "en"
        mock_doc1.custom_metadata = {}
        mock_doc1.document_created_at = None
        mock_doc1.document_modified_at = None
        mock_doc1.chunks = []

        mock_doc2 = MagicMock(spec=DocumentModel)
        mock_doc2.id = "doc-2"
        mock_doc2.content = "Content 2"
        mock_doc2.status = "pending"
        mock_doc2.source = "test"
        mock_doc2.source_id = "test-2"
        mock_doc2.title = "Doc 2"
        mock_doc2.author = None
        mock_doc2.file_type = "pdf"
        mock_doc2.file_size = 2048
        mock_doc2.language = "en"
        mock_doc2.custom_metadata = {}
        mock_doc2.document_created_at = None
        mock_doc2.document_modified_at = None
        mock_doc2.chunks = []

        mock_result = MagicMock()
        mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[mock_doc1, mock_doc2])))
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = DocumentRepository(mock_session)
        docs = await repo.list(limit=10, offset=0)

        assert len(docs) == 2
        assert docs[0].id == "doc-1"
        assert docs[1].id == "doc-2"

    @pytest.mark.asyncio
    async def test_document_repository_count(self, mock_session):
        """Test DocumentRepository count method."""
        from src.storage.document.repository import DocumentRepository

        mock_result = MagicMock()
        mock_result.scalar = MagicMock(return_value=5)
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = DocumentRepository(mock_session)
        count = await repo.count()

        assert count == 5


class TestChunkRepository:
    """Tests for the ChunkRepository class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.flush = AsyncMock()
        session.add = MagicMock()
        session.add_all = MagicMock()
        return session

    @pytest.fixture
    def sample_chunks(self):
        """Create sample Chunk objects for testing."""
        from src.core.types import Chunk, ContentType

        return [
            Chunk(
                id="chunk-1",
                document_id="doc-1",
                content="Chunk content 1",
                content_type=ContentType.TEXT,
                position=0,
                metadata={},
            ),
            Chunk(
                id="chunk-2",
                document_id="doc-1",
                content="Chunk content 2",
                content_type=ContentType.TEXT,
                position=1,
                metadata={},
            ),
        ]

    @pytest.mark.asyncio
    async def test_chunk_repository_create_many(self, mock_session, sample_chunks):
        """Test ChunkRepository create_many method."""
        from src.storage.document.repository import ChunkRepository

        repo = ChunkRepository(mock_session)
        chunk_ids = await repo.create_many(sample_chunks)

        assert len(chunk_ids) == 2
        mock_session.add_all.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_chunk_repository_get(self, mock_session):
        """Test ChunkRepository get method."""
        from src.storage.document.repository import ChunkRepository
        from src.storage.document.models import ChunkModel

        mock_chunk = MagicMock(spec=ChunkModel)
        mock_chunk.id = "chunk-1"
        mock_chunk.document_id = "doc-1"
        mock_chunk.content = "Chunk content"
        mock_chunk.content_type = "text"
        mock_chunk.position = 0
        mock_chunk.chunk_metadata = {}
        mock_chunk.parent_id = None

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=mock_chunk)
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = ChunkRepository(mock_session)
        chunk = await repo.get("chunk-1")

        assert chunk is not None
        assert chunk.id == "chunk-1"

    @pytest.mark.asyncio
    async def test_chunk_repository_get_by_document(self, mock_session):
        """Test ChunkRepository get_by_document method."""
        from src.storage.document.repository import ChunkRepository
        from src.storage.document.models import ChunkModel

        mock_chunk1 = MagicMock(spec=ChunkModel)
        mock_chunk1.id = "chunk-1"
        mock_chunk1.document_id = "doc-1"
        mock_chunk1.content = "Content 1"
        mock_chunk1.content_type = "text"
        mock_chunk1.position = 0
        mock_chunk1.chunk_metadata = {}
        mock_chunk1.parent_id = None

        mock_chunk2 = MagicMock(spec=ChunkModel)
        mock_chunk2.id = "chunk-2"
        mock_chunk2.document_id = "doc-1"
        mock_chunk2.content = "Content 2"
        mock_chunk2.content_type = "text"
        mock_chunk2.position = 1
        mock_chunk2.chunk_metadata = {}
        mock_chunk2.parent_id = None

        mock_result = MagicMock()
        mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[mock_chunk1, mock_chunk2])))
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = ChunkRepository(mock_session)
        chunks = await repo.get_by_document("doc-1")

        assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_chunk_repository_get_many(self, mock_session):
        """Test ChunkRepository get_many method."""
        from src.storage.document.repository import ChunkRepository
        from src.storage.document.models import ChunkModel

        mock_chunk = MagicMock(spec=ChunkModel)
        mock_chunk.id = "chunk-1"
        mock_chunk.document_id = "doc-1"
        mock_chunk.content = "Content"
        mock_chunk.content_type = "text"
        mock_chunk.position = 0
        mock_chunk.chunk_metadata = {}
        mock_chunk.parent_id = None

        mock_result = MagicMock()
        mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[mock_chunk])))
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = ChunkRepository(mock_session)
        chunks = await repo.get_many(["chunk-1"])

        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_chunk_repository_delete_by_document(self, mock_session):
        """Test ChunkRepository delete_by_document method."""
        from src.storage.document.repository import ChunkRepository

        mock_result = MagicMock()
        mock_result.rowcount = 3
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = ChunkRepository(mock_session)
        count = await repo.delete_by_document("doc-1")

        assert count == 3


class TestQueryLogRepository:
    """Tests for the QueryLogRepository class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.flush = AsyncMock()
        session.add = MagicMock()
        return session

    @pytest.mark.asyncio
    async def test_query_log_repository_create(self, mock_session):
        """Test QueryLogRepository create method."""
        from src.storage.document.repository import QueryLogRepository

        repo = QueryLogRepository(mock_session)

        query_id = await repo.create(
            query="What is the policy?",
            answer="The policy states...",
            latency_ms=150.0,
            tokens_used=200,
        )

        assert query_id is not None
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_log_repository_get(self, mock_session):
        """Test QueryLogRepository get method."""
        from src.storage.document.repository import QueryLogRepository
        from src.storage.document.models import QueryLogModel

        mock_log = MagicMock(spec=QueryLogModel)
        mock_log.id = "query-1"
        mock_log.query = "Test query"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=mock_log)
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = QueryLogRepository(mock_session)
        log = await repo.get("query-1")

        assert log is not None
        assert log.id == "query-1"

    @pytest.mark.asyncio
    async def test_query_log_repository_update_feedback(self, mock_session):
        """Test QueryLogRepository update_feedback method."""
        from src.storage.document.repository import QueryLogRepository

        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = QueryLogRepository(mock_session)
        updated = await repo.update_feedback("query-1", rating=5, feedback="Great answer!")

        assert updated is True
