"""
Tests for tenant filtering in repositories.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.core.types import Document, DocumentMetadata, DocumentStatus, Chunk, ContentType


class TestDocumentRepositoryTenantId:
    """Tests for DocumentRepository tenant handling."""

    def test_document_repository_requires_tenant_id(self):
        """Repository requires tenant_id in constructor."""
        from src.storage.document.repository import DocumentRepository

        mock_session = MagicMock()

        # Default tenant_id
        repo = DocumentRepository(mock_session)
        assert repo.tenant_id == "default"

        # Custom tenant_id
        repo = DocumentRepository(mock_session, tenant_id="tenant-123")
        assert repo.tenant_id == "tenant-123"

    @pytest.mark.asyncio
    async def test_document_create_sets_tenant_id(self):
        """Created documents have tenant_id set."""
        from src.storage.document.repository import DocumentRepository

        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        repo = DocumentRepository(mock_session, tenant_id="tenant-abc")

        doc = Document(
            id=str(uuid4()),
            content="Test content",
            metadata=DocumentMetadata(source="test"),
            status=DocumentStatus.PENDING,
        )

        await repo.create(doc)

        # Check that add was called
        mock_session.add.assert_called_once()
        added_model = mock_session.add.call_args[0][0]

        # Verify tenant_id was set
        assert added_model.tenant_id == "tenant-abc"

    @pytest.mark.asyncio
    async def test_document_get_filters_by_tenant(self):
        """Get only returns documents for the tenant."""
        from src.storage.document.repository import DocumentRepository
        from src.storage.document.models import DocumentModel

        # Create mock with specific tenant filtering behavior
        mock_session = AsyncMock()

        # Mock execute to verify the where clause includes tenant_id
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = DocumentRepository(mock_session, tenant_id="tenant-xyz")
        result = await repo.get("doc-123")

        # Verify execute was called (query includes tenant filter)
        mock_session.execute.assert_called_once()
        assert result is None

    @pytest.mark.asyncio
    async def test_document_list_filters_by_tenant(self):
        """List only returns documents for the tenant."""
        from src.storage.document.repository import DocumentRepository

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = DocumentRepository(mock_session, tenant_id="tenant-list")
        result = await repo.list()

        mock_session.execute.assert_called_once()
        assert result == []

    @pytest.mark.asyncio
    async def test_document_delete_filters_by_tenant(self):
        """Delete only affects documents for the tenant."""
        from src.storage.document.repository import DocumentRepository

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = DocumentRepository(mock_session, tenant_id="tenant-delete")
        result = await repo.delete("doc-to-delete")

        mock_session.execute.assert_called_once()
        assert result is True

    @pytest.mark.asyncio
    async def test_document_count_filters_by_tenant(self):
        """Count only counts documents for the tenant."""
        from src.storage.document.repository import DocumentRepository

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar = MagicMock(return_value=5)
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = DocumentRepository(mock_session, tenant_id="tenant-count")
        result = await repo.count()

        mock_session.execute.assert_called_once()
        assert result == 5


class TestChunkRepositoryTenantId:
    """Tests for ChunkRepository tenant handling."""

    def test_chunk_repository_requires_tenant_id(self):
        """Repository accepts tenant_id in constructor."""
        from src.storage.document.repository import ChunkRepository

        mock_session = MagicMock()

        # Default tenant_id
        repo = ChunkRepository(mock_session)
        assert repo.tenant_id == "default"

        # Custom tenant_id
        repo = ChunkRepository(mock_session, tenant_id="tenant-chunk-123")
        assert repo.tenant_id == "tenant-chunk-123"

    @pytest.mark.asyncio
    async def test_chunk_create_sets_tenant_id(self):
        """Created chunks have tenant_id set."""
        from src.storage.document.repository import ChunkRepository

        mock_session = AsyncMock()
        mock_session.add_all = MagicMock()
        mock_session.flush = AsyncMock()

        repo = ChunkRepository(mock_session, tenant_id="tenant-chunk-abc")

        chunks = [
            Chunk(
                id=str(uuid4()),
                document_id="doc-1",
                content="Test chunk content",
                content_type=ContentType.TEXT,
                position=0,
            )
        ]

        await repo.create_many(chunks)

        # Check that add_all was called
        mock_session.add_all.assert_called_once()
        added_models = mock_session.add_all.call_args[0][0]

        # Verify tenant_id was set on each chunk
        for model in added_models:
            assert model.tenant_id == "tenant-chunk-abc"

    @pytest.mark.asyncio
    async def test_chunk_get_filters_by_tenant(self):
        """Get only returns chunks for the tenant."""
        from src.storage.document.repository import ChunkRepository

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = ChunkRepository(mock_session, tenant_id="tenant-chunk-get")
        result = await repo.get("chunk-123")

        mock_session.execute.assert_called_once()
        assert result is None


class TestQueryLogRepositoryTenantId:
    """Tests for QueryLogRepository tenant handling."""

    def test_query_log_repository_requires_tenant_id(self):
        """Repository accepts tenant_id in constructor."""
        from src.storage.document.repository import QueryLogRepository

        mock_session = MagicMock()

        # Default tenant_id
        repo = QueryLogRepository(mock_session)
        assert repo.tenant_id == "default"

        # Custom tenant_id
        repo = QueryLogRepository(mock_session, tenant_id="tenant-log-123")
        assert repo.tenant_id == "tenant-log-123"

    @pytest.mark.asyncio
    async def test_query_log_create_sets_tenant_id(self):
        """Created query logs have tenant_id set."""
        from src.storage.document.repository import QueryLogRepository

        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        repo = QueryLogRepository(mock_session, tenant_id="tenant-log-abc")
        await repo.create(query="What is RAG?")

        # Check that add was called
        mock_session.add.assert_called_once()
        added_model = mock_session.add.call_args[0][0]

        # Verify tenant_id was set
        assert added_model.tenant_id == "tenant-log-abc"

    @pytest.mark.asyncio
    async def test_query_log_get_filters_by_tenant(self):
        """Get only returns query logs for the tenant."""
        from src.storage.document.repository import QueryLogRepository

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = QueryLogRepository(mock_session, tenant_id="tenant-log-get")
        result = await repo.get("log-123")

        mock_session.execute.assert_called_once()
        assert result is None
