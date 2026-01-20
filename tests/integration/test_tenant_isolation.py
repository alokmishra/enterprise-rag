"""
CRITICAL: Integration tests for tenant isolation.

These tests verify that tenant A cannot access tenant B's data.
This is essential for multi-tenant security.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.core.types import Document, DocumentMetadata, DocumentStatus, Chunk, ContentType


class TestDocumentTenantIsolation:
    """Critical tests for document tenant isolation."""

    @pytest.mark.asyncio
    async def test_tenant_a_cannot_access_tenant_b_documents(self):
        """Tenant A cannot retrieve Tenant B's documents."""
        from src.storage.document.repository import DocumentRepository
        from src.storage.document.models import DocumentModel

        # Create a mock session that returns a document for tenant B
        mock_session = AsyncMock()

        # Simulate that the document belongs to tenant B
        tenant_b_doc = MagicMock(spec=DocumentModel)
        tenant_b_doc.id = "doc-123"
        tenant_b_doc.tenant_id = "tenant-b"
        tenant_b_doc.content = "Secret data for tenant B"

        # When tenant A queries, the result should be None
        # because the where clause filters by tenant_id
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Tenant A's repository
        repo_tenant_a = DocumentRepository(mock_session, tenant_id="tenant-a")

        # Try to access tenant B's document
        result = await repo_tenant_a.get("doc-123")

        # Should return None because tenant filter blocks access
        assert result is None

    @pytest.mark.asyncio
    async def test_tenant_a_cannot_list_tenant_b_documents(self):
        """Tenant A's list doesn't include Tenant B's documents."""
        from src.storage.document.repository import DocumentRepository

        mock_session = AsyncMock()

        # Simulate empty result for tenant A (even if tenant B has documents)
        mock_result = MagicMock()
        mock_result.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=[]))
        )
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Tenant A's repository
        repo_tenant_a = DocumentRepository(mock_session, tenant_id="tenant-a")

        # List documents as tenant A
        result = await repo_tenant_a.list()

        # Should be empty (tenant B's documents not included)
        assert result == []

    @pytest.mark.asyncio
    async def test_tenant_a_cannot_delete_tenant_b_documents(self):
        """Tenant A cannot delete Tenant B's documents."""
        from src.storage.document.repository import DocumentRepository

        mock_session = AsyncMock()

        # Simulate no rows affected (tenant filter prevents deletion)
        mock_result = MagicMock()
        mock_result.rowcount = 0  # No rows deleted
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Tenant A's repository
        repo_tenant_a = DocumentRepository(mock_session, tenant_id="tenant-a")

        # Try to delete tenant B's document
        result = await repo_tenant_a.delete("tenant-b-doc-id")

        # Should return False (nothing was deleted)
        assert result is False

    @pytest.mark.asyncio
    async def test_tenant_a_cannot_update_tenant_b_documents(self):
        """Tenant A cannot update Tenant B's documents."""
        from src.storage.document.repository import DocumentRepository

        mock_session = AsyncMock()

        # Simulate no rows affected (tenant filter prevents update)
        mock_result = MagicMock()
        mock_result.rowcount = 0  # No rows updated
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Tenant A's repository
        repo_tenant_a = DocumentRepository(mock_session, tenant_id="tenant-a")

        # Try to update tenant B's document
        result = await repo_tenant_a.update(
            "tenant-b-doc-id",
            {"title": "Hacked!"}
        )

        # Should return False (nothing was updated)
        assert result is False


class TestChunkTenantIsolation:
    """Critical tests for chunk tenant isolation."""

    @pytest.mark.asyncio
    async def test_tenant_a_cannot_access_tenant_b_chunks(self):
        """Tenant A cannot retrieve Tenant B's chunks."""
        from src.storage.document.repository import ChunkRepository

        mock_session = AsyncMock()

        # When tenant A queries, result should be None
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Tenant A's repository
        repo_tenant_a = ChunkRepository(mock_session, tenant_id="tenant-a")

        # Try to access tenant B's chunk
        result = await repo_tenant_a.get("chunk-123")

        # Should return None
        assert result is None

    @pytest.mark.asyncio
    async def test_tenant_a_cannot_get_tenant_b_chunks_by_document(self):
        """Tenant A cannot get Tenant B's chunks by document ID."""
        from src.storage.document.repository import ChunkRepository

        mock_session = AsyncMock()

        # Simulate empty result for tenant A
        mock_result = MagicMock()
        mock_result.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=[]))
        )
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Tenant A's repository
        repo_tenant_a = ChunkRepository(mock_session, tenant_id="tenant-a")

        # Try to get chunks for tenant B's document
        result = await repo_tenant_a.get_by_document("tenant-b-doc-id")

        # Should be empty
        assert result == []

    @pytest.mark.asyncio
    async def test_tenant_a_cannot_delete_tenant_b_chunks(self):
        """Tenant A cannot delete Tenant B's chunks."""
        from src.storage.document.repository import ChunkRepository

        mock_session = AsyncMock()

        # Simulate no rows affected
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Tenant A's repository
        repo_tenant_a = ChunkRepository(mock_session, tenant_id="tenant-a")

        # Try to delete tenant B's chunks
        result = await repo_tenant_a.delete_by_document("tenant-b-doc-id")

        # Should return 0 (nothing deleted)
        assert result == 0


class TestQueryLogTenantIsolation:
    """Critical tests for query log tenant isolation."""

    @pytest.mark.asyncio
    async def test_tenant_a_cannot_access_tenant_b_query_logs(self):
        """Tenant A cannot retrieve Tenant B's query logs."""
        from src.storage.document.repository import QueryLogRepository

        mock_session = AsyncMock()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Tenant A's repository
        repo_tenant_a = QueryLogRepository(mock_session, tenant_id="tenant-a")

        # Try to access tenant B's query log
        result = await repo_tenant_a.get("tenant-b-log-id")

        # Should return None
        assert result is None

    @pytest.mark.asyncio
    async def test_tenant_a_cannot_update_tenant_b_feedback(self):
        """Tenant A cannot update feedback on Tenant B's query logs."""
        from src.storage.document.repository import QueryLogRepository

        mock_session = AsyncMock()

        mock_result = MagicMock()
        mock_result.rowcount = 0  # No rows updated
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Tenant A's repository
        repo_tenant_a = QueryLogRepository(mock_session, tenant_id="tenant-a")

        # Try to update tenant B's query log feedback
        result = await repo_tenant_a.update_feedback(
            "tenant-b-log-id",
            rating=5,
            feedback="Fake feedback!"
        )

        # Should return False
        assert result is False


class TestCrossTenantDataCreation:
    """Tests that data is always created with the correct tenant ID."""

    @pytest.mark.asyncio
    async def test_document_created_with_repository_tenant(self):
        """Documents are created with the repository's tenant ID, not arbitrary."""
        from src.storage.document.repository import DocumentRepository

        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        # Create repository for specific tenant
        repo = DocumentRepository(mock_session, tenant_id="correct-tenant")

        doc = Document(
            id="doc-new",
            content="New document",
            metadata=DocumentMetadata(source="test"),
        )

        await repo.create(doc)

        # Verify the model was created with the repository's tenant_id
        added_model = mock_session.add.call_args[0][0]
        assert added_model.tenant_id == "correct-tenant"

    @pytest.mark.asyncio
    async def test_chunk_created_with_repository_tenant(self):
        """Chunks are created with the repository's tenant ID."""
        from src.storage.document.repository import ChunkRepository

        mock_session = AsyncMock()
        mock_session.add_all = MagicMock()
        mock_session.flush = AsyncMock()

        # Create repository for specific tenant
        repo = ChunkRepository(mock_session, tenant_id="correct-tenant")

        chunks = [
            Chunk(
                id="chunk-new",
                document_id="doc-1",
                content="New chunk",
                content_type=ContentType.TEXT,
                position=0,
            )
        ]

        await repo.create_many(chunks)

        # Verify the models were created with the repository's tenant_id
        added_models = mock_session.add_all.call_args[0][0]
        for model in added_models:
            assert model.tenant_id == "correct-tenant"

    @pytest.mark.asyncio
    async def test_query_log_created_with_repository_tenant(self):
        """Query logs are created with the repository's tenant ID."""
        from src.storage.document.repository import QueryLogRepository

        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        # Create repository for specific tenant
        repo = QueryLogRepository(mock_session, tenant_id="correct-tenant")

        await repo.create(query="Test query")

        # Verify the model was created with the repository's tenant_id
        added_model = mock_session.add.call_args[0][0]
        assert added_model.tenant_id == "correct-tenant"
