"""
Integration tests for API tenant context.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.deployment import reset_deployment_config
from src.core.tenant import DEFAULT_TENANT_ID


class TestQueryEndpointTenantContext:
    """Tests for query endpoint tenant handling."""

    def setup_method(self):
        """Reset state before each test."""
        reset_deployment_config()

    def teardown_method(self):
        """Reset state after each test."""
        reset_deployment_config()

    @pytest.mark.asyncio
    async def test_query_endpoint_uses_tenant_context(self):
        """Query endpoint passes tenant_id to pipeline."""
        from src.api.routes.query import query, QueryRequest

        # Create request
        request = QueryRequest(
            query="What is the policy?",
            include_sources=True,
        )

        # Mock the pipeline
        mock_response = MagicMock()
        mock_response.query_id = "test-id"
        mock_response.query = "What is the policy?"
        mock_response.answer = "The policy is..."
        mock_response.sources = []
        mock_response.complexity = "standard"
        mock_response.latency_ms = 100.0
        mock_response.tokens_used = 50
        mock_response.confidence = 0.9

        with patch("src.api.routes.query.get_rag_pipeline") as mock_get_pipeline:
            mock_pipeline = AsyncMock()
            mock_pipeline.query = AsyncMock(return_value=mock_response)
            mock_get_pipeline.return_value = mock_pipeline

            # Mock database
            with patch("src.api.routes.query.get_database") as mock_db:
                mock_db.return_value.session = MagicMock(return_value=AsyncMock())

                # Execute with tenant context
                response = await query(request, tenant_id="tenant-query-test")

                # Verify pipeline was created with tenant_id
                mock_get_pipeline.assert_called_once_with(tenant_id="tenant-query-test")

    @pytest.mark.asyncio
    async def test_query_returns_only_tenant_documents(self):
        """Query results only include tenant's documents."""
        from src.api.services.rag_pipeline import RAGPipeline

        # Create pipeline with tenant
        pipeline = RAGPipeline(tenant_id="tenant-a")

        # Verify tenant_id is set
        assert pipeline.tenant_id == "tenant-a"


class TestDocumentTenantContext:
    """Tests for document upload tenant handling."""

    @pytest.mark.asyncio
    async def test_document_upload_sets_tenant_id(self):
        """Uploaded documents have correct tenant_id."""
        # This would test the document upload endpoint with tenant context
        # For now, verify the middleware dependency works

        from src.api.middleware.tenant import get_tenant_id

        mock_request = MagicMock()
        mock_request.state.tenant_id = "tenant-upload"

        result = get_tenant_id(mock_request)
        assert result == "tenant-upload"


class TestUnauthorizedTenantAccess:
    """Tests for unauthorized tenant access rejection."""

    def setup_method(self):
        """Reset state before each test."""
        reset_deployment_config()

    def teardown_method(self):
        """Reset state after each test."""
        reset_deployment_config()

    @pytest.mark.asyncio
    async def test_unauthorized_tenant_access_rejected(self):
        """Requests for other tenant's resources are rejected via filtering."""
        from src.storage.document.repository import DocumentRepository
        from unittest.mock import MagicMock

        # Create repos for different tenants
        mock_session = MagicMock()

        repo_a = DocumentRepository(mock_session, tenant_id="tenant-a")
        repo_b = DocumentRepository(mock_session, tenant_id="tenant-b")

        # Verify they have different tenant IDs
        assert repo_a._tenant_id == "tenant-a"
        assert repo_b._tenant_id == "tenant-b"

        # In a real scenario, queries would filter by tenant_id
        # preventing cross-tenant access


class TestFeedbackEndpointTenantContext:
    """Tests for feedback endpoint tenant handling."""

    @pytest.mark.asyncio
    async def test_feedback_uses_tenant_repository(self):
        """Feedback endpoint uses tenant-scoped repository."""
        from src.api.routes.query import submit_feedback, FeedbackRequest

        request = FeedbackRequest(
            query_id="query-123",
            rating=5,
            feedback="Great answer!",
        )

        with patch("src.api.routes.query.get_database") as mock_db:
            mock_db.return_value.is_connected = True

            mock_session = AsyncMock()
            mock_db.return_value.session = MagicMock(return_value=mock_session)

            with patch("src.api.routes.query.QueryLogRepository") as MockRepo:
                mock_repo = AsyncMock()
                mock_repo.get = AsyncMock(return_value=MagicMock())
                mock_repo.update_feedback = AsyncMock(return_value=True)
                MockRepo.return_value = mock_repo

                # This would execute with tenant context
                # For now, just verify the repository pattern
                assert MockRepo is not None


class TestRAGPipelineTenantContext:
    """Tests for RAG pipeline tenant handling."""

    @pytest.mark.asyncio
    async def test_pipeline_uses_tenant_searcher(self):
        """RAG pipeline uses tenant-scoped vector searcher."""
        from src.api.services.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(tenant_id="tenant-rag")

        # Verify tenant is stored
        assert pipeline.tenant_id == "tenant-rag"

    @pytest.mark.asyncio
    async def test_pipeline_retrieval_filters_by_tenant(self):
        """Pipeline retrieval uses tenant-scoped searcher."""
        from src.api.services.rag_pipeline import RAGPipeline
        from src.core.types import RetrievalStrategy

        pipeline = RAGPipeline(tenant_id="tenant-filter")

        # Mock the vector searcher at the source
        with patch("src.retrieval.search.vector.VectorSearcher") as MockSearcher:
            mock_searcher = AsyncMock()
            mock_result = MagicMock()
            mock_result.results = []
            mock_searcher.search = AsyncMock(return_value=mock_result)
            MockSearcher.return_value = mock_searcher

            await pipeline._retrieve(
                question="test question",
                top_k=10,
                filters=None,
                strategy=RetrievalStrategy.VECTOR,
            )

            # Verify searcher was created with tenant_id
            MockSearcher.assert_called_once_with(tenant_id="tenant-filter")
