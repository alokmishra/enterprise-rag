"""
Tests for tenant isolation strategies.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.deployment import (
    DeploymentConfig,
    DeploymentMode,
    TenantIsolation,
    reset_deployment_config,
)
from src.storage.isolation import (
    SharedIsolation,
    SchemaIsolation,
    DatabaseIsolation,
    InstanceIsolation,
    get_isolation_strategy,
    reset_isolation_strategy,
)


class TestGetIsolationStrategy:
    """Tests for get_isolation_strategy factory."""

    def setup_method(self):
        """Reset state before each test."""
        reset_deployment_config()
        reset_isolation_strategy()

    def teardown_method(self):
        """Reset state after each test."""
        reset_deployment_config()
        reset_isolation_strategy()

    def test_get_isolation_strategy_shared(self):
        """SHARED isolation returns SharedIsolation."""
        with patch.dict(
            os.environ,
            {"DEPLOYMENT_MODE": "saas_multi_tenant", "TENANT_ISOLATION": "shared"},
        ):
            reset_deployment_config()
            reset_isolation_strategy()

            strategy = get_isolation_strategy()
            assert isinstance(strategy, SharedIsolation)

    def test_get_isolation_strategy_schema(self):
        """SCHEMA isolation returns SchemaIsolation."""
        with patch.dict(
            os.environ,
            {"DEPLOYMENT_MODE": "saas_dedicated", "TENANT_ISOLATION": "schema"},
        ):
            reset_deployment_config()
            reset_isolation_strategy()

            strategy = get_isolation_strategy()
            assert isinstance(strategy, SchemaIsolation)

    def test_get_isolation_strategy_database(self):
        """DATABASE isolation returns DatabaseIsolation."""
        with patch.dict(
            os.environ,
            {"DEPLOYMENT_MODE": "saas_dedicated", "TENANT_ISOLATION": "database"},
        ):
            reset_deployment_config()
            reset_isolation_strategy()

            strategy = get_isolation_strategy()
            assert isinstance(strategy, DatabaseIsolation)

    def test_get_isolation_strategy_instance(self):
        """INSTANCE isolation returns InstanceIsolation."""
        with patch.dict(
            os.environ,
            {"DEPLOYMENT_MODE": "on_premise", "TENANT_ISOLATION": "instance"},
        ):
            reset_deployment_config()
            reset_isolation_strategy()

            strategy = get_isolation_strategy()
            assert isinstance(strategy, InstanceIsolation)

    def test_get_isolation_strategy_from_deployment_config(self):
        """Strategy can be created from explicit DeploymentConfig."""
        config = DeploymentConfig(
            mode=DeploymentMode.SAAS_DEDICATED,
            tenant_isolation=TenantIsolation.SCHEMA,
        )

        reset_isolation_strategy()
        strategy = get_isolation_strategy(deployment=config)
        assert isinstance(strategy, SchemaIsolation)

    def test_get_isolation_strategy_cached(self):
        """Strategy is cached after first call."""
        with patch.dict(
            os.environ,
            {"DEPLOYMENT_MODE": "saas_multi_tenant", "TENANT_ISOLATION": "shared"},
        ):
            reset_deployment_config()
            reset_isolation_strategy()

            strategy1 = get_isolation_strategy()
            strategy2 = get_isolation_strategy()

            assert strategy1 is strategy2


class TestSharedIsolation:
    """Tests for SharedIsolation strategy."""

    @pytest.mark.asyncio
    async def test_shared_isolation_returns_filtered_repository(self):
        """SharedIsolation returns repository with tenant filter."""
        strategy = SharedIsolation()
        mock_session = AsyncMock()

        with patch("src.storage.document.repository.DocumentRepository") as MockRepo:
            repo = await strategy.get_document_repository(
                tenant_id="tenant-shared",
                session=mock_session,
            )

            MockRepo.assert_called_once_with(
                session=mock_session,
                tenant_id="tenant-shared",
            )

    @pytest.mark.asyncio
    async def test_shared_isolation_returns_filtered_chunk_repository(self):
        """SharedIsolation returns chunk repository with tenant filter."""
        strategy = SharedIsolation()
        mock_session = AsyncMock()

        with patch("src.storage.document.repository.ChunkRepository") as MockRepo:
            repo = await strategy.get_chunk_repository(
                tenant_id="tenant-shared",
                session=mock_session,
            )

            MockRepo.assert_called_once_with(
                session=mock_session,
                tenant_id="tenant-shared",
            )

    @pytest.mark.asyncio
    async def test_shared_isolation_returns_shared_vector_store(self):
        """SharedIsolation returns shared vector store."""
        strategy = SharedIsolation()

        with patch("src.storage.vector.get_vector_store") as mock_get_store:
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            store = await strategy.get_vector_store(tenant_id="tenant-shared")

            mock_get_store.assert_called_once()
            assert store == mock_store

    def test_shared_isolation_collection_name(self):
        """SharedIsolation uses default collection name."""
        strategy = SharedIsolation()

        with patch("src.storage.isolation.settings") as mock_settings:
            mock_settings.QDRANT_COLLECTION_NAME = "rag_documents"

            name = strategy.get_collection_name("tenant-shared")
            assert name == "rag_documents"


class TestSchemaIsolation:
    """Tests for SchemaIsolation strategy."""

    def test_schema_isolation_schema_name(self):
        """SchemaIsolation generates correct schema name."""
        strategy = SchemaIsolation()

        schema = strategy._get_schema_name("tenant-123")
        assert schema == "tenant_tenant_123"

    def test_schema_isolation_collection_name(self):
        """SchemaIsolation uses tenant-specific collection name."""
        strategy = SchemaIsolation()

        with patch("src.storage.isolation.settings") as mock_settings:
            mock_settings.QDRANT_COLLECTION_NAME = "rag_documents"

            name = strategy.get_collection_name("tenant-123")
            assert name == "rag_documents_tenant_123"

    @pytest.mark.asyncio
    async def test_schema_isolation_returns_repository(self):
        """SchemaIsolation returns repository (with row-level fallback)."""
        strategy = SchemaIsolation()
        mock_session = AsyncMock()

        with patch("src.storage.document.repository.DocumentRepository") as MockRepo:
            repo = await strategy.get_document_repository(
                tenant_id="tenant-schema",
                session=mock_session,
            )

            MockRepo.assert_called_once_with(
                session=mock_session,
                tenant_id="tenant-schema",
            )


class TestDatabaseIsolation:
    """Tests for DatabaseIsolation strategy."""

    def test_database_isolation_database_url(self):
        """DatabaseIsolation generates correct database URL."""
        strategy = DatabaseIsolation()

        with patch("src.storage.isolation.settings") as mock_settings:
            mock_settings.DATABASE_URL = "postgresql+asyncpg://user:pass@localhost:5432/rag"

            url = strategy._get_database_url("tenant-123")
            assert url == "postgresql+asyncpg://user:pass@localhost:5432/rag_tenant_123"

    def test_database_isolation_collection_name(self):
        """DatabaseIsolation uses tenant-specific collection name."""
        strategy = DatabaseIsolation()

        with patch("src.storage.isolation.settings") as mock_settings:
            mock_settings.QDRANT_COLLECTION_NAME = "rag_documents"

            name = strategy.get_collection_name("tenant-db")
            assert name == "rag_documents_tenant_db"

    @pytest.mark.asyncio
    async def test_database_isolation_returns_repository(self):
        """DatabaseIsolation returns repository."""
        strategy = DatabaseIsolation()
        mock_session = AsyncMock()

        with patch("src.storage.document.repository.DocumentRepository") as MockRepo:
            repo = await strategy.get_document_repository(
                tenant_id="tenant-db",
                session=mock_session,
            )

            MockRepo.assert_called_once_with(
                session=mock_session,
                tenant_id="tenant-db",
            )


class TestInstanceIsolation:
    """Tests for InstanceIsolation strategy."""

    def test_instance_isolation_collection_name(self):
        """InstanceIsolation uses default collection name."""
        strategy = InstanceIsolation()

        with patch("src.storage.isolation.settings") as mock_settings:
            mock_settings.QDRANT_COLLECTION_NAME = "rag_documents"

            name = strategy.get_collection_name("tenant-instance")
            # Instance isolation uses default collection since instance is single-tenant
            assert name == "rag_documents"

    @pytest.mark.asyncio
    async def test_instance_isolation_returns_repository(self):
        """InstanceIsolation returns repository."""
        strategy = InstanceIsolation()
        mock_session = AsyncMock()

        with patch("src.storage.document.repository.DocumentRepository") as MockRepo:
            repo = await strategy.get_document_repository(
                tenant_id="tenant-instance",
                session=mock_session,
            )

            MockRepo.assert_called_once_with(
                session=mock_session,
                tenant_id="tenant-instance",
            )

    @pytest.mark.asyncio
    async def test_instance_isolation_returns_default_vector_store(self):
        """InstanceIsolation returns default vector store."""
        strategy = InstanceIsolation()

        with patch("src.storage.vector.get_vector_store") as mock_get_store:
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            store = await strategy.get_vector_store(tenant_id="tenant-instance")

            mock_get_store.assert_called_once()
            assert store == mock_store


class TestIsolationStrategyIntegration:
    """Integration tests for isolation strategy selection."""

    def setup_method(self):
        """Reset state before each test."""
        reset_deployment_config()
        reset_isolation_strategy()

    def teardown_method(self):
        """Reset state after each test."""
        reset_deployment_config()
        reset_isolation_strategy()

    def test_saas_multi_tenant_uses_shared_isolation(self):
        """saas_multi_tenant mode defaults to shared isolation."""
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "saas_multi_tenant"}):
            reset_deployment_config()
            reset_isolation_strategy()

            strategy = get_isolation_strategy()
            assert isinstance(strategy, SharedIsolation)

    def test_on_premise_can_use_instance_isolation(self):
        """on_premise mode can use instance isolation."""
        with patch.dict(
            os.environ,
            {"DEPLOYMENT_MODE": "on_premise", "TENANT_ISOLATION": "instance"},
        ):
            reset_deployment_config()
            reset_isolation_strategy()

            strategy = get_isolation_strategy()
            assert isinstance(strategy, InstanceIsolation)

    def test_air_gapped_can_use_instance_isolation(self):
        """air_gapped mode can use instance isolation."""
        with patch.dict(
            os.environ,
            {"DEPLOYMENT_MODE": "air_gapped", "TENANT_ISOLATION": "instance"},
        ):
            reset_deployment_config()
            reset_isolation_strategy()

            strategy = get_isolation_strategy()
            assert isinstance(strategy, InstanceIsolation)
