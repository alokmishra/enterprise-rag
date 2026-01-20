"""
Enterprise RAG System - Tenant Isolation Strategies

Provides different isolation strategies based on deployment configuration:
- SharedIsolation: Row-level isolation in shared resources
- SchemaIsolation: Schema-per-tenant in same database
- DatabaseIsolation: Separate database per tenant
- InstanceIsolation: Completely separate deployment
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.deployment import (
    DeploymentConfig,
    TenantIsolation,
    get_deployment_config,
)
from src.core.logging import LoggerMixin

if TYPE_CHECKING:
    from src.storage.document.repository import DocumentRepository, ChunkRepository
    from src.storage.vector.qdrant import QdrantVectorStore


class IsolationStrategy(ABC, LoggerMixin):
    """
    Abstract base class for tenant isolation strategies.

    Each strategy provides methods to get tenant-scoped resources
    (repositories, vector stores, etc.) based on the isolation level.
    """

    @abstractmethod
    async def get_document_repository(
        self,
        tenant_id: str,
        session: AsyncSession,
    ) -> "DocumentRepository":
        """
        Get a document repository scoped to the tenant.

        Args:
            tenant_id: The tenant identifier
            session: Database session

        Returns:
            DocumentRepository configured for the tenant
        """
        pass

    @abstractmethod
    async def get_chunk_repository(
        self,
        tenant_id: str,
        session: AsyncSession,
    ) -> "ChunkRepository":
        """
        Get a chunk repository scoped to the tenant.

        Args:
            tenant_id: The tenant identifier
            session: Database session

        Returns:
            ChunkRepository configured for the tenant
        """
        pass

    @abstractmethod
    async def get_vector_store(
        self,
        tenant_id: str,
    ) -> "QdrantVectorStore":
        """
        Get a vector store scoped to the tenant.

        Args:
            tenant_id: The tenant identifier

        Returns:
            VectorStore configured for the tenant
        """
        pass

    @abstractmethod
    def get_collection_name(self, tenant_id: str) -> str:
        """
        Get the vector collection name for the tenant.

        Args:
            tenant_id: The tenant identifier

        Returns:
            Collection name string
        """
        pass


class SharedIsolation(IsolationStrategy):
    """
    Row-level isolation in shared resources.

    All tenants share the same database tables and vector collections.
    Isolation is enforced by filtering on tenant_id in all queries.

    This is the default isolation strategy for saas_multi_tenant mode.
    """

    async def get_document_repository(
        self,
        tenant_id: str,
        session: AsyncSession,
    ) -> "DocumentRepository":
        """Get repository with tenant filter."""
        from src.storage.document.repository import DocumentRepository

        return DocumentRepository(session=session, tenant_id=tenant_id)

    async def get_chunk_repository(
        self,
        tenant_id: str,
        session: AsyncSession,
    ) -> "ChunkRepository":
        """Get chunk repository with tenant filter."""
        from src.storage.document.repository import ChunkRepository

        return ChunkRepository(session=session, tenant_id=tenant_id)

    async def get_vector_store(
        self,
        tenant_id: str,
    ) -> "QdrantVectorStore":
        """
        Get shared vector store.

        The tenant_id is used as a filter in search/insert operations,
        not as a separate collection.
        """
        from src.storage.vector import get_vector_store

        return get_vector_store()

    def get_collection_name(self, tenant_id: str) -> str:
        """
        Get shared collection name.

        All tenants use the same collection in shared isolation.
        """
        return settings.QDRANT_COLLECTION_NAME


class SchemaIsolation(IsolationStrategy):
    """
    Schema-per-tenant isolation in same database.

    Each tenant has their own database schema but shares the same
    database server. Provides stronger isolation than row-level
    while still sharing infrastructure.

    Note: Full implementation requires schema management and
    dynamic connection handling. This is a simplified version
    that falls back to row-level isolation with schema awareness.
    """

    def __init__(self):
        self._schema_prefix = "tenant_"

    def _get_schema_name(self, tenant_id: str) -> str:
        """Get the schema name for a tenant."""
        # Sanitize tenant_id for use in schema name
        safe_id = tenant_id.replace("-", "_").lower()
        return f"{self._schema_prefix}{safe_id}"

    async def get_document_repository(
        self,
        tenant_id: str,
        session: AsyncSession,
    ) -> "DocumentRepository":
        """
        Get repository with schema isolation.

        Note: Full schema isolation requires schema-aware sessions.
        This implementation uses row-level filtering as a fallback.
        """
        from src.storage.document.repository import DocumentRepository

        # TODO: Implement full schema switching
        # For now, use row-level isolation
        self.logger.debug(
            "Schema isolation using row-level fallback",
            tenant_id=tenant_id,
            schema=self._get_schema_name(tenant_id),
        )
        return DocumentRepository(session=session, tenant_id=tenant_id)

    async def get_chunk_repository(
        self,
        tenant_id: str,
        session: AsyncSession,
    ) -> "ChunkRepository":
        """Get chunk repository with schema isolation."""
        from src.storage.document.repository import ChunkRepository

        return ChunkRepository(session=session, tenant_id=tenant_id)

    async def get_vector_store(
        self,
        tenant_id: str,
    ) -> "QdrantVectorStore":
        """
        Get vector store with tenant-specific collection.

        In schema isolation, each tenant gets their own collection.
        """
        from src.storage.vector import get_vector_store

        return get_vector_store()

    def get_collection_name(self, tenant_id: str) -> str:
        """
        Get tenant-specific collection name.

        Each tenant has their own vector collection.
        """
        safe_id = tenant_id.replace("-", "_").lower()
        return f"{settings.QDRANT_COLLECTION_NAME}_{safe_id}"


class DatabaseIsolation(IsolationStrategy):
    """
    Separate database per tenant.

    Each tenant has their own database. Provides strong isolation
    but requires more infrastructure management.

    Note: Full implementation requires dynamic database connections.
    This is a simplified version showing the pattern.
    """

    def __init__(self):
        self._connection_cache: dict[str, Any] = {}

    def _get_database_url(self, tenant_id: str) -> str:
        """
        Get database URL for a tenant.

        In production, this would look up the tenant's database
        configuration from a management database or config service.
        """
        # Pattern: base_url with tenant-specific database name
        base_url = settings.DATABASE_URL
        safe_id = tenant_id.replace("-", "_").lower()

        # Replace database name in URL
        if "/rag" in base_url:
            return base_url.replace("/rag", f"/rag_{safe_id}")
        return f"{base_url}_{safe_id}"

    async def get_document_repository(
        self,
        tenant_id: str,
        session: AsyncSession,
    ) -> "DocumentRepository":
        """
        Get repository connected to tenant's database.

        Note: Full implementation requires tenant-specific sessions.
        This uses the provided session with row-level filtering.
        """
        from src.storage.document.repository import DocumentRepository

        # TODO: Implement tenant-specific database connections
        self.logger.debug(
            "Database isolation using shared session",
            tenant_id=tenant_id,
            target_db=self._get_database_url(tenant_id),
        )
        return DocumentRepository(session=session, tenant_id=tenant_id)

    async def get_chunk_repository(
        self,
        tenant_id: str,
        session: AsyncSession,
    ) -> "ChunkRepository":
        """Get chunk repository from tenant's database."""
        from src.storage.document.repository import ChunkRepository

        return ChunkRepository(session=session, tenant_id=tenant_id)

    async def get_vector_store(
        self,
        tenant_id: str,
    ) -> "QdrantVectorStore":
        """
        Get vector store for tenant.

        Could be a separate Qdrant instance per tenant in full implementation.
        """
        from src.storage.vector import get_vector_store

        return get_vector_store()

    def get_collection_name(self, tenant_id: str) -> str:
        """Get tenant-specific collection name."""
        safe_id = tenant_id.replace("-", "_").lower()
        return f"{settings.QDRANT_COLLECTION_NAME}_{safe_id}"


class InstanceIsolation(IsolationStrategy):
    """
    Completely separate deployment per tenant.

    Each tenant has their own infrastructure (database, vector store,
    application instances). This is typically managed at the
    infrastructure/deployment level rather than in application code.

    This strategy assumes the application is deployed in a
    single-tenant context and simply returns the default resources.
    """

    async def get_document_repository(
        self,
        tenant_id: str,
        session: AsyncSession,
    ) -> "DocumentRepository":
        """
        Get repository for isolated instance.

        In instance isolation, each deployment is single-tenant,
        so we use the default tenant_id.
        """
        from src.storage.document.repository import DocumentRepository

        # In instance isolation, tenant_id is still used for consistency
        # but the entire instance is dedicated to one tenant
        return DocumentRepository(session=session, tenant_id=tenant_id)

    async def get_chunk_repository(
        self,
        tenant_id: str,
        session: AsyncSession,
    ) -> "ChunkRepository":
        """Get chunk repository for isolated instance."""
        from src.storage.document.repository import ChunkRepository

        return ChunkRepository(session=session, tenant_id=tenant_id)

    async def get_vector_store(
        self,
        tenant_id: str,
    ) -> "QdrantVectorStore":
        """
        Get vector store for isolated instance.

        Returns the default vector store since each instance
        is dedicated to one tenant.
        """
        from src.storage.vector import get_vector_store

        return get_vector_store()

    def get_collection_name(self, tenant_id: str) -> str:
        """
        Get collection name for isolated instance.

        Uses default collection since instance is single-tenant.
        """
        return settings.QDRANT_COLLECTION_NAME


# Strategy cache
_isolation_strategy: Optional[IsolationStrategy] = None


def get_isolation_strategy(
    deployment: Optional[DeploymentConfig] = None,
) -> IsolationStrategy:
    """
    Get the appropriate isolation strategy based on deployment config.

    Args:
        deployment: Optional deployment config (uses global if not provided)

    Returns:
        IsolationStrategy instance for the configured isolation level
    """
    global _isolation_strategy

    if _isolation_strategy is not None:
        return _isolation_strategy

    if deployment is None:
        deployment = get_deployment_config()

    if deployment.tenant_isolation == TenantIsolation.SHARED:
        _isolation_strategy = SharedIsolation()
    elif deployment.tenant_isolation == TenantIsolation.SCHEMA:
        _isolation_strategy = SchemaIsolation()
    elif deployment.tenant_isolation == TenantIsolation.DATABASE:
        _isolation_strategy = DatabaseIsolation()
    elif deployment.tenant_isolation == TenantIsolation.INSTANCE:
        _isolation_strategy = InstanceIsolation()
    else:
        # Default to shared isolation
        _isolation_strategy = SharedIsolation()

    return _isolation_strategy


def reset_isolation_strategy() -> None:
    """Reset the cached isolation strategy. Used in testing."""
    global _isolation_strategy
    _isolation_strategy = None
