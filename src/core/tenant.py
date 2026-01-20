"""
Enterprise RAG System - Tenant Configuration

Manages tenant-specific configuration including:
- Resource limits (documents, queries)
- Feature enablement
- LLM provider settings
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import BaseModel, Field

from src.core.deployment import TenantIsolation


class TenantConfig(BaseModel):
    """Configuration for a specific tenant."""

    tenant_id: str = Field(
        description="Unique tenant identifier"
    )
    tenant_name: str = Field(
        description="Human-readable tenant name"
    )
    isolation_level: TenantIsolation = Field(
        default=TenantIsolation.SHARED,
        description="Isolation level for this tenant"
    )

    # LLM Configuration
    llm_provider: str = Field(
        default="anthropic",
        description="Default LLM provider for this tenant"
    )
    llm_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Default LLM model for this tenant"
    )
    allowed_providers: list[str] = Field(
        default_factory=lambda: ["anthropic", "openai"],
        description="LLM providers this tenant can use"
    )

    # Resource Limits
    max_documents: int = Field(
        default=10000,
        description="Maximum documents allowed"
    )
    max_queries_per_day: int = Field(
        default=1000,
        description="Maximum queries per day"
    )
    max_tokens_per_query: int = Field(
        default=8000,
        description="Maximum tokens per query"
    )
    max_file_size_mb: int = Field(
        default=50,
        description="Maximum file size in MB"
    )
    max_concurrent_ingestions: int = Field(
        default=5,
        description="Maximum concurrent document ingestions"
    )

    # Feature Flags
    features_enabled: list[str] = Field(
        default_factory=lambda: [
            "multi_agent",
            "knowledge_graph",
            "hybrid_search",
            "streaming",
            "citations",
        ],
        description="Features enabled for this tenant"
    )

    # Storage Configuration
    vector_collection_prefix: Optional[str] = Field(
        default=None,
        description="Prefix for vector collections (defaults to tenant_id)"
    )
    database_schema: Optional[str] = Field(
        default=None,
        description="Database schema for schema isolation"
    )
    database_url_override: Optional[str] = Field(
        default=None,
        description="Custom database URL for database isolation"
    )

    # Metadata
    metadata: dict = Field(
        default_factory=dict,
        description="Additional tenant metadata"
    )

    @property
    def effective_collection_prefix(self) -> str:
        """Get the effective vector collection prefix."""
        return self.vector_collection_prefix or self.tenant_id

    def has_feature(self, feature: str) -> bool:
        """Check if a feature is enabled for this tenant."""
        return feature in self.features_enabled

    def can_use_provider(self, provider: str) -> bool:
        """Check if tenant can use a specific LLM provider."""
        return provider in self.allowed_providers

    def get_resource_limit(self, resource: str) -> int:
        """Get resource limit by name."""
        limits = {
            "documents": self.max_documents,
            "queries_per_day": self.max_queries_per_day,
            "tokens_per_query": self.max_tokens_per_query,
            "file_size_mb": self.max_file_size_mb,
            "concurrent_ingestions": self.max_concurrent_ingestions,
        }
        return limits.get(resource, 0)

    @classmethod
    def default_single_tenant(cls) -> "TenantConfig":
        """
        Create default configuration for single-tenant deployments.

        Returns unlimited configuration for on-premise/air-gapped modes.
        """
        return cls(
            tenant_id="default",
            tenant_name="Default Tenant",
            isolation_level=TenantIsolation.INSTANCE,
            max_documents=-1,  # Unlimited
            max_queries_per_day=-1,  # Unlimited
            max_tokens_per_query=-1,  # Unlimited
            features_enabled=[
                "multi_agent",
                "knowledge_graph",
                "hybrid_search",
                "streaming",
                "citations",
                "multimodal",
                "admin_dashboard",
            ],
        )

    @classmethod
    def for_trial(cls, tenant_id: str, tenant_name: str) -> "TenantConfig":
        """Create configuration for trial tenants with reduced limits."""
        return cls(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            max_documents=100,
            max_queries_per_day=50,
            max_tokens_per_query=4000,
            max_file_size_mb=10,
            features_enabled=[
                "hybrid_search",
                "streaming",
                "citations",
            ],
        )

    @classmethod
    def for_enterprise(cls, tenant_id: str, tenant_name: str) -> "TenantConfig":
        """Create configuration for enterprise tenants with high limits."""
        return cls(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            isolation_level=TenantIsolation.DATABASE,
            max_documents=1000000,
            max_queries_per_day=100000,
            max_tokens_per_query=16000,
            max_file_size_mb=200,
            max_concurrent_ingestions=20,
            features_enabled=[
                "multi_agent",
                "knowledge_graph",
                "hybrid_search",
                "streaming",
                "citations",
                "multimodal",
                "admin_dashboard",
                "custom_models",
                "priority_support",
            ],
        )


class TenantContext:
    """Context holder for current tenant in request scope."""

    def __init__(
        self,
        tenant_id: str,
        tenant_config: Optional[TenantConfig] = None,
    ):
        self._tenant_id = tenant_id
        self._config = tenant_config

    @property
    def tenant_id(self) -> str:
        """Get the current tenant ID."""
        return self._tenant_id

    @property
    def config(self) -> Optional[TenantConfig]:
        """Get the tenant configuration."""
        return self._config

    def has_feature(self, feature: str) -> bool:
        """Check if current tenant has a feature enabled."""
        if self._config is None:
            return True  # Default to enabled if no config
        return self._config.has_feature(feature)


# Default tenant for single-tenant deployments
DEFAULT_TENANT_ID = "default"
_default_tenant_config: Optional[TenantConfig] = None


@lru_cache
def get_default_tenant_config() -> TenantConfig:
    """Get the default tenant configuration for single-tenant mode."""
    global _default_tenant_config
    if _default_tenant_config is None:
        _default_tenant_config = TenantConfig.default_single_tenant()
    return _default_tenant_config


def reset_default_tenant_config() -> None:
    """Reset the default tenant configuration (for testing)."""
    global _default_tenant_config
    _default_tenant_config = None
    get_default_tenant_config.cache_clear()
