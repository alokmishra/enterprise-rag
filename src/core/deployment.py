"""
Enterprise RAG System - Deployment Configuration

Supports 4 deployment modes:
- saas_multi_tenant: Shared infrastructure, row-level isolation
- saas_dedicated: Separate infrastructure per tenant
- on_premise: Customer data center deployment
- air_gapped: Zero internet, local LLMs only
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Optional
import os

from pydantic import BaseModel, Field, model_validator


class DeploymentMode(str, Enum):
    """Available deployment modes."""
    SAAS_MULTI_TENANT = "saas_multi_tenant"
    SAAS_DEDICATED = "saas_dedicated"
    ON_PREMISE = "on_premise"
    AIR_GAPPED = "air_gapped"


class TenantIsolation(str, Enum):
    """Tenant isolation levels."""
    SHARED = "shared"       # Row-level isolation in shared resources
    SCHEMA = "schema"       # Schema-per-tenant in same database
    DATABASE = "database"   # Separate database per tenant
    INSTANCE = "instance"   # Completely separate deployment


class DeploymentConfig(BaseModel):
    """Configuration for deployment mode and tenant isolation."""

    mode: DeploymentMode = Field(
        default=DeploymentMode.SAAS_MULTI_TENANT,
        description="Deployment mode"
    )
    tenant_isolation: TenantIsolation = Field(
        default=TenantIsolation.SHARED,
        description="Tenant isolation level"
    )
    enable_cloud_llm: bool = Field(
        default=True,
        description="Enable cloud-based LLM providers (Anthropic, OpenAI)"
    )
    enable_telemetry: bool = Field(
        default=True,
        description="Enable telemetry and usage tracking"
    )
    enable_auto_updates: bool = Field(
        default=True,
        description="Enable automatic updates"
    )
    license_key: Optional[str] = Field(
        default=None,
        description="License key for on-premise/air-gapped deployments"
    )

    @model_validator(mode="after")
    def validate_deployment_constraints(self) -> "DeploymentConfig":
        """Validate deployment configuration constraints."""
        # Air-gapped mode must disable cloud LLM and telemetry
        if self.mode == DeploymentMode.AIR_GAPPED:
            self.enable_cloud_llm = False
            self.enable_telemetry = False
            self.enable_auto_updates = False

        # On-premise defaults to disabled telemetry
        if self.mode == DeploymentMode.ON_PREMISE:
            # Telemetry can be explicitly enabled but defaults to False
            if os.getenv("TELEMETRY_ENABLED", "").lower() != "true":
                self.enable_telemetry = False

        # SaaS dedicated typically uses instance isolation
        if self.mode == DeploymentMode.SAAS_DEDICATED:
            if self.tenant_isolation == TenantIsolation.SHARED:
                self.tenant_isolation = TenantIsolation.DATABASE

        return self

    @classmethod
    def from_env(cls) -> "DeploymentConfig":
        """Create configuration from environment variables."""
        mode_str = os.getenv("DEPLOYMENT_MODE", "saas_multi_tenant")
        isolation_str = os.getenv("TENANT_ISOLATION", "shared")

        try:
            mode = DeploymentMode(mode_str)
        except ValueError:
            mode = DeploymentMode.SAAS_MULTI_TENANT

        try:
            isolation = TenantIsolation(isolation_str)
        except ValueError:
            isolation = TenantIsolation.SHARED

        return cls(
            mode=mode,
            tenant_isolation=isolation,
            enable_cloud_llm=os.getenv("ENABLE_CLOUD_LLM", "true").lower() == "true",
            enable_telemetry=os.getenv("TELEMETRY_ENABLED", "true").lower() == "true",
            enable_auto_updates=os.getenv("ENABLE_AUTO_UPDATES", "true").lower() == "true",
            license_key=os.getenv("LICENSE_KEY"),
        )

    @classmethod
    def for_saas_multi_tenant(cls) -> "DeploymentConfig":
        """Create default config for SaaS multi-tenant deployment."""
        return cls(
            mode=DeploymentMode.SAAS_MULTI_TENANT,
            tenant_isolation=TenantIsolation.SHARED,
            enable_cloud_llm=True,
            enable_telemetry=True,
            enable_auto_updates=True,
        )

    @classmethod
    def for_saas_dedicated(cls) -> "DeploymentConfig":
        """Create default config for SaaS dedicated deployment."""
        return cls(
            mode=DeploymentMode.SAAS_DEDICATED,
            tenant_isolation=TenantIsolation.DATABASE,
            enable_cloud_llm=True,
            enable_telemetry=True,
            enable_auto_updates=True,
        )

    @classmethod
    def for_on_premise(cls) -> "DeploymentConfig":
        """Create default config for on-premise deployment."""
        return cls(
            mode=DeploymentMode.ON_PREMISE,
            tenant_isolation=TenantIsolation.INSTANCE,
            enable_cloud_llm=True,
            enable_telemetry=False,
            enable_auto_updates=False,
        )

    @classmethod
    def for_air_gapped(cls) -> "DeploymentConfig":
        """Create default config for air-gapped deployment."""
        return cls(
            mode=DeploymentMode.AIR_GAPPED,
            tenant_isolation=TenantIsolation.INSTANCE,
            enable_cloud_llm=False,
            enable_telemetry=False,
            enable_auto_updates=False,
        )

    def is_multi_tenant(self) -> bool:
        """Check if deployment supports multiple tenants."""
        return self.mode in (
            DeploymentMode.SAAS_MULTI_TENANT,
            DeploymentMode.SAAS_DEDICATED,
        )

    def requires_tenant_context(self) -> bool:
        """Check if requests require tenant context."""
        return self.mode == DeploymentMode.SAAS_MULTI_TENANT

    def allows_cloud_providers(self) -> bool:
        """Check if cloud LLM providers are allowed."""
        return self.enable_cloud_llm


# Singleton deployment configuration
_deployment_config: Optional[DeploymentConfig] = None


@lru_cache
def get_deployment_config() -> DeploymentConfig:
    """Get the deployment configuration singleton."""
    global _deployment_config
    if _deployment_config is None:
        _deployment_config = DeploymentConfig.from_env()
    return _deployment_config


def reset_deployment_config() -> None:
    """Reset the deployment configuration (for testing)."""
    global _deployment_config
    _deployment_config = None
    get_deployment_config.cache_clear()
