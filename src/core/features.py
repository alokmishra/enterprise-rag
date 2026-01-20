"""
Enterprise RAG System - Feature Flags

Manages feature availability based on:
- Deployment mode
- Tenant configuration
- System settings
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from src.core.deployment import DeploymentMode, get_deployment_config
from src.core.tenant import TenantConfig


class FeatureFlags:
    """
    Feature flag manager that considers deployment mode and tenant settings.

    Features can be disabled at:
    1. Deployment level (air-gapped disables cloud LLM)
    2. Tenant level (based on TenantConfig.features_enabled)
    3. System level (via Settings.FEATURE_* flags)
    """

    # Feature definitions with deployment constraints
    _DEPLOYMENT_CONSTRAINTS = {
        # Cloud LLM features - disabled in air-gapped mode
        "cloud_llm": {
            "disabled_modes": [DeploymentMode.AIR_GAPPED],
            "description": "Cloud-based LLM providers (Anthropic, OpenAI)",
        },
        "telemetry": {
            "disabled_modes": [DeploymentMode.AIR_GAPPED],
            "description": "Usage telemetry and analytics",
        },
        "auto_updates": {
            "disabled_modes": [DeploymentMode.AIR_GAPPED, DeploymentMode.ON_PREMISE],
            "description": "Automatic software updates",
        },
        # Features available in all modes
        "multi_agent": {
            "disabled_modes": [],
            "description": "Multi-agent RAG pipeline",
        },
        "knowledge_graph": {
            "disabled_modes": [],
            "description": "Knowledge graph integration",
        },
        "hybrid_search": {
            "disabled_modes": [],
            "description": "Hybrid vector + BM25 search",
        },
        "streaming": {
            "disabled_modes": [],
            "description": "Streaming responses",
        },
        "citations": {
            "disabled_modes": [],
            "description": "Source citations",
        },
        "multimodal": {
            "disabled_modes": [],
            "description": "Multi-modal document processing",
        },
        "local_llm": {
            "disabled_modes": [],
            "description": "Local LLM support (Llama, Mistral)",
        },
    }

    def __init__(self, tenant_config: Optional[TenantConfig] = None):
        """
        Initialize feature flags.

        Args:
            tenant_config: Optional tenant configuration for tenant-specific overrides
        """
        self._deployment = get_deployment_config()
        self._tenant_config = tenant_config

    def is_enabled(self, feature: str) -> bool:
        """
        Check if a feature is enabled.

        Checks in order:
        1. Deployment constraints (e.g., air-gapped disables cloud_llm)
        2. Tenant configuration (if provided)
        3. System settings

        Args:
            feature: Feature name to check

        Returns:
            True if feature is enabled
        """
        # Check deployment constraints
        constraint = self._DEPLOYMENT_CONSTRAINTS.get(feature)
        if constraint:
            if self._deployment.mode in constraint.get("disabled_modes", []):
                return False

        # Check tenant-specific settings
        if self._tenant_config is not None:
            if not self._tenant_config.has_feature(feature):
                return False

        # Default to enabled if no constraints block it
        return True

    def multi_agent_enabled(self) -> bool:
        """Check if multi-agent pipeline is enabled."""
        return self.is_enabled("multi_agent")

    def knowledge_graph_enabled(self) -> bool:
        """Check if knowledge graph is enabled."""
        return self.is_enabled("knowledge_graph")

    def hybrid_search_enabled(self) -> bool:
        """Check if hybrid search is enabled."""
        return self.is_enabled("hybrid_search")

    def streaming_enabled(self) -> bool:
        """Check if streaming responses are enabled."""
        return self.is_enabled("streaming")

    def cloud_llm_enabled(self) -> bool:
        """Check if cloud LLM providers are enabled."""
        return self.is_enabled("cloud_llm")

    def local_llm_enabled(self) -> bool:
        """Check if local LLM is enabled."""
        return self.is_enabled("local_llm")

    def multimodal_enabled(self) -> bool:
        """Check if multimodal processing is enabled."""
        return self.is_enabled("multimodal")

    def telemetry_enabled(self) -> bool:
        """Check if telemetry is enabled."""
        return self.is_enabled("telemetry")

    def get_all_features(self) -> dict[str, bool]:
        """Get status of all features."""
        return {
            feature: self.is_enabled(feature)
            for feature in self._DEPLOYMENT_CONSTRAINTS.keys()
        }

    def get_enabled_features(self) -> list[str]:
        """Get list of enabled features."""
        return [
            feature
            for feature in self._DEPLOYMENT_CONSTRAINTS.keys()
            if self.is_enabled(feature)
        ]

    def get_disabled_features(self) -> list[str]:
        """Get list of disabled features."""
        return [
            feature
            for feature in self._DEPLOYMENT_CONSTRAINTS.keys()
            if not self.is_enabled(feature)
        ]


# Default feature flags instance (no tenant context)
_default_feature_flags: Optional[FeatureFlags] = None


def get_feature_flags(tenant_config: Optional[TenantConfig] = None) -> FeatureFlags:
    """
    Get feature flags instance.

    Args:
        tenant_config: Optional tenant configuration for tenant-specific flags

    Returns:
        FeatureFlags instance
    """
    if tenant_config is not None:
        # Don't cache tenant-specific instances
        return FeatureFlags(tenant_config)

    global _default_feature_flags
    if _default_feature_flags is None:
        _default_feature_flags = FeatureFlags()
    return _default_feature_flags


@lru_cache
def _get_default_feature_flags() -> FeatureFlags:
    """Internal cached getter for default feature flags."""
    return FeatureFlags()


def reset_feature_flags() -> None:
    """Reset the feature flags singleton (for testing)."""
    global _default_feature_flags
    _default_feature_flags = None
    _get_default_feature_flags.cache_clear()
