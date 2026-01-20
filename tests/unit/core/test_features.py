"""
Tests for src/core/features.py
"""

import os
from unittest.mock import patch

import pytest

from src.core.deployment import (
    DeploymentMode,
    DeploymentConfig,
    reset_deployment_config,
)
from src.core.tenant import TenantConfig
from src.core.features import (
    FeatureFlags,
    get_feature_flags,
    reset_feature_flags,
)


class TestFeatureFlagsAirGapped:
    """Tests for feature flags in air-gapped mode."""

    def setup_method(self):
        """Reset singletons before each test."""
        reset_deployment_config()
        reset_feature_flags()

    def teardown_method(self):
        """Reset singletons after each test."""
        reset_deployment_config()
        reset_feature_flags()

    def test_air_gapped_no_cloud_llm(self):
        """Cloud LLM feature disabled in air-gapped mode."""
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "air_gapped"}):
            reset_deployment_config()
            reset_feature_flags()
            flags = FeatureFlags()
            assert flags.cloud_llm_enabled() is False

    def test_air_gapped_no_telemetry(self):
        """Telemetry feature disabled in air-gapped mode."""
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "air_gapped"}):
            reset_deployment_config()
            reset_feature_flags()
            flags = FeatureFlags()
            assert flags.telemetry_enabled() is False

    def test_air_gapped_no_auto_updates(self):
        """Auto updates disabled in air-gapped mode."""
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "air_gapped"}):
            reset_deployment_config()
            reset_feature_flags()
            flags = FeatureFlags()
            assert flags.is_enabled("auto_updates") is False

    def test_air_gapped_local_llm_enabled(self):
        """Local LLM should be enabled in air-gapped mode."""
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "air_gapped"}):
            reset_deployment_config()
            reset_feature_flags()
            flags = FeatureFlags()
            assert flags.local_llm_enabled() is True

    def test_air_gapped_multi_agent_enabled(self):
        """Multi-agent should still work in air-gapped mode."""
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "air_gapped"}):
            reset_deployment_config()
            reset_feature_flags()
            flags = FeatureFlags()
            assert flags.multi_agent_enabled() is True


class TestFeatureFlagsOnPremise:
    """Tests for feature flags in on-premise mode."""

    def setup_method(self):
        """Reset singletons before each test."""
        reset_deployment_config()
        reset_feature_flags()

    def teardown_method(self):
        """Reset singletons after each test."""
        reset_deployment_config()
        reset_feature_flags()

    def test_on_premise_cloud_llm_enabled(self):
        """Cloud LLM should be available in on-premise mode."""
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "on_premise"}):
            reset_deployment_config()
            reset_feature_flags()
            flags = FeatureFlags()
            assert flags.cloud_llm_enabled() is True

    def test_on_premise_no_auto_updates(self):
        """Auto updates disabled in on-premise mode."""
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "on_premise"}):
            reset_deployment_config()
            reset_feature_flags()
            flags = FeatureFlags()
            assert flags.is_enabled("auto_updates") is False


class TestFeatureFlagsSaaS:
    """Tests for feature flags in SaaS modes."""

    def setup_method(self):
        """Reset singletons before each test."""
        reset_deployment_config()
        reset_feature_flags()

    def teardown_method(self):
        """Reset singletons after each test."""
        reset_deployment_config()
        reset_feature_flags()

    def test_saas_all_features_enabled(self):
        """SaaS multi-tenant should have all features enabled."""
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "saas_multi_tenant"}):
            reset_deployment_config()
            reset_feature_flags()
            flags = FeatureFlags()
            assert flags.cloud_llm_enabled() is True
            assert flags.telemetry_enabled() is True
            assert flags.is_enabled("auto_updates") is True
            assert flags.multi_agent_enabled() is True
            assert flags.knowledge_graph_enabled() is True


class TestFeatureFlagsTenantOverride:
    """Tests for tenant-specific feature overrides."""

    def setup_method(self):
        """Reset singletons before each test."""
        reset_deployment_config()
        reset_feature_flags()

    def teardown_method(self):
        """Reset singletons after each test."""
        reset_deployment_config()
        reset_feature_flags()

    def test_tenant_override_disables_feature(self):
        """Tenant-specific feature flags override defaults."""
        # Tenant without multi_agent feature
        tenant_config = TenantConfig(
            tenant_id="test",
            tenant_name="Test",
            features_enabled=["streaming", "citations"],  # No multi_agent
        )
        flags = FeatureFlags(tenant_config=tenant_config)
        assert flags.multi_agent_enabled() is False
        assert flags.streaming_enabled() is True

    def test_tenant_override_with_all_features(self):
        """Tenant with all features enabled."""
        tenant_config = TenantConfig(
            tenant_id="test",
            tenant_name="Test",
            features_enabled=[
                "multi_agent",
                "knowledge_graph",
                "hybrid_search",
                "streaming",
                "citations",
                "multimodal",
            ],
        )
        flags = FeatureFlags(tenant_config=tenant_config)
        assert flags.multi_agent_enabled() is True
        assert flags.knowledge_graph_enabled() is True
        assert flags.hybrid_search_enabled() is True
        assert flags.streaming_enabled() is True
        assert flags.multimodal_enabled() is True

    def test_deployment_constraint_overrides_tenant(self):
        """Deployment constraints should override tenant settings."""
        # Air-gapped mode with tenant trying to enable cloud_llm
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "air_gapped"}):
            reset_deployment_config()
            reset_feature_flags()

            tenant_config = TenantConfig(
                tenant_id="test",
                tenant_name="Test",
                features_enabled=["cloud_llm", "telemetry"],  # Try to enable
            )
            flags = FeatureFlags(tenant_config=tenant_config)
            # Should still be disabled due to deployment constraint
            assert flags.cloud_llm_enabled() is False
            assert flags.telemetry_enabled() is False


class TestFeatureFlagsHelperMethods:
    """Tests for feature flag helper methods."""

    def setup_method(self):
        """Reset singletons before each test."""
        reset_deployment_config()
        reset_feature_flags()

    def teardown_method(self):
        """Reset singletons after each test."""
        reset_deployment_config()
        reset_feature_flags()

    def test_get_all_features(self):
        """Test get_all_features returns dict of all features."""
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "saas_multi_tenant"}):
            reset_deployment_config()
            reset_feature_flags()
            flags = FeatureFlags()
            all_features = flags.get_all_features()

            assert isinstance(all_features, dict)
            assert "cloud_llm" in all_features
            assert "multi_agent" in all_features
            assert "telemetry" in all_features

    def test_get_enabled_features(self):
        """Test get_enabled_features returns list of enabled features."""
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "saas_multi_tenant"}):
            reset_deployment_config()
            reset_feature_flags()
            flags = FeatureFlags()
            enabled = flags.get_enabled_features()

            assert isinstance(enabled, list)
            assert "cloud_llm" in enabled
            assert "multi_agent" in enabled

    def test_get_disabled_features_air_gapped(self):
        """Test get_disabled_features in air-gapped mode."""
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "air_gapped"}):
            reset_deployment_config()
            reset_feature_flags()
            flags = FeatureFlags()
            disabled = flags.get_disabled_features()

            assert isinstance(disabled, list)
            assert "cloud_llm" in disabled
            assert "telemetry" in disabled
            assert "auto_updates" in disabled


class TestGetFeatureFlags:
    """Tests for get_feature_flags singleton."""

    def setup_method(self):
        """Reset singletons before each test."""
        reset_deployment_config()
        reset_feature_flags()

    def teardown_method(self):
        """Reset singletons after each test."""
        reset_deployment_config()
        reset_feature_flags()

    def test_get_feature_flags_singleton(self):
        """Verify singleton returns same instance without tenant."""
        flags1 = get_feature_flags()
        flags2 = get_feature_flags()
        assert flags1 is flags2

    def test_get_feature_flags_with_tenant_not_cached(self):
        """Verify tenant-specific flags are not cached."""
        tenant_config = TenantConfig(
            tenant_id="test",
            tenant_name="Test",
        )
        flags1 = get_feature_flags(tenant_config)
        flags2 = get_feature_flags(tenant_config)
        # Should be different instances since we don't cache tenant-specific
        assert flags1 is not flags2

    def test_reset_feature_flags(self):
        """Verify reset clears singleton."""
        flags1 = get_feature_flags()
        reset_feature_flags()
        flags2 = get_feature_flags()
        assert flags1 is not flags2


class TestFeatureFlagsIsEnabled:
    """Tests for is_enabled method."""

    def setup_method(self):
        """Reset singletons before each test."""
        reset_deployment_config()
        reset_feature_flags()

    def teardown_method(self):
        """Reset singletons after each test."""
        reset_deployment_config()
        reset_feature_flags()

    def test_is_enabled_unknown_feature(self):
        """Unknown features should default to enabled."""
        flags = FeatureFlags()
        assert flags.is_enabled("unknown_feature") is True

    def test_is_enabled_known_feature(self):
        """Known features should check constraints."""
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "saas_multi_tenant"}):
            reset_deployment_config()
            reset_feature_flags()
            flags = FeatureFlags()
            assert flags.is_enabled("multi_agent") is True
            assert flags.is_enabled("cloud_llm") is True
