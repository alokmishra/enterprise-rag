"""
Tests for src/core/deployment.py
"""

import os
from unittest.mock import patch

import pytest

from src.core.deployment import (
    DeploymentMode,
    TenantIsolation,
    DeploymentConfig,
    get_deployment_config,
    reset_deployment_config,
)


class TestDeploymentModeEnum:
    """Tests for DeploymentMode enum."""

    def test_deployment_mode_enum_values(self):
        """Verify all 4 deployment modes exist."""
        assert DeploymentMode.SAAS_MULTI_TENANT == "saas_multi_tenant"
        assert DeploymentMode.SAAS_DEDICATED == "saas_dedicated"
        assert DeploymentMode.ON_PREMISE == "on_premise"
        assert DeploymentMode.AIR_GAPPED == "air_gapped"

    def test_deployment_mode_enum_count(self):
        """Verify exactly 4 deployment modes."""
        assert len(DeploymentMode) == 4

    def test_deployment_mode_from_string(self):
        """Test creating enum from string value."""
        assert DeploymentMode("saas_multi_tenant") == DeploymentMode.SAAS_MULTI_TENANT
        assert DeploymentMode("air_gapped") == DeploymentMode.AIR_GAPPED


class TestTenantIsolationEnum:
    """Tests for TenantIsolation enum."""

    def test_tenant_isolation_enum_values(self):
        """Verify all 4 isolation levels exist."""
        assert TenantIsolation.SHARED == "shared"
        assert TenantIsolation.SCHEMA == "schema"
        assert TenantIsolation.DATABASE == "database"
        assert TenantIsolation.INSTANCE == "instance"

    def test_tenant_isolation_enum_count(self):
        """Verify exactly 4 isolation levels."""
        assert len(TenantIsolation) == 4


class TestDeploymentConfig:
    """Tests for DeploymentConfig class."""

    def test_deployment_config_defaults(self):
        """Verify default config values."""
        config = DeploymentConfig()
        assert config.mode == DeploymentMode.SAAS_MULTI_TENANT
        assert config.tenant_isolation == TenantIsolation.SHARED
        assert config.enable_cloud_llm is True
        assert config.enable_telemetry is True
        assert config.enable_auto_updates is True
        assert config.license_key is None

    def test_air_gapped_disables_cloud_llm(self):
        """Air-gapped mode should set enable_cloud_llm=False."""
        config = DeploymentConfig(
            mode=DeploymentMode.AIR_GAPPED,
            enable_cloud_llm=True,  # Try to enable
        )
        # Should be forced to False
        assert config.enable_cloud_llm is False
        assert config.enable_telemetry is False
        assert config.enable_auto_updates is False

    def test_air_gapped_disables_telemetry(self):
        """Air-gapped mode should disable telemetry."""
        config = DeploymentConfig(mode=DeploymentMode.AIR_GAPPED)
        assert config.enable_telemetry is False

    def test_air_gapped_disables_auto_updates(self):
        """Air-gapped mode should disable auto updates."""
        config = DeploymentConfig(mode=DeploymentMode.AIR_GAPPED)
        assert config.enable_auto_updates is False

    def test_saas_dedicated_defaults_to_database_isolation(self):
        """SaaS dedicated should default to database isolation when shared."""
        config = DeploymentConfig(
            mode=DeploymentMode.SAAS_DEDICATED,
            tenant_isolation=TenantIsolation.SHARED,
        )
        assert config.tenant_isolation == TenantIsolation.DATABASE

    def test_saas_dedicated_keeps_explicit_isolation(self):
        """SaaS dedicated should keep explicitly set isolation levels."""
        config = DeploymentConfig(
            mode=DeploymentMode.SAAS_DEDICATED,
            tenant_isolation=TenantIsolation.INSTANCE,
        )
        assert config.tenant_isolation == TenantIsolation.INSTANCE

    def test_deployment_config_from_env(self):
        """Verify config loads from environment variables."""
        reset_deployment_config()

        with patch.dict(os.environ, {
            "DEPLOYMENT_MODE": "on_premise",
            "TENANT_ISOLATION": "instance",
            "ENABLE_CLOUD_LLM": "true",
            "TELEMETRY_ENABLED": "false",
            "ENABLE_AUTO_UPDATES": "false",
            "LICENSE_KEY": "test-license-key",
        }):
            config = DeploymentConfig.from_env()
            assert config.mode == DeploymentMode.ON_PREMISE
            assert config.tenant_isolation == TenantIsolation.INSTANCE
            assert config.enable_cloud_llm is True
            assert config.enable_telemetry is False
            assert config.enable_auto_updates is False
            assert config.license_key == "test-license-key"

    def test_deployment_config_from_env_invalid_mode(self):
        """Invalid deployment mode should default to saas_multi_tenant."""
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "invalid_mode"}):
            config = DeploymentConfig.from_env()
            assert config.mode == DeploymentMode.SAAS_MULTI_TENANT

    def test_deployment_config_from_env_invalid_isolation(self):
        """Invalid isolation level should default to shared."""
        with patch.dict(os.environ, {"TENANT_ISOLATION": "invalid_isolation"}):
            config = DeploymentConfig.from_env()
            assert config.tenant_isolation == TenantIsolation.SHARED


class TestDeploymentConfigFactoryMethods:
    """Tests for factory methods."""

    def test_for_saas_multi_tenant(self):
        """Verify SaaS multi-tenant factory method."""
        config = DeploymentConfig.for_saas_multi_tenant()
        assert config.mode == DeploymentMode.SAAS_MULTI_TENANT
        assert config.tenant_isolation == TenantIsolation.SHARED
        assert config.enable_cloud_llm is True
        assert config.enable_telemetry is True

    def test_for_saas_dedicated(self):
        """Verify SaaS dedicated factory method."""
        config = DeploymentConfig.for_saas_dedicated()
        assert config.mode == DeploymentMode.SAAS_DEDICATED
        assert config.tenant_isolation == TenantIsolation.DATABASE
        assert config.enable_cloud_llm is True

    def test_for_on_premise(self):
        """Verify on-premise factory method."""
        config = DeploymentConfig.for_on_premise()
        assert config.mode == DeploymentMode.ON_PREMISE
        assert config.tenant_isolation == TenantIsolation.INSTANCE
        assert config.enable_cloud_llm is True
        assert config.enable_telemetry is False
        assert config.enable_auto_updates is False

    def test_for_air_gapped(self):
        """Verify air-gapped factory method."""
        config = DeploymentConfig.for_air_gapped()
        assert config.mode == DeploymentMode.AIR_GAPPED
        assert config.tenant_isolation == TenantIsolation.INSTANCE
        assert config.enable_cloud_llm is False
        assert config.enable_telemetry is False
        assert config.enable_auto_updates is False


class TestDeploymentConfigHelperMethods:
    """Tests for helper methods."""

    def test_is_multi_tenant_saas_multi_tenant(self):
        """SaaS multi-tenant should return True for is_multi_tenant."""
        config = DeploymentConfig.for_saas_multi_tenant()
        assert config.is_multi_tenant() is True

    def test_is_multi_tenant_saas_dedicated(self):
        """SaaS dedicated should return True for is_multi_tenant."""
        config = DeploymentConfig.for_saas_dedicated()
        assert config.is_multi_tenant() is True

    def test_is_multi_tenant_on_premise(self):
        """On-premise should return False for is_multi_tenant."""
        config = DeploymentConfig.for_on_premise()
        assert config.is_multi_tenant() is False

    def test_is_multi_tenant_air_gapped(self):
        """Air-gapped should return False for is_multi_tenant."""
        config = DeploymentConfig.for_air_gapped()
        assert config.is_multi_tenant() is False

    def test_requires_tenant_context_saas_multi_tenant(self):
        """SaaS multi-tenant should require tenant context."""
        config = DeploymentConfig.for_saas_multi_tenant()
        assert config.requires_tenant_context() is True

    def test_requires_tenant_context_others(self):
        """Other modes should not require tenant context."""
        assert DeploymentConfig.for_saas_dedicated().requires_tenant_context() is False
        assert DeploymentConfig.for_on_premise().requires_tenant_context() is False
        assert DeploymentConfig.for_air_gapped().requires_tenant_context() is False

    def test_allows_cloud_providers(self):
        """Test allows_cloud_providers based on config."""
        assert DeploymentConfig.for_saas_multi_tenant().allows_cloud_providers() is True
        assert DeploymentConfig.for_air_gapped().allows_cloud_providers() is False


class TestGetDeploymentConfig:
    """Tests for singleton getter."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_deployment_config()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_deployment_config()

    def test_get_deployment_config_returns_same_instance(self):
        """Verify singleton returns same instance."""
        config1 = get_deployment_config()
        config2 = get_deployment_config()
        assert config1 is config2

    def test_get_deployment_config_uses_env(self):
        """Verify singleton loads from environment."""
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "air_gapped"}):
            reset_deployment_config()
            config = get_deployment_config()
            assert config.mode == DeploymentMode.AIR_GAPPED

    def test_reset_deployment_config(self):
        """Verify reset clears singleton."""
        config1 = get_deployment_config()
        reset_deployment_config()

        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "on_premise"}):
            config2 = get_deployment_config()

        assert config1 is not config2
        assert config2.mode == DeploymentMode.ON_PREMISE
