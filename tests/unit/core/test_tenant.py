"""
Tests for src/core/tenant.py
"""

import pytest

from src.core.deployment import TenantIsolation
from src.core.tenant import (
    TenantConfig,
    TenantContext,
    DEFAULT_TENANT_ID,
    get_default_tenant_config,
    reset_default_tenant_config,
)


class TestTenantConfig:
    """Tests for TenantConfig class."""

    def test_tenant_config_defaults(self):
        """Verify default tenant config values."""
        config = TenantConfig(
            tenant_id="test-tenant",
            tenant_name="Test Tenant",
        )
        assert config.tenant_id == "test-tenant"
        assert config.tenant_name == "Test Tenant"
        assert config.isolation_level == TenantIsolation.SHARED
        assert config.llm_provider == "anthropic"
        assert config.llm_model == "claude-sonnet-4-20250514"
        assert config.max_documents == 10000
        assert config.max_queries_per_day == 1000
        assert config.max_tokens_per_query == 8000
        assert config.max_file_size_mb == 50
        assert config.max_concurrent_ingestions == 5

    def test_tenant_config_allowed_providers(self):
        """Verify default allowed providers."""
        config = TenantConfig(
            tenant_id="test-tenant",
            tenant_name="Test Tenant",
        )
        assert "anthropic" in config.allowed_providers
        assert "openai" in config.allowed_providers

    def test_tenant_config_default_features(self):
        """Verify default features enabled."""
        config = TenantConfig(
            tenant_id="test-tenant",
            tenant_name="Test Tenant",
        )
        assert "multi_agent" in config.features_enabled
        assert "knowledge_graph" in config.features_enabled
        assert "hybrid_search" in config.features_enabled
        assert "streaming" in config.features_enabled
        assert "citations" in config.features_enabled

    def test_tenant_config_custom_limits(self):
        """Test setting custom resource limits."""
        config = TenantConfig(
            tenant_id="test-tenant",
            tenant_name="Test Tenant",
            max_documents=50000,
            max_queries_per_day=10000,
        )
        assert config.max_documents == 50000
        assert config.max_queries_per_day == 10000


class TestTenantConfigDefaultSingleTenant:
    """Tests for default_single_tenant factory method."""

    def test_default_single_tenant_returns_unlimited(self):
        """Verify default_single_tenant() returns unlimited config."""
        config = TenantConfig.default_single_tenant()
        assert config.tenant_id == "default"
        assert config.tenant_name == "Default Tenant"
        assert config.isolation_level == TenantIsolation.INSTANCE
        assert config.max_documents == -1  # Unlimited
        assert config.max_queries_per_day == -1  # Unlimited
        assert config.max_tokens_per_query == -1  # Unlimited

    def test_default_single_tenant_has_all_features(self):
        """Default single tenant should have all features."""
        config = TenantConfig.default_single_tenant()
        assert "multi_agent" in config.features_enabled
        assert "knowledge_graph" in config.features_enabled
        assert "multimodal" in config.features_enabled
        assert "admin_dashboard" in config.features_enabled


class TestTenantConfigFactoryMethods:
    """Tests for factory methods."""

    def test_for_trial(self):
        """Verify trial tenant config."""
        config = TenantConfig.for_trial("trial-123", "Trial User")
        assert config.tenant_id == "trial-123"
        assert config.tenant_name == "Trial User"
        assert config.max_documents == 100
        assert config.max_queries_per_day == 50
        assert config.max_tokens_per_query == 4000
        assert config.max_file_size_mb == 10
        # Trial has limited features
        assert "hybrid_search" in config.features_enabled
        assert "streaming" in config.features_enabled
        assert "multi_agent" not in config.features_enabled
        assert "knowledge_graph" not in config.features_enabled

    def test_for_enterprise(self):
        """Verify enterprise tenant config."""
        config = TenantConfig.for_enterprise("enterprise-456", "Enterprise Corp")
        assert config.tenant_id == "enterprise-456"
        assert config.tenant_name == "Enterprise Corp"
        assert config.isolation_level == TenantIsolation.DATABASE
        assert config.max_documents == 1000000
        assert config.max_queries_per_day == 100000
        assert config.max_concurrent_ingestions == 20
        # Enterprise has all features
        assert "multi_agent" in config.features_enabled
        assert "knowledge_graph" in config.features_enabled
        assert "multimodal" in config.features_enabled
        assert "custom_models" in config.features_enabled
        assert "priority_support" in config.features_enabled


class TestTenantConfigHelperMethods:
    """Tests for helper methods."""

    def test_has_feature_enabled(self):
        """Test has_feature returns True for enabled features."""
        config = TenantConfig(
            tenant_id="test",
            tenant_name="Test",
            features_enabled=["multi_agent", "streaming"],
        )
        assert config.has_feature("multi_agent") is True
        assert config.has_feature("streaming") is True

    def test_has_feature_disabled(self):
        """Test has_feature returns False for disabled features."""
        config = TenantConfig(
            tenant_id="test",
            tenant_name="Test",
            features_enabled=["multi_agent"],
        )
        assert config.has_feature("knowledge_graph") is False
        assert config.has_feature("nonexistent") is False

    def test_can_use_provider_allowed(self):
        """Test can_use_provider returns True for allowed providers."""
        config = TenantConfig(
            tenant_id="test",
            tenant_name="Test",
            allowed_providers=["anthropic", "openai"],
        )
        assert config.can_use_provider("anthropic") is True
        assert config.can_use_provider("openai") is True

    def test_can_use_provider_not_allowed(self):
        """Test can_use_provider returns False for non-allowed providers."""
        config = TenantConfig(
            tenant_id="test",
            tenant_name="Test",
            allowed_providers=["anthropic"],
        )
        assert config.can_use_provider("openai") is False
        assert config.can_use_provider("local") is False

    def test_get_resource_limit(self):
        """Test get_resource_limit returns correct values."""
        config = TenantConfig(
            tenant_id="test",
            tenant_name="Test",
            max_documents=5000,
            max_queries_per_day=500,
        )
        assert config.get_resource_limit("documents") == 5000
        assert config.get_resource_limit("queries_per_day") == 500
        assert config.get_resource_limit("nonexistent") == 0

    def test_effective_collection_prefix_default(self):
        """Test effective_collection_prefix uses tenant_id by default."""
        config = TenantConfig(
            tenant_id="my-tenant",
            tenant_name="My Tenant",
        )
        assert config.effective_collection_prefix == "my-tenant"

    def test_effective_collection_prefix_override(self):
        """Test effective_collection_prefix uses override if set."""
        config = TenantConfig(
            tenant_id="my-tenant",
            tenant_name="My Tenant",
            vector_collection_prefix="custom_prefix",
        )
        assert config.effective_collection_prefix == "custom_prefix"


class TestTenantContext:
    """Tests for TenantContext class."""

    def test_tenant_context_basic(self):
        """Test basic TenantContext creation."""
        context = TenantContext(tenant_id="test-123")
        assert context.tenant_id == "test-123"
        assert context.config is None

    def test_tenant_context_with_config(self):
        """Test TenantContext with config."""
        config = TenantConfig(
            tenant_id="test-123",
            tenant_name="Test",
            features_enabled=["multi_agent"],
        )
        context = TenantContext(tenant_id="test-123", tenant_config=config)
        assert context.tenant_id == "test-123"
        assert context.config is config

    def test_tenant_context_has_feature_with_config(self):
        """Test has_feature uses config when available."""
        config = TenantConfig(
            tenant_id="test-123",
            tenant_name="Test",
            features_enabled=["multi_agent"],
        )
        context = TenantContext(tenant_id="test-123", tenant_config=config)
        assert context.has_feature("multi_agent") is True
        assert context.has_feature("knowledge_graph") is False

    def test_tenant_context_has_feature_without_config(self):
        """Test has_feature defaults to True without config."""
        context = TenantContext(tenant_id="test-123")
        assert context.has_feature("any_feature") is True


class TestDefaultTenantConfig:
    """Tests for default tenant config singleton."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_default_tenant_config()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_default_tenant_config()

    def test_default_tenant_id_constant(self):
        """Verify DEFAULT_TENANT_ID constant."""
        assert DEFAULT_TENANT_ID == "default"

    def test_get_default_tenant_config_returns_same_instance(self):
        """Verify singleton returns same instance."""
        config1 = get_default_tenant_config()
        config2 = get_default_tenant_config()
        assert config1 is config2

    def test_get_default_tenant_config_returns_default_single_tenant(self):
        """Verify singleton returns default_single_tenant config."""
        config = get_default_tenant_config()
        assert config.tenant_id == "default"
        assert config.max_documents == -1  # Unlimited

    def test_reset_default_tenant_config(self):
        """Verify reset clears singleton."""
        config1 = get_default_tenant_config()
        reset_default_tenant_config()
        config2 = get_default_tenant_config()
        # Should be equal but not the same object
        assert config1 is not config2
        assert config1.tenant_id == config2.tenant_id
