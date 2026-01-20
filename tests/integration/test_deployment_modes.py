"""
Integration tests for deployment mode-specific behavior.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.deployment import (
    DeploymentConfig,
    DeploymentMode,
    TenantIsolation,
    get_deployment_config,
    reset_deployment_config,
)
from src.core.features import reset_feature_flags
from src.core.tenant import DEFAULT_TENANT_ID


class TestSaasMultiTenantMode:
    """Tests for saas_multi_tenant deployment mode."""

    def setup_method(self):
        """Reset state before each test."""
        reset_deployment_config()

    def teardown_method(self):
        """Reset state after each test."""
        reset_deployment_config()

    def test_saas_multi_tenant_requires_tenant(self):
        """Multi-tenant SaaS mode requires tenant context."""
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "saas_multi_tenant"}):
            reset_deployment_config()

            config = get_deployment_config()

            assert config.mode == DeploymentMode.SAAS_MULTI_TENANT
            # In multi-tenant mode, requests without tenant should be rejected
            # (tested in middleware tests)

    def test_saas_multi_tenant_uses_shared_isolation(self):
        """Multi-tenant mode uses shared isolation by default."""
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "saas_multi_tenant"}):
            reset_deployment_config()

            config = get_deployment_config()

            assert config.tenant_isolation == TenantIsolation.SHARED

    def test_saas_multi_tenant_enables_cloud_llm(self):
        """Multi-tenant mode enables cloud LLM."""
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "saas_multi_tenant"}):
            reset_deployment_config()

            config = get_deployment_config()

            assert config.enable_cloud_llm is True


class TestOnPremiseMode:
    """Tests for on_premise deployment mode."""

    def setup_method(self):
        """Reset state before each test."""
        reset_deployment_config()

    def teardown_method(self):
        """Reset state after each test."""
        reset_deployment_config()

    def test_on_premise_uses_default_tenant(self):
        """On-premise mode uses default tenant."""
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "on_premise"}):
            reset_deployment_config()

            config = get_deployment_config()

            assert config.mode == DeploymentMode.ON_PREMISE
            # On-premise uses default tenant (verified in middleware tests)

    def test_on_premise_can_disable_telemetry(self):
        """On-premise mode can disable telemetry."""
        with patch.dict(
            os.environ,
            {"DEPLOYMENT_MODE": "on_premise", "TELEMETRY_ENABLED": "false"},
        ):
            reset_deployment_config()

            config = get_deployment_config()

            assert config.enable_telemetry is False

    def test_on_premise_enables_cloud_llm_by_default(self):
        """On-premise mode enables cloud LLM by default."""
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "on_premise"}):
            reset_deployment_config()

            config = get_deployment_config()

            assert config.enable_cloud_llm is True


class TestAirGappedMode:
    """Tests for air_gapped deployment mode."""

    def setup_method(self):
        """Reset state before each test."""
        reset_deployment_config()

    def teardown_method(self):
        """Reset state after each test."""
        reset_deployment_config()

    def test_air_gapped_uses_local_llm(self):
        """Air-gapped mode uses local LLM, not cloud."""
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "air_gapped"}):
            reset_deployment_config()

            config = get_deployment_config()

            assert config.mode == DeploymentMode.AIR_GAPPED
            assert config.enable_cloud_llm is False

    def test_air_gapped_disables_telemetry(self):
        """Air-gapped mode disables telemetry."""
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "air_gapped"}):
            reset_deployment_config()

            config = get_deployment_config()

            assert config.enable_telemetry is False

    def test_air_gapped_disables_auto_updates(self):
        """Air-gapped mode disables auto updates."""
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "air_gapped"}):
            reset_deployment_config()

            config = get_deployment_config()

            assert config.enable_auto_updates is False

    @pytest.mark.asyncio
    async def test_air_gapped_llm_factory_returns_local(self):
        """Air-gapped mode LLM factory returns local client."""
        from src.generation.llm.factory import get_llm_client
        from src.generation.llm.local import LocalLLMClient, reset_local_llm_client

        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "air_gapped"}):
            reset_deployment_config()
            reset_local_llm_client()

            with patch("src.generation.llm.local.settings") as mock_settings:
                mock_settings.LOCAL_LLM_MODEL_PATH = "/models/local.gguf"
                mock_settings.LOCAL_LLM_MODEL_TYPE = "llama"

                client = get_llm_client()
                assert isinstance(client, LocalLLMClient)


class TestSaasDedicatedMode:
    """Tests for saas_dedicated deployment mode."""

    def setup_method(self):
        """Reset state before each test."""
        reset_deployment_config()

    def teardown_method(self):
        """Reset state after each test."""
        reset_deployment_config()

    def test_saas_dedicated_supports_schema_isolation(self):
        """SaaS dedicated mode supports schema isolation."""
        with patch.dict(
            os.environ,
            {"DEPLOYMENT_MODE": "saas_dedicated", "TENANT_ISOLATION": "schema"},
        ):
            reset_deployment_config()

            config = get_deployment_config()

            assert config.mode == DeploymentMode.SAAS_DEDICATED
            assert config.tenant_isolation == TenantIsolation.SCHEMA

    def test_saas_dedicated_supports_database_isolation(self):
        """SaaS dedicated mode supports database isolation."""
        with patch.dict(
            os.environ,
            {"DEPLOYMENT_MODE": "saas_dedicated", "TENANT_ISOLATION": "database"},
        ):
            reset_deployment_config()

            config = get_deployment_config()

            assert config.tenant_isolation == TenantIsolation.DATABASE


class TestFeatureFlagsByDeploymentMode:
    """Tests for feature flags based on deployment mode."""

    def setup_method(self):
        """Reset state before each test."""
        reset_deployment_config()
        reset_feature_flags()

    def teardown_method(self):
        """Reset state after each test."""
        reset_deployment_config()
        reset_feature_flags()

    def test_air_gapped_disables_cloud_llm_feature(self):
        """Air-gapped mode disables cloud LLM feature."""
        from src.core.features import get_feature_flags

        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "air_gapped"}):
            reset_deployment_config()
            reset_feature_flags()

            flags = get_feature_flags()

            assert flags.cloud_llm_enabled() is False

    def test_saas_multi_tenant_enables_cloud_llm_feature(self):
        """SaaS multi-tenant enables cloud LLM feature."""
        from src.core.features import get_feature_flags

        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "saas_multi_tenant"}):
            reset_deployment_config()
            reset_feature_flags()

            flags = get_feature_flags()

            assert flags.cloud_llm_enabled() is True

    def test_air_gapped_disables_telemetry_feature(self):
        """Air-gapped mode disables telemetry feature."""
        from src.core.features import get_feature_flags

        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "air_gapped"}):
            reset_deployment_config()
            reset_feature_flags()

            flags = get_feature_flags()

            assert flags.telemetry_enabled() is False


class TestMiddlewareTenantBehaviorByMode:
    """Tests for middleware tenant handling by deployment mode."""

    def setup_method(self):
        """Reset state before each test."""
        reset_deployment_config()

    def teardown_method(self):
        """Reset state after each test."""
        reset_deployment_config()

    @pytest.mark.asyncio
    async def test_single_tenant_mode_uses_default(self):
        """Single-tenant modes use default tenant."""
        from src.api.middleware.tenant import tenant_context_middleware

        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "on_premise"}):
            reset_deployment_config()

            mock_request = MagicMock()
            mock_request.state = MagicMock(spec=[])
            mock_request.headers = {}
            mock_request.query_params = {}
            mock_request.url.path = "/api/v1/query"

            mock_response = MagicMock()
            mock_response.headers = {}

            async def mock_call_next(request):
                return mock_response

            await tenant_context_middleware(mock_request, mock_call_next)

            assert mock_request.state.tenant_id == DEFAULT_TENANT_ID

    @pytest.mark.asyncio
    async def test_multi_tenant_mode_extracts_tenant(self):
        """Multi-tenant mode extracts tenant from header."""
        from src.api.middleware.tenant import tenant_context_middleware

        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "saas_multi_tenant"}):
            reset_deployment_config()

            mock_request = MagicMock()
            mock_request.state = MagicMock(spec=[])
            mock_request.headers = {"X-Tenant-ID": "extracted-tenant"}
            mock_request.query_params = {}
            mock_request.url.path = "/api/v1/query"

            mock_response = MagicMock()
            mock_response.headers = {}

            async def mock_call_next(request):
                return mock_response

            await tenant_context_middleware(mock_request, mock_call_next)

            assert mock_request.state.tenant_id == "extracted-tenant"
