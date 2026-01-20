"""
Tests for LLM client factory.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.core.deployment import reset_deployment_config
from src.core.exceptions import ConfigurationError
from src.core.tenant import TenantConfig


class TestLLMFactory:
    """Tests for get_llm_client factory."""

    def setup_method(self):
        """Reset state before each test."""
        reset_deployment_config()
        # Reset LLM client cache
        from src.generation.llm.factory import reset_llm_client
        reset_llm_client()

    def teardown_method(self):
        """Reset state after each test."""
        reset_deployment_config()
        from src.generation.llm.factory import reset_llm_client
        reset_llm_client()

    def test_factory_returns_anthropic_by_default(self):
        """Default provider returns AnthropicClient."""
        from src.generation.llm.factory import get_llm_client
        from src.generation.llm.anthropic import AnthropicClient

        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "saas_multi_tenant"}):
            reset_deployment_config()

            client = get_llm_client("anthropic")
            assert isinstance(client, AnthropicClient)

    def test_factory_returns_openai_when_specified(self):
        """OpenAI provider returns OpenAIClient."""
        from src.generation.llm.factory import get_llm_client
        from src.generation.llm.openai import OpenAIClient

        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "saas_multi_tenant"}):
            reset_deployment_config()

            client = get_llm_client("openai")
            assert isinstance(client, OpenAIClient)

    def test_factory_returns_local_for_air_gapped(self):
        """Air-gapped mode returns LocalLLMClient."""
        from src.generation.llm.factory import get_llm_client
        from src.generation.llm.local import LocalLLMClient, reset_local_llm_client

        with patch.dict(
            os.environ,
            {
                "DEPLOYMENT_MODE": "air_gapped",
            },
        ):
            reset_deployment_config()
            reset_local_llm_client()

            with patch("src.generation.llm.local.settings") as mock_settings:
                mock_settings.LOCAL_LLM_MODEL_PATH = "/models/test.gguf"
                mock_settings.LOCAL_LLM_MODEL_TYPE = "llama"

                client = get_llm_client()
                assert isinstance(client, LocalLLMClient)

    def test_factory_respects_tenant_provider_override(self):
        """Tenant config can override default provider."""
        from src.generation.llm.factory import get_llm_client
        from src.generation.llm.openai import OpenAIClient

        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "saas_multi_tenant"}):
            reset_deployment_config()

            tenant_config = TenantConfig(
                tenant_id="test-tenant",
                tenant_name="Test Tenant",
                llm_provider="openai",
            )

            client = get_llm_client(tenant_config=tenant_config)
            assert isinstance(client, OpenAIClient)

    def test_factory_raises_for_cloud_in_air_gapped(self):
        """Air-gapped mode rejects cloud provider requests."""
        from src.generation.llm.factory import get_llm_client

        with patch.dict(
            os.environ,
            {
                "DEPLOYMENT_MODE": "air_gapped",
                "LOCAL_LLM_MODEL_PATH": "/models/test.gguf",
            },
        ):
            reset_deployment_config()

            with pytest.raises(ConfigurationError) as exc_info:
                get_llm_client("anthropic")

            assert "not available in air-gapped mode" in str(exc_info.value)

    def test_factory_raises_for_cloud_when_disabled(self):
        """Cloud LLM disabled raises error for cloud providers."""
        from src.generation.llm.factory import get_llm_client
        from src.core.deployment import DeploymentConfig, DeploymentMode, TenantIsolation

        # Create config with cloud LLM disabled
        config = DeploymentConfig(
            mode=DeploymentMode.ON_PREMISE,
            tenant_isolation=TenantIsolation.SHARED,
            enable_cloud_llm=False,
        )

        with patch("src.generation.llm.factory.get_deployment_config", return_value=config):
            with patch.dict(
                os.environ,
                {
                    "LOCAL_LLM_MODEL_PATH": "/models/test.gguf",
                },
            ):
                with pytest.raises(ConfigurationError) as exc_info:
                    get_llm_client("openai")

                assert "disabled" in str(exc_info.value)

    def test_factory_raises_for_unsupported_provider(self):
        """Unsupported provider raises ConfigurationError."""
        from src.generation.llm.factory import get_llm_client

        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "saas_multi_tenant"}):
            reset_deployment_config()

            with pytest.raises(ConfigurationError) as exc_info:
                get_llm_client("unsupported_provider")

            assert "Unsupported LLM provider" in str(exc_info.value)

    def test_factory_returns_local_when_explicitly_requested(self):
        """Local provider can be explicitly requested."""
        from src.generation.llm.factory import get_llm_client
        from src.generation.llm.local import LocalLLMClient, reset_local_llm_client

        with patch.dict(
            os.environ,
            {
                "DEPLOYMENT_MODE": "saas_multi_tenant",
            },
        ):
            reset_deployment_config()
            reset_local_llm_client()

            with patch("src.generation.llm.local.settings") as mock_settings:
                mock_settings.LOCAL_LLM_MODEL_PATH = "/models/test.gguf"
                mock_settings.LOCAL_LLM_MODEL_TYPE = "llama"

                client = get_llm_client("local")
                assert isinstance(client, LocalLLMClient)


class TestDefaultLLMClient:
    """Tests for get_default_llm_client."""

    def setup_method(self):
        """Reset state before each test."""
        reset_deployment_config()
        from src.generation.llm.factory import reset_llm_client
        reset_llm_client()

    def teardown_method(self):
        """Reset state after each test."""
        reset_deployment_config()
        from src.generation.llm.factory import reset_llm_client
        reset_llm_client()

    def test_default_client_is_cached(self):
        """Default client is cached after first call."""
        from src.generation.llm.factory import get_default_llm_client

        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "saas_multi_tenant"}):
            reset_deployment_config()

            client1 = get_default_llm_client()
            client2 = get_default_llm_client()

            assert client1 is client2

    def test_reset_clears_cache(self):
        """reset_llm_client clears the cached client."""
        from src.generation.llm import factory

        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "saas_multi_tenant"}):
            reset_deployment_config()

            # Set a mock client
            mock_client = MagicMock()
            factory._default_client = mock_client

            # Verify it's cached
            assert factory._default_client is mock_client

            # Reset and verify it's cleared
            factory.reset_llm_client()
            assert factory._default_client is None
