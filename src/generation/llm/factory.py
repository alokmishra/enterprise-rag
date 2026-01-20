"""
Enterprise RAG System - LLM Client Factory

Supports multiple LLM providers and deployment modes:
- Cloud providers: Anthropic, OpenAI
- Local providers: llama-cpp-python, transformers (for air-gapped)
- Deployment-aware: Automatically uses local LLM in air-gapped mode
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from src.core.config import settings
from src.core.deployment import DeploymentMode, get_deployment_config
from src.core.exceptions import ConfigurationError
from src.generation.llm.base import LLMClient
from src.generation.llm.anthropic import AnthropicClient, get_anthropic_client
from src.generation.llm.openai import OpenAIClient, get_openai_client

if TYPE_CHECKING:
    from src.core.tenant import TenantConfig


def get_llm_client(
    provider: Optional[str] = None,
    tenant_config: Optional["TenantConfig"] = None,
) -> LLMClient:
    """
    Get an LLM client for the specified provider.

    This factory is deployment-aware:
    - In air-gapped mode, always returns local LLM
    - In other modes, respects provider selection
    - Tenant config can override the default provider

    Args:
        provider: Provider name ("anthropic", "openai", "local").
                 Defaults to settings.DEFAULT_LLM_PROVIDER
        tenant_config: Optional tenant configuration that may specify
                      a different LLM provider

    Returns:
        LLMClient instance

    Raises:
        ConfigurationError: If provider is not supported or
                          cloud provider requested in air-gapped mode
    """
    deployment = get_deployment_config()

    # In air-gapped mode, always use local LLM
    if deployment.mode == DeploymentMode.AIR_GAPPED:
        if provider and provider not in ("local", None):
            raise ConfigurationError(
                f"Cloud LLM provider '{provider}' not available in air-gapped mode. "
                "Only 'local' provider is supported."
            )
        from src.generation.llm.local import get_local_llm_client
        return get_local_llm_client()

    # Check if cloud LLM is disabled
    if not deployment.enable_cloud_llm:
        if provider and provider not in ("local", None):
            raise ConfigurationError(
                f"Cloud LLM provider '{provider}' is disabled in this deployment. "
                "Enable with enable_cloud_llm=True or use 'local' provider."
            )
        from src.generation.llm.local import get_local_llm_client
        return get_local_llm_client()

    # Determine provider from tenant config or default
    if provider is None:
        if tenant_config is not None:
            provider = tenant_config.llm_provider
        else:
            provider = settings.DEFAULT_LLM_PROVIDER

    # Return appropriate client
    if provider == "anthropic":
        return get_anthropic_client()
    elif provider == "openai":
        return get_openai_client()
    elif provider == "local":
        from src.generation.llm.local import get_local_llm_client
        return get_local_llm_client()
    else:
        raise ConfigurationError(f"Unsupported LLM provider: {provider}")


# Default client based on configuration
_default_client: Optional[LLMClient] = None


def get_default_llm_client() -> LLMClient:
    """Get the default LLM client based on configuration."""
    global _default_client
    if _default_client is None:
        _default_client = get_llm_client()
    return _default_client


def reset_llm_client() -> None:
    """Reset the cached LLM client. Used in testing."""
    global _default_client
    _default_client = None
