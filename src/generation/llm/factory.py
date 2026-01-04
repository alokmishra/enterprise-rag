"""
Enterprise RAG System - LLM Client Factory
"""

from typing import Optional

from src.core.config import settings
from src.core.exceptions import ConfigurationError
from src.generation.llm.base import LLMClient
from src.generation.llm.anthropic import AnthropicClient, get_anthropic_client
from src.generation.llm.openai import OpenAIClient, get_openai_client


def get_llm_client(provider: Optional[str] = None) -> LLMClient:
    """
    Get an LLM client for the specified provider.

    Args:
        provider: Provider name ("anthropic" or "openai").
                 Defaults to settings.DEFAULT_LLM_PROVIDER

    Returns:
        LLMClient instance

    Raises:
        ConfigurationError: If provider is not supported
    """
    provider = provider or settings.DEFAULT_LLM_PROVIDER

    if provider == "anthropic":
        return get_anthropic_client()
    elif provider == "openai":
        return get_openai_client()
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
