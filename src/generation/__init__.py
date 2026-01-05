"""
Enterprise RAG System - Generation Module

This module handles response generation:
- LLM clients (Anthropic, OpenAI)
- Prompt templates
- Streaming responses
"""

from src.generation.llm import (
    LLMClient,
    LLMMessage,
    LLMResponse,
    StreamChunk,
    AnthropicClient,
    OpenAIClient,
    get_llm_client,
    get_default_llm_client,
)
from src.generation.prompts import (
    PromptTemplate,
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT,
    build_rag_prompt,
    format_context,
)

__all__ = [
    # LLM
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "StreamChunk",
    "AnthropicClient",
    "OpenAIClient",
    "get_llm_client",
    "get_default_llm_client",
    # Prompts
    "PromptTemplate",
    "RAG_SYSTEM_PROMPT",
    "RAG_USER_PROMPT",
    "build_rag_prompt",
    "format_context",
]
