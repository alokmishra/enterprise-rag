"""
Enterprise RAG System - Prompts Module
"""

from __future__ import annotations

from src.generation.prompts.templates import (
    PromptTemplate,
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT,
    RAG_USER_PROMPT_WITH_HISTORY,
    QUERY_ANALYSIS_PROMPT,
    VERIFICATION_PROMPT,
    format_context,
    format_conversation_history,
    build_rag_prompt,
)

__all__ = [
    "PromptTemplate",
    "RAG_SYSTEM_PROMPT",
    "RAG_USER_PROMPT",
    "RAG_USER_PROMPT_WITH_HISTORY",
    "QUERY_ANALYSIS_PROMPT",
    "VERIFICATION_PROMPT",
    "format_context",
    "format_conversation_history",
    "build_rag_prompt",
]
