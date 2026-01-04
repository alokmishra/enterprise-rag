"""
Enterprise RAG System - Core Module

This module provides core functionality used throughout the application:
- Configuration management
- Logging
- Custom exceptions
- Shared type definitions
"""

from src.core.config import Settings, get_settings, settings
from src.core.exceptions import (
    RAGException,
    IngestionError,
    RetrievalError,
    GenerationError,
    AgentError,
    AuthError,
    ValidationError,
    ConfigurationError,
)
from src.core.logging import get_logger, setup_logging, LoggerMixin
from src.core.types import (
    # Enums
    QueryComplexity,
    RetrievalStrategy,
    DocumentStatus,
    AgentType,
    MessageType,
    VerificationStatus,
    ContentType,
    # Document types
    DocumentMetadata,
    Chunk,
    Document,
    # Retrieval types
    SearchResult,
    RetrievalResult,
    ContextItem,
    # Generation types
    Citation,
    GeneratedResponse,
    # Agent types
    AgentMessage,
    AgentState,
    VerificationResult,
    CriticFeedback,
    # Knowledge graph types
    Entity,
    Relationship,
    GraphQueryResult,
)

__all__ = [
    # Config
    "Settings",
    "get_settings",
    "settings",
    # Logging
    "get_logger",
    "setup_logging",
    "LoggerMixin",
    # Exceptions
    "RAGException",
    "IngestionError",
    "RetrievalError",
    "GenerationError",
    "AgentError",
    "AuthError",
    "ValidationError",
    "ConfigurationError",
    # Enums
    "QueryComplexity",
    "RetrievalStrategy",
    "DocumentStatus",
    "AgentType",
    "MessageType",
    "VerificationStatus",
    "ContentType",
    # Document types
    "DocumentMetadata",
    "Chunk",
    "Document",
    # Retrieval types
    "SearchResult",
    "RetrievalResult",
    "ContextItem",
    # Generation types
    "Citation",
    "GeneratedResponse",
    # Agent types
    "AgentMessage",
    "AgentState",
    "VerificationResult",
    "CriticFeedback",
    # Knowledge graph types
    "Entity",
    "Relationship",
    "GraphQueryResult",
]
