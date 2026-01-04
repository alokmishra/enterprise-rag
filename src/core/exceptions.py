"""
Enterprise RAG System - Custom Exceptions
"""

from typing import Any, Optional


class RAGException(Exception):
    """Base exception for all RAG system errors."""
    
    def __init__(
        self,
        message: str,
        code: str = "RAG_ERROR",
        details: Optional[dict[str, Any]] = None,
    ):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)


# =============================================================================
# Ingestion Exceptions
# =============================================================================

class IngestionError(RAGException):
    """Base exception for ingestion errors."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, code="INGESTION_ERROR", details=details)


class UnsupportedFileTypeError(IngestionError):
    """Raised when an unsupported file type is encountered."""
    
    def __init__(self, file_type: str):
        super().__init__(
            message=f"Unsupported file type: {file_type}",
            details={"file_type": file_type}
        )


class DocumentProcessingError(IngestionError):
    """Raised when document processing fails."""
    
    def __init__(self, document_id: str, reason: str):
        super().__init__(
            message=f"Failed to process document {document_id}: {reason}",
            details={"document_id": document_id, "reason": reason}
        )


class ChunkingError(IngestionError):
    """Raised when chunking fails."""
    pass


class EmbeddingError(IngestionError):
    """Raised when embedding generation fails."""
    pass


# =============================================================================
# Retrieval Exceptions
# =============================================================================

class RetrievalError(RAGException):
    """Base exception for retrieval errors."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, code="RETRIEVAL_ERROR", details=details)


class VectorStoreError(RetrievalError):
    """Raised when vector store operations fail."""
    pass


class GraphStoreError(RetrievalError):
    """Raised when graph store operations fail."""
    pass


class SearchError(RetrievalError):
    """Raised when search operations fail."""
    pass


class RerankingError(RetrievalError):
    """Raised when reranking fails."""
    pass


# =============================================================================
# Generation Exceptions
# =============================================================================

class GenerationError(RAGException):
    """Base exception for generation errors."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, code="GENERATION_ERROR", details=details)


class LLMError(GenerationError):
    """Raised when LLM operations fail."""
    
    def __init__(self, provider: str, message: str):
        super().__init__(
            message=f"LLM error ({provider}): {message}",
            details={"provider": provider}
        )


class LLMRateLimitError(LLMError):
    """Raised when LLM rate limit is exceeded."""
    
    def __init__(self, provider: str, retry_after: Optional[int] = None):
        super().__init__(
            provider=provider,
            message="Rate limit exceeded"
        )
        self.retry_after = retry_after


class LLMContextLengthError(LLMError):
    """Raised when context exceeds LLM limits."""
    
    def __init__(self, provider: str, max_tokens: int, requested_tokens: int):
        super().__init__(
            provider=provider,
            message=f"Context length {requested_tokens} exceeds limit {max_tokens}"
        )
        self.details.update({
            "max_tokens": max_tokens,
            "requested_tokens": requested_tokens
        })


class PromptError(GenerationError):
    """Raised when prompt construction fails."""
    pass


# =============================================================================
# Agent Exceptions
# =============================================================================

class AgentError(RAGException):
    """Base exception for agent errors."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, code="AGENT_ERROR", details=details)


class AgentTimeoutError(AgentError):
    """Raised when agent execution times out."""
    
    def __init__(self, agent_name: str, timeout_seconds: int):
        super().__init__(
            message=f"Agent {agent_name} timed out after {timeout_seconds}s",
            details={"agent_name": agent_name, "timeout_seconds": timeout_seconds}
        )


class AgentIterationLimitError(AgentError):
    """Raised when agent exceeds iteration limit."""
    
    def __init__(self, agent_name: str, max_iterations: int):
        super().__init__(
            message=f"Agent {agent_name} exceeded max iterations ({max_iterations})",
            details={"agent_name": agent_name, "max_iterations": max_iterations}
        )


class VerificationError(AgentError):
    """Raised when fact verification fails."""
    pass


# =============================================================================
# Knowledge Graph Exceptions
# =============================================================================

class KnowledgeGraphError(RAGException):
    """Base exception for knowledge graph errors."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, code="KNOWLEDGE_GRAPH_ERROR", details=details)


class EntityExtractionError(KnowledgeGraphError):
    """Raised when entity extraction fails."""
    pass


class RelationshipExtractionError(KnowledgeGraphError):
    """Raised when relationship extraction fails."""
    pass


class EntityLinkingError(KnowledgeGraphError):
    """Raised when entity linking fails."""
    pass


# =============================================================================
# Authentication & Authorization Exceptions
# =============================================================================

class AuthError(RAGException):
    """Base exception for authentication errors."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, code="AUTH_ERROR", details=details)


class AuthenticationError(AuthError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(AuthError):
    """Raised when authorization fails."""
    
    def __init__(self, resource: str, action: str):
        super().__init__(
            message=f"Not authorized to {action} on {resource}",
            details={"resource": resource, "action": action}
        )


# =============================================================================
# Validation Exceptions
# =============================================================================

class ValidationError(RAGException):
    """Raised when validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            details={"field": field} if field else {}
        )


# =============================================================================
# Configuration Exceptions
# =============================================================================

class ConfigurationError(RAGException):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str):
        super().__init__(message, code="CONFIGURATION_ERROR")
