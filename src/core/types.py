"""
Enterprise RAG System - Shared Type Definitions
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class QueryComplexity(str, Enum):
    """Query complexity classification."""
    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""
    VECTOR = "vector"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    GRAPH = "graph"
    MULTI_QUERY = "multi_query"
    HYDE = "hyde"


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentType(str, Enum):
    """Types of agents in the system."""
    ORCHESTRATOR = "orchestrator"
    PLANNER = "planner"
    RETRIEVER = "retriever"
    RESEARCHER = "researcher"
    SYNTHESIZER = "synthesizer"
    VERIFIER = "verifier"
    CRITIC = "critic"
    CITATION = "citation"
    FORMATTER = "formatter"


class MessageType(str, Enum):
    """Types of agent messages."""
    REQUEST = "request"
    RESPONSE = "response"
    FEEDBACK = "feedback"
    ERROR = "error"


class VerificationStatus(str, Enum):
    """Claim verification status."""
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_FOUND = "not_found"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"


class ContentType(str, Enum):
    """Types of content/modalities."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TABLE = "table"
    CODE = "code"


# =============================================================================
# Base Models
# =============================================================================

class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


class IDMixin(BaseModel):
    """Mixin for ID field."""
    id: UUID


# =============================================================================
# Document Types
# =============================================================================

class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    source: str
    source_id: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    language: Optional[str] = None
    custom: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A chunk of document content."""
    id: str
    document_id: str
    content: str
    content_type: ContentType = ContentType.TEXT
    position: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list[float]] = None
    parent_id: Optional[str] = None


class Document(BaseModel):
    """A document in the system."""
    id: str
    content: str
    metadata: DocumentMetadata
    chunks: list[Chunk] = Field(default_factory=list)
    status: DocumentStatus = DocumentStatus.PENDING


# =============================================================================
# Retrieval Types
# =============================================================================

class SearchResult(BaseModel):
    """A single search result."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = None


class RetrievalResult(BaseModel):
    """Results from a retrieval operation."""
    query: str
    strategy: RetrievalStrategy
    results: list[SearchResult]
    total_found: int
    latency_ms: float


class ContextItem(BaseModel):
    """An item in the assembled context."""
    content: str
    source: str
    chunk_id: str
    document_id: str
    relevance_score: float
    content_type: ContentType = ContentType.TEXT


# =============================================================================
# Generation Types
# =============================================================================

class Citation(BaseModel):
    """A citation in the response."""
    id: str
    document_id: str
    chunk_id: str
    source: str
    title: Optional[str] = None
    text_span: Optional[str] = None
    url: Optional[str] = None


class GeneratedResponse(BaseModel):
    """A generated response with citations."""
    content: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: Optional[float] = None
    model: str
    tokens_used: int


# =============================================================================
# Agent Types
# =============================================================================

class AgentMessage(BaseModel):
    """Message passed between agents."""
    message_id: str
    trace_id: str
    from_agent: AgentType
    to_agent: AgentType
    message_type: MessageType
    payload: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentState(BaseModel):
    """Shared state during agent execution."""
    trace_id: str
    original_query: str
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    execution_plan: Optional[dict[str, Any]] = None
    retrieved_context: list[ContextItem] = Field(default_factory=list)
    draft_responses: list[str] = Field(default_factory=list)
    verification_results: list[dict[str, Any]] = Field(default_factory=list)
    critic_feedback: list[dict[str, Any]] = Field(default_factory=list)
    iteration_count: int = 0
    token_budget_remaining: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    """Result of fact verification."""
    claim: str
    status: VerificationStatus
    evidence: Optional[str] = None
    source: Optional[str] = None
    confidence: float


class CriticFeedback(BaseModel):
    """Feedback from the critic agent."""
    relevance_score: float
    completeness_score: float
    accuracy_score: float
    coherence_score: float
    citation_score: float
    overall_score: float
    feedback: str
    suggestions: list[str] = Field(default_factory=list)
    decision: str  # PASS, MINOR_REVISION, MAJOR_REVISION, RETRIEVAL_NEEDED, REJECT


# =============================================================================
# Knowledge Graph Types
# =============================================================================

class Entity(BaseModel):
    """An entity in the knowledge graph."""
    id: str
    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    description: Optional[str] = None
    properties: dict[str, Any] = Field(default_factory=dict)
    source_chunks: list[str] = Field(default_factory=list)
    confidence: float = 1.0


class Relationship(BaseModel):
    """A relationship between entities."""
    id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    properties: dict[str, Any] = Field(default_factory=dict)
    source_chunk_id: Optional[str] = None
    confidence: float = 1.0


class GraphQueryResult(BaseModel):
    """Result from a graph query."""
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    paths: list[list[str]] = Field(default_factory=list)
