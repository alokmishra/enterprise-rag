"""
Tests for src/core/types.py
"""

from datetime import datetime
from uuid import uuid4

import pytest


class TestEnums:
    """Tests for enum types."""

    def test_query_complexity_values(self):
        """Test QueryComplexity enum values."""
        from src.core.types import QueryComplexity

        assert QueryComplexity.SIMPLE == "simple"
        assert QueryComplexity.STANDARD == "standard"
        assert QueryComplexity.COMPLEX == "complex"

    def test_retrieval_strategy_values(self):
        """Test RetrievalStrategy enum values."""
        from src.core.types import RetrievalStrategy

        assert RetrievalStrategy.VECTOR == "vector"
        assert RetrievalStrategy.SPARSE == "sparse"
        assert RetrievalStrategy.HYBRID == "hybrid"
        assert RetrievalStrategy.GRAPH == "graph"

    def test_document_status_values(self):
        """Test DocumentStatus enum values."""
        from src.core.types import DocumentStatus

        assert DocumentStatus.PENDING == "pending"
        assert DocumentStatus.PROCESSING == "processing"
        assert DocumentStatus.COMPLETED == "completed"
        assert DocumentStatus.FAILED == "failed"

    def test_agent_type_values(self):
        """Test AgentType enum values."""
        from src.core.types import AgentType

        assert AgentType.ORCHESTRATOR == "orchestrator"
        assert AgentType.PLANNER == "planner"
        assert AgentType.RETRIEVER == "retriever"
        assert AgentType.SYNTHESIZER == "synthesizer"
        assert AgentType.VERIFIER == "verifier"
        assert AgentType.CRITIC == "critic"
        assert AgentType.CITATION == "citation"
        assert AgentType.FORMATTER == "formatter"

    def test_message_type_values(self):
        """Test MessageType enum values."""
        from src.core.types import MessageType

        assert MessageType.REQUEST == "request"
        assert MessageType.RESPONSE == "response"
        assert MessageType.FEEDBACK == "feedback"
        assert MessageType.ERROR == "error"

    def test_verification_status_values(self):
        """Test VerificationStatus enum values."""
        from src.core.types import VerificationStatus

        assert VerificationStatus.SUPPORTED == "supported"
        assert VerificationStatus.PARTIALLY_SUPPORTED == "partially_supported"
        assert VerificationStatus.NOT_FOUND == "not_found"
        assert VerificationStatus.CONTRADICTED == "contradicted"


class TestDocumentModels:
    """Tests for document-related models."""

    def test_document_metadata_creation(self):
        """Test DocumentMetadata creation."""
        from src.core.types import DocumentMetadata

        metadata = DocumentMetadata(
            source="test.pdf",
            title="Test Document",
            author="Test Author",
        )
        assert metadata.source == "test.pdf"
        assert metadata.title == "Test Document"

    def test_document_metadata_optional_fields(self):
        """Test DocumentMetadata with optional fields."""
        from src.core.types import DocumentMetadata

        metadata = DocumentMetadata(source="test.pdf")
        assert metadata.source == "test.pdf"
        assert metadata.title is None
        assert metadata.author is None

    def test_chunk_creation(self):
        """Test Chunk creation."""
        from src.core.types import Chunk

        chunk = Chunk(
            id="chunk-1",
            document_id="doc-1",
            content="Test content",
            position=0,
        )
        assert chunk.id == "chunk-1"
        assert chunk.content == "Test content"

    def test_document_creation(self):
        """Test Document creation."""
        from src.core.types import Document, DocumentMetadata

        doc = Document(
            id="doc-1",
            content="Full document content",
            metadata=DocumentMetadata(source="test.pdf"),
        )
        assert doc.id == "doc-1"
        assert doc.metadata.source == "test.pdf"


class TestRetrievalModels:
    """Tests for retrieval-related models."""

    def test_search_result_creation(self):
        """Test SearchResult creation."""
        from src.core.types import SearchResult

        result = SearchResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Test content",
            score=0.95,
        )
        assert result.chunk_id == "chunk-1"
        assert result.score == 0.95

    def test_retrieval_result_creation(self):
        """Test RetrievalResult creation."""
        from src.core.types import RetrievalResult, RetrievalStrategy, SearchResult

        result = RetrievalResult(
            query="test query",
            strategy=RetrievalStrategy.VECTOR,
            results=[
                SearchResult(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    content="Test",
                    score=0.9,
                )
            ],
            total_found=1,
            latency_ms=50.0,
        )
        assert result.query == "test query"
        assert len(result.results) == 1

    def test_context_item_creation(self):
        """Test ContextItem creation."""
        from src.core.types import ContextItem

        item = ContextItem(
            content="Test content",
            source="test.pdf",
            chunk_id="chunk-1",
            document_id="doc-1",
            relevance_score=0.95,
        )
        assert item.content == "Test content"
        assert item.relevance_score == 0.95


class TestAgentModels:
    """Tests for agent-related models."""

    def test_agent_message_creation(self):
        """Test AgentMessage creation."""
        from src.core.types import AgentMessage, AgentType, MessageType

        message = AgentMessage(
            message_id="msg-1",
            trace_id="trace-1",
            from_agent=AgentType.PLANNER,
            to_agent=AgentType.RETRIEVER,
            message_type=MessageType.REQUEST,
            payload={"query": "test"},
        )
        assert message.from_agent == AgentType.PLANNER
        assert message.to_agent == AgentType.RETRIEVER

    def test_agent_state_creation(self):
        """Test AgentState creation."""
        from src.core.types import AgentState

        state = AgentState(
            trace_id="trace-1",
            original_query="What is the policy?",
        )
        assert state.trace_id == "trace-1"
        assert state.original_query == "What is the policy?"
        assert state.draft_responses == []
        assert state.iteration_count == 0

    def test_agent_state_mutable_lists(self):
        """Test that AgentState lists are mutable."""
        from src.core.types import AgentState

        state = AgentState(
            trace_id="trace-1",
            original_query="Test query",
        )
        state.draft_responses.append("Response 1")
        assert len(state.draft_responses) == 1

    def test_verification_result_creation(self):
        """Test VerificationResult creation."""
        from src.core.types import VerificationResult, VerificationStatus

        result = VerificationResult(
            claim="The company was founded in 2020",
            status=VerificationStatus.SUPPORTED,
            evidence="Founded in 2020 according to...",
            confidence=0.95,
        )
        assert result.status == VerificationStatus.SUPPORTED
        assert result.confidence == 0.95

    def test_critic_feedback_creation(self):
        """Test CriticFeedback creation."""
        from src.core.types import CriticFeedback

        feedback = CriticFeedback(
            relevance_score=0.9,
            completeness_score=0.8,
            accuracy_score=0.95,
            coherence_score=0.85,
            citation_score=0.9,
            overall_score=0.88,
            feedback="Good response overall",
            decision="PASS",
        )
        assert feedback.overall_score == 0.88
        assert feedback.decision == "PASS"


class TestCitationModel:
    """Tests for Citation model."""

    def test_citation_creation(self):
        """Test Citation creation."""
        from src.core.types import Citation

        citation = Citation(
            id="cite-1",
            document_id="doc-1",
            chunk_id="chunk-1",
            source="policy.pdf",
            title="Privacy Policy",
        )
        assert citation.id == "cite-1"
        assert citation.source == "policy.pdf"

    def test_citation_optional_fields(self):
        """Test Citation with optional fields."""
        from src.core.types import Citation

        citation = Citation(
            id="cite-1",
            document_id="doc-1",
            chunk_id="chunk-1",
            source="test.pdf",
        )
        assert citation.title is None
        assert citation.url is None
