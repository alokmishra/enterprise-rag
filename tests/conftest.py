"""
Enterprise RAG System - Test Configuration and Fixtures
"""
from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Generator, List, Dict
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import BaseModel

# =============================================================================
# Async Event Loop
# =============================================================================

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Core Fixtures
# =============================================================================

@pytest.fixture
def trace_id() -> str:
    """Generate a unique trace ID."""
    return str(uuid4())


@pytest.fixture
def sample_query() -> str:
    """Sample user query."""
    return "What is the company's data privacy policy?"


@pytest.fixture
def sample_document_content() -> str:
    """Sample document content for testing."""
    return """
    Data Privacy Policy

    Our company is committed to protecting your personal data. We collect only
    the information necessary to provide our services. All data is encrypted
    at rest and in transit. We do not sell your data to third parties.

    Data Retention

    We retain your data for as long as your account is active or as needed to
    provide you services. You may request deletion of your data at any time.

    Your Rights

    You have the right to access, correct, or delete your personal data.
    Contact our privacy team at privacy@company.com for assistance.
    """


@pytest.fixture
def sample_chunks() -> list[dict]:
    """Sample document chunks."""
    return [
        {
            "id": "chunk-1",
            "document_id": "doc-1",
            "content": "Our company is committed to protecting your personal data.",
            "source": "privacy_policy.pdf",
            "position": 0,
        },
        {
            "id": "chunk-2",
            "document_id": "doc-1",
            "content": "We retain your data for as long as your account is active.",
            "source": "privacy_policy.pdf",
            "position": 1,
        },
        {
            "id": "chunk-3",
            "document_id": "doc-1",
            "content": "You have the right to access, correct, or delete your personal data.",
            "source": "privacy_policy.pdf",
            "position": 2,
        },
    ]


@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    """Sample embedding vectors (1536 dimensions, truncated for testing)."""
    import random
    random.seed(42)
    return [[random.random() for _ in range(1536)] for _ in range(3)]


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Mock LLM client."""
    client = AsyncMock()
    client.generate = AsyncMock(return_value=MagicMock(
        content="This is a generated response based on the context.",
        tokens_used=150,
    ))
    client.generate_stream = AsyncMock()
    return client


@pytest.fixture
def mock_embedding_provider() -> MagicMock:
    """Mock embedding provider."""
    import random
    random.seed(42)

    provider = AsyncMock()
    provider.embed = AsyncMock(return_value=MagicMock(
        embeddings=[[random.random() for _ in range(1536)]],
        tokens_used=10,
    ))
    provider.embed_batch = AsyncMock(return_value=MagicMock(
        embeddings=[[random.random() for _ in range(1536)] for _ in range(3)],
        tokens_used=30,
    ))
    return provider


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Mock vector store."""
    store = AsyncMock()
    store.search = AsyncMock(return_value=[
        {"id": "chunk-1", "score": 0.95, "payload": {"content": "Test content 1"}},
        {"id": "chunk-2", "score": 0.87, "payload": {"content": "Test content 2"}},
    ])
    store.upsert = AsyncMock(return_value=True)
    store.delete = AsyncMock(return_value=True)
    return store


@pytest.fixture
def mock_cache() -> MagicMock:
    """Mock cache."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    cache.delete = AsyncMock(return_value=True)
    return cache


@pytest.fixture
def mock_database() -> MagicMock:
    """Mock database session."""
    db = AsyncMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    return db


# =============================================================================
# Agent Fixtures
# =============================================================================

@pytest.fixture
def sample_agent_state():
    """Create a sample agent state for testing."""
    from src.core.types import AgentState, ContextItem

    return AgentState(
        trace_id=str(uuid4()),
        original_query="What is our data privacy policy?",
        conversation_history=[],
        retrieved_context=[
            ContextItem(
                content="Our company is committed to protecting your personal data.",
                source="privacy_policy.pdf",
                chunk_id="chunk-1",
                document_id="doc-1",
                relevance_score=0.95,
            ),
            ContextItem(
                content="We retain your data for as long as your account is active.",
                source="privacy_policy.pdf",
                chunk_id="chunk-2",
                document_id="doc-1",
                relevance_score=0.87,
            ),
        ],
        draft_responses=[],
        verification_results=[],
        critic_feedback=[],
        iteration_count=0,
        token_budget_remaining=8000,
    )


@pytest.fixture
def sample_agent_state_with_response(sample_agent_state):
    """Agent state with a draft response."""
    sample_agent_state.draft_responses.append(
        "Based on our privacy policy [1], the company is committed to protecting "
        "your personal data. Data is retained as long as your account is active [2]."
    )
    return sample_agent_state


# =============================================================================
# API Fixtures
# =============================================================================

@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    from fastapi.testclient import TestClient
    from src.app import app

    return TestClient(app)


@pytest.fixture
async def async_test_client():
    """Create an async test client."""
    from httpx import AsyncClient, ASGITransport
    from src.app import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client
