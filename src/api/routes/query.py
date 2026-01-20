"""
Enterprise RAG System - Query API Routes
"""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.core.logging import get_logger
from src.core.types import QueryComplexity, RetrievalStrategy
from src.api.services.rag_pipeline import get_rag_pipeline
from src.api.middleware.tenant import get_tenant_id
from src.agents import Orchestrator, OrchestratorConfig, StreamingOrchestrator, OutputFormat
from src.storage import get_database, QueryLogRepository


router = APIRouter()
logger = get_logger(__name__)


# =============================================================================
# Request/Response Schemas
# =============================================================================

class QueryRequest(BaseModel):
    """Request schema for query endpoint."""
    query: str = Field(..., min_length=1, max_length=10000, description="The user's question")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    retrieval_strategy: Optional[RetrievalStrategy] = Field(
        None, description="Override retrieval strategy"
    )
    top_k: Optional[int] = Field(None, ge=1, le=50, description="Number of results to retrieve")
    include_sources: bool = Field(True, description="Include source citations")
    stream: bool = Field(False, description="Stream the response")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What is our data privacy policy?",
                    "include_sources": True,
                    "stream": False
                }
            ]
        }
    }


class SourceReference(BaseModel):
    """A source reference in the response."""
    id: str
    document_id: str
    title: Optional[str] = None
    source: Optional[str] = None
    relevance_score: float
    excerpt: Optional[str] = None


class QueryResponse(BaseModel):
    """Response schema for query endpoint."""
    query_id: str
    query: str
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    confidence: Optional[float] = None
    complexity: QueryComplexity
    latency_ms: float
    tokens_used: int


class FeedbackRequest(BaseModel):
    """Request schema for query feedback."""
    query_id: str
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    feedback: Optional[str] = Field(None, max_length=1000, description="Optional text feedback")


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    tenant_id: str = Depends(get_tenant_id),
):
    """
    Execute a RAG query.

    This endpoint processes a natural language query through the full RAG pipeline:
    1. Query analysis and classification
    2. Multi-strategy retrieval
    3. Context assembly
    4. Response generation with citations
    5. Optional fact verification

    Returns a response with source citations.
    """
    try:
        pipeline = get_rag_pipeline(tenant_id=tenant_id)

        result = await pipeline.query(
            question=request.query,
            retrieval_strategy=request.retrieval_strategy,
            top_k=request.top_k,
        )

        # Build source references
        sources = []
        sources_for_storage = []
        if request.include_sources:
            for s in result.sources:
                sources.append(SourceReference(
                    id=s["id"],
                    document_id=s["document_id"],
                    title=s.get("title"),
                    source=s.get("source"),
                    relevance_score=s["relevance_score"],
                    excerpt=s.get("excerpt"),
                ))
                sources_for_storage.append({
                    "id": s["id"],
                    "document_id": s["document_id"],
                    "title": s.get("title"),
                    "source": s.get("source"),
                    "relevance_score": s["relevance_score"],
                })

        # Persist query to database
        try:
            async for session in get_database().session():
                repo = QueryLogRepository(session, tenant_id=tenant_id)
                await repo.create(
                    query=result.query,
                    answer=result.answer,
                    conversation_id=request.conversation_id,
                    latency_ms=result.latency_ms,
                    tokens_used=result.tokens_used,
                    confidence=result.confidence,
                    sources_used=sources_for_storage,
                )
                await session.commit()
        except Exception as persist_error:
            # Log but don't fail the request if persistence fails
            logger.warning("Failed to persist query", error=str(persist_error), tenant_id=tenant_id)

        return QueryResponse(
            query_id=result.query_id,
            query=result.query,
            answer=result.answer,
            sources=sources,
            confidence=result.confidence,
            complexity=result.complexity,
            latency_ms=result.latency_ms,
            tokens_used=result.tokens_used,
        )

    except Exception as e:
        logger.error("Query failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


@router.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    tenant_id: str = Depends(get_tenant_id),
):
    """
    Execute a RAG query with streaming response.

    Returns a streaming response where the answer is streamed token by token.
    Sources are provided at the end of the stream.
    """
    if not request.stream:
        raise HTTPException(
            status_code=400,
            detail="Use /query endpoint for non-streaming queries"
        )

    async def generate():
        try:
            pipeline = get_rag_pipeline(tenant_id=tenant_id)

            async for chunk in pipeline.query_stream(
                question=request.query,
                retrieval_strategy=request.retrieval_strategy,
                top_k=request.top_k,
            ):
                yield f"data: {json.dumps(chunk)}\n\n"

        except Exception as e:
            logger.error("Streaming query failed", error=str(e), tenant_id=tenant_id)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@router.post("/query/{query_id}/feedback")
async def submit_feedback(
    query_id: str,
    request: FeedbackRequest,
    tenant_id: str = Depends(get_tenant_id),
):
    """
    Submit feedback for a query response.

    This feedback is used to improve the system over time.
    """
    if request.query_id != query_id:
        raise HTTPException(
            status_code=400,
            detail="Query ID in path does not match body"
        )

    # Check if database is connected
    db = get_database()
    if not db.is_connected:
        # Graceful degradation - accept feedback without persisting
        logger.warning("Database not connected, feedback not persisted", query_id=query_id, tenant_id=tenant_id)
        return {"status": "accepted", "message": "Feedback received (not persisted)", "query_id": query_id}

    try:
        async for session in db.session():
            repo = QueryLogRepository(session, tenant_id=tenant_id)

            # Check if query exists
            query_log = await repo.get(query_id)
            if query_log is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Query {query_id} not found"
                )

            # Update feedback
            success = await repo.update_feedback(
                query_id=query_id,
                rating=request.rating,
                feedback=request.feedback,
            )

            if not success:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to record feedback"
                )

            await session.commit()

        logger.info(
            "Feedback recorded",
            query_id=query_id,
            rating=request.rating,
        )

        return {"status": "success", "message": "Feedback recorded", "query_id": query_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to store feedback", query_id=query_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store feedback: {str(e)}"
        )


class QueryDetailResponse(BaseModel):
    """Response schema for query detail endpoint."""
    query_id: str
    query: str
    answer: Optional[str] = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float
    tokens_used: int
    confidence: Optional[float] = None
    rating: Optional[int] = None
    feedback: Optional[str] = None
    created_at: str


@router.get("/query/{query_id}", response_model=QueryDetailResponse)
async def get_query(
    query_id: str,
    tenant_id: str = Depends(get_tenant_id),
):
    """
    Get details of a previous query.

    Returns the full query details including the response and sources.
    """
    try:
        async for session in get_database().session():
            repo = QueryLogRepository(session, tenant_id=tenant_id)
            query_log = await repo.get(query_id)

            if query_log is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Query {query_id} not found"
                )

            return QueryDetailResponse(
                query_id=query_log.id,
                query=query_log.query,
                answer=query_log.answer,
                sources=query_log.sources_used or [],
                latency_ms=query_log.latency_ms,
                tokens_used=query_log.tokens_used,
                confidence=query_log.confidence,
                rating=query_log.rating,
                feedback=query_log.feedback,
                created_at=query_log.created_at.isoformat(),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retrieve query", query_id=query_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve query: {str(e)}"
        )


class QueryTraceResponse(BaseModel):
    """Response schema for query trace endpoint."""
    query_id: str
    trace: dict[str, Any]


@router.get("/query/{query_id}/trace", response_model=QueryTraceResponse)
async def get_query_trace(
    query_id: str,
    tenant_id: str = Depends(get_tenant_id),
):
    """
    Get the execution trace for a query.

    Returns detailed information about how the query was processed,
    including agent interactions and timing.
    """
    try:
        async for session in get_database().session():
            repo = QueryLogRepository(session, tenant_id=tenant_id)
            query_log = await repo.get(query_id)

            if query_log is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Query {query_id} not found"
                )

            if not query_log.trace_data:
                raise HTTPException(
                    status_code=404,
                    detail=f"Trace for query {query_id} not found"
                )

            return QueryTraceResponse(
                query_id=query_log.id,
                trace=query_log.trace_data,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retrieve trace", query_id=query_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve trace: {str(e)}"
        )


# =============================================================================
# Multi-Agent Endpoints
# =============================================================================

class AgentQueryRequest(BaseModel):
    """Request schema for multi-agent query endpoint."""
    query: str = Field(..., min_length=1, max_length=10000, description="The user's question")
    conversation_history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Previous conversation turns"
    )
    output_format: Optional[str] = Field("markdown", description="Output format: markdown, plain_text, html, json")
    max_iterations: Optional[int] = Field(3, ge=1, le=5, description="Max refinement iterations")
    enable_verification: bool = Field(True, description="Enable fact verification")
    enable_critic: bool = Field(True, description="Enable quality evaluation")


class AgentQueryResponse(BaseModel):
    """Response schema for multi-agent query endpoint."""
    response: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    trace_id: str
    iterations: int
    latency_ms: float
    trace: Optional[dict[str, Any]] = None


@router.post("/query/agent", response_model=AgentQueryResponse)
async def agent_query(
    request: AgentQueryRequest,
    tenant_id: str = Depends(get_tenant_id),
):
    """
    Execute a query using the multi-agent RAG system.

    This endpoint processes queries through a sophisticated multi-agent pipeline:
    1. **Planner**: Analyzes query complexity and creates execution plan
    2. **Retriever**: Retrieves relevant context using the planned strategy
    3. **Synthesizer**: Generates response from context
    4. **Verifier**: Fact-checks claims against sources (optional)
    5. **Critic**: Evaluates response quality and decides on refinement (optional)
    6. **Citation**: Links response to source documents
    7. **Formatter**: Formats output in requested format

    The system iterates through synthesis/verification/critic until
    quality thresholds are met or max iterations reached.
    """
    try:
        # Parse output format
        output_format = OutputFormat.MARKDOWN
        if request.output_format:
            try:
                output_format = OutputFormat(request.output_format.lower())
            except ValueError:
                pass

        # Configure orchestrator
        config = OrchestratorConfig(
            max_iterations=request.max_iterations or 3,
            enable_verification=request.enable_verification,
            enable_critic=request.enable_critic,
            output_format=output_format,
        )

        orchestrator = Orchestrator(config, tenant_id=tenant_id)

        # Execute query
        result = await orchestrator.execute(
            query=request.query,
            conversation_history=request.conversation_history,
        )

        # Persist query to database with trace data
        try:
            async for session in get_database().session():
                repo = QueryLogRepository(session, tenant_id=tenant_id)
                await repo.create(
                    query=request.query,
                    answer=result["response"],
                    latency_ms=result["metadata"]["latency_ms"],
                    tokens_used=0,  # TODO: track tokens in orchestrator
                    trace_data=result.get("trace", {}),
                    sources_used=result.get("citations", []),
                )
                await session.commit()
        except Exception as persist_error:
            logger.warning("Failed to persist agent query", error=str(persist_error), tenant_id=tenant_id)

        return AgentQueryResponse(
            response=result["response"],
            citations=result.get("citations", []),
            trace_id=result["metadata"]["trace_id"],
            iterations=result["metadata"]["iterations"],
            latency_ms=result["metadata"]["latency_ms"],
            trace=result.get("trace"),
        )

    except Exception as e:
        logger.error("Agent query failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Agent query processing failed: {str(e)}"
        )


@router.post("/query/agent/stream")
async def agent_query_stream(
    request: AgentQueryRequest,
    tenant_id: str = Depends(get_tenant_id),
):
    """
    Execute a multi-agent query with streaming response.

    Returns a Server-Sent Events stream with:
    - Status updates as each agent processes
    - Response content chunks as they are generated
    - Final done event with trace ID
    """
    async def generate():
        try:
            orchestrator = StreamingOrchestrator(tenant_id=tenant_id)

            async for event in orchestrator.execute_streaming(
                query=request.query,
                conversation_history=request.conversation_history,
            ):
                yield f"data: {json.dumps(event)}\n\n"

        except Exception as e:
            logger.error("Streaming agent query failed", error=str(e), tenant_id=tenant_id)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
