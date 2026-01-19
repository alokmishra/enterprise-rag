"""
Enterprise RAG System - Health Check Routes
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.storage import get_database, get_vector_store, get_cache


router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str


class DetailedHealthResponse(BaseModel):
    """Detailed health check response."""
    status: str
    timestamp: str
    version: str
    components: dict[str, dict[str, Any]]


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="0.1.0"
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """
    Detailed health check with component status.

    Checks connectivity to all dependent services.
    """
    components: dict[str, dict[str, Any]] = {}

    # Check database
    try:
        db = get_database()
        if db.is_connected:
            components["database"] = await db.health_check()
        else:
            components["database"] = {"status": "disconnected", "latency_ms": 0}
    except Exception as e:
        components["database"] = {"status": "error", "error": str(e), "latency_ms": 0}

    # Check vector store
    try:
        vector_store = get_vector_store()
        if vector_store.is_connected:
            components["vector_store"] = await vector_store.health_check()
        else:
            components["vector_store"] = {"status": "disconnected", "latency_ms": 0}
    except Exception as e:
        components["vector_store"] = {"status": "error", "error": str(e), "latency_ms": 0}

    # Check cache
    try:
        cache = get_cache()
        if cache.is_connected:
            components["cache"] = await cache.health_check()
        else:
            components["cache"] = {"status": "disconnected", "latency_ms": 0}
    except Exception as e:
        components["cache"] = {"status": "error", "error": str(e), "latency_ms": 0}

    # Determine overall status
    overall_status = "healthy" if all(
        c.get("status") == "healthy" for c in components.values()
    ) else "degraded"

    # Mark as unhealthy if database is down (critical)
    if components.get("database", {}).get("status") != "healthy":
        overall_status = "unhealthy"

    return DetailedHealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        version="0.1.0",
        components=components
    )


@router.get("/ready")
async def readiness_check():
    """
    Kubernetes readiness probe.

    Returns 200 if all critical services are connected, 503 otherwise.
    """
    checks = {}
    all_ready = True

    # Check database (critical)
    try:
        db = get_database()
        if db.is_connected:
            checks["database"] = "connected"
        else:
            checks["database"] = "disconnected"
            all_ready = False
    except Exception as e:
        checks["database"] = f"error: {str(e)}"
        all_ready = False

    # Check vector store (critical)
    try:
        vector_store = get_vector_store()
        if vector_store.is_connected:
            checks["vector_store"] = "connected"
        else:
            checks["vector_store"] = "disconnected"
            all_ready = False
    except Exception as e:
        checks["vector_store"] = f"error: {str(e)}"
        all_ready = False

    # Check cache (optional - degraded mode OK)
    try:
        cache = get_cache()
        if cache.is_connected:
            checks["cache"] = "connected"
        else:
            checks["cache"] = "disconnected"
    except Exception as e:
        checks["cache"] = f"error: {str(e)}"

    response_data = {"ready": all_ready, "checks": checks}
    status_code = 200 if all_ready else 503

    return JSONResponse(content=response_data, status_code=status_code)


@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe."""
    return {"alive": True}
