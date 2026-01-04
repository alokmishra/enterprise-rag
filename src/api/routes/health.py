"""
Enterprise RAG System - Health Check Routes
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter
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
    """Kubernetes readiness probe."""
    # Check if critical services are connected
    db = get_database()

    if not db.is_connected:
        return {"ready": False, "reason": "database not connected"}

    return {"ready": True}


@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe."""
    return {"alive": True}
