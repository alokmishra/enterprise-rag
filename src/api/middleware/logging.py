"""
Enterprise RAG System - Logging Middleware
"""

import time
from uuid import uuid4

from fastapi import Request, Response

from src.core.logging import get_logger

logger = get_logger(__name__)


async def logging_middleware(request: Request, call_next) -> Response:
    """Request/response logging middleware."""
    request_id = str(uuid4())
    start_time = time.perf_counter()
    
    # Add request ID to state
    request.state.request_id = request_id
    
    # Log request
    logger.info(
        "Request started",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration_ms = (time.perf_counter() - start_time) * 1000
    
    # Log response
    logger.info(
        "Request completed",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
    )
    
    # Add headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
    
    return response
