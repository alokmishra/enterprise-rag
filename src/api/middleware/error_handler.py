"""
Enterprise RAG System - Error Handler Middleware
"""

from __future__ import annotations

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from src.core.exceptions import RAGException
from src.core.logging import get_logger

logger = get_logger(__name__)


async def error_handler_middleware(request: Request, call_next) -> Response:
    """Global error handler middleware."""
    try:
        return await call_next(request)
    except RAGException as e:
        logger.error(
            "RAG exception",
            code=e.code,
            message=e.message,
            details=e.details,
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": e.code,
                "message": e.message,
                "details": e.details,
            }
        )
    except Exception as e:
        logger.exception("Unhandled exception", error=str(e))
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
            }
        )
