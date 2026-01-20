"""
Enterprise RAG System - Application Factory
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.core.logging import get_logger, setup_logging
from src.api.routes import query, documents, admin, health
from src.api.middleware.error_handler import error_handler_middleware
from src.api.middleware.logging import logging_middleware
from src.api.middleware.rate_limit import rate_limit_middleware
from src.api.middleware.tenant import tenant_context_middleware
from src.storage import (
    init_database,
    close_database,
    init_vector_store,
    close_vector_store,
    init_cache,
    close_cache,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler for startup and shutdown events."""
    # Startup
    setup_logging()
    logger.info("Starting Enterprise RAG System", env=settings.RAG_ENV)

    # Initialize connections
    try:
        await init_database()
        logger.info("Database initialized")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))

    try:
        await init_vector_store()
        logger.info("Vector store initialized")
    except Exception as e:
        logger.error("Failed to initialize vector store", error=str(e))

    try:
        await init_cache()
        logger.info("Cache initialized")
    except Exception as e:
        logger.error("Failed to initialize cache", error=str(e))

    logger.info("All services initialized")

    yield

    # Shutdown
    logger.info("Shutting down Enterprise RAG System")
    await close_cache()
    await close_vector_store()
    await close_database()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Enterprise RAG System",
        description="State-of-the-art Retrieval-Augmented Generation API",
        version="0.1.0",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
    )
    
    # Add custom middleware (order matters: last added runs first)
    # Execution order: logging -> tenant_context -> rate_limit -> error_handler
    app.middleware("http")(error_handler_middleware)
    app.middleware("http")(rate_limit_middleware)
    app.middleware("http")(tenant_context_middleware)
    app.middleware("http")(logging_middleware)
    
    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(query.router, prefix="/api/v1", tags=["Query"])
    app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
    app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])
    
    return app
