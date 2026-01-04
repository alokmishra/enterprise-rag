"""
Enterprise RAG System - Application Factory
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.core.logging import setup_logging
from src.api.routes import query, documents, admin, health
from src.api.middleware.error_handler import error_handler_middleware
from src.api.middleware.logging import logging_middleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler for startup and shutdown events."""
    # Startup
    setup_logging()
    
    # Initialize connections
    # await init_database()
    # await init_vector_store()
    # await init_cache()
    
    yield
    
    # Shutdown
    # await close_database()
    # await close_vector_store()
    # await close_cache()


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
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom middleware
    app.middleware("http")(error_handler_middleware)
    app.middleware("http")(logging_middleware)
    
    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(query.router, prefix="/api/v1", tags=["Query"])
    app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
    app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])
    
    return app
