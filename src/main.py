"""
Enterprise RAG System - Main Entry Point
"""

import uvicorn

from src.app import create_app
from src.core.config import settings

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS,
    )
