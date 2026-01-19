"""
Enterprise RAG System - Embeddings Module
"""

from __future__ import annotations

from src.ingestion.embeddings.base import EmbeddingProvider, EmbeddingResult
from src.ingestion.embeddings.openai import OpenAIEmbeddings, get_embedding_provider

__all__ = [
    # Base
    "EmbeddingProvider",
    "EmbeddingResult",
    # OpenAI
    "OpenAIEmbeddings",
    "get_embedding_provider",
]
