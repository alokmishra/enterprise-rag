"""
Enterprise RAG System - Ingestion Module

This module handles document ingestion:
- Document processing (PDF, DOCX, HTML, etc.)
- Text chunking
- Embedding generation
"""

from __future__ import annotations

from src.ingestion.processors import (
    DocumentProcessor,
    ProcessedDocument,
    ProcessorRegistry,
    get_processor_registry,
)
from src.ingestion.chunking import (
    TextChunk,
    TextChunker,
    RecursiveTextChunker,
    MarkdownChunker,
    SentenceChunker,
    ParagraphChunker,
)
from src.ingestion.embeddings import (
    EmbeddingProvider,
    EmbeddingResult,
    OpenAIEmbeddings,
    get_embedding_provider,
)

__all__ = [
    # Processors
    "DocumentProcessor",
    "ProcessedDocument",
    "ProcessorRegistry",
    "get_processor_registry",
    # Chunking
    "TextChunk",
    "TextChunker",
    "RecursiveTextChunker",
    "MarkdownChunker",
    "SentenceChunker",
    "ParagraphChunker",
    # Embeddings
    "EmbeddingProvider",
    "EmbeddingResult",
    "OpenAIEmbeddings",
    "get_embedding_provider",
]
