"""
Enterprise RAG System - Document Processors Module
"""

from __future__ import annotations

from src.ingestion.processors.base import (
    DocumentProcessor,
    ProcessedDocument,
)
from src.ingestion.processors.text import (
    PlainTextProcessor,
    MarkdownProcessor,
    CodeProcessor,
)
from src.ingestion.processors.pdf import PDFProcessor
from src.ingestion.processors.docx import DocxProcessor
from src.ingestion.processors.html import HTMLProcessor
from src.ingestion.processors.registry import (
    ProcessorRegistry,
    get_processor_registry,
)

__all__ = [
    # Base
    "DocumentProcessor",
    "ProcessedDocument",
    # Processors
    "PlainTextProcessor",
    "MarkdownProcessor",
    "CodeProcessor",
    "PDFProcessor",
    "DocxProcessor",
    "HTMLProcessor",
    # Registry
    "ProcessorRegistry",
    "get_processor_registry",
]
