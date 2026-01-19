"""
Enterprise RAG System - PDF Document Processor
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

from pypdf import PdfReader

from src.core.types import ContentType
from src.ingestion.processors.base import DocumentProcessor, ProcessedDocument


class PDFProcessor(DocumentProcessor):
    """Processor for PDF documents."""

    supported_extensions = [".pdf"]
    supported_mimetypes = ["application/pdf"]

    async def process(
        self,
        source: Union[str, Path, BinaryIO],
        metadata: Optional[dict[str, Any]] = None,
    ) -> ProcessedDocument:
        """Process a PDF file and extract text content."""
        # Read PDF content
        if isinstance(source, (str, Path)):
            reader = PdfReader(source)
        else:
            content = source.read()
            reader = PdfReader(io.BytesIO(content))

        # Extract text from all pages
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")

        content = "\n\n".join(text_parts)

        # Extract metadata from PDF
        pdf_metadata = {}
        if reader.metadata:
            if reader.metadata.title:
                pdf_metadata["pdf_title"] = reader.metadata.title
            if reader.metadata.author:
                pdf_metadata["pdf_author"] = reader.metadata.author
            if reader.metadata.subject:
                pdf_metadata["pdf_subject"] = reader.metadata.subject
            if reader.metadata.creator:
                pdf_metadata["pdf_creator"] = reader.metadata.creator
            if reader.metadata.creation_date:
                pdf_metadata["pdf_created"] = str(reader.metadata.creation_date)

        # Merge with provided metadata
        combined_metadata = {**pdf_metadata, **(metadata or {})}
        combined_metadata["page_count"] = len(reader.pages)

        # Use PDF title or first line as title
        title = pdf_metadata.get("pdf_title")
        if not title and content:
            first_line = content.split("\n")[0].strip()
            if first_line and not first_line.startswith("---"):
                title = first_line[:100]

        return ProcessedDocument(
            content=content,
            content_type=ContentType.TEXT,
            metadata=combined_metadata,
            title=title,
        )
