"""
Enterprise RAG System - Document Processor Registry
"""

from pathlib import Path
from typing import Any, BinaryIO, Optional, Type, Union

from src.core.exceptions import UnsupportedFileTypeError
from src.core.logging import LoggerMixin
from src.ingestion.processors.base import DocumentProcessor, ProcessedDocument
from src.ingestion.processors.text import (
    PlainTextProcessor,
    MarkdownProcessor,
    CodeProcessor,
)
from src.ingestion.processors.pdf import PDFProcessor
from src.ingestion.processors.docx import DocxProcessor
from src.ingestion.processors.html import HTMLProcessor


class ProcessorRegistry(LoggerMixin):
    """Registry for document processors."""

    def __init__(self):
        self._processors: list[Type[DocumentProcessor]] = []
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default processors."""
        self.register(PDFProcessor)
        self.register(DocxProcessor)
        self.register(HTMLProcessor)
        self.register(MarkdownProcessor)
        self.register(CodeProcessor)
        self.register(PlainTextProcessor)  # Last as fallback for text

    def register(self, processor_class: Type[DocumentProcessor]) -> None:
        """Register a new processor."""
        self._processors.append(processor_class)
        self.logger.debug(
            "Registered processor",
            processor=processor_class.__name__,
            extensions=processor_class.supported_extensions,
        )

    def get_processor(
        self,
        extension: str,
        mimetype: Optional[str] = None,
    ) -> DocumentProcessor:
        """Get a processor for the given file type."""
        ext = extension.lower().lstrip(".")

        for processor_class in self._processors:
            if processor_class.can_process(ext, mimetype):
                return processor_class()

        raise UnsupportedFileTypeError(extension)

    async def process(
        self,
        source: Union[str, Path, BinaryIO],
        extension: Optional[str] = None,
        mimetype: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ProcessedDocument:
        """
        Process a document using the appropriate processor.

        Args:
            source: File path or file-like object
            extension: File extension (auto-detected from path if not provided)
            mimetype: MIME type of the file
            metadata: Additional metadata to include

        Returns:
            ProcessedDocument with extracted content
        """
        # Auto-detect extension from path
        if extension is None and isinstance(source, (str, Path)):
            extension = Path(source).suffix

        if extension is None:
            raise UnsupportedFileTypeError("unknown")

        processor = self.get_processor(extension, mimetype)
        self.logger.debug(
            "Processing document",
            processor=processor.__class__.__name__,
            extension=extension,
        )

        result = await processor.process(source, metadata)

        self.logger.info(
            "Document processed",
            processor=processor.__class__.__name__,
            content_length=len(result.content),
            title=result.title,
        )

        return result

    def supported_extensions(self) -> list[str]:
        """Get list of all supported file extensions."""
        extensions = set()
        for processor_class in self._processors:
            extensions.update(processor_class.supported_extensions)
        return sorted(extensions)


# Global registry instance
_registry: Optional[ProcessorRegistry] = None


def get_processor_registry() -> ProcessorRegistry:
    """Get the global processor registry."""
    global _registry
    if _registry is None:
        _registry = ProcessorRegistry()
    return _registry
