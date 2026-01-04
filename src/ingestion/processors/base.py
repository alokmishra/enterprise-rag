"""
Enterprise RAG System - Document Processor Base Classes
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

from src.core.logging import LoggerMixin
from src.core.types import ContentType


@dataclass
class ProcessedDocument:
    """Result of document processing."""
    content: str
    content_type: ContentType
    metadata: dict[str, Any]
    title: Optional[str] = None
    language: Optional[str] = None


class DocumentProcessor(ABC, LoggerMixin):
    """Abstract base class for document processors."""

    supported_extensions: list[str] = []
    supported_mimetypes: list[str] = []

    @abstractmethod
    async def process(
        self,
        source: Union[str, Path, BinaryIO],
        metadata: Optional[dict[str, Any]] = None,
    ) -> ProcessedDocument:
        """
        Process a document and extract its content.

        Args:
            source: File path or file-like object
            metadata: Optional additional metadata

        Returns:
            ProcessedDocument with extracted content
        """
        pass

    @classmethod
    def can_process(cls, extension: str, mimetype: Optional[str] = None) -> bool:
        """Check if this processor can handle the given file type."""
        ext = extension.lower().lstrip(".")
        if ext in [e.lstrip(".") for e in cls.supported_extensions]:
            return True
        if mimetype and mimetype in cls.supported_mimetypes:
            return True
        return False

    def _read_file(self, source: Union[str, Path, BinaryIO]) -> bytes:
        """Read file content from various source types."""
        if isinstance(source, (str, Path)):
            with open(source, "rb") as f:
                return f.read()
        else:
            return source.read()

    def _read_text(
        self,
        source: Union[str, Path, BinaryIO],
        encoding: str = "utf-8",
    ) -> str:
        """Read text content from various source types."""
        content = self._read_file(source)
        return content.decode(encoding)
