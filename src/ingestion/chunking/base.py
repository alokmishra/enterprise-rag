"""
Enterprise RAG System - Chunking Base Classes
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

from src.core.logging import LoggerMixin


@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    content: str
    position: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))


class TextChunker(ABC, LoggerMixin):
    """Abstract base class for text chunkers."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[TextChunk]:
        """
        Split text into chunks.

        Args:
            text: The text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of TextChunk objects
        """
        pass

    def _create_chunk(
        self,
        content: str,
        position: int,
        start_char: int,
        end_char: int,
        metadata: Optional[dict] = None,
    ) -> TextChunk:
        """Create a TextChunk with metadata."""
        chunk_metadata = {
            "chunk_size": len(content),
            "chunker": self.__class__.__name__,
            **(metadata or {}),
        }
        return TextChunk(
            content=content,
            position=position,
            start_char=start_char,
            end_char=end_char,
            metadata=chunk_metadata,
        )
