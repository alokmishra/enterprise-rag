"""Base classes for multi-modal processing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union
from pathlib import Path
import hashlib
from datetime import datetime


class ModalityType(str, Enum):
    """Supported modality types."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"  # PDF, DOCX, etc.


@dataclass
class MultiModalContent:
    """Represents multi-modal content with extracted information."""

    id: str
    modality: ModalityType
    source_path: Optional[str] = None
    source_url: Optional[str] = None

    # Extracted text content
    text_content: str = ""

    # Embeddings
    text_embedding: Optional[list[float]] = None
    visual_embedding: Optional[list[float]] = None
    audio_embedding: Optional[list[float]] = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Processing info
    processed_at: Optional[datetime] = None
    processing_time_ms: Optional[float] = None

    # Content hash for deduplication
    content_hash: Optional[str] = None

    # Original file info
    file_size_bytes: Optional[int] = None
    mime_type: Optional[str] = None

    # For images
    width: Optional[int] = None
    height: Optional[int] = None

    # For audio
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None

    # Visual analysis results (for images)
    detected_objects: list[dict] = field(default_factory=list)
    detected_text_regions: list[dict] = field(default_factory=list)
    image_description: Optional[str] = None

    # Audio analysis results
    transcription: Optional[str] = None
    speaker_segments: list[dict] = field(default_factory=list)

    def compute_hash(self, content: bytes) -> str:
        """Compute content hash for deduplication."""
        self.content_hash = hashlib.sha256(content).hexdigest()
        return self.content_hash


class MultiModalProcessor(ABC):
    """Abstract base class for multi-modal processors."""

    def __init__(self, config: Optional[dict] = None):
        """Initialize processor with optional configuration."""
        self.config = config or {}
        self._initialized = False

    @property
    @abstractmethod
    def supported_modalities(self) -> list[ModalityType]:
        """Return list of supported modalities."""
        pass

    @property
    @abstractmethod
    def supported_formats(self) -> list[str]:
        """Return list of supported file formats/extensions."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize processor resources."""
        pass

    @abstractmethod
    async def process(
        self,
        content: Union[bytes, str, Path],
        modality: Optional[ModalityType] = None,
        **kwargs,
    ) -> MultiModalContent:
        """Process content and extract information."""
        pass

    @abstractmethod
    async def extract_text(
        self,
        content: Union[bytes, str, Path],
        **kwargs,
    ) -> str:
        """Extract text from content."""
        pass

    @abstractmethod
    async def generate_embedding(
        self,
        content: Union[bytes, str, Path],
        **kwargs,
    ) -> list[float]:
        """Generate embedding for content."""
        pass

    async def cleanup(self) -> None:
        """Cleanup processor resources."""
        self._initialized = False

    def detect_modality(self, file_path: Union[str, Path]) -> ModalityType:
        """Detect modality from file extension."""
        path = Path(file_path)
        ext = path.suffix.lower()

        image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg"}
        audio_exts = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
        doc_exts = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"}

        if ext in image_exts:
            return ModalityType.IMAGE
        elif ext in audio_exts:
            return ModalityType.AUDIO
        elif ext in video_exts:
            return ModalityType.VIDEO
        elif ext in doc_exts:
            return ModalityType.DOCUMENT
        else:
            return ModalityType.TEXT

    def get_mime_type(self, file_path: Union[str, Path]) -> str:
        """Get MIME type from file extension."""
        import mimetypes
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"
