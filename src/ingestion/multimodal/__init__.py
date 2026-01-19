# Multi-modal processing for images, audio, and documents
from __future__ import annotations

from src.ingestion.multimodal.image import (
    ImageProcessor,
    OCREngine,
    ImageEmbedder,
    VisualAnalyzer,
)
from src.ingestion.multimodal.audio import (
    AudioProcessor,
    Transcriber,
    AudioEmbedder,
    SpeakerDiarizer,
)
from src.ingestion.multimodal.document import (
    PDFProcessor,
    MultiModalDocumentProcessor,
    DocumentPage,
    DocumentStructure,
)
from src.ingestion.multimodal.base import (
    MultiModalProcessor,
    ModalityType,
    MultiModalContent,
)

__all__ = [
    # Image processing
    "ImageProcessor",
    "OCREngine",
    "ImageEmbedder",
    "VisualAnalyzer",
    # Audio processing
    "AudioProcessor",
    "Transcriber",
    "AudioEmbedder",
    "SpeakerDiarizer",
    # Document processing
    "PDFProcessor",
    "MultiModalDocumentProcessor",
    "DocumentPage",
    "DocumentStructure",
    # Base classes
    "MultiModalProcessor",
    "ModalityType",
    "MultiModalContent",
]
