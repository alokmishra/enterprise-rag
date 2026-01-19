"""Tests for multi-modal document processing."""

from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.multimodal.base import ModalityType, MultiModalContent
from src.ingestion.multimodal.document import (
    DocumentPage,
    DocumentStructure,
    MultiModalDocumentProcessor,
    PDFProcessor,
)


class TestPDFProcessor:
    """Tests for PDF processor."""

    @pytest.fixture
    def pdf_processor(self):
        """Create PDF processor."""
        return PDFProcessor(
            extract_images=True,
            extract_tables=True,
            ocr_scanned_pages=False,
        )

    @pytest.fixture
    def sample_pdf_bytes(self):
        """Create minimal PDF bytes."""
        # Minimal valid PDF
        pdf_content = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj
4 0 obj << /Length 44 >> stream
BT /F1 12 Tf 100 700 Td (Hello World) Tj ET
endstream endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000214 00000 n
trailer << /Size 5 /Root 1 0 R >>
startxref
306
%%EOF"""
        return pdf_content

    @pytest.mark.asyncio
    async def test_initialize(self, pdf_processor):
        """Test PDF processor initialization."""
        await pdf_processor.initialize()
        assert pdf_processor._initialized

    @pytest.mark.asyncio
    async def test_process_returns_pages_and_structure(self, pdf_processor):
        """Test PDF processing returns pages and structure."""
        with patch.object(pdf_processor, "_process_with_pdfplumber") as mock_fallback:
            mock_fallback.return_value = (
                [DocumentPage(page_number=1, text_content="Page content")],
                DocumentStructure(title="Test PDF", author="Test Author"),
            )

            pdf_processor._initialized = True
            pages, structure = await pdf_processor.process(b"pdf_bytes")

            assert isinstance(pages, list)
            assert isinstance(structure, DocumentStructure)
            assert structure.title == "Test PDF"

    def test_document_page_structure(self):
        """Test DocumentPage structure."""
        page = DocumentPage(
            page_number=1,
            text_content="Sample text",
            images=[],
            tables=[{"headers": ["A", "B"], "rows": [["1", "2"]]}],
        )

        assert page.page_number == 1
        assert page.text_content == "Sample text"
        assert len(page.tables) == 1

    def test_document_structure(self):
        """Test DocumentStructure."""
        structure = DocumentStructure(
            title="Test Document",
            author="Test Author",
            table_of_contents=[
                {"level": 1, "title": "Introduction", "page": 1}
            ],
            metadata={"keywords": "test, document"},
        )

        assert structure.title == "Test Document"
        assert len(structure.table_of_contents) == 1


class TestMultiModalDocumentProcessor:
    """Tests for multi-modal document processor."""

    @pytest.fixture
    def processor(self):
        """Create multi-modal document processor."""
        return MultiModalDocumentProcessor()

    def test_supported_modalities(self, processor):
        """Test supported modalities."""
        modalities = processor.supported_modalities
        assert ModalityType.TEXT in modalities
        assert ModalityType.IMAGE in modalities
        assert ModalityType.AUDIO in modalities
        assert ModalityType.DOCUMENT in modalities

    def test_supported_formats(self, processor):
        """Test supported formats."""
        formats = processor.supported_formats
        assert ".txt" in formats
        assert ".pdf" in formats
        assert ".jpg" in formats
        assert ".mp3" in formats

    def test_detect_modality(self, processor):
        """Test modality detection."""
        assert processor.detect_modality("doc.pdf") == ModalityType.DOCUMENT
        assert processor.detect_modality("image.jpg") == ModalityType.IMAGE
        assert processor.detect_modality("audio.mp3") == ModalityType.AUDIO
        assert processor.detect_modality("text.txt") == ModalityType.TEXT

    @pytest.mark.asyncio
    async def test_process_text(self, processor):
        """Test processing text content."""
        processor._initialized = True

        result = await processor._process_text("Sample text content")

        assert result.modality == ModalityType.TEXT
        assert result.text_content == "Sample text content"

    @pytest.mark.asyncio
    async def test_process_text_file(self, processor, tmp_path):
        """Test processing text file."""
        text_file = tmp_path / "test.txt"
        text_file.write_text("File content here")

        processor._initialized = True
        result = await processor._process_text(text_file)

        assert result.text_content == "File content here"
        assert result.source_path == str(text_file)

    @pytest.mark.asyncio
    async def test_process_routes_to_image_processor(self, processor):
        """Test processing routes image to image processor."""
        with patch.object(processor.image_processor, "process") as mock_process:
            mock_process.return_value = MultiModalContent(
                id="img123",
                modality=ModalityType.IMAGE,
                text_content="OCR text",
            )
            processor._initialized = True

            result = await processor.process(
                b"image_bytes",
                modality=ModalityType.IMAGE,
            )

            mock_process.assert_called_once()
            assert result.modality == ModalityType.IMAGE

    @pytest.mark.asyncio
    async def test_process_routes_to_audio_processor(self, processor):
        """Test processing routes audio to audio processor."""
        with patch.object(processor.audio_processor, "process") as mock_process:
            mock_process.return_value = MultiModalContent(
                id="audio123",
                modality=ModalityType.AUDIO,
                transcription="Transcribed text",
            )
            processor._initialized = True

            result = await processor.process(
                b"audio_bytes",
                modality=ModalityType.AUDIO,
            )

            mock_process.assert_called_once()
            assert result.modality == ModalityType.AUDIO

    @pytest.mark.asyncio
    async def test_process_document(self, processor):
        """Test processing document."""
        with patch.object(processor.pdf_processor, "process") as mock_pdf:
            mock_pdf.return_value = (
                [DocumentPage(page_number=1, text_content="PDF text")],
                DocumentStructure(title="Test PDF"),
            )
            processor._initialized = True

            result = await processor._process_document(b"pdf_bytes")

            assert result.modality == ModalityType.DOCUMENT
            assert "PDF text" in result.text_content

    @pytest.mark.asyncio
    async def test_process_docx(self, processor):
        """Test processing DOCX file."""
        with patch("docx.Document") as mock_doc:
            mock_doc_instance = MagicMock()
            mock_doc_instance.paragraphs = [
                MagicMock(text="Paragraph 1"),
                MagicMock(text="Paragraph 2"),
            ]
            mock_doc_instance.tables = []
            mock_doc.return_value = mock_doc_instance

            processor._initialized = True
            result = MultiModalContent(
                id="test",
                modality=ModalityType.DOCUMENT,
            )
            result = await processor._process_docx(b"docx_bytes", result)

            assert "Paragraph 1" in result.text_content

    @pytest.mark.asyncio
    async def test_extract_text(self, processor):
        """Test text extraction from any content."""
        with patch.object(processor, "process") as mock_process:
            mock_process.return_value = MultiModalContent(
                id="test",
                modality=ModalityType.TEXT,
                text_content="Extracted text",
            )
            processor._initialized = True

            text = await processor.extract_text("content")

            assert text == "Extracted text"

    def test_format_table_as_text(self, processor):
        """Test table formatting."""
        table = {
            "headers": ["Name", "Age"],
            "rows": [["Alice", "30"], ["Bob", "25"]],
        }

        text = processor._format_table_as_text(table)

        assert "Name" in text
        assert "Alice" in text
        assert "|" in text
