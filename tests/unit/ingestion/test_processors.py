"""
Tests for src/ingestion/processors/
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestDocumentProcessor:
    """Tests for the base DocumentProcessor."""

    def test_processor_base_is_abstract(self):
        """Test that DocumentProcessor is abstract."""
        from src.ingestion.processors.base import DocumentProcessor
        from abc import ABC

        assert issubclass(DocumentProcessor, ABC)

    def test_processor_requires_process_method(self):
        """Test that processors require a process method."""
        from src.ingestion.processors.base import DocumentProcessor

        assert hasattr(DocumentProcessor, 'process')

    def test_processor_has_supported_extensions(self):
        """Test that processors define supported_extensions."""
        from src.ingestion.processors.base import DocumentProcessor

        assert hasattr(DocumentProcessor, 'supported_extensions')

    def test_processor_has_can_process(self):
        """Test that processors have can_process classmethod."""
        from src.ingestion.processors.base import DocumentProcessor

        assert hasattr(DocumentProcessor, 'can_process')


class TestPlainTextProcessor:
    """Tests for the PlainTextProcessor."""

    def test_plain_text_processor_creation(self):
        """Test PlainTextProcessor can be created."""
        from src.ingestion.processors.text import PlainTextProcessor

        processor = PlainTextProcessor()
        assert processor is not None

    def test_plain_text_processor_supported_extensions(self):
        """Test PlainTextProcessor supported extensions."""
        from src.ingestion.processors.text import PlainTextProcessor

        processor = PlainTextProcessor()
        assert ".txt" in processor.supported_extensions

    @pytest.mark.asyncio
    async def test_plain_text_processor_process_file(self):
        """Test PlainTextProcessor process method with file."""
        from src.ingestion.processors.text import PlainTextProcessor

        processor = PlainTextProcessor()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\nWith multiple lines.")
            f.flush()

            result = await processor.process(f.name)
            assert result.content is not None
            assert "test document" in result.content

    @pytest.mark.asyncio
    async def test_plain_text_processor_process_binary_io(self):
        """Test PlainTextProcessor process with BinaryIO."""
        from src.ingestion.processors.text import PlainTextProcessor

        processor = PlainTextProcessor()
        content = b"Test content from binary"
        source = io.BytesIO(content)

        result = await processor.process(source)
        assert result.content == "Test content from binary"

    @pytest.mark.asyncio
    async def test_plain_text_processor_extracts_metadata(self):
        """Test that processor includes provided metadata."""
        from src.ingestion.processors.text import PlainTextProcessor

        processor = PlainTextProcessor()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            f.flush()

            result = await processor.process(
                f.name,
                metadata={"author": "Test Author"},
            )
            assert result.metadata.get("author") == "Test Author"


class TestMarkdownProcessor:
    """Tests for the MarkdownProcessor."""

    def test_markdown_processor_creation(self):
        """Test MarkdownProcessor can be created."""
        from src.ingestion.processors.text import MarkdownProcessor

        processor = MarkdownProcessor()
        assert processor is not None

    def test_markdown_processor_supported_extensions(self):
        """Test MarkdownProcessor supported extensions."""
        from src.ingestion.processors.text import MarkdownProcessor

        processor = MarkdownProcessor()
        assert ".md" in processor.supported_extensions

    @pytest.mark.asyncio
    async def test_markdown_processor_extracts_title(self):
        """Test MarkdownProcessor extracts title from heading."""
        from src.ingestion.processors.text import MarkdownProcessor

        processor = MarkdownProcessor()
        content = b"# My Title\n\nParagraph text.\n\n## Section\n\nMore text."
        source = io.BytesIO(content)

        result = await processor.process(source)
        assert result.title == "My Title"


class TestPDFProcessor:
    """Tests for the PDFProcessor."""

    def test_pdf_processor_creation(self):
        """Test PDFProcessor can be created."""
        from src.ingestion.processors.pdf import PDFProcessor

        processor = PDFProcessor()
        assert processor is not None

    def test_pdf_processor_supported_extensions(self):
        """Test PDFProcessor supported extensions."""
        from src.ingestion.processors.pdf import PDFProcessor

        processor = PDFProcessor()
        assert ".pdf" in processor.supported_extensions

    def test_pdf_processor_can_process(self):
        """Test PDFProcessor can_process method."""
        from src.ingestion.processors.pdf import PDFProcessor

        assert PDFProcessor.can_process("pdf")
        assert PDFProcessor.can_process(".pdf")
        assert not PDFProcessor.can_process(".txt")


class TestHTMLProcessor:
    """Tests for the HTMLProcessor."""

    def test_html_processor_creation(self):
        """Test HTMLProcessor can be created."""
        from src.ingestion.processors.html import HTMLProcessor

        processor = HTMLProcessor()
        assert processor is not None

    def test_html_processor_supported_extensions(self):
        """Test HTMLProcessor supported extensions."""
        from src.ingestion.processors.html import HTMLProcessor

        processor = HTMLProcessor()
        assert ".html" in processor.supported_extensions or ".htm" in processor.supported_extensions

    @pytest.mark.asyncio
    async def test_html_processor_strips_tags(self):
        """Test that HTML processor strips tags."""
        from src.ingestion.processors.html import HTMLProcessor

        processor = HTMLProcessor()
        html_content = b"<html><body><p>Test paragraph</p></body></html>"
        source = io.BytesIO(html_content)

        result = await processor.process(source)
        assert "<p>" not in result.content
        assert "Test paragraph" in result.content

    @pytest.mark.asyncio
    async def test_html_processor_removes_scripts(self):
        """Test that HTML processor removes scripts."""
        from src.ingestion.processors.html import HTMLProcessor

        processor = HTMLProcessor()
        html_content = b"<html><script>alert('test')</script><body>Content</body></html>"
        source = io.BytesIO(html_content)

        result = await processor.process(source)
        assert "alert" not in result.content

    @pytest.mark.asyncio
    async def test_html_processor_extracts_title(self):
        """Test that HTML processor extracts title."""
        from src.ingestion.processors.html import HTMLProcessor

        processor = HTMLProcessor()
        html_content = b"<html><head><title>Page Title</title></head><body>Content</body></html>"
        source = io.BytesIO(html_content)

        result = await processor.process(source)
        assert result.title == "Page Title"


class TestProcessorRegistry:
    """Tests for the ProcessorRegistry."""

    def test_registry_creation(self):
        """Test ProcessorRegistry can be created."""
        from src.ingestion.processors.registry import ProcessorRegistry

        registry = ProcessorRegistry()
        assert registry is not None

    def test_registry_get_processor(self):
        """Test getting processor by file extension."""
        from src.ingestion.processors.registry import ProcessorRegistry
        from src.ingestion.processors.text import PlainTextProcessor

        registry = ProcessorRegistry()
        processor = registry.get_processor(".txt")
        assert processor is not None
        assert isinstance(processor, PlainTextProcessor)

    def test_registry_raises_for_unknown_type(self):
        """Test registry raises exception for unknown type."""
        from src.ingestion.processors.registry import ProcessorRegistry
        from src.core.exceptions import UnsupportedFileTypeError

        registry = ProcessorRegistry()
        with pytest.raises(UnsupportedFileTypeError):
            registry.get_processor(".xyz_unknown")

    def test_registry_register_processor(self):
        """Test registering a custom processor."""
        from src.ingestion.processors.registry import ProcessorRegistry
        from src.ingestion.processors.text import PlainTextProcessor

        registry = ProcessorRegistry()
        registry.register(PlainTextProcessor)

        # PlainTextProcessor should still work for .txt
        processor = registry.get_processor(".txt")
        assert processor is not None

    def test_registry_supported_extensions(self):
        """Test getting list of supported extensions."""
        from src.ingestion.processors.registry import ProcessorRegistry

        registry = ProcessorRegistry()
        extensions = registry.supported_extensions()

        assert ".txt" in extensions
        assert ".pdf" in extensions
        assert ".html" in extensions
        assert ".md" in extensions

    @pytest.mark.asyncio
    async def test_registry_process_method(self):
        """Test registry process method."""
        from src.ingestion.processors.registry import ProcessorRegistry

        registry = ProcessorRegistry()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for processing")
            f.flush()

            result = await registry.process(f.name)
            assert result.content is not None
            assert "Test content" in result.content


class TestProcessedDocument:
    """Tests for the ProcessedDocument model."""

    def test_processed_document_creation(self):
        """Test ProcessedDocument can be created."""
        from src.ingestion.processors.base import ProcessedDocument
        from src.core.types import ContentType

        doc = ProcessedDocument(
            content="Test content",
            content_type=ContentType.TEXT,
            metadata={"source": "test.txt"},
        )
        assert doc.content == "Test content"
        assert doc.content_type == ContentType.TEXT

    def test_processed_document_with_title(self):
        """Test ProcessedDocument with title."""
        from src.ingestion.processors.base import ProcessedDocument
        from src.core.types import ContentType

        doc = ProcessedDocument(
            content="Test content",
            content_type=ContentType.TEXT,
            metadata={},
            title="My Document",
        )
        assert doc.title == "My Document"
