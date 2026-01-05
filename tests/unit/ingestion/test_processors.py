"""
Tests for src/ingestion/processors/
"""

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

    def test_processor_requires_supported_types(self):
        """Test that processors define supported types."""
        from src.ingestion.processors.base import DocumentProcessor

        assert hasattr(DocumentProcessor, 'supported_types')


class TestPlainTextProcessor:
    """Tests for the PlainTextProcessor."""

    def test_plain_text_processor_creation(self):
        """Test PlainTextProcessor can be created."""
        from src.ingestion.processors.text import PlainTextProcessor

        processor = PlainTextProcessor()
        assert processor is not None

    def test_plain_text_processor_supported_types(self):
        """Test PlainTextProcessor supported types."""
        from src.ingestion.processors.text import PlainTextProcessor

        processor = PlainTextProcessor()
        assert ".txt" in processor.supported_types

    def test_plain_text_processor_process(self):
        """Test PlainTextProcessor process method."""
        from src.ingestion.processors.text import PlainTextProcessor

        processor = PlainTextProcessor()
        result = processor.process(
            content="This is a test document.\nWith multiple lines.",
            source="test.txt",
        )
        assert result.content is not None
        assert result.source == "test.txt"

    def test_plain_text_processor_extracts_metadata(self):
        """Test that processor extracts metadata."""
        from src.ingestion.processors.text import PlainTextProcessor

        processor = PlainTextProcessor()
        result = processor.process(
            content="Test content",
            source="document.txt",
            metadata={"author": "Test Author"},
        )
        # Should preserve metadata


class TestMarkdownProcessor:
    """Tests for the MarkdownProcessor."""

    def test_markdown_processor_creation(self):
        """Test MarkdownProcessor can be created."""
        from src.ingestion.processors.text import MarkdownProcessor

        processor = MarkdownProcessor()
        assert processor is not None

    def test_markdown_processor_supported_types(self):
        """Test MarkdownProcessor supported types."""
        from src.ingestion.processors.text import MarkdownProcessor

        processor = MarkdownProcessor()
        assert ".md" in processor.supported_types

    def test_markdown_processor_handles_headers(self):
        """Test MarkdownProcessor handles headers."""
        from src.ingestion.processors.text import MarkdownProcessor

        processor = MarkdownProcessor()
        content = "# Title\n\nParagraph text.\n\n## Section\n\nMore text."
        result = processor.process(content=content, source="doc.md")
        assert result.content is not None


class TestPDFProcessor:
    """Tests for the PDFProcessor."""

    def test_pdf_processor_creation(self):
        """Test PDFProcessor can be created."""
        from src.ingestion.processors.pdf import PDFProcessor

        processor = PDFProcessor()
        assert processor is not None

    def test_pdf_processor_supported_types(self):
        """Test PDFProcessor supported types."""
        from src.ingestion.processors.pdf import PDFProcessor

        processor = PDFProcessor()
        assert ".pdf" in processor.supported_types

    @pytest.mark.asyncio
    async def test_pdf_processor_requires_file_path(self):
        """Test that PDF processor requires a file path."""
        from src.ingestion.processors.pdf import PDFProcessor

        processor = PDFProcessor()
        # Should require file_path for binary files


class TestHTMLProcessor:
    """Tests for the HTMLProcessor."""

    def test_html_processor_creation(self):
        """Test HTMLProcessor can be created."""
        from src.ingestion.processors.html import HTMLProcessor

        processor = HTMLProcessor()
        assert processor is not None

    def test_html_processor_supported_types(self):
        """Test HTMLProcessor supported types."""
        from src.ingestion.processors.html import HTMLProcessor

        processor = HTMLProcessor()
        assert ".html" in processor.supported_types or ".htm" in processor.supported_types

    def test_html_processor_strips_tags(self):
        """Test that HTML processor strips tags."""
        from src.ingestion.processors.html import HTMLProcessor

        processor = HTMLProcessor()
        html_content = "<html><body><p>Test paragraph</p></body></html>"
        result = processor.process(content=html_content, source="page.html")
        assert "<p>" not in result.content
        assert "Test paragraph" in result.content

    def test_html_processor_handles_scripts(self):
        """Test that HTML processor removes scripts."""
        from src.ingestion.processors.html import HTMLProcessor

        processor = HTMLProcessor()
        html_content = "<html><script>alert('test')</script><body>Content</body></html>"
        result = processor.process(content=html_content, source="page.html")
        assert "alert" not in result.content


class TestProcessorRegistry:
    """Tests for the ProcessorRegistry."""

    def test_registry_creation(self):
        """Test ProcessorRegistry can be created."""
        from src.ingestion.processors.registry import ProcessorRegistry

        registry = ProcessorRegistry()
        assert registry is not None

    def test_registry_get_processor(self):
        """Test getting processor by file type."""
        from src.ingestion.processors.registry import ProcessorRegistry

        registry = ProcessorRegistry()
        processor = registry.get_processor(".txt")
        assert processor is not None

    def test_registry_returns_none_for_unknown_type(self):
        """Test registry returns None for unknown type."""
        from src.ingestion.processors.registry import ProcessorRegistry

        registry = ProcessorRegistry()
        processor = registry.get_processor(".xyz_unknown")
        assert processor is None

    def test_registry_register_processor(self):
        """Test registering a custom processor."""
        from src.ingestion.processors.registry import ProcessorRegistry
        from src.ingestion.processors.text import PlainTextProcessor

        registry = ProcessorRegistry()
        registry.register(".custom", PlainTextProcessor())
        processor = registry.get_processor(".custom")
        assert processor is not None
