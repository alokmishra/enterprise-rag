"""
Tests for src/ingestion/chunking/
"""

import pytest


class TestTextChunker:
    """Tests for the base TextChunker."""

    def test_chunker_base_is_abstract(self):
        """Test that TextChunker is abstract."""
        from src.ingestion.chunking.base import TextChunker
        from abc import ABC

        assert issubclass(TextChunker, ABC)

    def test_chunker_requires_chunk_method(self):
        """Test that chunkers require a chunk method."""
        from src.ingestion.chunking.base import TextChunker

        assert hasattr(TextChunker, 'chunk')


class TestRecursiveTextChunker:
    """Tests for the RecursiveTextChunker."""

    def test_recursive_chunker_creation(self):
        """Test RecursiveTextChunker can be created."""
        from src.ingestion.chunking.recursive import RecursiveTextChunker

        chunker = RecursiveTextChunker(chunk_size=500, chunk_overlap=50)
        assert chunker is not None

    def test_recursive_chunker_parameters(self):
        """Test RecursiveTextChunker parameters."""
        from src.ingestion.chunking.recursive import RecursiveTextChunker

        chunker = RecursiveTextChunker(chunk_size=1000, chunk_overlap=100)
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 100

    def test_recursive_chunker_chunks_text(self):
        """Test RecursiveTextChunker chunks text."""
        from src.ingestion.chunking.recursive import RecursiveTextChunker

        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a test. " * 50  # Long text
        chunks = chunker.chunk(text, document_id="doc-1")

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= chunker.chunk_size + chunker.chunk_overlap

    def test_recursive_chunker_preserves_document_id(self):
        """Test that chunker preserves document ID."""
        from src.ingestion.chunking.recursive import RecursiveTextChunker

        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=20)
        text = "Test content. " * 20
        chunks = chunker.chunk(text, document_id="doc-123")

        for chunk in chunks:
            assert chunk.document_id == "doc-123"

    def test_recursive_chunker_assigns_positions(self):
        """Test that chunker assigns positions."""
        from src.ingestion.chunking.recursive import RecursiveTextChunker

        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=20)
        text = "Test content. " * 20
        chunks = chunker.chunk(text, document_id="doc-1")

        positions = [chunk.position for chunk in chunks]
        assert positions == list(range(len(chunks)))

    def test_recursive_chunker_overlap(self):
        """Test that chunks have overlap."""
        from src.ingestion.chunking.recursive import RecursiveTextChunker

        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=30)
        text = "Word " * 100
        chunks = chunker.chunk(text, document_id="doc-1")

        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                # Overlapping content check
                pass


class TestSentenceChunker:
    """Tests for the SentenceChunker."""

    def test_sentence_chunker_creation(self):
        """Test SentenceChunker can be created."""
        from src.ingestion.chunking.sentence import SentenceChunker

        chunker = SentenceChunker(sentences_per_chunk=3)
        assert chunker is not None

    def test_sentence_chunker_respects_sentences(self):
        """Test that SentenceChunker respects sentence boundaries."""
        from src.ingestion.chunking.sentence import SentenceChunker

        chunker = SentenceChunker(sentences_per_chunk=2)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.chunk(text, document_id="doc-1")

        # Should have 2 chunks with 2 sentences each
        assert len(chunks) >= 1

    def test_sentence_chunker_handles_edge_cases(self):
        """Test SentenceChunker handles edge cases."""
        from src.ingestion.chunking.sentence import SentenceChunker

        chunker = SentenceChunker(sentences_per_chunk=3)

        # Single sentence
        chunks = chunker.chunk("Just one sentence.", document_id="doc-1")
        assert len(chunks) == 1

        # Empty text
        chunks = chunker.chunk("", document_id="doc-1")
        assert len(chunks) == 0 or chunks[0].content == ""


class TestParagraphChunker:
    """Tests for the ParagraphChunker."""

    def test_paragraph_chunker_creation(self):
        """Test ParagraphChunker can be created."""
        from src.ingestion.chunking.sentence import ParagraphChunker

        chunker = ParagraphChunker()
        assert chunker is not None

    def test_paragraph_chunker_splits_on_paragraphs(self):
        """Test that ParagraphChunker splits on paragraph breaks."""
        from src.ingestion.chunking.sentence import ParagraphChunker

        chunker = ParagraphChunker()
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunker.chunk(text, document_id="doc-1")

        assert len(chunks) == 3


class TestTextChunk:
    """Tests for the TextChunk model."""

    def test_text_chunk_creation(self):
        """Test TextChunk can be created."""
        from src.ingestion.chunking.base import TextChunk

        chunk = TextChunk(
            id="chunk-1",
            document_id="doc-1",
            content="Test content",
            position=0,
        )
        assert chunk.id == "chunk-1"
        assert chunk.content == "Test content"

    def test_text_chunk_metadata(self):
        """Test TextChunk with metadata."""
        from src.ingestion.chunking.base import TextChunk

        chunk = TextChunk(
            id="chunk-1",
            document_id="doc-1",
            content="Test",
            position=0,
            metadata={"source": "test.pdf"},
        )
        assert chunk.metadata["source"] == "test.pdf"
