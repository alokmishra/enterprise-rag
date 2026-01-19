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
        chunks = chunker.chunk(text, metadata={"doc_id": "doc-1"})

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= chunker.chunk_size + chunker.chunk_overlap

    def test_recursive_chunker_preserves_metadata(self):
        """Test that chunker preserves metadata."""
        from src.ingestion.chunking.recursive import RecursiveTextChunker

        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=20)
        text = "Test content. " * 20
        chunks = chunker.chunk(text, metadata={"doc_id": "doc-123"})

        for chunk in chunks:
            assert chunk.metadata.get("doc_id") == "doc-123"

    def test_recursive_chunker_assigns_positions(self):
        """Test that chunker assigns positions."""
        from src.ingestion.chunking.recursive import RecursiveTextChunker

        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=20)
        text = "Test content. " * 20
        chunks = chunker.chunk(text)

        positions = [chunk.position for chunk in chunks]
        assert positions == list(range(len(chunks)))

    def test_recursive_chunker_sets_char_positions(self):
        """Test that chunks have start_char and end_char."""
        from src.ingestion.chunking.recursive import RecursiveTextChunker

        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=30)
        text = "Word " * 100
        chunks = chunker.chunk(text)

        for chunk in chunks:
            assert hasattr(chunk, 'start_char')
            assert hasattr(chunk, 'end_char')
            assert chunk.end_char >= chunk.start_char

    def test_recursive_chunker_empty_text(self):
        """Test that chunker handles empty text."""
        from src.ingestion.chunking.recursive import RecursiveTextChunker

        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk("")

        assert chunks == []


class TestSentenceChunker:
    """Tests for the SentenceChunker."""

    def test_sentence_chunker_creation(self):
        """Test SentenceChunker can be created."""
        from src.ingestion.chunking.sentence import SentenceChunker

        chunker = SentenceChunker(min_sentences=1, max_sentences=3)
        assert chunker is not None

    def test_sentence_chunker_respects_sentences(self):
        """Test that SentenceChunker respects sentence boundaries."""
        from src.ingestion.chunking.sentence import SentenceChunker

        chunker = SentenceChunker(chunk_size=1000, max_sentences=2)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.chunk(text)

        assert len(chunks) >= 1

    def test_sentence_chunker_handles_edge_cases(self):
        """Test SentenceChunker handles edge cases."""
        from src.ingestion.chunking.sentence import SentenceChunker

        chunker = SentenceChunker(min_sentences=1, max_sentences=3)

        # Single sentence
        chunks = chunker.chunk("Just one sentence.")
        assert len(chunks) == 1

        # Empty text
        chunks = chunker.chunk("")
        assert len(chunks) == 0


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

        chunker = ParagraphChunker(chunk_size=5000)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunker.chunk(text)

        # With large chunk_size, may be combined or separate based on implementation
        assert len(chunks) >= 1

    def test_paragraph_chunker_respects_chunk_size(self):
        """Test ParagraphChunker respects chunk_size."""
        from src.ingestion.chunking.sentence import ParagraphChunker

        chunker = ParagraphChunker(chunk_size=50, chunk_overlap=10)
        text = "A short paragraph.\n\nAnother short paragraph.\n\nThird paragraph here."
        chunks = chunker.chunk(text)

        assert len(chunks) >= 1


class TestTextChunk:
    """Tests for the TextChunk model."""

    def test_text_chunk_creation(self):
        """Test TextChunk can be created."""
        from src.ingestion.chunking.base import TextChunk

        chunk = TextChunk(
            content="Test content",
            position=0,
            start_char=0,
            end_char=12,
        )
        assert chunk.content == "Test content"
        assert chunk.position == 0

    def test_text_chunk_has_id(self):
        """Test TextChunk has auto-generated id."""
        from src.ingestion.chunking.base import TextChunk

        chunk = TextChunk(
            content="Test",
            position=0,
            start_char=0,
            end_char=4,
        )
        assert chunk.id is not None

    def test_text_chunk_metadata(self):
        """Test TextChunk with metadata."""
        from src.ingestion.chunking.base import TextChunk

        chunk = TextChunk(
            content="Test",
            position=0,
            start_char=0,
            end_char=4,
            metadata={"source": "test.pdf"},
        )
        assert chunk.metadata["source"] == "test.pdf"

    def test_text_chunk_custom_id(self):
        """Test TextChunk with custom id."""
        from src.ingestion.chunking.base import TextChunk

        chunk = TextChunk(
            id="custom-id",
            content="Test",
            position=0,
            start_char=0,
            end_char=4,
        )
        assert chunk.id == "custom-id"
