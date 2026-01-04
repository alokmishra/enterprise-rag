"""
Enterprise RAG System - Sentence-Based Text Chunker
"""

import re
from typing import Optional

from src.ingestion.chunking.base import TextChunk, TextChunker


class SentenceChunker(TextChunker):
    """
    Splits text into chunks based on sentence boundaries.

    Groups sentences together until chunk_size is reached.
    """

    # Sentence ending patterns
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    # Abbreviations that shouldn't end sentences
    ABBREVIATIONS = {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
        "inc", "ltd", "corp", "vs", "etc", "al",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
        "st", "nd", "rd", "th",
        "fig", "eq", "ref", "sec", "ch", "vol", "no",
        "i.e", "e.g", "cf", "viz",
    }

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_sentences: int = 1,
        max_sentences: int = 10,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[TextChunk]:
        """Split text into sentence-based chunks."""
        if not text.strip():
            return []

        # Split into sentences
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        # Group sentences into chunks
        chunks = []
        current_sentences = []
        current_length = 0
        current_start = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # Check if adding this sentence exceeds limit
            would_exceed = (
                current_length + sentence_length > self.chunk_size and
                len(current_sentences) >= self.min_sentences
            )

            # Check if we have max sentences
            at_max = len(current_sentences) >= self.max_sentences

            if would_exceed or at_max:
                # Create chunk from current sentences
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    end_char = current_start + len(chunk_text)

                    chunks.append(self._create_chunk(
                        content=chunk_text,
                        position=len(chunks),
                        start_char=current_start,
                        end_char=end_char,
                        metadata={
                            **(metadata or {}),
                            "sentence_count": len(current_sentences),
                        },
                    ))

                    # Calculate overlap
                    if self.chunk_overlap > 0:
                        overlap_sentences = self._get_overlap_sentences(
                            current_sentences, self.chunk_overlap
                        )
                        current_sentences = overlap_sentences
                        current_length = sum(len(s) for s in current_sentences)
                        # Adjust start position
                        current_start = end_char - sum(len(s) + 1 for s in overlap_sentences)
                    else:
                        current_sentences = []
                        current_length = 0
                        current_start = end_char + 1

            current_sentences.append(sentence)
            current_length += sentence_length

        # Don't forget the last chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(self._create_chunk(
                content=chunk_text,
                position=len(chunks),
                start_char=current_start,
                end_char=current_start + len(chunk_text),
                metadata={
                    **(metadata or {}),
                    "sentence_count": len(current_sentences),
                },
            ))

        self.logger.debug(
            "Chunked text by sentences",
            input_length=len(text),
            total_sentences=len(sentences),
            num_chunks=len(chunks),
        )

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Simple sentence splitting
        # Handle common abbreviations
        for abbrev in self.ABBREVIATIONS:
            # Replace periods after abbreviations temporarily
            pattern = rf'\b{abbrev}\.'
            text = re.sub(pattern, f'{abbrev}<PERIOD>', text, flags=re.IGNORECASE)

        # Split on sentence endings
        raw_sentences = self.SENTENCE_ENDINGS.split(text)

        # Restore periods
        sentences = []
        for s in raw_sentences:
            s = s.replace('<PERIOD>', '.').strip()
            if s:
                sentences.append(s)

        return sentences

    def _get_overlap_sentences(
        self,
        sentences: list[str],
        overlap_chars: int,
    ) -> list[str]:
        """Get sentences that fit within overlap character limit."""
        overlap = []
        total_length = 0

        for sentence in reversed(sentences):
            if total_length + len(sentence) <= overlap_chars:
                overlap.insert(0, sentence)
                total_length += len(sentence)
            else:
                break

        return overlap


class ParagraphChunker(TextChunker):
    """
    Splits text into chunks based on paragraph boundaries.

    Groups paragraphs together until chunk_size is reached.
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        paragraph_separator: str = "\n\n",
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.paragraph_separator = paragraph_separator

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[TextChunk]:
        """Split text into paragraph-based chunks."""
        if not text.strip():
            return []

        # Split into paragraphs
        paragraphs = [
            p.strip() for p in text.split(self.paragraph_separator)
            if p.strip()
        ]

        if not paragraphs:
            return []

        # Group paragraphs into chunks
        chunks = []
        current_paragraphs = []
        current_length = 0
        current_start = 0

        for paragraph in paragraphs:
            para_length = len(paragraph) + len(self.paragraph_separator)

            # Check if adding this paragraph exceeds limit
            if current_length + para_length > self.chunk_size and current_paragraphs:
                # Create chunk
                chunk_text = self.paragraph_separator.join(current_paragraphs)
                end_char = current_start + len(chunk_text)

                chunks.append(self._create_chunk(
                    content=chunk_text,
                    position=len(chunks),
                    start_char=current_start,
                    end_char=end_char,
                    metadata={
                        **(metadata or {}),
                        "paragraph_count": len(current_paragraphs),
                    },
                ))

                # Handle overlap
                if self.chunk_overlap > 0 and current_paragraphs:
                    # Keep last paragraph for overlap
                    last_para = current_paragraphs[-1]
                    if len(last_para) <= self.chunk_overlap:
                        current_paragraphs = [last_para]
                        current_length = len(last_para)
                        current_start = text.find(last_para, end_char - len(last_para) - 10)
                        if current_start == -1:
                            current_start = end_char
                    else:
                        current_paragraphs = []
                        current_length = 0
                        current_start = end_char + len(self.paragraph_separator)
                else:
                    current_paragraphs = []
                    current_length = 0
                    current_start = end_char + len(self.paragraph_separator)

            current_paragraphs.append(paragraph)
            current_length += para_length

        # Last chunk
        if current_paragraphs:
            chunk_text = self.paragraph_separator.join(current_paragraphs)
            chunks.append(self._create_chunk(
                content=chunk_text,
                position=len(chunks),
                start_char=current_start,
                end_char=current_start + len(chunk_text),
                metadata={
                    **(metadata or {}),
                    "paragraph_count": len(current_paragraphs),
                },
            ))

        return chunks
