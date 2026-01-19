"""
Enterprise RAG System - Recursive Text Chunker
"""

from __future__ import annotations

import re
from typing import Optional

from src.ingestion.chunking.base import TextChunk, TextChunker


class RecursiveTextChunker(TextChunker):
    """
    Recursively splits text using a hierarchy of separators.

    Tries to keep semantic units together by first splitting on
    larger separators (paragraphs), then smaller ones (sentences, words).
    """

    DEFAULT_SEPARATORS = [
        "\n\n\n",     # Multiple paragraph breaks
        "\n\n",       # Paragraph break
        "\n",         # Line break
        ". ",         # Sentence
        "? ",         # Question
        "! ",         # Exclamation
        "; ",         # Semicolon
        ", ",         # Comma
        " ",          # Word
        "",           # Character (last resort)
    ]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[list[str]] = None,
        keep_separator: bool = True,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.keep_separator = keep_separator

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[TextChunk]:
        """Split text recursively using separator hierarchy."""
        if not text.strip():
            return []

        chunks = self._split_recursive(text, self.separators)

        # Merge small chunks
        merged = self._merge_chunks(chunks)

        # Create TextChunk objects with positions
        result = []
        current_pos = 0
        for i, chunk_text in enumerate(merged):
            start_char = text.find(chunk_text, current_pos)
            if start_char == -1:
                start_char = current_pos
            end_char = start_char + len(chunk_text)

            result.append(self._create_chunk(
                content=chunk_text,
                position=i,
                start_char=start_char,
                end_char=end_char,
                metadata=metadata,
            ))
            current_pos = end_char

        self.logger.debug(
            "Chunked text",
            input_length=len(text),
            num_chunks=len(result),
            avg_chunk_size=sum(len(c.content) for c in result) / len(result) if result else 0,
        )

        return result

    def _split_recursive(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:
        """Recursively split text using separators."""
        if not separators:
            # No more separators, split by character
            return self._split_by_size(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        # If text is small enough, return as is
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Split by current separator
        if separator:
            splits = text.split(separator)
        else:
            # Empty separator means split by character
            return self._split_by_size(text)

        # If no splits occurred, try next separator
        if len(splits) == 1:
            return self._split_recursive(text, remaining_separators)

        # Process each split
        chunks = []
        for i, split in enumerate(splits):
            if not split.strip():
                continue

            # Add separator back if needed
            if self.keep_separator and separator and i < len(splits) - 1:
                split = split + separator

            if len(split) <= self.chunk_size:
                chunks.append(split)
            else:
                # Recursively split if still too large
                sub_chunks = self._split_recursive(split, remaining_separators)
                chunks.extend(sub_chunks)

        return chunks

    def _split_by_size(self, text: str) -> list[str]:
        """Split text into fixed-size chunks as last resort."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - self.chunk_overlap if end < len(text) else end
        return chunks

    def _merge_chunks(self, chunks: list[str]) -> list[str]:
        """Merge small chunks together up to chunk_size."""
        if not chunks:
            return []

        merged = []
        current = chunks[0]

        for chunk in chunks[1:]:
            # Check if we can merge
            combined_length = len(current) + len(chunk)
            if combined_length <= self.chunk_size:
                current = current + chunk
            else:
                if current.strip():
                    merged.append(current.strip())
                current = chunk

        if current.strip():
            merged.append(current.strip())

        return merged


class MarkdownChunker(RecursiveTextChunker):
    """
    Chunker optimized for Markdown documents.

    Preserves heading structure and code blocks.
    """

    MARKDOWN_SEPARATORS = [
        "\n## ",      # H2 headers
        "\n### ",     # H3 headers
        "\n#### ",    # H4 headers
        "\n```",      # Code blocks
        "\n\n",       # Paragraphs
        "\n- ",       # List items
        "\n* ",       # List items
        "\n",         # Lines
        ". ",         # Sentences
        " ",          # Words
    ]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.MARKDOWN_SEPARATORS,
            keep_separator=True,
        )

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[TextChunk]:
        """Split Markdown text while preserving structure."""
        # Handle code blocks specially - don't split them
        code_block_pattern = r"```[\s\S]*?```"

        # Find all code blocks
        code_blocks = list(re.finditer(code_block_pattern, text))

        if not code_blocks:
            return super().chunk(text, metadata)

        # Split around code blocks
        chunks = []
        last_end = 0
        position = 0

        for match in code_blocks:
            # Process text before code block
            before_text = text[last_end:match.start()]
            if before_text.strip():
                text_chunks = super().chunk(before_text, metadata)
                for tc in text_chunks:
                    tc.position = position
                    tc.start_char += last_end
                    tc.end_char += last_end
                    position += 1
                chunks.extend(text_chunks)

            # Add code block as single chunk (if not too large)
            code_block = match.group()
            if len(code_block) <= self.chunk_size * 2:  # Allow larger code blocks
                chunks.append(self._create_chunk(
                    content=code_block,
                    position=position,
                    start_char=match.start(),
                    end_char=match.end(),
                    metadata={**(metadata or {}), "is_code_block": True},
                ))
                position += 1
            else:
                # Split large code blocks
                code_chunks = self._split_by_size(code_block)
                for i, cc in enumerate(code_chunks):
                    chunks.append(self._create_chunk(
                        content=cc,
                        position=position,
                        start_char=match.start() + i * (self.chunk_size - self.chunk_overlap),
                        end_char=match.start() + i * (self.chunk_size - self.chunk_overlap) + len(cc),
                        metadata={**(metadata or {}), "is_code_block": True},
                    ))
                    position += 1

            last_end = match.end()

        # Process remaining text
        remaining = text[last_end:]
        if remaining.strip():
            remaining_chunks = super().chunk(remaining, metadata)
            for tc in remaining_chunks:
                tc.position = position
                tc.start_char += last_end
                tc.end_char += last_end
                position += 1
            chunks.extend(remaining_chunks)

        return chunks
