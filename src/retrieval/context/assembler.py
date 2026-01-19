"""
Enterprise RAG System - Context Assembly

Assembles and optimizes context from retrieved documents
for inclusion in LLM prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.core.config import settings
from src.core.logging import LoggerMixin
from src.core.types import ContextItem, SearchResult


@dataclass
class AssembledContext:
    """Result of context assembly."""
    items: list[ContextItem]
    total_tokens: int
    truncated: bool
    sources_count: int


class ContextAssembler(LoggerMixin):
    """
    Assembles context from search results for LLM prompts.

    Handles:
    - Token budget management
    - Deduplication
    - Context ordering
    - Source grouping
    """

    # Approximate tokens per character (conservative estimate)
    TOKENS_PER_CHAR = 0.25

    def __init__(
        self,
        max_tokens: int = 8000,
        max_items: int = 20,
        deduplicate: bool = True,
        similarity_threshold: float = 0.9,
    ):
        """
        Initialize context assembler.

        Args:
            max_tokens: Maximum tokens for context
            max_items: Maximum number of context items
            deduplicate: Whether to remove similar chunks
            similarity_threshold: Threshold for deduplication
        """
        self.max_tokens = max_tokens
        self.max_items = max_items
        self.deduplicate = deduplicate
        self.similarity_threshold = similarity_threshold

    def assemble(
        self,
        results: list[SearchResult],
        strategy: str = "relevance",  # "relevance", "diversity", "recency"
    ) -> AssembledContext:
        """
        Assemble context from search results.

        Args:
            results: Search results to assemble
            strategy: Assembly strategy

        Returns:
            AssembledContext with optimized items
        """
        if not results:
            return AssembledContext(
                items=[],
                total_tokens=0,
                truncated=False,
                sources_count=0,
            )

        # Convert to context items
        items = [self._to_context_item(r) for r in results]

        # Deduplicate if enabled
        if self.deduplicate:
            items = self._deduplicate(items)

        # Order based on strategy
        if strategy == "diversity":
            items = self._order_by_diversity(items)
        elif strategy == "recency":
            items = self._order_by_recency(items)
        # Default: keep relevance order

        # Apply token budget
        final_items, total_tokens, truncated = self._apply_token_budget(items)

        # Count unique sources
        sources = set(item.source for item in final_items)

        self.logger.info(
            "Context assembled",
            input_items=len(results),
            output_items=len(final_items),
            total_tokens=total_tokens,
            truncated=truncated,
            sources=len(sources),
        )

        return AssembledContext(
            items=final_items,
            total_tokens=total_tokens,
            truncated=truncated,
            sources_count=len(sources),
        )

    def _to_context_item(self, result: SearchResult) -> ContextItem:
        """Convert SearchResult to ContextItem."""
        return ContextItem(
            content=result.content,
            source=result.source or f"Document {result.document_id[:8]}",
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            relevance_score=result.score,
        )

    def _deduplicate(self, items: list[ContextItem]) -> list[ContextItem]:
        """Remove near-duplicate items."""
        if len(items) <= 1:
            return items

        unique_items = [items[0]]

        for item in items[1:]:
            is_duplicate = False

            for unique in unique_items:
                similarity = self._text_similarity(item.content, unique.content)
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_items.append(item)

        if len(unique_items) < len(items):
            self.logger.debug(
                "Deduplicated context",
                original=len(items),
                unique=len(unique_items),
            )

        return unique_items

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity using Jaccard index."""
        # Simple word-based Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _order_by_diversity(self, items: list[ContextItem]) -> list[ContextItem]:
        """Reorder items to maximize diversity."""
        if len(items) <= 2:
            return items

        # Use maximal marginal relevance (MMR) approach
        selected = [items[0]]
        remaining = items[1:]

        while remaining and len(selected) < len(items):
            # Find item most different from selected
            best_item = None
            best_score = -1

            for item in remaining:
                # Calculate minimum similarity to any selected item
                min_sim = min(
                    self._text_similarity(item.content, s.content)
                    for s in selected
                )
                # MMR: balance relevance and diversity
                diversity_score = item.relevance_score * (1 - min_sim)

                if diversity_score > best_score:
                    best_score = diversity_score
                    best_item = item

            if best_item:
                selected.append(best_item)
                remaining.remove(best_item)

        return selected

    def _order_by_recency(self, items: list[ContextItem]) -> list[ContextItem]:
        """Order items by recency (if date available in metadata)."""
        # This would require date information in metadata
        # For now, return original order
        return items

    def _apply_token_budget(
        self,
        items: list[ContextItem],
    ) -> tuple[list[ContextItem], int, bool]:
        """
        Apply token budget and item limit.

        Returns: (items, total_tokens, was_truncated)
        """
        final_items = []
        total_tokens = 0
        truncated = False

        for item in items:
            if len(final_items) >= self.max_items:
                truncated = True
                break

            item_tokens = int(len(item.content) * self.TOKENS_PER_CHAR)

            if total_tokens + item_tokens > self.max_tokens:
                # Try to include partial content
                remaining_tokens = self.max_tokens - total_tokens
                if remaining_tokens > 100:  # Only include if meaningful
                    max_chars = int(remaining_tokens / self.TOKENS_PER_CHAR)
                    truncated_content = item.content[:max_chars] + "..."

                    final_items.append(ContextItem(
                        content=truncated_content,
                        source=item.source,
                        chunk_id=item.chunk_id,
                        document_id=item.document_id,
                        relevance_score=item.relevance_score,
                    ))
                    total_tokens += int(len(truncated_content) * self.TOKENS_PER_CHAR)

                truncated = True
                break

            final_items.append(item)
            total_tokens += item_tokens

        return final_items, total_tokens, truncated

    def group_by_source(
        self,
        items: list[ContextItem],
    ) -> dict[str, list[ContextItem]]:
        """Group context items by source."""
        groups: dict[str, list[ContextItem]] = {}

        for item in items:
            source = item.source
            if source not in groups:
                groups[source] = []
            groups[source].append(item)

        return groups


# Singleton instance
_assembler: Optional[ContextAssembler] = None


def get_context_assembler() -> ContextAssembler:
    """Get the global context assembler instance."""
    global _assembler
    if _assembler is None:
        _assembler = ContextAssembler()
    return _assembler
