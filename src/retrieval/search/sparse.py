"""
Enterprise RAG System - Sparse (BM25) Search

BM25 is a keyword-based ranking function used for text retrieval.
This implementation provides an in-memory BM25 index for sparse search.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from src.core.logging import LoggerMixin


@dataclass
class SparseSearchResult:
    """Result from sparse search."""
    id: str
    score: float
    content: str
    metadata: dict


class BM25Index(LoggerMixin):
    """
    BM25 index for sparse retrieval.

    BM25 Parameters:
    - k1: Term frequency saturation (default 1.5)
    - b: Document length normalization (default 0.75)
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b

        # Index structures
        self._documents: dict[str, dict] = {}  # id -> {content, metadata}
        self._doc_lengths: dict[str, int] = {}  # id -> length
        self._avg_doc_length: float = 0.0
        self._doc_count: int = 0

        # Inverted index: term -> {doc_id: term_frequency}
        self._inverted_index: dict[str, dict[str, int]] = defaultdict(dict)
        self._doc_freqs: dict[str, int] = defaultdict(int)  # term -> num docs containing term

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a document to the index."""
        tokens = self._tokenize(content)

        # Store document
        self._documents[doc_id] = {
            "content": content,
            "metadata": metadata or {},
        }
        self._doc_lengths[doc_id] = len(tokens)

        # Count term frequencies
        term_counts: dict[str, int] = defaultdict(int)
        for token in tokens:
            term_counts[token] += 1

        # Update inverted index
        for term, count in term_counts.items():
            if doc_id not in self._inverted_index[term]:
                self._doc_freqs[term] += 1
            self._inverted_index[term][doc_id] = count

        # Update statistics
        self._doc_count = len(self._documents)
        self._avg_doc_length = sum(self._doc_lengths.values()) / self._doc_count

    def add_documents(
        self,
        documents: list[dict],
    ) -> None:
        """
        Add multiple documents to the index.

        Each document should have: id, content, metadata (optional)
        """
        for doc in documents:
            self.add_document(
                doc_id=doc["id"],
                content=doc["content"],
                metadata=doc.get("metadata"),
            )

        self.logger.info(
            "Added documents to BM25 index",
            num_documents=len(documents),
            total_documents=self._doc_count,
        )

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the index."""
        if doc_id not in self._documents:
            return False

        # Get document tokens
        content = self._documents[doc_id]["content"]
        tokens = self._tokenize(content)

        # Remove from inverted index
        for token in set(tokens):
            if doc_id in self._inverted_index[token]:
                del self._inverted_index[token][doc_id]
                self._doc_freqs[token] -= 1
                if self._doc_freqs[token] == 0:
                    del self._doc_freqs[token]
                    del self._inverted_index[token]

        # Remove document
        del self._documents[doc_id]
        del self._doc_lengths[doc_id]

        # Update statistics
        self._doc_count = len(self._documents)
        if self._doc_count > 0:
            self._avg_doc_length = sum(self._doc_lengths.values()) / self._doc_count
        else:
            self._avg_doc_length = 0.0

        return True

    def _calculate_idf(self, term: str) -> float:
        """Calculate IDF for a term."""
        doc_freq = self._doc_freqs.get(term, 0)
        if doc_freq == 0:
            return 0.0

        # IDF with smoothing
        return math.log((self._doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def _calculate_bm25_score(
        self,
        query_terms: list[str],
        doc_id: str,
    ) -> float:
        """Calculate BM25 score for a document."""
        score = 0.0
        doc_length = self._doc_lengths[doc_id]

        for term in query_terms:
            if term not in self._inverted_index:
                continue

            term_freq = self._inverted_index[term].get(doc_id, 0)
            if term_freq == 0:
                continue

            idf = self._calculate_idf(term)

            # BM25 formula
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (
                1 - self.b + self.b * (doc_length / self._avg_doc_length)
            )

            score += idf * (numerator / denominator)

        return score

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[SparseSearchResult]:
        """
        Search the index using BM25.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of SparseSearchResult sorted by score
        """
        if self._doc_count == 0:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        # Find candidate documents (any term matches)
        candidate_docs = set()
        for term in query_terms:
            if term in self._inverted_index:
                candidate_docs.update(self._inverted_index[term].keys())

        if not candidate_docs:
            return []

        # Score all candidates
        scored_docs = []
        for doc_id in candidate_docs:
            score = self._calculate_bm25_score(query_terms, doc_id)
            if score > 0:
                scored_docs.append((doc_id, score))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Build results
        results = []
        for doc_id, score in scored_docs[:top_k]:
            doc = self._documents[doc_id]
            results.append(SparseSearchResult(
                id=doc_id,
                score=score,
                content=doc["content"],
                metadata=doc["metadata"],
            ))

        return results

    def clear(self) -> None:
        """Clear the entire index."""
        self._documents.clear()
        self._doc_lengths.clear()
        self._inverted_index.clear()
        self._doc_freqs.clear()
        self._doc_count = 0
        self._avg_doc_length = 0.0


# Singleton instance
_bm25_index: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    """Get the global BM25 index instance."""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
    return _bm25_index
