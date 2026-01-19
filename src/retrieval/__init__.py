"""
Enterprise RAG System - Retrieval Module

This module handles document retrieval:
- Vector similarity search
- Hybrid search (vector + keyword)
- Query expansion
- Reranking
- Context assembly
"""

from __future__ import annotations

from src.retrieval.search import (
    VectorSearcher,
    MultiQuerySearcher,
    get_vector_searcher,
    BM25Index,
    get_bm25_index,
    HybridSearcher,
    get_hybrid_searcher,
)
from src.retrieval.query import (
    QueryExpander,
    SubQuestionDecomposer,
    HyDEGenerator,
)
from src.retrieval.reranking import (
    Reranker,
    RerankResult,
    LLMReranker,
    CrossEncoderReranker,
)
from src.retrieval.filters import (
    MetadataFilter,
    FilterOperator,
    build_source_filter,
    build_date_filter,
)
from src.retrieval.context import (
    ContextAssembler,
    AssembledContext,
    get_context_assembler,
)

__all__ = [
    # Search
    "VectorSearcher",
    "MultiQuerySearcher",
    "get_vector_searcher",
    "BM25Index",
    "get_bm25_index",
    "HybridSearcher",
    "get_hybrid_searcher",
    # Query
    "QueryExpander",
    "SubQuestionDecomposer",
    "HyDEGenerator",
    # Reranking
    "Reranker",
    "RerankResult",
    "LLMReranker",
    "CrossEncoderReranker",
    # Filters
    "MetadataFilter",
    "FilterOperator",
    "build_source_filter",
    "build_date_filter",
    # Context
    "ContextAssembler",
    "AssembledContext",
    "get_context_assembler",
]
