"""
Enterprise RAG System - RAG Pipeline Service
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional
from uuid import uuid4

from src.core.config import settings
from src.core.logging import LoggerMixin
from src.core.types import (
    ContextItem,
    QueryComplexity,
    RetrievalStrategy,
    SearchResult,
    Citation,
)
from src.retrieval import get_vector_searcher
from src.generation import (
    get_default_llm_client,
    LLMMessage,
    build_rag_prompt,
)


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""
    query_id: str
    query: str
    answer: str
    sources: list[dict[str, Any]]
    citations: list[Citation]
    complexity: QueryComplexity
    retrieval_strategy: RetrievalStrategy
    confidence: Optional[float]
    latency_ms: float
    tokens_used: int


class RAGPipeline(LoggerMixin):
    """
    Main RAG pipeline that orchestrates retrieval and generation.

    This is a simplified pipeline for Phase 2. More advanced features
    (multi-agent, verification, etc.) will be added in later phases.
    """

    def __init__(
        self,
        top_k: Optional[int] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.top_k = top_k or settings.DEFAULT_TOP_K
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def query(
        self,
        question: str,
        conversation_history: Optional[list[dict[str, str]]] = None,
        retrieval_strategy: Optional[RetrievalStrategy] = None,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> RAGResponse:
        """
        Execute the full RAG pipeline.

        Args:
            question: User's question
            conversation_history: Previous conversation turns
            retrieval_strategy: Override retrieval strategy
            top_k: Override number of results to retrieve
            filters: Metadata filters for retrieval

        Returns:
            RAGResponse with answer and sources
        """
        start_time = time.time()
        query_id = str(uuid4())

        self.logger.info(
            "Starting RAG pipeline",
            query_id=query_id,
            question_length=len(question),
        )

        # Step 1: Classify query complexity (simplified for now)
        complexity = self._classify_complexity(question)

        # Step 2: Retrieve relevant context
        strategy = retrieval_strategy or RetrievalStrategy.VECTOR
        top_k = top_k or self.top_k

        retrieval_result = await self._retrieve(
            question, top_k, filters, strategy
        )

        # Step 3: Build context items
        context_items = self._build_context_items(retrieval_result.results)

        # Step 4: Generate response
        llm_response = await self._generate(
            question, context_items, conversation_history
        )

        # Step 5: Extract citations
        citations = self._extract_citations(llm_response.content, context_items)

        # Step 6: Build source references
        sources = self._build_sources(retrieval_result.results)

        latency_ms = (time.time() - start_time) * 1000

        self.logger.info(
            "RAG pipeline complete",
            query_id=query_id,
            latency_ms=round(latency_ms, 2),
            tokens_used=llm_response.total_tokens,
            sources_count=len(sources),
        )

        return RAGResponse(
            query_id=query_id,
            query=question,
            answer=llm_response.content,
            sources=sources,
            citations=citations,
            complexity=complexity,
            retrieval_strategy=strategy,
            confidence=self._calculate_confidence(retrieval_result.results),
            latency_ms=latency_ms,
            tokens_used=llm_response.total_tokens,
        )

    async def query_stream(
        self,
        question: str,
        conversation_history: Optional[list[dict[str, str]]] = None,
        retrieval_strategy: Optional[RetrievalStrategy] = None,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Execute the RAG pipeline with streaming response.

        Yields chunks of the response as they're generated.
        """
        query_id = str(uuid4())
        start_time = time.time()

        # Retrieve context first (non-streaming)
        strategy = retrieval_strategy or RetrievalStrategy.VECTOR
        top_k = top_k or self.top_k

        retrieval_result = await self._retrieve(
            question, top_k, filters, strategy
        )

        context_items = self._build_context_items(retrieval_result.results)
        sources = self._build_sources(retrieval_result.results)

        # Build prompt
        system_prompt, user_prompt = build_rag_prompt(
            question=question,
            context_items=context_items,
            history=conversation_history,
        )

        # Stream generation
        llm_client = get_default_llm_client()
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

        full_content = ""
        total_tokens = 0

        async for chunk in llm_client.generate_stream(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ):
            if chunk.content:
                full_content += chunk.content
                yield {
                    "type": "token",
                    "content": chunk.content,
                }

            if chunk.is_final:
                total_tokens = chunk.metadata.get("output_tokens", 0)

        # Send final message with metadata
        latency_ms = (time.time() - start_time) * 1000

        yield {
            "type": "done",
            "query_id": query_id,
            "sources": sources,
            "latency_ms": round(latency_ms, 2),
            "tokens_used": total_tokens,
        }

    def _classify_complexity(self, question: str) -> QueryComplexity:
        """
        Classify query complexity.

        Simple heuristic for Phase 2. Will be replaced by LLM-based
        classification in Phase 4.
        """
        # Simple heuristics
        question_lower = question.lower()

        # Complex indicators
        complex_indicators = [
            "compare", "contrast", "analyze", "evaluate",
            "how does", "why did", "what are the implications",
            "relationship between", "explain the difference",
        ]

        # Simple indicators
        simple_indicators = [
            "what is", "who is", "when did", "where is",
            "define", "list",
        ]

        for indicator in complex_indicators:
            if indicator in question_lower:
                return QueryComplexity.COMPLEX

        for indicator in simple_indicators:
            if indicator in question_lower:
                return QueryComplexity.SIMPLE

        # Default to standard
        return QueryComplexity.STANDARD

    async def _retrieve(
        self,
        question: str,
        top_k: int,
        filters: Optional[dict[str, Any]],
        strategy: RetrievalStrategy,
    ):
        """Retrieve relevant documents."""
        searcher = get_vector_searcher()
        return await searcher.search(
            query=question,
            top_k=top_k,
            filters=filters,
        )

    def _build_context_items(
        self,
        results: list[SearchResult],
    ) -> list[ContextItem]:
        """Convert search results to context items."""
        return [
            ContextItem(
                content=r.content,
                source=r.source or f"Document {r.document_id[:8]}",
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                relevance_score=r.score,
            )
            for r in results
        ]

    async def _generate(
        self,
        question: str,
        context_items: list[ContextItem],
        history: Optional[list[dict[str, str]]],
    ):
        """Generate response using LLM."""
        system_prompt, user_prompt = build_rag_prompt(
            question=question,
            context_items=context_items,
            history=history,
        )

        llm_client = get_default_llm_client()

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

        return await llm_client.generate(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _extract_citations(
        self,
        content: str,
        context_items: list[ContextItem],
    ) -> list[Citation]:
        """Extract citation references from generated content."""
        import re

        citations = []
        # Find all [N] references
        matches = re.findall(r'\[(\d+)\]', content)

        for match in matches:
            idx = int(match) - 1  # Convert to 0-based index
            if 0 <= idx < len(context_items):
                item = context_items[idx]
                citations.append(Citation(
                    id=f"cite-{idx + 1}",
                    document_id=item.document_id,
                    chunk_id=item.chunk_id,
                    source=item.source,
                ))

        # Deduplicate
        seen = set()
        unique_citations = []
        for c in citations:
            if c.id not in seen:
                seen.add(c.id)
                unique_citations.append(c)

        return unique_citations

    def _build_sources(self, results: list[SearchResult]) -> list[dict[str, Any]]:
        """Build source reference list."""
        return [
            {
                "id": f"source-{i + 1}",
                "document_id": r.document_id,
                "chunk_id": r.chunk_id,
                "title": r.metadata.get("title"),
                "source": r.source,
                "relevance_score": round(r.score, 4),
                "excerpt": r.content[:200] + "..." if len(r.content) > 200 else r.content,
            }
            for i, r in enumerate(results)
        ]

    def _calculate_confidence(self, results: list[SearchResult]) -> Optional[float]:
        """Calculate confidence score based on retrieval results."""
        if not results:
            return 0.0

        # Average of top 3 scores
        top_scores = [r.score for r in results[:3]]
        return round(sum(top_scores) / len(top_scores), 4) if top_scores else None


# Singleton instance
_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get the global RAG pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
