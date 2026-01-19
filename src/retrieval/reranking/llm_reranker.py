"""
Enterprise RAG System - LLM-Based Reranker

Uses an LLM to score and rerank search results based on relevance.
More expensive but can handle nuanced relevance judgments.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from src.core.config import settings
from src.core.types import SearchResult
from src.generation.llm import get_default_llm_client, LLMMessage
from src.retrieval.reranking.base import Reranker, RerankResult


class LLMReranker(Reranker):
    """
    LLM-based reranker that uses an LLM to score relevance.
    """

    RERANK_PROMPT = """You are a relevance scoring assistant. Score how relevant each passage is to answering the given query.

Query: {query}

Passages to score:
{passages}

For each passage, provide a relevance score from 0.0 to 1.0 where:
- 1.0 = Highly relevant, directly answers the query
- 0.7-0.9 = Relevant, contains useful information
- 0.4-0.6 = Partially relevant, tangentially related
- 0.1-0.3 = Marginally relevant, mostly off-topic
- 0.0 = Not relevant at all

Return a JSON array with objects containing "id" and "score" for each passage.
Example: [{{"id": "1", "score": 0.85}}, {{"id": "2", "score": 0.45}}]

Only return the JSON array, no other text."""

    def __init__(
        self,
        batch_size: int = 10,
        score_threshold: float = 0.3,
    ):
        """
        Initialize LLM reranker.

        Args:
            batch_size: Number of results to score in one LLM call
            score_threshold: Minimum score to include in results
        """
        self.batch_size = batch_size
        self.score_threshold = score_threshold

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        """Rerank results using LLM scoring."""
        start_time = time.time()

        if not results:
            return RerankResult(
                results=[],
                original_count=0,
                reranked_count=0,
                latency_ms=0.0,
            )

        top_k = top_k or settings.RERANK_TOP_K
        original_count = len(results)

        # Score results in batches
        all_scores = {}

        for i in range(0, len(results), self.batch_size):
            batch = results[i:i + self.batch_size]
            batch_scores = await self._score_batch(query, batch)
            all_scores.update(batch_scores)

        # Apply scores and filter
        scored_results = []
        for result in results:
            score = all_scores.get(result.chunk_id, 0.0)
            if score >= self.score_threshold:
                # Create new result with updated score
                scored_results.append(SearchResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    content=result.content,
                    score=score,  # Use LLM score
                    metadata={
                        **result.metadata,
                        "original_score": result.score,
                        "rerank_score": score,
                    },
                    source=result.source,
                ))

        # Sort by new score
        scored_results.sort(key=lambda x: x.score, reverse=True)
        final_results = scored_results[:top_k]

        latency_ms = (time.time() - start_time) * 1000

        self.logger.info(
            "LLM reranking complete",
            original_count=original_count,
            reranked_count=len(final_results),
            latency_ms=round(latency_ms, 2),
        )

        return RerankResult(
            results=final_results,
            original_count=original_count,
            reranked_count=len(final_results),
            latency_ms=latency_ms,
        )

    async def _score_batch(
        self,
        query: str,
        results: list[SearchResult],
    ) -> dict[str, float]:
        """Score a batch of results."""
        # Format passages
        passages = []
        for i, result in enumerate(results):
            truncated = result.content[:500]  # Truncate for prompt
            passages.append(f"[{result.chunk_id}] {truncated}")

        passages_text = "\n\n".join(passages)

        prompt = self.RERANK_PROMPT.format(
            query=query,
            passages=passages_text,
        )

        try:
            llm = get_default_llm_client()
            response = await llm.generate(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.0,
                max_tokens=500,
            )

            # Parse JSON response
            scores = self._parse_scores(response.content, results)
            return scores

        except Exception as e:
            self.logger.warning("LLM reranking batch failed", error=str(e))
            # Return original scores as fallback
            return {r.chunk_id: r.score for r in results}

    def _parse_scores(
        self,
        response: str,
        results: list[SearchResult],
    ) -> dict[str, float]:
        """Parse scores from LLM response."""
        try:
            # Find JSON array in response
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                scores_list = json.loads(json_str)

                scores = {}
                for item in scores_list:
                    chunk_id = str(item.get("id", ""))
                    score = float(item.get("score", 0.0))
                    scores[chunk_id] = min(max(score, 0.0), 1.0)  # Clamp to [0, 1]

                return scores

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.warning("Failed to parse rerank scores", error=str(e))

        # Fallback to original scores
        return {r.chunk_id: r.score for r in results}
