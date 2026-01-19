"""
Enterprise RAG System - Critic Agent

The Critic Agent evaluates response quality and determines
whether revisions are needed.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from src.agents.base import BaseAgent, AgentConfig, AgentResult
from src.core.types import (
    AgentState,
    AgentType,
    CriticFeedback,
)
from src.generation.llm import get_default_llm_client, LLMMessage


class CriticDecision:
    """Decisions the critic can make."""
    PASS = "PASS"
    MINOR_REVISION = "MINOR_REVISION"
    MAJOR_REVISION = "MAJOR_REVISION"
    RETRIEVAL_NEEDED = "RETRIEVAL_NEEDED"
    REJECT = "REJECT"


class CriticAgent(BaseAgent):
    """
    Agent responsible for evaluating response quality.

    The critic:
    1. Scores responses on multiple dimensions
    2. Provides specific feedback for improvement
    3. Decides whether to pass, revise, or reject
    4. Considers verification results in evaluation
    """

    EVALUATION_PROMPT = """Evaluate the following response to a user query.

Query: {query}

Response:
{response}

Context Used:
{context_summary}

Verification Summary:
- Supported claims: {supported}
- Partially supported: {partial}
- Unsupported claims: {unsupported}
- Contradicted claims: {contradicted}

Evaluate the response on these dimensions (0.0-1.0):
1. Relevance: Does it answer the question?
2. Completeness: Does it cover all aspects?
3. Accuracy: Is the information correct (based on verification)?
4. Coherence: Is it well-organized and clear?
5. Citation: Are sources properly referenced?

Then decide:
- PASS: Response is good enough to return
- MINOR_REVISION: Small improvements needed
- MAJOR_REVISION: Significant rewrite needed
- RETRIEVAL_NEEDED: Need more context from sources
- REJECT: Cannot provide acceptable response

Return JSON:
{{
    "relevance_score": 0.0-1.0,
    "completeness_score": 0.0-1.0,
    "accuracy_score": 0.0-1.0,
    "coherence_score": 0.0-1.0,
    "citation_score": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "decision": "PASS|MINOR_REVISION|MAJOR_REVISION|RETRIEVAL_NEEDED|REJECT",
    "feedback": "Overall assessment",
    "suggestions": ["specific suggestion 1", "specific suggestion 2"]
}}

Return ONLY the JSON object."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        thresholds: Optional[dict] = None,
    ):
        config = config or AgentConfig(
            name="critic",
            agent_type=AgentType.CRITIC,
            temperature=0.0,
            max_tokens=1000,
        )
        super().__init__(config)

        # Quality thresholds
        self.thresholds = thresholds or {
            "relevance": 0.7,
            "completeness": 0.6,
            "accuracy": 0.8,
            "coherence": 0.7,
            "overall": 0.7,
        }

    async def execute(self, state: AgentState, **kwargs) -> AgentResult:
        """
        Evaluate the generated response.

        Args:
            state: Current agent state with response and verification

        Returns:
            AgentResult containing CriticFeedback
        """
        start_time = time.time()

        try:
            if not state.draft_responses:
                return AgentResult(
                    success=False,
                    output=None,
                    error="No response to evaluate",
                    latency_ms=0,
                )

            response = state.draft_responses[-1]

            # Get verification summary
            verification = self._summarize_verification(state.verification_results)

            # Format context summary
            context_summary = self._summarize_context(state.retrieved_context)

            # Evaluate response
            feedback = await self._evaluate(
                query=state.original_query,
                response=response,
                context_summary=context_summary,
                verification=verification,
            )

            # Update state
            state.critic_feedback.append(feedback.__dict__)

            latency_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "Evaluation complete",
                overall_score=feedback.overall_score,
                decision=feedback.decision,
                trace_id=state.trace_id,
            )

            return AgentResult(
                success=True,
                output=feedback,
                latency_ms=latency_ms,
            )

        except Exception as e:
            self.logger.error(
                "Evaluation failed",
                error=str(e),
                trace_id=state.trace_id,
            )
            # Return default pass on failure
            return AgentResult(
                success=False,
                output=CriticFeedback(
                    relevance_score=0.0,
                    completeness_score=0.0,
                    accuracy_score=0.0,
                    coherence_score=0.0,
                    citation_score=0.0,
                    overall_score=0.0,
                    feedback="Evaluation failed",
                    suggestions=[],
                    decision=CriticDecision.PASS,
                ),
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def _evaluate(
        self,
        query: str,
        response: str,
        context_summary: str,
        verification: dict,
    ) -> CriticFeedback:
        """Evaluate response using LLM."""
        prompt = self.EVALUATION_PROMPT.format(
            query=query,
            response=response,
            context_summary=context_summary,
            supported=verification.get("supported", 0),
            partial=verification.get("partial", 0),
            unsupported=verification.get("unsupported", 0),
            contradicted=verification.get("contradicted", 0),
        )

        llm = get_default_llm_client()
        result = await llm.generate(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=800,
        )

        # Parse result
        try:
            start = result.content.find("{")
            end = result.content.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(result.content[start:end])

                return CriticFeedback(
                    relevance_score=float(data.get("relevance_score", 0.5)),
                    completeness_score=float(data.get("completeness_score", 0.5)),
                    accuracy_score=float(data.get("accuracy_score", 0.5)),
                    coherence_score=float(data.get("coherence_score", 0.5)),
                    citation_score=float(data.get("citation_score", 0.5)),
                    overall_score=float(data.get("overall_score", 0.5)),
                    feedback=data.get("feedback", ""),
                    suggestions=data.get("suggestions", []),
                    decision=data.get("decision", CriticDecision.PASS),
                )
        except (json.JSONDecodeError, ValueError):
            pass

        # Default feedback
        return CriticFeedback(
            relevance_score=0.5,
            completeness_score=0.5,
            accuracy_score=0.5,
            coherence_score=0.5,
            citation_score=0.5,
            overall_score=0.5,
            feedback="Unable to evaluate response",
            suggestions=[],
            decision=CriticDecision.PASS,
        )

    def _summarize_verification(self, results: list) -> dict:
        """Summarize verification results."""
        if not results:
            return {
                "supported": 0,
                "partial": 0,
                "unsupported": 0,
                "contradicted": 0,
            }

        supported = sum(1 for r in results if r.get("status") == "SUPPORTED")
        partial = sum(1 for r in results if r.get("status") == "PARTIALLY_SUPPORTED")
        unsupported = sum(1 for r in results if r.get("status") == "NOT_FOUND")
        contradicted = sum(1 for r in results if r.get("status") == "CONTRADICTED")

        return {
            "supported": supported,
            "partial": partial,
            "unsupported": unsupported,
            "contradicted": contradicted,
        }

    def _summarize_context(self, context: list) -> str:
        """Summarize retrieved context."""
        if not context:
            return "No context available"

        sources = set()
        for item in context:
            sources.add(item.source if hasattr(item, 'source') else str(item))

        return f"{len(context)} chunks from {len(sources)} sources"

    def should_pass(self, feedback: CriticFeedback) -> bool:
        """Check if response passes quality thresholds."""
        return (
            feedback.relevance_score >= self.thresholds["relevance"] and
            feedback.completeness_score >= self.thresholds["completeness"] and
            feedback.accuracy_score >= self.thresholds["accuracy"] and
            feedback.coherence_score >= self.thresholds["coherence"] and
            feedback.overall_score >= self.thresholds["overall"]
        )

    async def validate_input(self, state: AgentState) -> bool:
        """Validate that we have a response to evaluate."""
        return bool(state.draft_responses)
