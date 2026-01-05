"""
Enterprise RAG System - Verifier Agent

The Verifier Agent fact-checks claims in the generated response
against the retrieved source documents.
"""

import json
import re
import time
from typing import Any, Optional

from src.agents.base import BaseAgent, AgentConfig, AgentResult
from src.core.types import (
    AgentState,
    AgentType,
    VerificationResult,
    VerificationStatus,
)
from src.generation.llm import get_default_llm_client, LLMMessage


class VerifierAgent(BaseAgent):
    """
    Agent responsible for fact-checking generated responses.

    The verifier:
    1. Extracts claims from the response
    2. Checks each claim against source documents
    3. Classifies verification status
    4. Flags unsupported or contradicted claims
    """

    CLAIM_EXTRACTION_PROMPT = """Extract the main factual claims from the following response. Focus on verifiable statements.

Response:
{response}

Return a JSON array of claims. Each claim should be a single, verifiable statement.
Example: ["The company was founded in 2020", "Revenue increased by 15%"]

Return ONLY the JSON array, no other text."""

    VERIFICATION_PROMPT = """Verify the following claim against the provided source documents.

Claim: {claim}

Source Documents:
{sources}

Determine the verification status:
- SUPPORTED: Direct evidence supports the claim
- PARTIALLY_SUPPORTED: Some evidence exists, but incomplete
- NOT_FOUND: No relevant evidence in the sources
- CONTRADICTED: Evidence contradicts the claim

Return JSON:
{{
    "status": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "NOT_FOUND" | "CONTRADICTED",
    "evidence": "relevant quote or summary from sources",
    "confidence": 0.0-1.0,
    "source_index": 1-N (which source supports/contradicts)
}}

Return ONLY the JSON object."""

    def __init__(self, config: Optional[AgentConfig] = None):
        config = config or AgentConfig(
            name="verifier",
            agent_type=AgentType.VERIFIER,
            temperature=0.0,
            max_tokens=1000,
        )
        super().__init__(config)

    async def execute(self, state: AgentState, **kwargs) -> AgentResult:
        """
        Verify claims in the generated response.

        Args:
            state: Current agent state with response and context

        Returns:
            AgentResult containing verification results
        """
        start_time = time.time()

        try:
            # Get the latest response
            if not state.draft_responses:
                return AgentResult(
                    success=False,
                    output=[],
                    error="No response to verify",
                    latency_ms=0,
                )

            response = state.draft_responses[-1]

            # Extract claims
            claims = await self._extract_claims(response)

            if not claims:
                self.logger.info(
                    "No claims to verify",
                    trace_id=state.trace_id,
                )
                return AgentResult(
                    success=True,
                    output=[],
                    latency_ms=(time.time() - start_time) * 1000,
                )

            # Verify each claim
            results = []
            for claim in claims[:10]:  # Limit to 10 claims
                result = await self._verify_claim(claim, state.retrieved_context)
                results.append(result)

            # Update state
            state.verification_results = [r.__dict__ for r in results]

            latency_ms = (time.time() - start_time) * 1000

            # Calculate summary
            supported = sum(1 for r in results if r.status == VerificationStatus.SUPPORTED)
            partial = sum(1 for r in results if r.status == VerificationStatus.PARTIALLY_SUPPORTED)
            not_found = sum(1 for r in results if r.status == VerificationStatus.NOT_FOUND)
            contradicted = sum(1 for r in results if r.status == VerificationStatus.CONTRADICTED)

            self.logger.info(
                "Verification complete",
                claims_count=len(claims),
                supported=supported,
                partial=partial,
                not_found=not_found,
                contradicted=contradicted,
                trace_id=state.trace_id,
            )

            return AgentResult(
                success=True,
                output={
                    "results": results,
                    "summary": {
                        "total_claims": len(results),
                        "supported": supported,
                        "partially_supported": partial,
                        "not_found": not_found,
                        "contradicted": contradicted,
                    },
                },
                latency_ms=latency_ms,
            )

        except Exception as e:
            self.logger.error(
                "Verification failed",
                error=str(e),
                trace_id=state.trace_id,
            )
            return AgentResult(
                success=False,
                output=[],
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def _extract_claims(self, response: str) -> list[str]:
        """Extract factual claims from the response."""
        prompt = self.CLAIM_EXTRACTION_PROMPT.format(response=response)

        llm = get_default_llm_client()
        result = await llm.generate(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=500,
        )

        try:
            # Parse JSON array
            start = result.content.find("[")
            end = result.content.rfind("]") + 1
            if start >= 0 and end > start:
                claims = json.loads(result.content[start:end])
                return [str(c) for c in claims if c]
        except json.JSONDecodeError:
            pass

        return []

    async def _verify_claim(
        self,
        claim: str,
        context: list,
    ) -> VerificationResult:
        """Verify a single claim against source documents."""
        # Format sources
        sources_text = []
        for i, item in enumerate(context[:5], 1):  # Limit sources
            sources_text.append(f"[{i}] {item.source}:\n{item.content[:500]}")

        prompt = self.VERIFICATION_PROMPT.format(
            claim=claim,
            sources="\n\n".join(sources_text),
        )

        llm = get_default_llm_client()
        result = await llm.generate(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=300,
        )

        # Parse result
        try:
            start = result.content.find("{")
            end = result.content.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(result.content[start:end])

                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus(data.get("status", "NOT_FOUND")),
                    evidence=data.get("evidence"),
                    source=str(data.get("source_index", "")),
                    confidence=float(data.get("confidence", 0.5)),
                )
        except (json.JSONDecodeError, ValueError):
            pass

        # Default to uncertain
        return VerificationResult(
            claim=claim,
            status=VerificationStatus.UNCERTAIN,
            evidence=None,
            source=None,
            confidence=0.0,
        )

    async def validate_input(self, state: AgentState) -> bool:
        """Validate that we have a response to verify."""
        return bool(state.draft_responses)
