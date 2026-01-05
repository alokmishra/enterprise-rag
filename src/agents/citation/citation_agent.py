"""
Enterprise RAG System - Citation Agent

The Citation Agent links response claims to source documents
and formats proper citations.
"""

import re
import time
from typing import Optional

from src.agents.base import BaseAgent, AgentConfig, AgentResult
from src.core.types import (
    AgentState,
    AgentType,
    Citation,
    ContextItem,
)


class CitationAgent(BaseAgent):
    """
    Agent responsible for managing citations.

    The citation agent:
    1. Extracts citation references from the response
    2. Links them to source documents
    3. Validates citation accuracy
    4. Formats citations consistently
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        config = config or AgentConfig(
            name="citation",
            agent_type=AgentType.CITATION,
            timeout_seconds=10,
        )
        super().__init__(config)

    async def execute(self, state: AgentState, **kwargs) -> AgentResult:
        """
        Process citations in the response.

        Args:
            state: Current agent state with response and context

        Returns:
            AgentResult containing citations and annotated response
        """
        start_time = time.time()

        try:
            if not state.draft_responses:
                return AgentResult(
                    success=False,
                    output=None,
                    error="No response to process",
                    latency_ms=0,
                )

            response = state.draft_responses[-1]
            context = state.retrieved_context

            # Extract and validate citations
            citations = self._extract_citations(response, context)

            # Verify citation accuracy
            verified_citations = self._verify_citations(citations, context)

            # Add missing citations if needed
            enhanced_citations = self._add_missing_citations(
                response, context, verified_citations
            )

            latency_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "Citations processed",
                extracted=len(citations),
                verified=len(verified_citations),
                final=len(enhanced_citations),
                trace_id=state.trace_id,
            )

            return AgentResult(
                success=True,
                output={
                    "citations": enhanced_citations,
                    "citation_count": len(enhanced_citations),
                },
                latency_ms=latency_ms,
            )

        except Exception as e:
            self.logger.error(
                "Citation processing failed",
                error=str(e),
                trace_id=state.trace_id,
            )
            return AgentResult(
                success=False,
                output={"citations": []},
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _extract_citations(
        self,
        response: str,
        context: list[ContextItem],
    ) -> list[Citation]:
        """Extract citation references from response."""
        citations = []

        # Find all [N] references
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, response)

        seen_ids = set()
        for match in matches:
            idx = int(match) - 1  # Convert to 0-based index

            if idx < 0 or idx >= len(context):
                continue

            cite_id = f"cite-{match}"
            if cite_id in seen_ids:
                continue

            seen_ids.add(cite_id)
            item = context[idx]

            citations.append(Citation(
                id=cite_id,
                document_id=item.document_id,
                chunk_id=item.chunk_id,
                source=item.source,
                title=None,  # Could be extracted from metadata
                text_span=None,
                url=None,
            ))

        return citations

    def _verify_citations(
        self,
        citations: list[Citation],
        context: list[ContextItem],
    ) -> list[Citation]:
        """Verify that citations are valid."""
        verified = []

        context_ids = {item.chunk_id for item in context}

        for citation in citations:
            if citation.chunk_id in context_ids:
                verified.append(citation)
            else:
                self.logger.warning(
                    "Invalid citation reference",
                    citation_id=citation.id,
                    chunk_id=citation.chunk_id,
                )

        return verified

    def _add_missing_citations(
        self,
        response: str,
        context: list[ContextItem],
        existing_citations: list[Citation],
    ) -> list[Citation]:
        """
        Add citations for content that appears to come from sources
        but wasn't explicitly cited.

        This is a simple implementation - could be enhanced with
        semantic matching.
        """
        # For now, just return existing citations
        # A more advanced implementation would:
        # 1. Find text spans in response that match context
        # 2. Add citations for uncited matches
        return existing_citations

    def format_bibliography(
        self,
        citations: list[Citation],
        style: str = "numbered",
    ) -> str:
        """
        Format citations as a bibliography.

        Args:
            citations: List of citations
            style: "numbered", "author-date", or "footnote"

        Returns:
            Formatted bibliography string
        """
        if not citations:
            return ""

        lines = ["\n\n---\n**Sources:**\n"]

        for i, citation in enumerate(citations, 1):
            if style == "numbered":
                line = f"[{i}] {citation.source}"
                if citation.title:
                    line += f" - {citation.title}"
                if citation.url:
                    line += f" ({citation.url})"
                lines.append(line)

        return "\n".join(lines)

    async def validate_input(self, state: AgentState) -> bool:
        """Validate that we have a response and context."""
        return bool(state.draft_responses and state.retrieved_context)
