"""
Enterprise RAG System - Formatter Agent

The Formatter Agent transforms responses into the requested
output format with consistent styling.
"""

import json
import re
import time
from enum import Enum
from typing import Optional

from src.agents.base import BaseAgent, AgentConfig, AgentResult
from src.core.types import (
    AgentState,
    AgentType,
    Citation,
)


class OutputFormat(str, Enum):
    """Supported output formats."""
    MARKDOWN = "markdown"
    PLAIN_TEXT = "plain_text"
    HTML = "html"
    JSON = "json"
    STRUCTURED = "structured"


class FormatterAgent(BaseAgent):
    """
    Agent responsible for formatting responses.

    The formatter:
    1. Applies the requested output format
    2. Adds citations and sources
    3. Structures content consistently
    4. Handles special formatting requirements
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        config = config or AgentConfig(
            name="formatter",
            agent_type=AgentType.FORMATTER,
            timeout_seconds=5,
        )
        super().__init__(config)

    async def execute(self, state: AgentState, **kwargs) -> AgentResult:
        """
        Format the final response.

        Args:
            state: Current agent state with response and citations

        Returns:
            AgentResult containing formatted response
        """
        start_time = time.time()

        try:
            if not state.draft_responses:
                return AgentResult(
                    success=False,
                    output=None,
                    error="No response to format",
                    latency_ms=0,
                )

            response = state.draft_responses[-1]
            output_format = kwargs.get("format", OutputFormat.MARKDOWN)
            include_sources = kwargs.get("include_sources", True)

            # Get citations if available
            citations = self._get_citations(state)

            # Format based on requested format
            formatted = self._format_response(
                response=response,
                citations=citations,
                output_format=output_format,
                include_sources=include_sources,
            )

            latency_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "Response formatted",
                format=output_format.value if isinstance(output_format, OutputFormat) else output_format,
                has_citations=bool(citations),
                trace_id=state.trace_id,
            )

            return AgentResult(
                success=True,
                output={
                    "formatted_response": formatted,
                    "format": output_format.value if isinstance(output_format, OutputFormat) else output_format,
                    "citation_count": len(citations),
                },
                latency_ms=latency_ms,
            )

        except Exception as e:
            self.logger.error(
                "Formatting failed",
                error=str(e),
                trace_id=state.trace_id,
            )
            return AgentResult(
                success=False,
                output={"formatted_response": state.draft_responses[-1] if state.draft_responses else ""},
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _get_citations(self, state: AgentState) -> list[Citation]:
        """Extract citations from state."""
        citations = []

        # Check if citations are in the state
        if hasattr(state, 'citations') and state.citations:
            return state.citations

        # Try to extract from metadata
        if state.metadata and "citations" in state.metadata:
            citation_data = state.metadata["citations"]
            for c in citation_data:
                if isinstance(c, Citation):
                    citations.append(c)
                elif isinstance(c, dict):
                    citations.append(Citation(
                        id=c.get("id", ""),
                        document_id=c.get("document_id", ""),
                        chunk_id=c.get("chunk_id", ""),
                        source=c.get("source", ""),
                        title=c.get("title"),
                        text_span=c.get("text_span"),
                        url=c.get("url"),
                    ))

        return citations

    def _format_response(
        self,
        response: str,
        citations: list[Citation],
        output_format: OutputFormat,
        include_sources: bool,
    ) -> str:
        """Format response based on output format."""
        if output_format == OutputFormat.MARKDOWN:
            return self._format_markdown(response, citations, include_sources)
        elif output_format == OutputFormat.PLAIN_TEXT:
            return self._format_plain_text(response, citations, include_sources)
        elif output_format == OutputFormat.HTML:
            return self._format_html(response, citations, include_sources)
        elif output_format == OutputFormat.JSON:
            return self._format_json(response, citations)
        elif output_format == OutputFormat.STRUCTURED:
            return self._format_structured(response, citations)
        else:
            return response

    def _format_markdown(
        self,
        response: str,
        citations: list[Citation],
        include_sources: bool,
    ) -> str:
        """Format as Markdown."""
        formatted = response

        if include_sources and citations:
            formatted += "\n\n---\n\n**Sources:**\n"
            for i, citation in enumerate(citations, 1):
                source_line = f"{i}. "
                if citation.title:
                    source_line += f"**{citation.title}**"
                else:
                    source_line += f"*{citation.source}*"
                if citation.url:
                    source_line = f"{i}. [{citation.title or citation.source}]({citation.url})"
                formatted += source_line + "\n"

        return formatted

    def _format_plain_text(
        self,
        response: str,
        citations: list[Citation],
        include_sources: bool,
    ) -> str:
        """Format as plain text."""
        # Remove markdown formatting
        plain = response

        # Remove bold/italic markers
        plain = re.sub(r'\*\*(.+?)\*\*', r'\1', plain)
        plain = re.sub(r'\*(.+?)\*', r'\1', plain)
        plain = re.sub(r'__(.+?)__', r'\1', plain)
        plain = re.sub(r'_(.+?)_', r'\1', plain)

        # Remove markdown links, keep text
        plain = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', plain)

        # Remove headers markers
        plain = re.sub(r'^#{1,6}\s+', '', plain, flags=re.MULTILINE)

        # Remove code blocks
        plain = re.sub(r'```.*?```', '', plain, flags=re.DOTALL)
        plain = re.sub(r'`(.+?)`', r'\1', plain)

        if include_sources and citations:
            plain += "\n\nSources:\n"
            for i, citation in enumerate(citations, 1):
                source = citation.title or citation.source
                if citation.url:
                    plain += f"{i}. {source} ({citation.url})\n"
                else:
                    plain += f"{i}. {source}\n"

        return plain

    def _format_html(
        self,
        response: str,
        citations: list[Citation],
        include_sources: bool,
    ) -> str:
        """Format as HTML."""
        # Convert markdown to basic HTML
        html = response

        # Convert headers
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

        # Convert bold/italic
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

        # Convert links
        html = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', html)

        # Convert paragraphs
        paragraphs = html.split('\n\n')
        html = ''.join(f'<p>{p}</p>' if not p.startswith('<h') else p for p in paragraphs if p.strip())

        # Convert inline code
        html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)

        # Convert lists
        html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)

        if include_sources and citations:
            html += '<hr><h4>Sources</h4><ol>'
            for citation in citations:
                source = citation.title or citation.source
                if citation.url:
                    html += f'<li><a href="{citation.url}">{source}</a></li>'
                else:
                    html += f'<li>{source}</li>'
            html += '</ol>'

        return html

    def _format_json(
        self,
        response: str,
        citations: list[Citation],
    ) -> str:
        """Format as JSON."""
        result = {
            "response": response,
            "citations": [
                {
                    "id": c.id,
                    "source": c.source,
                    "title": c.title,
                    "url": c.url,
                    "document_id": c.document_id,
                }
                for c in citations
            ],
        }
        return json.dumps(result, indent=2)

    def _format_structured(
        self,
        response: str,
        citations: list[Citation],
    ) -> str:
        """Format as structured output with sections."""
        sections = self._extract_sections(response)

        structured = {
            "summary": sections.get("summary", response[:200] + "..." if len(response) > 200 else response),
            "main_content": response,
            "key_points": sections.get("key_points", []),
            "citations": [
                {
                    "id": c.id,
                    "source": c.source,
                    "title": c.title,
                }
                for c in citations
            ],
        }

        return json.dumps(structured, indent=2)

    def _extract_sections(self, response: str) -> dict:
        """Extract structured sections from response."""
        sections = {}

        # Look for numbered or bulleted lists as key points
        bullet_pattern = r'^[-*•]\s+(.+)$'
        numbered_pattern = r'^\d+\.\s+(.+)$'

        key_points = []
        for match in re.finditer(bullet_pattern, response, re.MULTILINE):
            key_points.append(match.group(1))
        for match in re.finditer(numbered_pattern, response, re.MULTILINE):
            key_points.append(match.group(1))

        if key_points:
            sections["key_points"] = key_points[:5]  # Limit to 5

        # Extract first paragraph as summary
        paragraphs = response.split('\n\n')
        if paragraphs:
            first_para = paragraphs[0].strip()
            # Remove markdown headers
            first_para = re.sub(r'^#{1,6}\s+', '', first_para)
            sections["summary"] = first_para

        return sections

    def clean_response(self, response: str) -> str:
        """
        Clean up response text.

        - Remove excessive whitespace
        - Normalize line endings
        - Fix common formatting issues
        """
        # Normalize line endings
        cleaned = response.replace('\r\n', '\n')

        # Remove excessive blank lines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

        # Remove trailing whitespace
        cleaned = '\n'.join(line.rstrip() for line in cleaned.split('\n'))

        # Ensure proper spacing around headers
        cleaned = re.sub(r'([^\n])\n(#{1,6}\s)', r'\1\n\n\2', cleaned)

        return cleaned.strip()

    async def validate_input(self, state: AgentState) -> bool:
        """Validate that we have a response to format."""
        return bool(state.draft_responses)
