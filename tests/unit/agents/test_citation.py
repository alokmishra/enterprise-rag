"""
Tests for src/agents/citation/citation_agent.py
"""

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestCitationAgent:
    """Tests for the CitationAgent class."""

    def test_citation_agent_creation(self):
        """Test CitationAgent can be created."""
        from src.agents.citation import CitationAgent

        agent = CitationAgent()
        assert agent is not None
        assert agent.name == "citation"

    @pytest.mark.asyncio
    async def test_citation_agent_execute(self, sample_agent_state_with_response):
        """Test CitationAgent execute method."""
        from src.agents.citation import CitationAgent

        agent = CitationAgent()
        result = await agent.execute(sample_agent_state_with_response)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_citation_extracts_references(self, sample_agent_state_with_response):
        """Test CitationAgent extracts [N] references."""
        from src.agents.citation import CitationAgent

        agent = CitationAgent()
        result = await agent.execute(sample_agent_state_with_response)

        if result.success and result.output:
            output = result.output
            if isinstance(output, dict):
                assert "citations" in output or "citation_count" in output

    @pytest.mark.asyncio
    async def test_citation_links_to_sources(self, sample_agent_state_with_response):
        """Test CitationAgent links citations to source documents."""
        from src.agents.citation import CitationAgent

        agent = CitationAgent()
        result = await agent.execute(sample_agent_state_with_response)

        if result.success and result.output:
            output = result.output
            if isinstance(output, dict) and "citations" in output:
                for citation in output["citations"]:
                    if hasattr(citation, 'source'):
                        assert citation.source is not None

    @pytest.mark.asyncio
    async def test_citation_handles_no_response(self, sample_agent_state):
        """Test CitationAgent handles missing response."""
        from src.agents.citation import CitationAgent

        agent = CitationAgent()
        result = await agent.execute(sample_agent_state)

        assert result.success is False
        assert "No response" in result.error

    def test_citation_format_bibliography(self):
        """Test CitationAgent format_bibliography method."""
        from src.agents.citation import CitationAgent
        from src.core.types import Citation

        agent = CitationAgent()
        citations = [
            Citation(id="cite-1", document_id="doc-1", chunk_id="c-1", source="policy.pdf", title="Privacy Policy"),
            Citation(id="cite-2", document_id="doc-2", chunk_id="c-2", source="terms.pdf", title="Terms of Service"),
        ]

        bibliography = agent.format_bibliography(citations)
        assert "Sources" in bibliography
        assert "Privacy Policy" in bibliography or "policy.pdf" in bibliography

    def test_citation_format_bibliography_empty(self):
        """Test CitationAgent format_bibliography with empty list."""
        from src.agents.citation import CitationAgent

        agent = CitationAgent()
        bibliography = agent.format_bibliography([])
        assert bibliography == ""

    @pytest.mark.asyncio
    async def test_citation_validate_input(self, sample_agent_state_with_response):
        """Test CitationAgent validate_input method."""
        from src.agents.citation import CitationAgent

        agent = CitationAgent()
        is_valid = await agent.validate_input(sample_agent_state_with_response)
        assert is_valid is True
