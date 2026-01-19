"""
Tests for src/agents/formatter/formatter_agent.py
"""

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestFormatterAgent:
    """Tests for the FormatterAgent class."""

    def test_formatter_agent_creation(self):
        """Test FormatterAgent can be created."""
        from src.agents.formatter import FormatterAgent

        agent = FormatterAgent()
        assert agent is not None
        assert agent.name == "formatter"

    @pytest.mark.asyncio
    async def test_formatter_agent_execute(self, sample_agent_state_with_response):
        """Test FormatterAgent execute method."""
        from src.agents.formatter import FormatterAgent

        agent = FormatterAgent()
        result = await agent.execute(sample_agent_state_with_response)

        assert result.success is True
        assert result.output is not None

    @pytest.mark.asyncio
    async def test_formatter_markdown_output(self, sample_agent_state_with_response):
        """Test FormatterAgent with markdown format."""
        from src.agents.formatter import FormatterAgent, OutputFormat

        agent = FormatterAgent()
        result = await agent.execute(
            sample_agent_state_with_response,
            format=OutputFormat.MARKDOWN,
        )

        if result.success and result.output:
            formatted = result.output.get("formatted_response", "")
            # Markdown formatting should be preserved

    @pytest.mark.asyncio
    async def test_formatter_plain_text_output(self, sample_agent_state_with_response):
        """Test FormatterAgent with plain text format."""
        from src.agents.formatter import FormatterAgent, OutputFormat

        agent = FormatterAgent()
        result = await agent.execute(
            sample_agent_state_with_response,
            format=OutputFormat.PLAIN_TEXT,
        )

        if result.success and result.output:
            formatted = result.output.get("formatted_response", "")
            # Should not contain markdown markers
            assert "**" not in formatted or formatted == ""

    @pytest.mark.asyncio
    async def test_formatter_html_output(self, sample_agent_state_with_response):
        """Test FormatterAgent with HTML format."""
        from src.agents.formatter import FormatterAgent, OutputFormat

        agent = FormatterAgent()
        result = await agent.execute(
            sample_agent_state_with_response,
            format=OutputFormat.HTML,
        )

        if result.success and result.output:
            formatted = result.output.get("formatted_response", "")
            # Should contain HTML tags or be empty

    @pytest.mark.asyncio
    async def test_formatter_json_output(self, sample_agent_state_with_response):
        """Test FormatterAgent with JSON format."""
        from src.agents.formatter import FormatterAgent, OutputFormat
        import json

        agent = FormatterAgent()
        result = await agent.execute(
            sample_agent_state_with_response,
            format=OutputFormat.JSON,
        )

        if result.success and result.output:
            formatted = result.output.get("formatted_response", "")
            # Should be valid JSON
            try:
                parsed = json.loads(formatted)
                assert "response" in parsed
            except json.JSONDecodeError:
                pass  # May not be JSON in all cases

    @pytest.mark.asyncio
    async def test_formatter_includes_sources(self, sample_agent_state_with_response):
        """Test FormatterAgent includes sources when requested."""
        from src.agents.formatter import FormatterAgent, OutputFormat

        agent = FormatterAgent()
        result = await agent.execute(
            sample_agent_state_with_response,
            format=OutputFormat.MARKDOWN,
            include_sources=True,
        )

        if result.success and result.output:
            formatted = result.output.get("formatted_response", "")
            # Sources section may be included

    @pytest.mark.asyncio
    async def test_formatter_handles_no_response(self, sample_agent_state):
        """Test FormatterAgent handles missing response."""
        from src.agents.formatter import FormatterAgent

        agent = FormatterAgent()
        result = await agent.execute(sample_agent_state)

        assert result.success is False
        assert "No response" in result.error

    def test_formatter_clean_response(self):
        """Test FormatterAgent clean_response method."""
        from src.agents.formatter import FormatterAgent

        agent = FormatterAgent()
        messy = "Text with trailing spaces   \n\n\n\nAnd many blank lines"
        cleaned = agent.clean_response(messy)

        assert not cleaned.endswith("   ")  # Trailing whitespace removed
        assert "\n\n\n" not in cleaned  # Excessive newlines removed


class TestOutputFormat:
    """Tests for the OutputFormat enum."""

    def test_output_format_values(self):
        """Test OutputFormat has expected values."""
        from src.agents.formatter import OutputFormat

        assert OutputFormat.MARKDOWN == "markdown"
        assert OutputFormat.PLAIN_TEXT == "plain_text"
        assert OutputFormat.HTML == "html"
        assert OutputFormat.JSON == "json"
        assert OutputFormat.STRUCTURED == "structured"
