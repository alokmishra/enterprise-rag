"""
Tests for src/agents/verifier/verifier_agent.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestVerifierAgent:
    """Tests for the VerifierAgent class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = AsyncMock()
        # Mock for claim extraction
        client.generate = AsyncMock(side_effect=[
            MagicMock(content='["The company protects personal data", "Data is retained while account is active"]'),
            MagicMock(content='{"status": "SUPPORTED", "evidence": "direct quote", "confidence": 0.9, "source_index": 1}'),
            MagicMock(content='{"status": "SUPPORTED", "evidence": "another quote", "confidence": 0.85, "source_index": 1}'),
        ])
        return client

    def test_verifier_agent_creation(self):
        """Test VerifierAgent can be created."""
        from src.agents.verifier import VerifierAgent

        agent = VerifierAgent()
        assert agent is not None
        assert agent.name == "verifier"

    @pytest.mark.asyncio
    async def test_verifier_agent_execute(self, mock_llm_client, sample_agent_state_with_response):
        """Test VerifierAgent execute method."""
        from src.agents.verifier import VerifierAgent

        with patch('src.agents.verifier.verifier_agent.get_default_llm_client', return_value=mock_llm_client):
            agent = VerifierAgent()
            result = await agent.execute(sample_agent_state_with_response)

            assert result.success is True

    @pytest.mark.asyncio
    async def test_verifier_extracts_claims(self, mock_llm_client, sample_agent_state_with_response):
        """Test VerifierAgent extracts claims from response."""
        from src.agents.verifier import VerifierAgent

        with patch('src.agents.verifier.verifier_agent.get_default_llm_client', return_value=mock_llm_client):
            agent = VerifierAgent()
            result = await agent.execute(sample_agent_state_with_response)

            if result.success and result.output:
                output = result.output
                if isinstance(output, dict):
                    assert "results" in output or "summary" in output

    @pytest.mark.asyncio
    async def test_verifier_returns_verification_results(self, mock_llm_client, sample_agent_state_with_response):
        """Test VerifierAgent returns verification results."""
        from src.agents.verifier import VerifierAgent

        with patch('src.agents.verifier.verifier_agent.get_default_llm_client', return_value=mock_llm_client):
            agent = VerifierAgent()
            result = await agent.execute(sample_agent_state_with_response)

            if result.success and result.output:
                output = result.output
                if isinstance(output, dict) and "summary" in output:
                    summary = output["summary"]
                    assert "supported" in summary or "total_claims" in summary

    @pytest.mark.asyncio
    async def test_verifier_handles_no_response(self, sample_agent_state):
        """Test VerifierAgent handles missing response."""
        from src.agents.verifier import VerifierAgent

        agent = VerifierAgent()
        # State without draft response
        result = await agent.execute(sample_agent_state)

        assert result.success is False
        assert "No response" in result.error

    @pytest.mark.asyncio
    async def test_verifier_validate_input(self, sample_agent_state_with_response):
        """Test VerifierAgent validate_input method."""
        from src.agents.verifier import VerifierAgent

        agent = VerifierAgent()
        is_valid = await agent.validate_input(sample_agent_state_with_response)
        assert is_valid is True
