"""
Tests for src/agents/synthesizer/synthesizer_agent.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSynthesizerAgent:
    """Tests for the SynthesizerAgent class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = AsyncMock()
        client.generate = AsyncMock(return_value=MagicMock(
            content="Based on the provided context, here is the answer...",
            tokens_used=150,
        ))
        return client

    def test_synthesizer_agent_creation(self):
        """Test SynthesizerAgent can be created."""
        from src.agents.synthesizer import SynthesizerAgent

        agent = SynthesizerAgent()
        assert agent is not None
        assert agent.name == "synthesizer"

    @pytest.mark.asyncio
    async def test_synthesizer_agent_execute(self, mock_llm_client, sample_agent_state):
        """Test SynthesizerAgent execute method."""
        from src.agents.synthesizer import SynthesizerAgent

        with patch('src.agents.synthesizer.synthesizer_agent.get_default_llm_client', return_value=mock_llm_client):
            agent = SynthesizerAgent()
            result = await agent.execute(sample_agent_state)

            assert result.success is True
            assert result.output is not None

    @pytest.mark.asyncio
    async def test_synthesizer_generates_response(self, mock_llm_client, sample_agent_state):
        """Test SynthesizerAgent generates a response."""
        from src.agents.synthesizer import SynthesizerAgent

        with patch('src.agents.synthesizer.synthesizer_agent.get_default_llm_client', return_value=mock_llm_client):
            agent = SynthesizerAgent()
            result = await agent.execute(sample_agent_state)

            if result.success and result.output:
                response = result.output.get("response", result.output)
                assert len(str(response)) > 0

    @pytest.mark.asyncio
    async def test_synthesizer_uses_context(self, mock_llm_client, sample_agent_state):
        """Test SynthesizerAgent uses provided context."""
        from src.agents.synthesizer import SynthesizerAgent

        with patch('src.agents.synthesizer.synthesizer_agent.get_default_llm_client', return_value=mock_llm_client):
            agent = SynthesizerAgent()
            await agent.execute(sample_agent_state)

            # Check that LLM was called with context
            mock_llm_client.generate.assert_called()
            call_args = mock_llm_client.generate.call_args
            # Context should be in the messages

    @pytest.mark.asyncio
    async def test_synthesizer_handles_revision(self, mock_llm_client, sample_agent_state_with_response):
        """Test SynthesizerAgent handles revision requests."""
        from src.agents.synthesizer import SynthesizerAgent

        sample_agent_state_with_response.critic_feedback.append({
            "decision": "MINOR_REVISION",
            "suggestions": ["Add more detail about privacy"],
        })

        with patch('src.agents.synthesizer.synthesizer_agent.get_default_llm_client', return_value=mock_llm_client):
            agent = SynthesizerAgent()
            result = await agent.execute(sample_agent_state_with_response)

            # Should generate revised response
            assert result.success is True

    @pytest.mark.asyncio
    async def test_synthesizer_validate_input(self, sample_agent_state):
        """Test SynthesizerAgent validate_input method."""
        from src.agents.synthesizer import SynthesizerAgent

        agent = SynthesizerAgent()
        is_valid = await agent.validate_input(sample_agent_state)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_synthesizer_no_context_fails(self):
        """Test SynthesizerAgent fails gracefully without context."""
        from src.agents.synthesizer import SynthesizerAgent
        from src.core.types import AgentState

        agent = SynthesizerAgent()
        state = AgentState(
            trace_id="test",
            original_query="test query",
            retrieved_context=[],
        )
        is_valid = await agent.validate_input(state)
        # No context should be invalid
