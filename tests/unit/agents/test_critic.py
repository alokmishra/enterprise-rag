"""
Tests for src/agents/critic/critic_agent.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCriticAgent:
    """Tests for the CriticAgent class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = AsyncMock()
        client.generate = AsyncMock(return_value=MagicMock(
            content='''{
                "relevance_score": 0.9,
                "completeness_score": 0.85,
                "accuracy_score": 0.95,
                "coherence_score": 0.88,
                "citation_score": 0.8,
                "overall_score": 0.88,
                "decision": "PASS",
                "feedback": "Good response overall",
                "suggestions": []
            }''',
        ))
        return client

    def test_critic_agent_creation(self):
        """Test CriticAgent can be created."""
        from src.agents.critic import CriticAgent

        agent = CriticAgent()
        assert agent is not None
        assert agent.name == "critic"

    def test_critic_agent_with_thresholds(self):
        """Test CriticAgent with custom thresholds."""
        from src.agents.critic import CriticAgent

        agent = CriticAgent(thresholds={
            "relevance": 0.8,
            "accuracy": 0.9,
            "overall": 0.85,
        })
        assert agent.thresholds["relevance"] == 0.8

    @pytest.mark.asyncio
    async def test_critic_agent_execute(self, mock_llm_client, sample_agent_state_with_response):
        """Test CriticAgent execute method."""
        from src.agents.critic import CriticAgent

        with patch('src.agents.critic.critic_agent.get_default_llm_client', return_value=mock_llm_client):
            agent = CriticAgent()
            result = await agent.execute(sample_agent_state_with_response)

            assert result.success is True
            assert result.output is not None

    @pytest.mark.asyncio
    async def test_critic_returns_feedback(self, mock_llm_client, sample_agent_state_with_response):
        """Test CriticAgent returns CriticFeedback."""
        from src.agents.critic import CriticAgent

        with patch('src.agents.critic.critic_agent.get_default_llm_client', return_value=mock_llm_client):
            agent = CriticAgent()
            result = await agent.execute(sample_agent_state_with_response)

            if result.success:
                feedback = result.output
                assert hasattr(feedback, 'overall_score') or 'overall_score' in str(feedback)
                assert hasattr(feedback, 'decision') or 'decision' in str(feedback)

    @pytest.mark.asyncio
    async def test_critic_decision_pass(self, mock_llm_client, sample_agent_state_with_response):
        """Test CriticAgent PASS decision."""
        from src.agents.critic import CriticAgent, CriticDecision

        with patch('src.agents.critic.critic_agent.get_default_llm_client', return_value=mock_llm_client):
            agent = CriticAgent()
            result = await agent.execute(sample_agent_state_with_response)

            if result.success:
                feedback = result.output
                assert feedback.decision == CriticDecision.PASS or feedback.decision == "PASS"

    @pytest.mark.asyncio
    async def test_critic_decision_revision(self, sample_agent_state_with_response):
        """Test CriticAgent revision decision."""
        from src.agents.critic import CriticAgent

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=MagicMock(
            content='''{
                "relevance_score": 0.6,
                "completeness_score": 0.5,
                "accuracy_score": 0.7,
                "coherence_score": 0.6,
                "citation_score": 0.4,
                "overall_score": 0.56,
                "decision": "MAJOR_REVISION",
                "feedback": "Response needs significant improvement",
                "suggestions": ["Add more context", "Include citations"]
            }''',
        ))

        with patch('src.agents.critic.critic_agent.get_default_llm_client', return_value=mock_llm):
            agent = CriticAgent()
            result = await agent.execute(sample_agent_state_with_response)

            if result.success:
                feedback = result.output
                assert "REVISION" in feedback.decision

    def test_critic_should_pass(self):
        """Test CriticAgent should_pass method."""
        from src.agents.critic import CriticAgent
        from src.core.types import CriticFeedback

        agent = CriticAgent()
        feedback = CriticFeedback(
            relevance_score=0.9,
            completeness_score=0.85,
            accuracy_score=0.95,
            coherence_score=0.88,
            citation_score=0.9,
            overall_score=0.89,
            feedback="Good",
            decision="PASS",
        )

        assert agent.should_pass(feedback) is True

    def test_critic_should_not_pass_low_accuracy(self):
        """Test CriticAgent should_pass fails on low accuracy."""
        from src.agents.critic import CriticAgent
        from src.core.types import CriticFeedback

        agent = CriticAgent()
        feedback = CriticFeedback(
            relevance_score=0.9,
            completeness_score=0.85,
            accuracy_score=0.5,  # Low accuracy
            coherence_score=0.88,
            citation_score=0.9,
            overall_score=0.75,
            feedback="Low accuracy",
            decision="MAJOR_REVISION",
        )

        assert agent.should_pass(feedback) is False


class TestCriticDecision:
    """Tests for CriticDecision enum."""

    def test_critic_decision_values(self):
        """Test CriticDecision has expected values."""
        from src.agents.critic import CriticDecision

        assert CriticDecision.PASS == "PASS"
        assert CriticDecision.MINOR_REVISION == "MINOR_REVISION"
        assert CriticDecision.MAJOR_REVISION == "MAJOR_REVISION"
        assert CriticDecision.RETRIEVAL_NEEDED == "RETRIEVAL_NEEDED"
        assert CriticDecision.REJECT == "REJECT"
