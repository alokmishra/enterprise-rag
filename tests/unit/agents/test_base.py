"""
Tests for src/agents/base.py
"""

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestBaseAgent:
    """Tests for the BaseAgent class."""

    def test_base_agent_is_abstract(self):
        """Test that BaseAgent is abstract."""
        from src.agents.base import BaseAgent
        from abc import ABC

        assert issubclass(BaseAgent, ABC)

    def test_base_agent_requires_execute(self):
        """Test that BaseAgent requires execute method."""
        from src.agents.base import BaseAgent

        assert hasattr(BaseAgent, 'execute')

    def test_base_agent_has_create_message(self):
        """Test that BaseAgent has create_message method."""
        from src.agents.base import BaseAgent

        assert hasattr(BaseAgent, 'create_message')

    def test_base_agent_has_validate_input(self):
        """Test that BaseAgent has validate_input method."""
        from src.agents.base import BaseAgent

        assert hasattr(BaseAgent, 'validate_input')


class TestAgentConfig:
    """Tests for the AgentConfig class."""

    def test_agent_config_creation(self):
        """Test AgentConfig can be created."""
        from src.agents.base import AgentConfig
        from src.core.types import AgentType

        config = AgentConfig(
            name="test_agent",
            agent_type=AgentType.PLANNER,
        )
        assert config.name == "test_agent"
        assert config.agent_type == AgentType.PLANNER

    def test_agent_config_defaults(self):
        """Test AgentConfig has sensible defaults."""
        from src.agents.base import AgentConfig
        from src.core.types import AgentType

        config = AgentConfig(
            name="test",
            agent_type=AgentType.SYNTHESIZER,
        )
        assert config.temperature == 0.0
        assert config.max_tokens == 4096
        assert config.timeout_seconds == 60

    def test_agent_config_custom_values(self):
        """Test AgentConfig with custom values."""
        from src.agents.base import AgentConfig
        from src.core.types import AgentType

        config = AgentConfig(
            name="custom",
            agent_type=AgentType.CRITIC,
            temperature=0.7,
            max_tokens=2000,
            timeout_seconds=30,
        )
        assert config.temperature == 0.7
        assert config.max_tokens == 2000


class TestAgentResult:
    """Tests for the AgentResult class."""

    def test_agent_result_success(self):
        """Test successful AgentResult."""
        from src.agents.base import AgentResult

        result = AgentResult(
            success=True,
            output={"key": "value"},
            latency_ms=50.0,
        )
        assert result.success is True
        assert result.error is None

    def test_agent_result_failure(self):
        """Test failed AgentResult."""
        from src.agents.base import AgentResult

        result = AgentResult(
            success=False,
            output=None,
            error="Something went wrong",
            latency_ms=10.0,
        )
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_agent_result_with_tokens(self):
        """Test AgentResult with token usage."""
        from src.agents.base import AgentResult

        result = AgentResult(
            success=True,
            output="response",
            tokens_used=150,
            latency_ms=100.0,
        )
        assert result.tokens_used == 150
