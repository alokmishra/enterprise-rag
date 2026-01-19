"""
Tests for src/generation/llm/
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLLMClient:
    """Tests for the base LLMClient."""

    def test_llm_client_is_abstract(self):
        """Test that LLMClient is abstract."""
        from src.generation.llm.base import LLMClient
        from abc import ABC

        assert issubclass(LLMClient, ABC)

    def test_llm_client_requires_generate(self):
        """Test that LLMClient requires generate method."""
        from src.generation.llm.base import LLMClient

        assert hasattr(LLMClient, 'generate')


class TestAnthropicClient:
    """Tests for the AnthropicClient."""

    def test_anthropic_client_creation(self):
        """Test AnthropicClient can be created."""
        from src.generation.llm.anthropic import AnthropicClient

        client = AnthropicClient(api_key='test-key')
        assert client is not None

    def test_anthropic_client_model_selection(self):
        """Test AnthropicClient model selection."""
        from src.generation.llm.anthropic import AnthropicClient

        client = AnthropicClient(api_key='test-key', default_model="claude-3-sonnet-20240229")
        assert "claude" in client.default_model

    @pytest.mark.asyncio
    async def test_anthropic_client_generate(self):
        """Test AnthropicClient generate method."""
        from src.generation.llm.anthropic import AnthropicClient
        from src.generation.llm.base import LLMMessage

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Generated response")]
        mock_response.usage = MagicMock(input_tokens=50, output_tokens=100)
        mock_response.stop_reason = "end_turn"
        mock_response.id = "msg_123"
        mock_response.type = "message"

        client = AnthropicClient(api_key='test-key')

        with patch.object(client, '_get_client') as mock_get_client:
            mock_anthropic = MagicMock()
            mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_anthropic

            result = await client.generate(
                messages=[LLMMessage(role="user", content="Test prompt")],
            )
            assert result.content is not None
            assert result.input_tokens == 50
            assert result.output_tokens == 100


class TestOpenAIClient:
    """Tests for the OpenAIClient."""

    def test_openai_client_creation(self):
        """Test OpenAIClient can be created."""
        from src.generation.llm.openai import OpenAIClient

        client = OpenAIClient(api_key='test-key')
        assert client is not None

    def test_openai_client_model_selection(self):
        """Test OpenAIClient model selection."""
        from src.generation.llm.openai import OpenAIClient

        client = OpenAIClient(api_key='test-key', default_model="gpt-4o")
        assert "gpt" in client.default_model

    @pytest.mark.asyncio
    async def test_openai_client_generate(self):
        """Test OpenAIClient generate method."""
        from src.generation.llm.openai import OpenAIClient
        from src.generation.llm.base import LLMMessage

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Generated response"), finish_reason="stop")]
        mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=100, total_tokens=150)
        mock_response.id = "chatcmpl_123"

        client = OpenAIClient(api_key='test-key')

        with patch.object(client, '_get_client') as mock_get_client:
            mock_openai = MagicMock()
            mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_openai

            result = await client.generate(
                messages=[LLMMessage(role="user", content="Test prompt")],
            )
            assert result.content is not None
            assert result.input_tokens == 50
            assert result.output_tokens == 100


class TestLLMFactory:
    """Tests for the LLM factory function."""

    def test_get_llm_client_anthropic(self):
        """Test getting Anthropic client."""
        from src.generation.llm.factory import get_llm_client
        from src.generation.llm.anthropic import AnthropicClient

        with patch('src.generation.llm.factory.get_anthropic_client') as mock_get:
            mock_get.return_value = MagicMock(spec=AnthropicClient)
            client = get_llm_client("anthropic")
            assert client is not None

    def test_get_llm_client_openai(self):
        """Test getting OpenAI client."""
        from src.generation.llm.factory import get_llm_client
        from src.generation.llm.openai import OpenAIClient

        with patch('src.generation.llm.factory.get_openai_client') as mock_get:
            mock_get.return_value = MagicMock(spec=OpenAIClient)
            client = get_llm_client("openai")
            assert client is not None

    def test_get_llm_client_invalid_provider(self):
        """Test getting invalid provider raises error."""
        from src.generation.llm.factory import get_llm_client
        from src.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            get_llm_client("invalid_provider")


class TestLLMMessage:
    """Tests for the LLMMessage model."""

    def test_llm_message_creation(self):
        """Test LLMMessage can be created."""
        from src.generation.llm.base import LLMMessage

        message = LLMMessage(role="user", content="Test message")
        assert message.role == "user"
        assert message.content == "Test message"

    def test_llm_message_roles(self):
        """Test LLMMessage supports different roles."""
        from src.generation.llm.base import LLMMessage

        user_msg = LLMMessage(role="user", content="User message")
        assistant_msg = LLMMessage(role="assistant", content="Assistant message")
        system_msg = LLMMessage(role="system", content="System message")

        assert user_msg.role == "user"
        assert assistant_msg.role == "assistant"
        assert system_msg.role == "system"


class TestLLMResponse:
    """Tests for the LLMResponse model."""

    def test_llm_response_creation(self):
        """Test LLMResponse can be created."""
        from src.generation.llm.base import LLMResponse

        response = LLMResponse(
            content="Generated content",
            model="gpt-4o",
            input_tokens=50,
            output_tokens=100,
            total_tokens=150,
            finish_reason="stop",
        )
        assert response.content == "Generated content"
        assert response.total_tokens == 150
        assert response.input_tokens == 50
        assert response.output_tokens == 100
