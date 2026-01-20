"""
Tests for local LLM client.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import ConfigurationError
from src.generation.llm.base import LLMMessage


class TestLocalLLMClientInit:
    """Tests for LocalLLMClient initialization."""

    def test_local_llm_client_init(self):
        """LocalLLMClient initializes with model path."""
        from src.generation.llm.local import LocalLLMClient

        with patch.dict(os.environ, {"LOCAL_LLM_MODEL_PATH": "/models/llama.gguf"}):
            client = LocalLLMClient(model_path="/models/llama.gguf", model_type="llama")

            assert client._model_path == "/models/llama.gguf"
            assert client._model_type == "llama"
            assert client._backend == "llama-cpp"

    def test_local_llm_requires_model_path(self):
        """LocalLLMClient raises if model path missing."""
        from src.generation.llm.local import LocalLLMClient

        with patch.dict(os.environ, {"LOCAL_LLM_MODEL_PATH": ""}):
            with pytest.raises(ConfigurationError) as exc_info:
                LocalLLMClient(model_path=None)

            assert "LOCAL_LLM_MODEL_PATH must be set" in str(exc_info.value)

    def test_local_llm_client_provider_name(self):
        """LocalLLMClient returns 'local' as provider."""
        from src.generation.llm.local import LocalLLMClient

        client = LocalLLMClient(model_path="/models/test.gguf")
        assert client.provider == "local"

    def test_local_llm_client_default_model(self):
        """LocalLLMClient returns model type as default model."""
        from src.generation.llm.local import LocalLLMClient

        client = LocalLLMClient(
            model_path="/models/test.gguf",
            model_type="mistral",
        )
        assert client.default_model == "mistral"

    def test_local_llm_client_accepts_backend_option(self):
        """LocalLLMClient accepts backend option."""
        from src.generation.llm.local import LocalLLMClient

        client = LocalLLMClient(
            model_path="/models/test",
            backend="transformers",
        )
        assert client._backend == "transformers"


class TestLocalLLMClientMessageFormatting:
    """Tests for message formatting."""

    def test_format_messages_default(self):
        """Default message formatting works."""
        from src.generation.llm.local import LocalLLMClient

        client = LocalLLMClient(model_path="/models/test.gguf")

        messages = [
            LLMMessage(role="system", content="You are helpful."),
            LLMMessage(role="user", content="Hello"),
        ]

        formatted = client._format_messages(messages)

        assert "<|system|>" in formatted
        assert "You are helpful." in formatted
        assert "<|user|>" in formatted
        assert "Hello" in formatted
        assert "<|assistant|>" in formatted

    def test_format_messages_llama(self):
        """Llama-style message formatting works."""
        from src.generation.llm.local import LocalLLMClient

        client = LocalLLMClient(model_path="/models/test.gguf", model_type="llama")

        messages = [
            LLMMessage(role="system", content="You are helpful."),
            LLMMessage(role="user", content="Hello"),
        ]

        formatted = client._format_messages_llama(messages)

        assert "[INST]" in formatted
        assert "<<SYS>>" in formatted
        assert "You are helpful." in formatted
        assert "Hello" in formatted


class TestLocalLLMClientGenerate:
    """Tests for generation methods."""

    @pytest.mark.asyncio
    async def test_local_llm_generate_llama_cpp(self):
        """LocalLLMClient generates response with llama-cpp backend."""
        from src.generation.llm.local import LocalLLMClient

        client = LocalLLMClient(
            model_path="/models/test.gguf",
            model_type="llama",
            backend="llama-cpp",
        )

        # Mock the model loading and inference
        mock_llama = MagicMock()
        mock_llama.return_value = {
            "choices": [{"text": "Hello! How can I help?", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        with patch.object(client, "_load_model", return_value=mock_llama):
            messages = [LLMMessage(role="user", content="Hello")]
            response = await client.generate(messages)

            assert response.content == "Hello! How can I help?"
            assert response.model == "llama"
            assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_local_llm_generate_with_temperature(self):
        """LocalLLMClient respects temperature parameter."""
        from src.generation.llm.local import LocalLLMClient

        client = LocalLLMClient(
            model_path="/models/test.gguf",
            backend="llama-cpp",
        )

        mock_llama = MagicMock()
        mock_llama.return_value = {
            "choices": [{"text": "Response", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }

        with patch.object(client, "_load_model", return_value=mock_llama):
            messages = [LLMMessage(role="user", content="Test")]
            await client.generate(messages, temperature=0.7)

            # Verify llama was called with temperature
            mock_llama.assert_called_once()
            call_kwargs = mock_llama.call_args
            assert call_kwargs[1]["temperature"] == 0.7


class TestLocalLLMClientStream:
    """Tests for streaming generation."""

    @pytest.mark.asyncio
    async def test_local_llm_generate_stream_llama_cpp(self):
        """LocalLLMClient supports streaming with llama-cpp."""
        from src.generation.llm.local import LocalLLMClient

        client = LocalLLMClient(
            model_path="/models/test.gguf",
            model_type="llama",
            backend="llama-cpp",
        )

        # Mock streaming response
        mock_stream = [
            {"choices": [{"text": "Hello", "finish_reason": None}]},
            {"choices": [{"text": " world", "finish_reason": None}]},
            {"choices": [{"text": "!", "finish_reason": "stop"}]},
        ]

        mock_llama = MagicMock()
        mock_llama.return_value = iter(mock_stream)

        with patch.object(client, "_load_model", return_value=mock_llama):
            messages = [LLMMessage(role="user", content="Hello")]

            chunks = []
            async for chunk in client.generate_stream(messages):
                chunks.append(chunk)

            assert len(chunks) == 3
            assert chunks[0].content == "Hello"
            assert chunks[1].content == " world"
            assert chunks[2].content == "!"
            assert chunks[2].is_final is True


class TestLocalLLMModelLoading:
    """Tests for model loading."""

    def test_load_model_unsupported_backend(self):
        """Unsupported backend raises ConfigurationError."""
        from src.generation.llm.local import LocalLLMClient

        client = LocalLLMClient(
            model_path="/models/test.gguf",
            backend="unsupported",
        )

        with pytest.raises(ConfigurationError) as exc_info:
            client._load_model()

        assert "Unsupported backend" in str(exc_info.value)

    def test_load_llama_cpp_missing_import(self):
        """Missing llama-cpp raises ConfigurationError."""
        from src.generation.llm.local import LocalLLMClient

        client = LocalLLMClient(
            model_path="/models/test.gguf",
            backend="llama-cpp",
        )

        with patch.dict("sys.modules", {"llama_cpp": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                with pytest.raises(ConfigurationError) as exc_info:
                    client._load_llama_cpp()

                assert "llama-cpp-python is required" in str(exc_info.value)

    def test_load_transformers_missing_import(self):
        """Missing transformers raises ConfigurationError."""
        from src.generation.llm.local import LocalLLMClient

        client = LocalLLMClient(
            model_path="/models/test",
            backend="transformers",
        )

        with patch("builtins.__import__", side_effect=ImportError("No module")):
            with pytest.raises(ConfigurationError) as exc_info:
                client._load_transformers()

                assert "transformers and torch are required" in str(exc_info.value)


class TestLocalLLMSingleton:
    """Tests for singleton pattern."""

    def test_get_local_llm_client_singleton(self):
        """get_local_llm_client returns singleton."""
        from src.generation.llm import local
        from src.generation.llm.local import (
            get_local_llm_client,
            reset_local_llm_client,
        )

        reset_local_llm_client()

        with patch("src.generation.llm.local.settings") as mock_settings:
            mock_settings.LOCAL_LLM_MODEL_PATH = "/models/test.gguf"
            mock_settings.LOCAL_LLM_MODEL_TYPE = "llama"

            client1 = get_local_llm_client()
            client2 = get_local_llm_client()

            assert client1 is client2

            reset_local_llm_client()

    def test_reset_local_llm_client(self):
        """reset_local_llm_client clears singleton."""
        from src.generation.llm import local
        from src.generation.llm.local import (
            get_local_llm_client,
            reset_local_llm_client,
        )

        reset_local_llm_client()

        with patch("src.generation.llm.local.settings") as mock_settings:
            mock_settings.LOCAL_LLM_MODEL_PATH = "/models/test.gguf"
            mock_settings.LOCAL_LLM_MODEL_TYPE = "llama"

            client1 = get_local_llm_client()
            reset_local_llm_client()
            client2 = get_local_llm_client()

            assert client1 is not client2

            reset_local_llm_client()
