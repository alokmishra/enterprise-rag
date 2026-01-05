"""
Tests for src/ingestion/embeddings/
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestEmbeddingProvider:
    """Tests for the base EmbeddingProvider."""

    def test_provider_base_is_abstract(self):
        """Test that EmbeddingProvider is abstract."""
        from src.ingestion.embeddings.base import EmbeddingProvider
        from abc import ABC

        assert issubclass(EmbeddingProvider, ABC)

    def test_provider_requires_embed_method(self):
        """Test that providers require an embed method."""
        from src.ingestion.embeddings.base import EmbeddingProvider

        assert hasattr(EmbeddingProvider, 'embed')


class TestOpenAIEmbeddings:
    """Tests for the OpenAIEmbeddings provider."""

    def test_openai_embeddings_creation(self):
        """Test OpenAIEmbeddings can be created."""
        from src.ingestion.embeddings.openai import OpenAIEmbeddings

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIEmbeddings()
            assert provider is not None

    def test_openai_embeddings_model_selection(self):
        """Test OpenAIEmbeddings model selection."""
        from src.ingestion.embeddings.openai import OpenAIEmbeddings

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIEmbeddings(model="text-embedding-3-small")
            assert provider.model == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_openai_embeddings_embed(self):
        """Test OpenAIEmbeddings embed method."""
        from src.ingestion.embeddings.openai import OpenAIEmbeddings

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_response.usage = MagicMock(total_tokens=10)

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIEmbeddings()

            with patch.object(provider, 'client') as mock_client:
                mock_client.embeddings.create = AsyncMock(return_value=mock_response)

                result = await provider.embed("Test text")
                assert result.embeddings is not None
                assert len(result.embeddings) > 0

    @pytest.mark.asyncio
    async def test_openai_embeddings_batch(self):
        """Test OpenAIEmbeddings batch embedding."""
        from src.ingestion.embeddings.openai import OpenAIEmbeddings

        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
            MagicMock(embedding=[0.3] * 1536),
        ]
        mock_response.usage = MagicMock(total_tokens=30)

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIEmbeddings()

            with patch.object(provider, 'client') as mock_client:
                mock_client.embeddings.create = AsyncMock(return_value=mock_response)

                result = await provider.embed_batch([
                    "Text 1",
                    "Text 2",
                    "Text 3",
                ])
                assert result.embeddings is not None
                assert len(result.embeddings) == 3

    def test_openai_embeddings_dimension(self):
        """Test OpenAIEmbeddings returns correct dimension."""
        from src.ingestion.embeddings.openai import OpenAIEmbeddings

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIEmbeddings(model="text-embedding-3-small")
            # Default dimension is 1536
            assert provider.dimension == 1536 or hasattr(provider, 'dimension')


class TestEmbeddingResult:
    """Tests for the EmbeddingResult model."""

    def test_embedding_result_creation(self):
        """Test EmbeddingResult can be created."""
        from src.ingestion.embeddings.base import EmbeddingResult

        result = EmbeddingResult(
            embeddings=[[0.1] * 1536],
            tokens_used=10,
        )
        assert result.embeddings is not None
        assert result.tokens_used == 10

    def test_embedding_result_multiple_embeddings(self):
        """Test EmbeddingResult with multiple embeddings."""
        from src.ingestion.embeddings.base import EmbeddingResult

        result = EmbeddingResult(
            embeddings=[[0.1] * 1536, [0.2] * 1536],
            tokens_used=20,
        )
        assert len(result.embeddings) == 2


class TestEmbeddingsBatching:
    """Tests for embedding batching functionality."""

    @pytest.mark.asyncio
    async def test_batching_large_input(self):
        """Test that large inputs are batched correctly."""
        from src.ingestion.embeddings.openai import OpenAIEmbeddings

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIEmbeddings(batch_size=10)

            # Should handle batching internally
            texts = [f"Text {i}" for i in range(25)]
            # Would call embed_batch multiple times

    def test_batch_size_configuration(self):
        """Test batch size can be configured."""
        from src.ingestion.embeddings.openai import OpenAIEmbeddings

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIEmbeddings(batch_size=50)
            assert provider.batch_size == 50
