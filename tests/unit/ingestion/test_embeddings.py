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

    def test_provider_requires_embed_texts_method(self):
        """Test that providers require an embed_texts method."""
        from src.ingestion.embeddings.base import EmbeddingProvider

        assert hasattr(EmbeddingProvider, 'embed_texts')

    def test_provider_requires_model_name_property(self):
        """Test that providers require a model_name property."""
        from src.ingestion.embeddings.base import EmbeddingProvider

        assert hasattr(EmbeddingProvider, 'model_name')

    def test_provider_requires_dimensions_property(self):
        """Test that providers require a dimensions property."""
        from src.ingestion.embeddings.base import EmbeddingProvider

        assert hasattr(EmbeddingProvider, 'dimensions')


class TestOpenAIEmbeddings:
    """Tests for the OpenAIEmbeddings provider."""

    def test_openai_embeddings_creation(self):
        """Test OpenAIEmbeddings can be created."""
        from src.ingestion.embeddings.openai import OpenAIEmbeddings

        provider = OpenAIEmbeddings(api_key='test-key')
        assert provider is not None

    def test_openai_embeddings_model_selection(self):
        """Test OpenAIEmbeddings model selection."""
        from src.ingestion.embeddings.openai import OpenAIEmbeddings

        provider = OpenAIEmbeddings(api_key='test-key', model="text-embedding-3-small")
        assert provider.model_name == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_openai_embeddings_embed_texts(self):
        """Test OpenAIEmbeddings embed_texts method."""
        from src.ingestion.embeddings.openai import OpenAIEmbeddings

        mock_response = MagicMock()
        mock_response.data = [MagicMock(index=0, embedding=[0.1] * 1536)]
        mock_response.usage = MagicMock(total_tokens=10)

        provider = OpenAIEmbeddings(api_key='test-key')

        with patch.object(provider, '_get_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await provider.embed_texts(["Test text"])
            assert result.embeddings is not None
            assert len(result.embeddings) == 1

    @pytest.mark.asyncio
    async def test_openai_embeddings_embed_text_single(self):
        """Test OpenAIEmbeddings embed_text for single text."""
        from src.ingestion.embeddings.openai import OpenAIEmbeddings

        mock_response = MagicMock()
        mock_response.data = [MagicMock(index=0, embedding=[0.1] * 1536)]
        mock_response.usage = MagicMock(total_tokens=10)

        provider = OpenAIEmbeddings(api_key='test-key')

        with patch.object(provider, '_get_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await provider.embed_text("Test text")
            assert isinstance(result, list)
            assert len(result) == 1536

    @pytest.mark.asyncio
    async def test_openai_embeddings_batch(self):
        """Test OpenAIEmbeddings batch embedding."""
        from src.ingestion.embeddings.openai import OpenAIEmbeddings

        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(index=0, embedding=[0.1] * 1536),
            MagicMock(index=1, embedding=[0.2] * 1536),
            MagicMock(index=2, embedding=[0.3] * 1536),
        ]
        mock_response.usage = MagicMock(total_tokens=30)

        provider = OpenAIEmbeddings(api_key='test-key')

        with patch.object(provider, '_get_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await provider.embed_texts([
                "Text 1",
                "Text 2",
                "Text 3",
            ])
            assert result.embeddings is not None
            assert len(result.embeddings) == 3

    def test_openai_embeddings_dimensions(self):
        """Test OpenAIEmbeddings returns correct dimensions."""
        from src.ingestion.embeddings.openai import OpenAIEmbeddings

        # When dimensions not specified, uses settings default or model default
        provider = OpenAIEmbeddings(api_key='test-key', model="text-embedding-3-small", dimensions=1536)
        assert provider.dimensions == 1536

    def test_openai_embeddings_custom_dimensions(self):
        """Test OpenAIEmbeddings with custom dimensions."""
        from src.ingestion.embeddings.openai import OpenAIEmbeddings

        provider = OpenAIEmbeddings(api_key='test-key', dimensions=512)
        assert provider.dimensions == 512


class TestEmbeddingResult:
    """Tests for the EmbeddingResult model."""

    def test_embedding_result_creation(self):
        """Test EmbeddingResult can be created."""
        from src.ingestion.embeddings.base import EmbeddingResult

        result = EmbeddingResult(
            embeddings=[[0.1] * 1536],
            model="text-embedding-3-small",
            dimensions=1536,
            tokens_used=10,
        )
        assert result.embeddings is not None
        assert result.tokens_used == 10
        assert result.model == "text-embedding-3-small"
        assert result.dimensions == 1536

    def test_embedding_result_multiple_embeddings(self):
        """Test EmbeddingResult with multiple embeddings."""
        from src.ingestion.embeddings.base import EmbeddingResult

        result = EmbeddingResult(
            embeddings=[[0.1] * 1536, [0.2] * 1536],
            model="text-embedding-3-small",
            dimensions=1536,
            tokens_used=20,
        )
        assert len(result.embeddings) == 2


class TestEmbeddingsBatching:
    """Tests for embedding batching functionality."""

    @pytest.mark.asyncio
    async def test_embed_texts_empty_list(self):
        """Test that empty input returns empty result."""
        from src.ingestion.embeddings.openai import OpenAIEmbeddings

        provider = OpenAIEmbeddings(api_key='test-key')
        result = await provider.embed_texts([])

        assert result.embeddings == []
        assert result.tokens_used == 0

    @pytest.mark.asyncio
    async def test_embed_query(self):
        """Test embed_query method."""
        from src.ingestion.embeddings.openai import OpenAIEmbeddings

        mock_response = MagicMock()
        mock_response.data = [MagicMock(index=0, embedding=[0.1] * 1536)]
        mock_response.usage = MagicMock(total_tokens=5)

        provider = OpenAIEmbeddings(api_key='test-key')

        with patch.object(provider, '_get_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await provider.embed_query("search query")
            assert isinstance(result, list)
            assert len(result) == 1536
