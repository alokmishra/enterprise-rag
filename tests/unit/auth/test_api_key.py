"""Tests for API key authentication."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.auth.api_key import (
    APIKeyAuth,
    generate_api_key,
    hash_api_key,
)


class TestGenerateAPIKey:
    """Tests for API key generation."""

    def test_generate_api_key_default_prefix(self):
        """Test API key generation with default prefix."""
        full_key, prefix, key_hash = generate_api_key()

        assert full_key.startswith("rag_")
        assert prefix == full_key[:8]  # First 8 chars
        assert len(key_hash) == 64  # SHA256 hex digest
        assert key_hash != full_key  # Hash should be different

    def test_generate_api_key_custom_prefix(self):
        """Test API key generation with custom prefix."""
        full_key, prefix, key_hash = generate_api_key(prefix="custom")

        assert full_key.startswith("custom_")
        assert prefix == full_key[:8]  # First 8 chars

    def test_generate_api_key_uniqueness(self):
        """Test that generated keys are unique."""
        keys = set()
        for _ in range(100):
            full_key, _, _ = generate_api_key()
            keys.add(full_key)

        assert len(keys) == 100  # All keys should be unique

    def test_hash_api_key(self):
        """Test API key hashing."""
        key = "rag_test123456789abcdef"
        hash1 = hash_api_key(key)
        hash2 = hash_api_key(key)

        assert hash1 == hash2  # Same key should produce same hash
        assert len(hash1) == 64

        different_key = "rag_different_key_here"
        different_hash = hash_api_key(different_key)
        assert different_hash != hash1


class TestAPIKeyAuth:
    """Tests for APIKeyAuth class."""

    @pytest.fixture
    def mock_key_repository(self):
        """Create mock key repository."""
        repo = AsyncMock()
        return repo

    @pytest.fixture
    def api_key_auth(self, mock_key_repository):
        """Create APIKeyAuth instance."""
        return APIKeyAuth(key_repository=mock_key_repository)

    @pytest.fixture
    def sample_api_key_record(self):
        """Sample API key database record."""
        record = MagicMock()
        record.id = uuid4()
        record.name = "Test API Key"
        record.key_prefix = "rag_abc1"
        record.key_hash = "0" * 64
        record.tenant_id = uuid4()
        record.user_id = None
        record.permissions = ["documents:read", "queries:execute"]
        record.is_active = True
        record.expires_at = datetime.utcnow() + timedelta(days=30)
        record.created_at = datetime.utcnow()
        record.last_used_at = None
        return record

    @pytest.mark.asyncio
    async def test_validate_key_success(self, api_key_auth, mock_key_repository, sample_api_key_record):
        """Test successful API key validation."""
        mock_key_repository.get_by_hash.return_value = sample_api_key_record
        mock_key_repository.update_last_used = AsyncMock()

        api_key = await api_key_auth.validate_key("rag_abc123_secretpart")

        assert api_key is not None
        assert api_key.name == "Test API Key"

    @pytest.mark.asyncio
    async def test_validate_key_not_found(self, api_key_auth, mock_key_repository):
        """Test API key validation when key not found."""
        mock_key_repository.get_by_hash.return_value = None

        with pytest.raises(Exception):  # AuthenticationError
            await api_key_auth.validate_key("rag_nonexistent_key")

    @pytest.mark.asyncio
    async def test_validate_key_inactive(self, api_key_auth, mock_key_repository, sample_api_key_record):
        """Test API key validation when key is inactive."""
        sample_api_key_record.is_active = False
        mock_key_repository.get_by_hash.return_value = sample_api_key_record

        with pytest.raises(Exception):  # AuthenticationError
            await api_key_auth.validate_key("rag_abc123_secretpart")

    @pytest.mark.asyncio
    async def test_validate_key_expired(self, api_key_auth, mock_key_repository, sample_api_key_record):
        """Test API key validation when key is expired."""
        sample_api_key_record.expires_at = datetime.utcnow() - timedelta(days=1)
        mock_key_repository.get_by_hash.return_value = sample_api_key_record

        with pytest.raises(Exception):  # AuthenticationError
            await api_key_auth.validate_key("rag_abc123_secretpart")

    @pytest.mark.asyncio
    async def test_create_key(self, api_key_auth, mock_key_repository):
        """Test API key creation."""
        mock_key_repository.create = AsyncMock()

        full_key, api_key = await api_key_auth.create_key(
            name="New API Key",
            tenant_id=uuid4(),
            permissions=["documents:read"],
            expires_in_days=30,
        )

        assert full_key.startswith("rag_")
        assert api_key.name == "New API Key"

    @pytest.mark.asyncio
    async def test_revoke_key(self, api_key_auth, mock_key_repository, sample_api_key_record):
        """Test API key revocation."""
        mock_key_repository.deactivate = AsyncMock()

        result = await api_key_auth.revoke_key(sample_api_key_record.id)

        assert result is True
        mock_key_repository.deactivate.assert_called_once_with(sample_api_key_record.id)

    def test_clear_cache(self, api_key_auth):
        """Test clearing the cache."""
        api_key_auth._cache["test_hash"] = MagicMock()
        assert len(api_key_auth._cache) == 1

        api_key_auth.clear_cache()

        assert len(api_key_auth._cache) == 0
