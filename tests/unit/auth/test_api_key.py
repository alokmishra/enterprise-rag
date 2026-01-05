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
        assert prefix == full_key[:12]
        assert len(key_hash) == 64  # SHA256 hex digest
        assert key_hash != full_key  # Hash should be different

    def test_generate_api_key_custom_prefix(self):
        """Test API key generation with custom prefix."""
        full_key, prefix, key_hash = generate_api_key(prefix="custom")

        assert full_key.startswith("custom_")
        assert prefix == full_key[:12]

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
    def mock_db_session(self):
        """Create mock database session."""
        session = AsyncMock()
        return session

    @pytest.fixture
    def api_key_auth(self, mock_db_session):
        """Create APIKeyAuth instance."""
        return APIKeyAuth(db_session=mock_db_session)

    @pytest.fixture
    def sample_api_key_record(self):
        """Sample API key database record."""
        return MagicMock(
            id=str(uuid4()),
            name="Test API Key",
            key_prefix="rag_abc123",
            key_hash="0" * 64,
            tenant_id=str(uuid4()),
            permissions=["documents:read", "queries:execute"],
            rate_limit=1000,
            is_active=True,
            expires_at=datetime.utcnow() + timedelta(days=30),
            created_at=datetime.utcnow(),
            last_used_at=None,
        )

    @pytest.mark.asyncio
    async def test_validate_key_success(self, api_key_auth, mock_db_session, sample_api_key_record):
        """Test successful API key validation."""
        # Setup mock to return the API key record
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_api_key_record
        mock_db_session.execute.return_value = mock_result

        api_key = await api_key_auth.validate_key("rag_abc123_secretpart")

        assert api_key is not None
        assert api_key.name == "Test API Key"

    @pytest.mark.asyncio
    async def test_validate_key_not_found(self, api_key_auth, mock_db_session):
        """Test API key validation when key not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        with pytest.raises(Exception):  # InvalidAPIKeyError
            await api_key_auth.validate_key("rag_nonexistent_key")

    @pytest.mark.asyncio
    async def test_validate_key_inactive(self, api_key_auth, mock_db_session, sample_api_key_record):
        """Test API key validation when key is inactive."""
        sample_api_key_record.is_active = False
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_api_key_record
        mock_db_session.execute.return_value = mock_result

        with pytest.raises(Exception):  # InactiveAPIKeyError
            await api_key_auth.validate_key("rag_abc123_secretpart")

    @pytest.mark.asyncio
    async def test_validate_key_expired(self, api_key_auth, mock_db_session, sample_api_key_record):
        """Test API key validation when key is expired."""
        sample_api_key_record.expires_at = datetime.utcnow() - timedelta(days=1)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_api_key_record
        mock_db_session.execute.return_value = mock_result

        with pytest.raises(Exception):  # ExpiredAPIKeyError
            await api_key_auth.validate_key("rag_abc123_secretpart")

    @pytest.mark.asyncio
    async def test_create_key(self, api_key_auth, mock_db_session):
        """Test API key creation."""
        mock_db_session.add = MagicMock()
        mock_db_session.commit = AsyncMock()
        mock_db_session.refresh = AsyncMock()

        full_key, api_key = await api_key_auth.create_key(
            name="New API Key",
            tenant_id=str(uuid4()),
            permissions=["documents:read"],
            rate_limit=500,
            expires_in_days=30,
        )

        assert full_key.startswith("rag_")
        assert api_key.name == "New API Key"
        mock_db_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_revoke_key(self, api_key_auth, mock_db_session, sample_api_key_record):
        """Test API key revocation."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_api_key_record
        mock_db_session.execute.return_value = mock_result
        mock_db_session.commit = AsyncMock()

        await api_key_auth.revoke_key(sample_api_key_record.id)

        assert sample_api_key_record.is_active is False

    @pytest.mark.asyncio
    async def test_update_last_used(self, api_key_auth, mock_db_session, sample_api_key_record):
        """Test updating last used timestamp."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_api_key_record
        mock_db_session.execute.return_value = mock_result
        mock_db_session.commit = AsyncMock()

        await api_key_auth.update_last_used(sample_api_key_record.id)

        assert sample_api_key_record.last_used_at is not None
