"""Tests for authentication dependencies."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from src.auth.dependencies import (
    get_current_user,
    get_current_tenant,
    get_api_key,
    get_token_payload,
    get_api_key_from_header,
)
from src.auth.jwt import TokenPayload
from datetime import datetime, timedelta


class TestGetCurrentUser:
    """Tests for get_current_user dependency."""

    @pytest.fixture
    def valid_token_payload(self):
        """Create valid JWT token payload."""
        return TokenPayload(
            sub=str(uuid4()),
            tenant_id=str(uuid4()),
            email="test@example.com",
            role="user",
            permissions=["documents:read", "queries:execute"],
            type="access",
            exp=datetime.utcnow() + timedelta(hours=1),
            iat=datetime.utcnow(),
        )

    @pytest.mark.asyncio
    async def test_get_current_user_valid_token(self, valid_token_payload):
        """Test getting current user with valid token payload."""
        user = await get_current_user(
            token_payload=valid_token_payload,
            api_key=None,
        )

        assert user is not None
        assert str(user.id) == valid_token_payload.sub
        assert user.email == valid_token_payload.email
        assert user.role == valid_token_payload.role

    @pytest.mark.asyncio
    async def test_get_current_user_no_auth(self):
        """Test getting current user without any authentication."""
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(
                token_payload=None,
                api_key=None,
            )

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_get_current_user_api_key_not_implemented(self):
        """Test that API key auth returns 501 (not implemented)."""
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(
                token_payload=None,
                api_key="rag_test_api_key",
            )

        assert exc_info.value.status_code == 501


class TestGetTokenPayload:
    """Tests for get_token_payload dependency."""

    @pytest.mark.asyncio
    async def test_get_token_payload_no_credentials(self):
        """Test getting token payload without credentials."""
        result = await get_token_payload(credentials=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_token_payload_invalid_token(self):
        """Test getting token payload with invalid token."""
        from src.auth.jwt import JWTAuth

        jwt_auth = JWTAuth(secret_key="test-secret-key-for-testing")
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="invalid.token.here",
        )

        with patch("src.auth.dependencies.verify_token") as mock_verify:
            from src.core.exceptions import AuthenticationError
            mock_verify.side_effect = AuthenticationError("Invalid token")
            result = await get_token_payload(credentials=credentials)

        assert result is None


class TestGetCurrentTenant:
    """Tests for get_current_tenant dependency."""

    @pytest.fixture
    def mock_user(self):
        """Create mock user."""
        from src.auth.models import User
        return User(
            id=uuid4(),
            email="test@example.com",
            name="Test User",
            tenant_id=uuid4(),
            role="user",
        )

    @pytest.mark.asyncio
    async def test_get_current_tenant_success(self, mock_user):
        """Test getting current tenant successfully."""
        tenant = await get_current_tenant(current_user=mock_user)

        assert tenant is not None
        assert tenant.id == mock_user.tenant_id
        assert tenant.name == "Default Tenant"
        assert tenant.slug == "default"


class TestGetAPIKey:
    """Tests for get_api_key dependency."""

    @pytest.mark.asyncio
    async def test_get_api_key_missing_header(self):
        """Test getting API key without header."""
        with pytest.raises(HTTPException) as exc_info:
            await get_api_key(api_key=None)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_get_api_key_not_implemented(self):
        """Test that API key validation returns 501 (not implemented)."""
        with pytest.raises(HTTPException) as exc_info:
            await get_api_key(api_key="rag_test_api_key")

        assert exc_info.value.status_code == 501


class TestGetAPIKeyFromHeader:
    """Tests for get_api_key_from_header dependency."""

    @pytest.mark.asyncio
    async def test_get_api_key_from_header_returns_key(self):
        """Test that API key from header is returned."""
        result = await get_api_key_from_header(api_key="rag_test_key")
        assert result == "rag_test_key"

    @pytest.mark.asyncio
    async def test_get_api_key_from_header_returns_none(self):
        """Test that None is returned when no API key."""
        result = await get_api_key_from_header(api_key=None)
        assert result is None
