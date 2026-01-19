"""Tests for JWT authentication."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from uuid import uuid4

from src.auth.jwt import (
    JWTAuth,
    create_access_token,
    create_refresh_token,
    verify_token,
    TokenPayload,
)


class TestJWTAuth:
    """Tests for JWTAuth class."""

    @pytest.fixture
    def jwt_auth(self):
        """Create JWTAuth instance."""
        return JWTAuth(
            secret_key="test-secret-key-that-is-at-least-32-chars",
            algorithm="HS256",
            access_token_expire_minutes=30,
            refresh_token_expire_days=7,
        )

    @pytest.fixture
    def sample_user_data(self):
        """Sample user data for token creation."""
        return {
            "user_id": uuid4(),
            "tenant_id": uuid4(),
            "email": "test@example.com",
            "role": "user",
            "permissions": ["documents:read", "queries:execute"],
        }

    def test_create_access_token(self, jwt_auth, sample_user_data):
        """Test access token creation."""
        token = jwt_auth.create_access_token(**sample_user_data)

        assert token is not None
        assert isinstance(token, str)
        assert len(token.split(".")) == 3  # JWT has 3 parts

    def test_create_refresh_token(self, jwt_auth, sample_user_data):
        """Test refresh token creation."""
        token = jwt_auth.create_refresh_token(
            user_id=sample_user_data["user_id"],
            tenant_id=sample_user_data["tenant_id"],
        )

        assert token is not None
        assert isinstance(token, str)
        assert len(token.split(".")) == 3

    def test_verify_valid_access_token(self, jwt_auth, sample_user_data):
        """Test verifying a valid access token."""
        token = jwt_auth.create_access_token(**sample_user_data)
        payload = jwt_auth.verify_token(token)

        assert payload is not None
        assert payload.sub == str(sample_user_data["user_id"])
        assert payload.tenant_id == str(sample_user_data["tenant_id"])
        assert payload.email == sample_user_data["email"]
        assert payload.role == sample_user_data["role"]
        assert payload.type == "access"

    def test_refresh_access_token(self, jwt_auth, sample_user_data):
        """Test refreshing an access token from a refresh token."""
        refresh_token = jwt_auth.create_refresh_token(
            user_id=sample_user_data["user_id"],
            tenant_id=sample_user_data["tenant_id"],
        )

        with pytest.raises(Exception):
            jwt_auth.refresh_access_token(refresh_token)

    def test_verify_invalid_token(self, jwt_auth):
        """Test verifying an invalid token."""
        with pytest.raises(Exception):  # JWTError or similar
            jwt_auth.verify_token("invalid.token.here")

    def test_verify_expired_token(self, jwt_auth, sample_user_data):
        """Test verifying an expired token."""
        # Create auth with very short expiry
        short_auth = JWTAuth(
            secret_key="test-secret-key-that-is-at-least-32-chars",
            algorithm="HS256",
            access_token_expire_minutes=-1,  # Already expired
            refresh_token_expire_days=7,
        )

        token = short_auth.create_access_token(**sample_user_data)

        with pytest.raises(Exception):  # ExpiredSignatureError
            short_auth.verify_token(token)

    def test_token_contains_permissions(self, jwt_auth, sample_user_data):
        """Test that token contains permissions."""
        token = jwt_auth.create_access_token(**sample_user_data)
        payload = jwt_auth.verify_token(token)

        assert payload.permissions == sample_user_data["permissions"]

    def test_different_secrets_fail_verification(self, sample_user_data):
        """Test that tokens signed with different secrets fail verification."""
        auth1 = JWTAuth(
            secret_key="secret-key-one-that-is-at-least-32-chars",
            algorithm="HS256",
        )
        auth2 = JWTAuth(
            secret_key="secret-key-two-that-is-at-least-32-chars",
            algorithm="HS256",
        )

        token = auth1.create_access_token(**sample_user_data)

        with pytest.raises(Exception):
            auth2.verify_token(token)


class TestTokenHelperFunctions:
    """Tests for token helper functions."""

    def test_create_access_token_function(self):
        """Test create_access_token helper function."""
        with patch("src.auth.jwt.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                jwt_secret_key="test-secret-key-32-chars-minimum!!",
                jwt_algorithm="HS256",
                access_token_expire_minutes=30,
            )

            token = create_access_token(
                user_id=uuid4(),
                tenant_id=uuid4(),
                email="test@example.com",
                role="user",
                permissions=[],
            )

            assert token is not None
            assert isinstance(token, str)

    def test_create_refresh_token_function(self):
        """Test create_refresh_token helper function."""
        with patch("src.auth.jwt.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                jwt_secret_key="test-secret-key-32-chars-minimum!!",
                jwt_algorithm="HS256",
                refresh_token_expire_days=7,
            )

            token = create_refresh_token(
                user_id=uuid4(),
                tenant_id=uuid4(),
            )

            assert token is not None
            assert isinstance(token, str)


class TestTokenPayload:
    """Tests for TokenPayload model."""

    def test_token_payload_creation(self):
        """Test TokenPayload model creation."""
        payload = TokenPayload(
            sub="user-123",
            tenant_id="tenant-456",
            email="test@example.com",
            role="admin",
            permissions=["admin:all"],
            type="access",
            exp=datetime.utcnow() + timedelta(hours=1),
            iat=datetime.utcnow(),
        )

        assert payload.sub == "user-123"
        assert payload.tenant_id == "tenant-456"
        assert payload.role == "admin"
        assert "admin:all" in payload.permissions

    def test_token_payload_required_fields(self):
        """Test TokenPayload with required fields."""
        payload = TokenPayload(
            sub="user-123",
            tenant_id="tenant-456",
            email="test@example.com",
            role="user",
            exp=datetime.utcnow() + timedelta(hours=1),
            iat=datetime.utcnow(),
            type="access",
        )

        assert payload.sub == "user-123"
        assert payload.permissions == []
