"""Tests for authentication models."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from src.auth.models import User, Tenant, APIKey


class TestUserModel:
    """Tests for User model."""

    def test_create_user(self):
        """Test creating a user."""
        user = User(
            id=uuid4(),
            email="test@example.com",
            name="Test User",
            tenant_id=uuid4(),
            role="user",
            permissions=["documents:read", "queries:execute"],
            created_at=datetime.utcnow(),
        )

        assert user.email == "test@example.com"
        assert user.role == "user"
        assert len(user.permissions) == 2

    def test_user_default_values(self):
        """Test user with default values."""
        user = User(
            email="minimal@example.com",
            name="Minimal User",
            tenant_id=uuid4(),
        )

        assert user.email == "minimal@example.com"
        assert user.role == "user"  # Default value
        assert user.permissions == []

    def test_user_name_field(self):
        """Test user name field."""
        user = User(
            email="test@example.com",
            name="John Doe",
            tenant_id=uuid4(),
        )

        assert user.name == "John Doe"

    def test_user_email_validation(self):
        """Test email validation."""
        user = User(
            email="valid.email@example.com",
            name="Valid Email User",
            tenant_id=uuid4(),
        )
        assert "@" in user.email

    def test_user_timestamps(self):
        """Test user timestamps."""
        now = datetime.utcnow()
        user = User(
            email="test@example.com",
            name="Test User",
            tenant_id=uuid4(),
            created_at=now,
            last_login=now,
        )

        assert user.created_at == now
        assert user.last_login == now


class TestTenantModel:
    """Tests for Tenant model."""

    def test_create_tenant(self):
        """Test creating a tenant."""
        tenant = Tenant(
            id=uuid4(),
            name="Test Organization",
            slug="test-org",
            is_active=True,
            created_at=datetime.utcnow(),
        )

        assert tenant.name == "Test Organization"
        assert tenant.slug == "test-org"
        assert tenant.is_active is True

    def test_tenant_settings(self):
        """Test tenant with settings."""
        tenant = Tenant(
            name="Configured Org",
            slug="configured-org",
            settings={
                "custom_setting": "value",
            },
        )

        assert tenant.settings["custom_setting"] == "value"

    def test_tenant_resource_limits(self):
        """Test tenant resource limit fields."""
        tenant = Tenant(
            name="Limited Org",
            slug="limited-org",
            max_documents=20000,
            max_queries_per_day=5000,
            max_storage_gb=50.0,
        )

        assert tenant.max_documents == 20000
        assert tenant.max_queries_per_day == 5000
        assert tenant.max_storage_gb == 50.0

    def test_tenant_default_limits(self):
        """Test tenant default resource limits."""
        tenant = Tenant(
            name="Default Org",
            slug="default-org",
        )

        assert tenant.max_documents == 10000
        assert tenant.max_queries_per_day == 1000
        assert tenant.max_storage_gb == 10.0


class TestAPIKeyModel:
    """Tests for APIKey model."""

    def test_create_api_key(self):
        """Test creating an API key."""
        api_key = APIKey(
            id=uuid4(),
            name="Production API Key",
            key_prefix="rag_abc1",
            key_hash="0" * 64,
            tenant_id=uuid4(),
            permissions=["documents:read", "queries:execute"],
            is_active=True,
            created_at=datetime.utcnow(),
        )

        assert api_key.name == "Production API Key"
        assert api_key.key_prefix == "rag_abc1"
        assert api_key.is_active is True

    def test_api_key_expiration(self):
        """Test API key with expiration."""
        api_key = APIKey(
            name="Expiring Key",
            key_prefix="rag_exp1",
            key_hash="1" * 64,
            tenant_id=uuid4(),
            expires_at=datetime.utcnow() + timedelta(days=30),
        )

        assert api_key.expires_at > datetime.utcnow()

    def test_api_key_expired(self):
        """Test expired API key."""
        api_key = APIKey(
            name="Expired Key",
            key_prefix="rag_old1",
            key_hash="2" * 64,
            tenant_id=uuid4(),
            expires_at=datetime.utcnow() - timedelta(days=1),
        )

        assert api_key.expires_at < datetime.utcnow()

    def test_api_key_last_used(self):
        """Test API key last used tracking."""
        now = datetime.utcnow()
        api_key = APIKey(
            name="Used Key",
            key_prefix="rag_used",
            key_hash="4" * 64,
            tenant_id=uuid4(),
            last_used_at=now,
        )

        assert api_key.last_used_at == now

    def test_api_key_scopes(self):
        """Test API key with scopes/permissions."""
        api_key = APIKey(
            name="Scoped Key",
            key_prefix="rag_scop",
            key_hash="5" * 64,
            tenant_id=uuid4(),
            permissions=[
                "documents:read",
                "documents:write",
                "queries:execute",
            ],
        )

        assert "documents:read" in api_key.permissions
        assert "admin:settings" not in api_key.permissions

    def test_api_key_user_association(self):
        """Test API key associated with a user."""
        user_id = uuid4()
        api_key = APIKey(
            name="User Key",
            key_prefix="rag_user",
            key_hash="7" * 64,
            tenant_id=uuid4(),
            user_id=user_id,
        )

        assert api_key.user_id == user_id

    def test_api_key_default_values(self):
        """Test API key default values."""
        api_key = APIKey(
            name="Default Key",
            key_prefix="rag_dflt",
            key_hash="8" * 64,
            tenant_id=uuid4(),
        )

        assert api_key.is_active is True
        assert api_key.permissions == []
        assert api_key.expires_at is None
        assert api_key.user_id is None
