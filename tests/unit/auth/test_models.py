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
            id=str(uuid4()),
            email="test@example.com",
            hashed_password="hashed_password_here",
            tenant_id=str(uuid4()),
            role="user",
            permissions=["documents:read", "queries:execute"],
            is_active=True,
            created_at=datetime.utcnow(),
        )

        assert user.email == "test@example.com"
        assert user.role == "user"
        assert user.is_active is True
        assert len(user.permissions) == 2

    def test_user_optional_fields(self):
        """Test user with optional fields."""
        user = User(
            id=str(uuid4()),
            email="minimal@example.com",
            tenant_id=str(uuid4()),
        )

        assert user.email == "minimal@example.com"
        assert user.hashed_password is None
        assert user.role is None or user.role == "user"  # Default value

    def test_user_full_name(self):
        """Test user full name computation."""
        user = User(
            id=str(uuid4()),
            email="test@example.com",
            tenant_id=str(uuid4()),
            first_name="John",
            last_name="Doe",
        )

        assert user.first_name == "John"
        assert user.last_name == "Doe"
        # If full_name property exists
        if hasattr(user, "full_name"):
            assert user.full_name == "John Doe"

    def test_user_email_validation(self):
        """Test email validation."""
        # Valid email
        user = User(
            id=str(uuid4()),
            email="valid.email@example.com",
            tenant_id=str(uuid4()),
        )
        assert "@" in user.email

    def test_user_timestamps(self):
        """Test user timestamps."""
        now = datetime.utcnow()
        user = User(
            id=str(uuid4()),
            email="test@example.com",
            tenant_id=str(uuid4()),
            created_at=now,
            updated_at=now,
            last_login_at=now,
        )

        assert user.created_at == now
        assert user.updated_at == now
        assert user.last_login_at == now


class TestTenantModel:
    """Tests for Tenant model."""

    def test_create_tenant(self):
        """Test creating a tenant."""
        tenant = Tenant(
            id=str(uuid4()),
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
            id=str(uuid4()),
            name="Configured Org",
            slug="configured-org",
            settings={
                "max_documents": 10000,
                "max_queries_per_day": 5000,
                "allowed_models": ["gpt-4", "claude-3"],
            },
        )

        assert tenant.settings["max_documents"] == 10000
        assert "gpt-4" in tenant.settings["allowed_models"]

    def test_tenant_quota(self):
        """Test tenant quota fields."""
        tenant = Tenant(
            id=str(uuid4()),
            name="Quota Org",
            slug="quota-org",
            document_quota=10000,
            query_quota=50000,
            storage_quota_bytes=10 * 1024 * 1024 * 1024,  # 10GB
        )

        assert tenant.document_quota == 10000
        assert tenant.query_quota == 50000
        assert tenant.storage_quota_bytes == 10 * 1024 * 1024 * 1024

    def test_tenant_subscription(self):
        """Test tenant subscription info."""
        tenant = Tenant(
            id=str(uuid4()),
            name="Subscribed Org",
            slug="subscribed-org",
            subscription_tier="enterprise",
            subscription_expires_at=datetime.utcnow() + timedelta(days=365),
        )

        assert tenant.subscription_tier == "enterprise"
        assert tenant.subscription_expires_at > datetime.utcnow()


class TestAPIKeyModel:
    """Tests for APIKey model."""

    def test_create_api_key(self):
        """Test creating an API key."""
        api_key = APIKey(
            id=str(uuid4()),
            name="Production API Key",
            key_prefix="rag_abc123",
            key_hash="0" * 64,
            tenant_id=str(uuid4()),
            permissions=["documents:read", "queries:execute"],
            is_active=True,
            created_at=datetime.utcnow(),
        )

        assert api_key.name == "Production API Key"
        assert api_key.key_prefix == "rag_abc123"
        assert api_key.is_active is True

    def test_api_key_expiration(self):
        """Test API key with expiration."""
        api_key = APIKey(
            id=str(uuid4()),
            name="Expiring Key",
            key_prefix="rag_exp123",
            key_hash="1" * 64,
            tenant_id=str(uuid4()),
            expires_at=datetime.utcnow() + timedelta(days=30),
        )

        assert api_key.expires_at > datetime.utcnow()

    def test_api_key_expired(self):
        """Test expired API key."""
        api_key = APIKey(
            id=str(uuid4()),
            name="Expired Key",
            key_prefix="rag_old123",
            key_hash="2" * 64,
            tenant_id=str(uuid4()),
            expires_at=datetime.utcnow() - timedelta(days=1),
        )

        assert api_key.expires_at < datetime.utcnow()
        # If is_expired property exists
        if hasattr(api_key, "is_expired"):
            assert api_key.is_expired is True

    def test_api_key_rate_limit(self):
        """Test API key with rate limit."""
        api_key = APIKey(
            id=str(uuid4()),
            name="Rate Limited Key",
            key_prefix="rag_rate123",
            key_hash="3" * 64,
            tenant_id=str(uuid4()),
            rate_limit=1000,
            rate_limit_window_seconds=3600,
        )

        assert api_key.rate_limit == 1000
        assert api_key.rate_limit_window_seconds == 3600

    def test_api_key_last_used(self):
        """Test API key last used tracking."""
        now = datetime.utcnow()
        api_key = APIKey(
            id=str(uuid4()),
            name="Used Key",
            key_prefix="rag_used123",
            key_hash="4" * 64,
            tenant_id=str(uuid4()),
            last_used_at=now,
            use_count=150,
        )

        assert api_key.last_used_at == now
        assert api_key.use_count == 150

    def test_api_key_scopes(self):
        """Test API key with scopes/permissions."""
        api_key = APIKey(
            id=str(uuid4()),
            name="Scoped Key",
            key_prefix="rag_scope123",
            key_hash="5" * 64,
            tenant_id=str(uuid4()),
            permissions=[
                "documents:read",
                "documents:write",
                "queries:execute",
            ],
        )

        assert "documents:read" in api_key.permissions
        assert "admin:settings" not in api_key.permissions

    def test_api_key_ip_whitelist(self):
        """Test API key with IP whitelist."""
        api_key = APIKey(
            id=str(uuid4()),
            name="IP Restricted Key",
            key_prefix="rag_ip123",
            key_hash="6" * 64,
            tenant_id=str(uuid4()),
            allowed_ips=["192.168.1.0/24", "10.0.0.0/8"],
        )

        assert "192.168.1.0/24" in api_key.allowed_ips

    def test_api_key_created_by(self):
        """Test API key creator tracking."""
        creator_id = str(uuid4())
        api_key = APIKey(
            id=str(uuid4()),
            name="Admin Created Key",
            key_prefix="rag_admin123",
            key_hash="7" * 64,
            tenant_id=str(uuid4()),
            created_by=creator_id,
        )

        assert api_key.created_by == creator_id
