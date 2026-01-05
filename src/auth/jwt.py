"""
Enterprise RAG System - JWT Authentication
"""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from jose import JWTError, jwt
from pydantic import BaseModel

from src.core.config import get_settings
from src.core.exceptions import AuthenticationError


class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str  # User ID
    tenant_id: str
    email: str
    role: str
    permissions: list[str] = []
    exp: datetime
    iat: datetime
    type: str = "access"  # access or refresh


class JWTAuth:
    """JWT authentication manager."""

    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
    ):
        settings = get_settings()
        self.secret_key = secret_key or settings.jwt_secret_key or "your-secret-key-change-in-production"
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days

    def create_access_token(
        self,
        user_id: UUID,
        tenant_id: UUID,
        email: str,
        role: str,
        permissions: list[str] = None,
    ) -> str:
        """Create an access token."""
        now = datetime.utcnow()
        expire = now + timedelta(minutes=self.access_token_expire_minutes)

        payload = {
            "sub": str(user_id),
            "tenant_id": str(tenant_id),
            "email": email,
            "role": role,
            "permissions": permissions or [],
            "exp": expire,
            "iat": now,
            "type": "access",
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(
        self,
        user_id: UUID,
        tenant_id: UUID,
    ) -> str:
        """Create a refresh token."""
        now = datetime.utcnow()
        expire = now + timedelta(days=self.refresh_token_expire_days)

        payload = {
            "sub": str(user_id),
            "tenant_id": str(tenant_id),
            "exp": expire,
            "iat": now,
            "type": "refresh",
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> TokenPayload:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return TokenPayload(**payload)
        except JWTError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")

    def refresh_access_token(self, refresh_token: str) -> str:
        """Create a new access token from a refresh token."""
        payload = self.verify_token(refresh_token)

        if payload.type != "refresh":
            raise AuthenticationError("Invalid token type")

        # In a real implementation, you would look up the user
        # and their current role/permissions
        return self.create_access_token(
            user_id=UUID(payload.sub),
            tenant_id=UUID(payload.tenant_id),
            email=payload.email,
            role=payload.role,
            permissions=payload.permissions,
        )


# Module-level convenience functions
_jwt_auth: Optional[JWTAuth] = None


def get_jwt_auth() -> JWTAuth:
    """Get the JWT auth singleton."""
    global _jwt_auth
    if _jwt_auth is None:
        _jwt_auth = JWTAuth()
    return _jwt_auth


def create_access_token(
    user_id: UUID,
    tenant_id: UUID,
    email: str,
    role: str,
    permissions: list[str] = None,
) -> str:
    """Create an access token."""
    return get_jwt_auth().create_access_token(
        user_id=user_id,
        tenant_id=tenant_id,
        email=email,
        role=role,
        permissions=permissions,
    )


def create_refresh_token(user_id: UUID, tenant_id: UUID) -> str:
    """Create a refresh token."""
    return get_jwt_auth().create_refresh_token(user_id=user_id, tenant_id=tenant_id)


def verify_token(token: str) -> TokenPayload:
    """Verify and decode a JWT token."""
    return get_jwt_auth().verify_token(token)
