"""
Enterprise RAG System - FastAPI Authentication Dependencies
"""

from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from src.auth.jwt import verify_token, TokenPayload
from src.auth.api_key import APIKeyAuth, hash_api_key
from src.auth.models import User, Tenant, APIKey
from src.core.exceptions import AuthenticationError


# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_token_payload(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> Optional[TokenPayload]:
    """Extract and verify JWT token from Authorization header."""
    if not credentials:
        return None

    try:
        return verify_token(credentials.credentials)
    except AuthenticationError:
        return None


async def get_api_key_from_header(
    api_key: Optional[str] = Security(api_key_header),
) -> Optional[str]:
    """Extract API key from X-API-Key header."""
    return api_key


async def get_current_user(
    token_payload: Optional[TokenPayload] = Depends(get_token_payload),
    api_key: Optional[str] = Depends(get_api_key_from_header),
) -> User:
    """
    Get the current authenticated user.

    Supports both JWT tokens and API keys.
    """
    # Try JWT first
    if token_payload:
        return User(
            id=UUID(token_payload.sub),
            email=token_payload.email,
            name=token_payload.email.split("@")[0],  # Default name from email
            tenant_id=UUID(token_payload.tenant_id),
            role=token_payload.role,
            permissions=token_payload.permissions,
        )

    # Try API key
    if api_key:
        # In a real implementation, this would look up the API key
        # and return the associated user or a service account
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="API key authentication requires database setup",
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user_optional(
    token_payload: Optional[TokenPayload] = Depends(get_token_payload),
    api_key: Optional[str] = Depends(get_api_key_from_header),
) -> Optional[User]:
    """
    Get the current user if authenticated, or None.

    Useful for endpoints that work with or without authentication.
    """
    try:
        return await get_current_user(token_payload, api_key)
    except HTTPException:
        return None


async def get_current_tenant(
    current_user: User = Depends(get_current_user),
) -> Tenant:
    """
    Get the current tenant from the authenticated user.

    In a real implementation, this would look up the full tenant details.
    """
    return Tenant(
        id=current_user.tenant_id,
        name="Default Tenant",
        slug="default",
    )


async def get_api_key(
    api_key: Optional[str] = Depends(get_api_key_from_header),
) -> APIKey:
    """
    Validate and return the API key details.

    Raises 401 if no valid API key is provided.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
        )

    # In a real implementation, this would validate against the database
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="API key validation requires database setup",
    )


def require_tenant(tenant_id: UUID):
    """
    Dependency that ensures the current user belongs to the specified tenant.

    Usage:
        @router.get("/tenants/{tenant_id}/documents")
        async def get_documents(
            tenant_id: UUID,
            current_user: User = Depends(get_current_user),
            _: None = Depends(require_tenant(tenant_id)),
        ):
            ...
    """
    async def check_tenant(
        current_user: User = Depends(get_current_user),
    ):
        if current_user.tenant_id != tenant_id:
            # Check if super admin
            if current_user.role != "super_admin":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this tenant",
                )
        return None

    return check_tenant


class TenantContext:
    """
    Context manager for tenant-scoped operations.

    Ensures all database queries are scoped to the current tenant.
    """

    def __init__(self, tenant_id: UUID):
        self.tenant_id = tenant_id

    def __enter__(self):
        # Set tenant context for current request
        # This could set a contextvar that repositories check
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clear tenant context
        pass
