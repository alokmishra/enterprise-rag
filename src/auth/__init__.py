"""
Enterprise RAG System - Authentication & Authorization Module

Provides:
- JWT token authentication
- API key authentication
- Role-based access control (RBAC)
- Multi-tenancy support
"""

from __future__ import annotations

from src.auth.jwt import (
    JWTAuth,
    create_access_token,
    create_refresh_token,
    verify_token,
    TokenPayload,
)
from src.auth.api_key import (
    APIKeyAuth,
    generate_api_key,
    hash_api_key,
)
from src.auth.rbac import (
    Permission,
    Role,
    RBACManager,
    require_permission,
    require_role,
)
from src.auth.dependencies import (
    get_current_user,
    get_current_tenant,
    get_api_key,
)
from src.auth.models import User, Tenant, APIKey

__all__ = [
    # JWT
    "JWTAuth",
    "create_access_token",
    "create_refresh_token",
    "verify_token",
    "TokenPayload",
    # API Key
    "APIKeyAuth",
    "generate_api_key",
    "hash_api_key",
    # RBAC
    "Permission",
    "Role",
    "RBACManager",
    "require_permission",
    "require_role",
    # Dependencies
    "get_current_user",
    "get_current_tenant",
    "get_api_key",
    # Models
    "User",
    "Tenant",
    "APIKey",
]
