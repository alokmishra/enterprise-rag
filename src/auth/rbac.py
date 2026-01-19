"""
Enterprise RAG System - Role-Based Access Control
"""

from __future__ import annotations

from enum import Enum
from functools import wraps
from typing import Callable, Optional

from fastapi import HTTPException, status

from src.core.exceptions import AuthorizationError


class Permission(str, Enum):
    """System permissions."""
    # Document permissions
    DOCUMENTS_READ = "documents:read"
    DOCUMENTS_WRITE = "documents:write"
    DOCUMENTS_DELETE = "documents:delete"
    DOCUMENTS_ADMIN = "documents:admin"

    # Query permissions
    QUERIES_EXECUTE = "queries:execute"
    QUERIES_READ_HISTORY = "queries:read_history"

    # User management
    USERS_READ = "users:read"
    USERS_WRITE = "users:write"
    USERS_DELETE = "users:delete"

    # API key management
    API_KEYS_READ = "api_keys:read"
    API_KEYS_WRITE = "api_keys:write"
    API_KEYS_DELETE = "api_keys:delete"

    # Admin permissions
    ADMIN_SETTINGS = "admin:settings"
    ADMIN_AUDIT = "admin:audit"
    ADMIN_METRICS = "admin:metrics"

    # Tenant management (super admin only)
    TENANTS_READ = "tenants:read"
    TENANTS_WRITE = "tenants:write"
    TENANTS_DELETE = "tenants:delete"


class Role(str, Enum):
    """System roles with predefined permission sets."""
    VIEWER = "viewer"
    USER = "user"
    EDITOR = "editor"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


# Role to permissions mapping
ROLE_PERMISSIONS: dict[Role, list[Permission]] = {
    Role.VIEWER: [
        Permission.DOCUMENTS_READ,
        Permission.QUERIES_EXECUTE,
    ],
    Role.USER: [
        Permission.DOCUMENTS_READ,
        Permission.QUERIES_EXECUTE,
        Permission.QUERIES_READ_HISTORY,
    ],
    Role.EDITOR: [
        Permission.DOCUMENTS_READ,
        Permission.DOCUMENTS_WRITE,
        Permission.QUERIES_EXECUTE,
        Permission.QUERIES_READ_HISTORY,
    ],
    Role.ADMIN: [
        Permission.DOCUMENTS_READ,
        Permission.DOCUMENTS_WRITE,
        Permission.DOCUMENTS_DELETE,
        Permission.DOCUMENTS_ADMIN,
        Permission.QUERIES_EXECUTE,
        Permission.QUERIES_READ_HISTORY,
        Permission.USERS_READ,
        Permission.USERS_WRITE,
        Permission.API_KEYS_READ,
        Permission.API_KEYS_WRITE,
        Permission.API_KEYS_DELETE,
        Permission.ADMIN_SETTINGS,
        Permission.ADMIN_AUDIT,
        Permission.ADMIN_METRICS,
    ],
    Role.SUPER_ADMIN: [
        # All permissions
        Permission.DOCUMENTS_READ,
        Permission.DOCUMENTS_WRITE,
        Permission.DOCUMENTS_DELETE,
        Permission.DOCUMENTS_ADMIN,
        Permission.QUERIES_EXECUTE,
        Permission.QUERIES_READ_HISTORY,
        Permission.USERS_READ,
        Permission.USERS_WRITE,
        Permission.USERS_DELETE,
        Permission.API_KEYS_READ,
        Permission.API_KEYS_WRITE,
        Permission.API_KEYS_DELETE,
        Permission.ADMIN_SETTINGS,
        Permission.ADMIN_AUDIT,
        Permission.ADMIN_METRICS,
        Permission.TENANTS_READ,
        Permission.TENANTS_WRITE,
        Permission.TENANTS_DELETE,
    ],
}


class RBACManager:
    """Role-based access control manager."""

    def __init__(self):
        self.role_permissions = ROLE_PERMISSIONS.copy()
        self._custom_permissions: dict[str, list[Permission]] = {}

    def get_role_permissions(self, role: Role) -> list[Permission]:
        """Get all permissions for a role."""
        return self.role_permissions.get(role, [])

    def has_permission(
        self,
        role: Role,
        permission: Permission,
        extra_permissions: list[str] = None,
    ) -> bool:
        """Check if a role has a specific permission."""
        role_perms = self.get_role_permissions(role)

        # Check role permissions
        if permission in role_perms:
            return True

        # Check extra permissions
        if extra_permissions and permission.value in extra_permissions:
            return True

        return False

    def has_any_permission(
        self,
        role: Role,
        permissions: list[Permission],
        extra_permissions: list[str] = None,
    ) -> bool:
        """Check if a role has any of the specified permissions."""
        return any(
            self.has_permission(role, p, extra_permissions)
            for p in permissions
        )

    def has_all_permissions(
        self,
        role: Role,
        permissions: list[Permission],
        extra_permissions: list[str] = None,
    ) -> bool:
        """Check if a role has all of the specified permissions."""
        return all(
            self.has_permission(role, p, extra_permissions)
            for p in permissions
        )

    def add_custom_permission(self, user_id: str, permission: Permission) -> None:
        """Add a custom permission for a specific user."""
        if user_id not in self._custom_permissions:
            self._custom_permissions[user_id] = []
        if permission not in self._custom_permissions[user_id]:
            self._custom_permissions[user_id].append(permission)

    def get_custom_permissions(self, user_id: str) -> list[Permission]:
        """Get custom permissions for a user."""
        return self._custom_permissions.get(user_id, [])


# Global RBAC manager instance
_rbac_manager: Optional[RBACManager] = None


def get_rbac_manager() -> RBACManager:
    """Get the RBAC manager singleton."""
    global _rbac_manager
    if _rbac_manager is None:
        _rbac_manager = RBACManager()
    return _rbac_manager


def require_permission(permission: Permission):
    """
    Decorator to require a specific permission.

    Usage:
        @require_permission(Permission.DOCUMENTS_WRITE)
        async def create_document(current_user: User = Depends(get_current_user)):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get current user from kwargs
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            rbac = get_rbac_manager()
            try:
                role = Role(current_user.role)
            except ValueError:
                role = Role.VIEWER

            if not rbac.has_permission(role, permission, current_user.permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: {permission.value} required",
                )

            return await func(*args, **kwargs)
        return wrapper
    return decorator


def require_role(required_role: Role):
    """
    Decorator to require a specific role or higher.

    Usage:
        @require_role(Role.ADMIN)
        async def admin_endpoint(current_user: User = Depends(get_current_user)):
            ...
    """
    role_hierarchy = [Role.VIEWER, Role.USER, Role.EDITOR, Role.ADMIN, Role.SUPER_ADMIN]

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            try:
                user_role = Role(current_user.role)
            except ValueError:
                user_role = Role.VIEWER

            required_idx = role_hierarchy.index(required_role)
            user_idx = role_hierarchy.index(user_role)

            if user_idx < required_idx:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role {required_role.value} or higher required",
                )

            return await func(*args, **kwargs)
        return wrapper
    return decorator
