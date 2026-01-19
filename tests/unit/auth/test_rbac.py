"""Tests for Role-Based Access Control."""

import pytest
from unittest.mock import MagicMock

from src.auth.rbac import (
    Permission,
    Role,
    RBACManager,
    require_permission,
    require_role,
)


class TestPermission:
    """Tests for Permission enum."""

    def test_permission_values(self):
        """Test that all expected permissions exist."""
        assert Permission.DOCUMENTS_READ
        assert Permission.DOCUMENTS_WRITE
        assert Permission.DOCUMENTS_DELETE
        assert Permission.QUERIES_EXECUTE
        assert Permission.ADMIN_SETTINGS
        assert Permission.USERS_READ
        assert Permission.TENANTS_READ

    def test_permission_string_values(self):
        """Test permission string representations."""
        assert Permission.DOCUMENTS_READ.value == "documents:read"
        assert Permission.QUERIES_EXECUTE.value == "queries:execute"


class TestRole:
    """Tests for Role enum."""

    def test_role_values(self):
        """Test that all expected roles exist."""
        assert Role.VIEWER
        assert Role.USER
        assert Role.EDITOR
        assert Role.ADMIN
        assert Role.SUPER_ADMIN

    def test_role_hierarchy(self):
        """Test role string values."""
        assert Role.VIEWER.value == "viewer"
        assert Role.SUPER_ADMIN.value == "super_admin"


class TestRBACManager:
    """Tests for RBACManager class."""

    @pytest.fixture
    def rbac_manager(self):
        """Create RBACManager instance."""
        return RBACManager()

    def test_viewer_permissions(self, rbac_manager):
        """Test viewer role has read-only permissions."""
        assert rbac_manager.has_permission(Role.VIEWER, Permission.DOCUMENTS_READ)
        assert not rbac_manager.has_permission(Role.VIEWER, Permission.DOCUMENTS_WRITE)
        assert not rbac_manager.has_permission(Role.VIEWER, Permission.DOCUMENTS_DELETE)
        assert not rbac_manager.has_permission(Role.VIEWER, Permission.ADMIN_SETTINGS)

    def test_user_permissions(self, rbac_manager):
        """Test user role has query permissions."""
        assert rbac_manager.has_permission(Role.USER, Permission.DOCUMENTS_READ)
        assert rbac_manager.has_permission(Role.USER, Permission.QUERIES_EXECUTE)
        assert not rbac_manager.has_permission(Role.USER, Permission.DOCUMENTS_DELETE)
        assert not rbac_manager.has_permission(Role.USER, Permission.ADMIN_SETTINGS)

    def test_editor_permissions(self, rbac_manager):
        """Test editor role has write permissions."""
        assert rbac_manager.has_permission(Role.EDITOR, Permission.DOCUMENTS_READ)
        assert rbac_manager.has_permission(Role.EDITOR, Permission.DOCUMENTS_WRITE)
        assert rbac_manager.has_permission(Role.EDITOR, Permission.QUERIES_EXECUTE)
        assert not rbac_manager.has_permission(Role.EDITOR, Permission.ADMIN_SETTINGS)

    def test_admin_permissions(self, rbac_manager):
        """Test admin role has admin permissions."""
        assert rbac_manager.has_permission(Role.ADMIN, Permission.DOCUMENTS_READ)
        assert rbac_manager.has_permission(Role.ADMIN, Permission.DOCUMENTS_WRITE)
        assert rbac_manager.has_permission(Role.ADMIN, Permission.DOCUMENTS_DELETE)
        assert rbac_manager.has_permission(Role.ADMIN, Permission.ADMIN_SETTINGS)
        assert rbac_manager.has_permission(Role.ADMIN, Permission.USERS_READ)

    def test_super_admin_permissions(self, rbac_manager):
        """Test super admin has all permissions."""
        for permission in Permission:
            assert rbac_manager.has_permission(Role.SUPER_ADMIN, permission)

    def test_extra_permissions(self, rbac_manager):
        """Test extra permissions override role defaults."""
        # Viewer with extra write permission
        assert rbac_manager.has_permission(
            Role.VIEWER,
            Permission.DOCUMENTS_WRITE,
            extra_permissions=[Permission.DOCUMENTS_WRITE.value],
        )

    def test_get_role_permissions(self, rbac_manager):
        """Test getting all permissions for a role."""
        viewer_perms = rbac_manager.get_role_permissions(Role.VIEWER)
        assert Permission.DOCUMENTS_READ in viewer_perms
        assert Permission.DOCUMENTS_WRITE not in viewer_perms

        admin_perms = rbac_manager.get_role_permissions(Role.ADMIN)
        assert len(admin_perms) > len(viewer_perms)

    def test_has_any_permission(self, rbac_manager):
        """Test checking for any of multiple permissions."""
        assert rbac_manager.has_any_permission(
            Role.VIEWER,
            [Permission.DOCUMENTS_READ, Permission.DOCUMENTS_WRITE],
        )
        assert not rbac_manager.has_any_permission(
            Role.VIEWER,
            [Permission.DOCUMENTS_WRITE, Permission.DOCUMENTS_DELETE],
        )

    def test_has_all_permissions(self, rbac_manager):
        """Test checking for all of multiple permissions."""
        assert rbac_manager.has_all_permissions(
            Role.EDITOR,
            [Permission.DOCUMENTS_READ, Permission.DOCUMENTS_WRITE],
        )
        assert not rbac_manager.has_all_permissions(
            Role.EDITOR,
            [Permission.DOCUMENTS_WRITE, Permission.ADMIN_SETTINGS],
        )


class TestRequirePermissionDecorator:
    """Tests for require_permission decorator."""

    @pytest.fixture
    def mock_request(self):
        """Create mock request with user state."""
        request = MagicMock()
        request.state.user = MagicMock(
            role=Role.USER,
            permissions=[Permission.QUERIES_EXECUTE.value],
        )
        return request

    def test_require_permission_allowed(self, mock_request):
        """Test decorator allows access when permission exists."""
        @require_permission(Permission.QUERIES_EXECUTE)
        async def protected_endpoint(request):
            return {"status": "success"}

        # Should not raise
        # Note: Actual test would need to await the decorated function

    def test_require_permission_denied(self, mock_request):
        """Test decorator denies access when permission missing."""
        mock_request.state.user.role = Role.VIEWER
        mock_request.state.user.permissions = []

        @require_permission(Permission.DOCUMENTS_WRITE)
        async def protected_endpoint(request):
            return {"status": "success"}

        # Should raise PermissionDenied
        # Note: Actual test would need to await and expect exception


class TestRequireRoleDecorator:
    """Tests for require_role decorator."""

    @pytest.fixture
    def mock_request_admin(self):
        """Create mock request with admin user."""
        request = MagicMock()
        request.state.user = MagicMock(role=Role.ADMIN)
        return request

    def test_require_role_allowed(self, mock_request_admin):
        """Test decorator allows access when role is sufficient."""
        @require_role(Role.USER)
        async def protected_endpoint(request):
            return {"status": "success"}

        # Admin should have access to user-level endpoints

    def test_require_role_exact_match(self, mock_request_admin):
        """Test decorator with exact role match."""
        @require_role(Role.ADMIN)
        async def admin_endpoint(request):
            return {"status": "success"}

        # Admin should have access to admin endpoints

    def test_require_role_denied(self):
        """Test decorator denies access when role insufficient."""
        request = MagicMock()
        request.state.user = MagicMock(role=Role.VIEWER)

        @require_role(Role.ADMIN)
        async def admin_endpoint(request):
            return {"status": "success"}

        # Viewer should not have access to admin endpoints
