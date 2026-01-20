"""
Tests for tenant context middleware.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.deployment import DeploymentMode, reset_deployment_config
from src.core.tenant import DEFAULT_TENANT_ID


class TestTenantMiddlewareMultiTenant:
    """Tests for multi-tenant mode."""

    def setup_method(self):
        """Reset deployment config before each test."""
        reset_deployment_config()

    def teardown_method(self):
        """Reset deployment config after each test."""
        reset_deployment_config()

    @pytest.mark.asyncio
    async def test_middleware_extracts_tenant_from_header(self):
        """Tenant ID extracted from X-Tenant-ID header."""
        from src.api.middleware.tenant import tenant_context_middleware

        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "saas_multi_tenant"}):
            reset_deployment_config()

            # Create mock request with tenant header
            mock_request = MagicMock()
            mock_request.state = MagicMock(spec=[])  # Empty spec to avoid auto-attributes
            mock_request.headers = {"X-Tenant-ID": "tenant-from-header"}
            mock_request.query_params = {}
            mock_request.url.path = "/api/v1/query"

            mock_response = MagicMock()
            mock_response.headers = {}

            async def mock_call_next(request):
                return mock_response

            response = await tenant_context_middleware(mock_request, mock_call_next)

            # Verify tenant_id was set on request state
            assert mock_request.state.tenant_id == "tenant-from-header"
            assert response.headers["X-Tenant-ID"] == "tenant-from-header"

    @pytest.mark.asyncio
    async def test_middleware_extracts_tenant_from_query_param(self):
        """Tenant ID extracted from query parameter."""
        from src.api.middleware.tenant import tenant_context_middleware

        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "saas_multi_tenant"}):
            reset_deployment_config()

            # Create mock request with query param
            mock_request = MagicMock()
            mock_request.state = MagicMock(spec=[])
            mock_request.headers = {}
            mock_request.query_params = {"tenant_id": "tenant-from-param"}
            mock_request.url.path = "/api/v1/query"

            mock_response = MagicMock()
            mock_response.headers = {}

            async def mock_call_next(request):
                return mock_response

            await tenant_context_middleware(mock_request, mock_call_next)

            # Verify tenant_id was set on request state
            assert mock_request.state.tenant_id == "tenant-from-param"

    @pytest.mark.asyncio
    async def test_middleware_rejects_missing_tenant_multi_tenant_mode(self):
        """Multi-tenant mode rejects requests without tenant context."""
        from src.api.middleware.tenant import tenant_context_middleware

        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "saas_multi_tenant"}):
            reset_deployment_config()

            # Create mock request without tenant info
            mock_request = MagicMock()
            mock_request.state = MagicMock(spec=[])
            mock_request.headers = {}
            mock_request.query_params = {}
            mock_request.url.path = "/api/v1/query"

            async def mock_call_next(request):
                return MagicMock()

            response = await tenant_context_middleware(mock_request, mock_call_next)

            # Should return 401 error
            assert response.status_code == 401


class TestTenantMiddlewareSingleTenant:
    """Tests for single-tenant modes."""

    def setup_method(self):
        """Reset deployment config before each test."""
        reset_deployment_config()

    def teardown_method(self):
        """Reset deployment config after each test."""
        reset_deployment_config()

    @pytest.mark.asyncio
    async def test_middleware_uses_default_tenant_on_premise(self):
        """On-premise mode uses 'default' tenant."""
        from src.api.middleware.tenant import tenant_context_middleware

        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "on_premise"}):
            reset_deployment_config()

            mock_request = MagicMock()
            mock_request.state = MagicMock(spec=[])
            mock_request.headers = {}
            mock_request.query_params = {}
            mock_request.url.path = "/api/v1/query"

            mock_response = MagicMock()
            mock_response.headers = {}

            async def mock_call_next(request):
                return mock_response

            response = await tenant_context_middleware(mock_request, mock_call_next)

            # Should use default tenant
            assert mock_request.state.tenant_id == DEFAULT_TENANT_ID

    @pytest.mark.asyncio
    async def test_middleware_uses_default_tenant_air_gapped(self):
        """Air-gapped mode uses 'default' tenant."""
        from src.api.middleware.tenant import tenant_context_middleware

        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "air_gapped"}):
            reset_deployment_config()

            mock_request = MagicMock()
            mock_request.state = MagicMock(spec=[])
            mock_request.headers = {}
            mock_request.query_params = {}
            mock_request.url.path = "/api/v1/query"

            mock_response = MagicMock()
            mock_response.headers = {}

            async def mock_call_next(request):
                return mock_response

            response = await tenant_context_middleware(mock_request, mock_call_next)

            # Should use default tenant
            assert mock_request.state.tenant_id == DEFAULT_TENANT_ID


class TestTenantMiddlewareRequestState:
    """Tests for request state management."""

    def setup_method(self):
        """Reset deployment config before each test."""
        reset_deployment_config()

    def teardown_method(self):
        """Reset deployment config after each test."""
        reset_deployment_config()

    @pytest.mark.asyncio
    async def test_middleware_sets_request_state(self):
        """Middleware sets request.state.tenant_id and tenant_config."""
        from src.api.middleware.tenant import tenant_context_middleware

        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "saas_multi_tenant"}):
            reset_deployment_config()

            mock_request = MagicMock()
            mock_request.state = MagicMock(spec=[])
            mock_request.headers = {"X-Tenant-ID": "test-tenant"}
            mock_request.query_params = {}
            mock_request.url.path = "/api/v1/query"

            mock_response = MagicMock()
            mock_response.headers = {}

            async def mock_call_next(request):
                return mock_response

            await tenant_context_middleware(mock_request, mock_call_next)

            # Verify all state attributes are set
            assert mock_request.state.tenant_id == "test-tenant"
            assert mock_request.state.tenant_config is not None
            assert mock_request.state.tenant_context is not None

    @pytest.mark.asyncio
    async def test_middleware_loads_tenant_config(self):
        """Middleware loads correct TenantConfig for tenant."""
        from src.api.middleware.tenant import tenant_context_middleware

        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "saas_multi_tenant"}):
            reset_deployment_config()

            mock_request = MagicMock()
            mock_request.state = MagicMock(spec=[])
            mock_request.headers = {"X-Tenant-ID": "enterprise-tenant"}
            mock_request.query_params = {}
            mock_request.url.path = "/api/v1/query"

            mock_response = MagicMock()
            mock_response.headers = {}

            async def mock_call_next(request):
                return mock_response

            await tenant_context_middleware(mock_request, mock_call_next)

            # Verify config has correct tenant_id
            config = mock_request.state.tenant_config
            assert config.tenant_id == "enterprise-tenant"


class TestGetTenantDependencies:
    """Tests for FastAPI dependencies."""

    def test_get_tenant_id_returns_from_state(self):
        """get_tenant_id returns tenant_id from request state."""
        from src.api.middleware.tenant import get_tenant_id

        mock_request = MagicMock()
        mock_request.state.tenant_id = "dep-tenant"

        result = get_tenant_id(mock_request)
        assert result == "dep-tenant"

    def test_get_tenant_id_returns_default_if_missing(self):
        """get_tenant_id returns default if not set."""
        from src.api.middleware.tenant import get_tenant_id

        mock_request = MagicMock()
        # Simulate missing tenant_id
        mock_request.state = MagicMock(spec=[])

        result = get_tenant_id(mock_request)
        assert result == DEFAULT_TENANT_ID

    def test_get_tenant_config_returns_from_state(self):
        """get_tenant_config returns config from request state."""
        from src.api.middleware.tenant import get_tenant_config
        from src.core.tenant import TenantConfig

        config = TenantConfig(tenant_id="test", tenant_name="Test")

        mock_request = MagicMock()
        mock_request.state.tenant_config = config

        result = get_tenant_config(mock_request)
        assert result == config

    def test_get_tenant_context_returns_from_state(self):
        """get_tenant_context returns context from request state."""
        from src.api.middleware.tenant import get_tenant_context
        from src.core.tenant import TenantContext

        context = TenantContext(tenant_id="ctx-tenant")

        mock_request = MagicMock()
        mock_request.state.tenant_context = context

        result = get_tenant_context(mock_request)
        assert result == context

    def test_get_tenant_context_returns_default_if_missing(self):
        """get_tenant_context returns default context if not set."""
        from src.api.middleware.tenant import get_tenant_context

        mock_request = MagicMock()
        mock_request.state = MagicMock(spec=[])

        result = get_tenant_context(mock_request)
        assert result.tenant_id == DEFAULT_TENANT_ID
