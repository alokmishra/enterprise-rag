"""
Enterprise RAG System - Tenant Context Middleware

Extracts tenant context from requests and adds it to request state.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from src.core.deployment import DeploymentMode, get_deployment_config
from src.core.logging import get_logger
from src.core.tenant import (
    TenantConfig,
    TenantContext,
    DEFAULT_TENANT_ID,
    get_default_tenant_config,
)

logger = get_logger(__name__)


async def tenant_context_middleware(request: Request, call_next) -> Response:
    """
    Middleware to extract and validate tenant context from requests.

    For multi-tenant SaaS mode, extracts tenant_id from:
    1. JWT token (Authorization header)
    2. API key header (X-API-Key)
    3. Query parameter (tenant_id)

    For single-tenant modes (on-premise, air-gapped), uses default tenant.
    """
    deployment = get_deployment_config()

    try:
        if deployment.mode == DeploymentMode.SAAS_MULTI_TENANT:
            # Multi-tenant mode - extract tenant from request
            tenant_id = await _extract_tenant_id(request)

            if tenant_id is None:
                logger.warning(
                    "Missing tenant context in multi-tenant mode",
                    path=request.url.path,
                )
                return JSONResponse(
                    status_code=401,
                    content={
                        "detail": "Tenant context required",
                        "error": "missing_tenant",
                    },
                )

            # Load tenant config (in production, this would query a database)
            tenant_config = await _load_tenant_config(tenant_id)

        else:
            # Single-tenant modes - use default tenant
            tenant_id = DEFAULT_TENANT_ID
            tenant_config = get_default_tenant_config()

        # Create tenant context and add to request state
        tenant_context = TenantContext(
            tenant_id=tenant_id,
            tenant_config=tenant_config,
        )

        request.state.tenant_id = tenant_id
        request.state.tenant_config = tenant_config
        request.state.tenant_context = tenant_context

        logger.debug(
            "Tenant context set",
            tenant_id=tenant_id,
            deployment_mode=deployment.mode.value,
        )

        # Process request
        response = await call_next(request)

        # Add tenant ID to response headers for debugging
        response.headers["X-Tenant-ID"] = tenant_id

        return response

    except TenantNotFoundError as e:
        logger.warning(
            "Tenant not found",
            tenant_id=str(e),
            path=request.url.path,
        )
        return JSONResponse(
            status_code=404,
            content={
                "detail": "Tenant not found",
                "error": "tenant_not_found",
            },
        )

    except TenantDisabledError as e:
        logger.warning(
            "Tenant disabled",
            tenant_id=str(e),
            path=request.url.path,
        )
        return JSONResponse(
            status_code=403,
            content={
                "detail": "Tenant account is disabled",
                "error": "tenant_disabled",
            },
        )


async def _extract_tenant_id(request: Request) -> Optional[str]:
    """
    Extract tenant ID from request.

    Checks in order:
    1. JWT token claims
    2. X-Tenant-ID header
    3. Query parameter
    """
    # Try to get from request state (set by auth middleware)
    if hasattr(request.state, "user"):
        user = getattr(request.state, "user", None)
        if user and hasattr(user, "tenant_id"):
            return user.tenant_id

    # Try to get from X-Tenant-ID header
    tenant_header = request.headers.get("X-Tenant-ID")
    if tenant_header:
        return tenant_header

    # Try to get from query parameter (useful for testing/admin)
    tenant_param = request.query_params.get("tenant_id")
    if tenant_param:
        return tenant_param

    return None


async def _load_tenant_config(tenant_id: str) -> TenantConfig:
    """
    Load tenant configuration.

    In production, this would query a tenant database or cache.
    For now, returns a default config with the tenant_id set.
    """
    # TODO: Implement actual tenant config loading from database
    # For now, return a default config
    return TenantConfig(
        tenant_id=tenant_id,
        tenant_name=f"Tenant {tenant_id}",
    )


class TenantNotFoundError(Exception):
    """Raised when a tenant is not found."""

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        super().__init__(tenant_id)


class TenantDisabledError(Exception):
    """Raised when a tenant is disabled."""

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        super().__init__(tenant_id)


# FastAPI dependency for getting tenant ID
def get_tenant_id(request: Request) -> str:
    """
    Dependency to get current tenant ID from request state.

    Usage:
        @router.get("/documents")
        async def list_documents(tenant_id: str = Depends(get_tenant_id)):
            ...
    """
    tenant_id = getattr(request.state, "tenant_id", None)
    if tenant_id is None:
        tenant_id = DEFAULT_TENANT_ID
    return tenant_id


def get_tenant_config(request: Request) -> Optional[TenantConfig]:
    """
    Dependency to get current tenant config from request state.

    Usage:
        @router.get("/documents")
        async def list_documents(tenant_config: TenantConfig = Depends(get_tenant_config)):
            ...
    """
    return getattr(request.state, "tenant_config", None)


def get_tenant_context(request: Request) -> TenantContext:
    """
    Dependency to get current tenant context from request state.

    Usage:
        @router.get("/documents")
        async def list_documents(tenant_ctx: TenantContext = Depends(get_tenant_context)):
            ...
    """
    tenant_context = getattr(request.state, "tenant_context", None)
    if tenant_context is None:
        # Return default context
        return TenantContext(tenant_id=DEFAULT_TENANT_ID)
    return tenant_context
