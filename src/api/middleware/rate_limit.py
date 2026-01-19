"""
Enterprise RAG System - Rate Limiting Middleware

Implements Redis-based rate limiting using a sliding window algorithm.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse

from src.core.config import settings
from src.core.logging import get_logger
from src.storage.cache import get_cache, CacheError


logger = get_logger(__name__)


class RateLimitConfig:
    """Configuration for rate limiting."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10,
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size


# Default rate limit configurations by endpoint type
RATE_LIMITS = {
    "query": RateLimitConfig(requests_per_minute=30, requests_per_hour=500),
    "agent_query": RateLimitConfig(requests_per_minute=10, requests_per_hour=200),
    "documents": RateLimitConfig(requests_per_minute=20, requests_per_hour=300),
    "admin": RateLimitConfig(requests_per_minute=60, requests_per_hour=1000),
    "default": RateLimitConfig(requests_per_minute=60, requests_per_hour=1000),
}


def get_client_identifier(request: Request) -> str:
    """
    Extract client identifier for rate limiting.

    Priority:
    1. API key from header
    2. User ID from JWT (if authenticated)
    3. Client IP address
    """
    # Check for API key
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key[:16]}"  # Use prefix of key

    # Check for authenticated user
    user = getattr(request.state, "user", None)
    if user and hasattr(user, "id"):
        return f"user:{user.id}"

    # Fall back to IP address
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"

    return f"ip:{client_ip}"


def get_endpoint_type(path: str) -> str:
    """Determine endpoint type for rate limit configuration."""
    if "/query/agent" in path:
        return "agent_query"
    elif "/query" in path:
        return "query"
    elif "/documents" in path:
        return "documents"
    elif "/admin" in path:
        return "admin"
    return "default"


async def check_rate_limit(
    client_id: str,
    endpoint_type: str,
    config: Optional[RateLimitConfig] = None,
) -> tuple[bool, dict]:
    """
    Check if request should be rate limited.

    Returns:
        Tuple of (is_allowed, rate_limit_info)
    """
    config = config or RATE_LIMITS.get(endpoint_type, RATE_LIMITS["default"])

    try:
        cache = get_cache()
        if not cache.is_connected:
            # If cache unavailable, allow request (fail open)
            return True, {}

        current_minute = int(time.time() // 60)
        current_hour = int(time.time() // 3600)

        # Keys for tracking
        minute_key = f"rate_limit:{client_id}:{endpoint_type}:minute:{current_minute}"
        hour_key = f"rate_limit:{client_id}:{endpoint_type}:hour:{current_hour}"

        # Get current counts
        client = cache._ensure_connected()

        # Use pipeline for efficiency
        async with client.pipeline(transaction=True) as pipe:
            pipe.get(minute_key)
            pipe.get(hour_key)
            results = await pipe.execute()

        minute_count = int(results[0] or 0)
        hour_count = int(results[1] or 0)

        # Check limits
        if minute_count >= config.requests_per_minute:
            return False, {
                "limit": config.requests_per_minute,
                "remaining": 0,
                "reset": 60 - (int(time.time()) % 60),
                "window": "minute",
            }

        if hour_count >= config.requests_per_hour:
            return False, {
                "limit": config.requests_per_hour,
                "remaining": 0,
                "reset": 3600 - (int(time.time()) % 3600),
                "window": "hour",
            }

        # Increment counters
        async with client.pipeline(transaction=True) as pipe:
            pipe.incr(minute_key)
            pipe.expire(minute_key, 60)
            pipe.incr(hour_key)
            pipe.expire(hour_key, 3600)
            await pipe.execute()

        return True, {
            "limit": config.requests_per_minute,
            "remaining": config.requests_per_minute - minute_count - 1,
            "reset": 60 - (int(time.time()) % 60),
            "window": "minute",
        }

    except (CacheError, Exception) as e:
        logger.warning("Rate limit check failed, allowing request", error=str(e))
        return True, {}


async def rate_limit_middleware(request: Request, call_next: Callable) -> Response:
    """
    Rate limiting middleware.

    Applies rate limits based on client identifier and endpoint type.
    Adds rate limit headers to responses.
    """
    # Skip rate limiting for health checks
    if request.url.path in ["/health", "/health/detailed", "/ready", "/live"]:
        return await call_next(request)

    # Skip if disabled (e.g., in development)
    if settings.RAG_ENV == "development" and not settings.DEBUG:
        return await call_next(request)

    client_id = get_client_identifier(request)
    endpoint_type = get_endpoint_type(request.url.path)

    is_allowed, rate_info = await check_rate_limit(client_id, endpoint_type)

    if not is_allowed:
        logger.warning(
            "Rate limit exceeded",
            client_id=client_id,
            endpoint_type=endpoint_type,
            path=request.url.path,
        )

        return JSONResponse(
            status_code=429,
            content={
                "detail": "Rate limit exceeded",
                "retry_after": rate_info.get("reset", 60),
            },
            headers={
                "Retry-After": str(rate_info.get("reset", 60)),
                "X-RateLimit-Limit": str(rate_info.get("limit", 0)),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(rate_info.get("reset", 0)),
            },
        )

    # Process request
    response = await call_next(request)

    # Add rate limit headers to response
    if rate_info:
        response.headers["X-RateLimit-Limit"] = str(rate_info.get("limit", 0))
        response.headers["X-RateLimit-Remaining"] = str(rate_info.get("remaining", 0))
        response.headers["X-RateLimit-Reset"] = str(rate_info.get("reset", 0))

    return response


# FastAPI dependency for route-level rate limiting
class RateLimiter:
    """
    Dependency for route-level rate limiting.

    Usage:
        @router.post("/query", dependencies=[Depends(RateLimiter(requests_per_minute=30))])
        async def query(request: QueryRequest):
            ...
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
    ):
        self.config = RateLimitConfig(
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour,
        )

    async def __call__(self, request: Request) -> None:
        client_id = get_client_identifier(request)
        endpoint_type = get_endpoint_type(request.url.path)

        is_allowed, rate_info = await check_rate_limit(
            client_id, endpoint_type, self.config
        )

        if not is_allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "message": "Rate limit exceeded",
                    "retry_after": rate_info.get("reset", 60),
                },
                headers={
                    "Retry-After": str(rate_info.get("reset", 60)),
                },
            )
