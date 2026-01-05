"""
Enterprise RAG System - Observability Middleware
"""

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.observability.metrics import get_metrics
from src.observability.tracing import create_span, add_span_attribute


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware to collect HTTP metrics.

    Tracks:
    - Request count by endpoint, method, and status
    - Request latency
    - Active request count
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        metrics = get_metrics()

        # Track active requests
        metrics.active_requests.inc()

        # Get endpoint path
        endpoint = request.url.path

        # Skip metrics endpoint to avoid self-tracking
        if endpoint == "/metrics":
            response = await call_next(request)
            metrics.active_requests.dec()
            return response

        start_time = time.time()

        try:
            response = await call_next(request)
            latency = time.time() - start_time

            # Track request
            metrics.track_request(
                endpoint=endpoint,
                method=request.method,
                status=response.status_code,
                latency=latency,
            )

            return response

        except Exception as e:
            latency = time.time() - start_time

            # Track error
            metrics.track_request(
                endpoint=endpoint,
                method=request.method,
                status=500,
                latency=latency,
            )
            metrics.track_error(
                error_type=type(e).__name__,
                component="http",
            )
            raise

        finally:
            metrics.active_requests.dec()


class TracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add distributed tracing.

    Creates a span for each HTTP request with:
    - Request path, method
    - Response status
    - Client info
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        # Extract trace context from headers (for distributed tracing)
        # traceparent = request.headers.get("traceparent")

        endpoint = request.url.path

        # Skip tracing for health checks and metrics
        if endpoint in ["/health", "/metrics"]:
            return await call_next(request)

        with create_span(
            f"HTTP {request.method} {endpoint}",
            {
                "http.method": request.method,
                "http.url": str(request.url),
                "http.route": endpoint,
                "http.user_agent": request.headers.get("user-agent", ""),
                "http.client_ip": request.client.host if request.client else "",
            },
        ) as span:
            try:
                response = await call_next(request)

                add_span_attribute("http.status_code", response.status_code)

                return response

            except Exception as e:
                add_span_attribute("http.status_code", 500)
                add_span_attribute("error.type", type(e).__name__)
                add_span_attribute("error.message", str(e))
                raise


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware to set up request context.

    Adds:
    - Request ID
    - Tenant context
    - User context
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        import uuid

        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Store in request state for access in handlers
        request.state.request_id = request_id

        # Add to response headers
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response
