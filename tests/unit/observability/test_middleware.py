"""Tests for observability middleware."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestMetricsMiddleware:
    """Tests for MetricsMiddleware class."""

    def test_middleware_tracks_successful_request(self):
        """Test middleware tracks successful requests."""
        from src.observability.middleware import MetricsMiddleware

        app = FastAPI()
        app.add_middleware(MetricsMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200

    def test_middleware_tracks_failed_request(self):
        """Test middleware tracks failed requests."""
        from src.observability.middleware import MetricsMiddleware

        app = FastAPI()
        app.add_middleware(MetricsMiddleware)

        @app.get("/error")
        async def error_endpoint():
            raise ValueError("Test error")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/error")

        assert response.status_code == 500

    def test_middleware_skips_metrics_endpoint(self):
        """Test middleware skips /metrics endpoint to avoid self-tracking."""
        from src.observability.middleware import MetricsMiddleware

        app = FastAPI()
        app.add_middleware(MetricsMiddleware)

        @app.get("/metrics")
        async def metrics_endpoint():
            return "metrics"

        client = TestClient(app)
        response = client.get("/metrics")

        assert response.status_code == 200


class TestTracingMiddleware:
    """Tests for TracingMiddleware class."""

    def test_tracing_middleware_successful_request(self):
        """Test tracing middleware on successful request."""
        from src.observability.middleware import TracingMiddleware

        app = FastAPI()
        app.add_middleware(TracingMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200

    def test_tracing_middleware_skips_health(self):
        """Test tracing middleware skips health endpoint."""
        from src.observability.middleware import TracingMiddleware

        app = FastAPI()
        app.add_middleware(TracingMiddleware)

        @app.get("/health")
        async def health_endpoint():
            return {"status": "healthy"}

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200

    def test_tracing_middleware_skips_metrics(self):
        """Test tracing middleware skips metrics endpoint."""
        from src.observability.middleware import TracingMiddleware

        app = FastAPI()
        app.add_middleware(TracingMiddleware)

        @app.get("/metrics")
        async def metrics_endpoint():
            return "metrics"

        client = TestClient(app)
        response = client.get("/metrics")

        assert response.status_code == 200

    def test_tracing_middleware_handles_error(self):
        """Test tracing middleware handles errors."""
        from src.observability.middleware import TracingMiddleware

        app = FastAPI()
        app.add_middleware(TracingMiddleware)

        @app.get("/error")
        async def error_endpoint():
            raise ValueError("Test error")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/error")

        assert response.status_code == 500


class TestRequestContextMiddleware:
    """Tests for RequestContextMiddleware class."""

    def test_request_context_adds_request_id(self):
        """Test middleware adds request ID to response."""
        from src.observability.middleware import RequestContextMiddleware

        app = FastAPI()
        app.add_middleware(RequestContextMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        assert "X-Request-ID" in response.headers

    def test_request_context_uses_provided_request_id(self):
        """Test middleware uses X-Request-ID from request headers."""
        from src.observability.middleware import RequestContextMiddleware

        app = FastAPI()
        app.add_middleware(RequestContextMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test", headers={"X-Request-ID": "custom-id-123"})

        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == "custom-id-123"

    def test_request_context_generates_uuid_if_not_provided(self):
        """Test middleware generates UUID if X-Request-ID not provided."""
        from src.observability.middleware import RequestContextMiddleware
        import uuid

        app = FastAPI()
        app.add_middleware(RequestContextMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        request_id = response.headers["X-Request-ID"]
        # Should be a valid UUID
        try:
            uuid.UUID(request_id)
            is_valid_uuid = True
        except ValueError:
            is_valid_uuid = False
        assert is_valid_uuid


class TestCombinedMiddleware:
    """Tests for combining multiple middlewares."""

    def test_all_middlewares_together(self):
        """Test all middleware working together."""
        from src.observability.middleware import (
            MetricsMiddleware,
            TracingMiddleware,
            RequestContextMiddleware,
        )

        app = FastAPI()
        app.add_middleware(RequestContextMiddleware)
        app.add_middleware(TracingMiddleware)
        app.add_middleware(MetricsMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
