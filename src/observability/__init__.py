"""
Enterprise RAG System - Observability Module

Provides:
- Prometheus metrics
- Distributed tracing (OpenTelemetry)
- Health checks
- Performance monitoring
"""

from __future__ import annotations

from src.observability.metrics import (
    MetricsCollector,
    get_metrics,
    track_request,
    track_latency,
    track_tokens,
    track_error,
)
from src.observability.tracing import (
    TracingManager,
    get_tracer,
    create_span,
    trace_function,
)
from src.observability.middleware import (
    MetricsMiddleware,
    TracingMiddleware,
    RequestContextMiddleware,
)

__all__ = [
    # Metrics
    "MetricsCollector",
    "get_metrics",
    "track_request",
    "track_latency",
    "track_tokens",
    "track_error",
    # Tracing
    "TracingManager",
    "get_tracer",
    "create_span",
    "trace_function",
    # Middleware
    "MetricsMiddleware",
    "TracingMiddleware",
    "RequestContextMiddleware",
]
