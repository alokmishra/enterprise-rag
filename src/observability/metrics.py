"""
Enterprise RAG System - Prometheus Metrics
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Callable, Optional

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)

from src.core.config import get_settings


class MetricsCollector:
    """
    Prometheus metrics collector for the RAG system.

    Tracks:
    - Request counts and latencies
    - Token usage
    - Agent execution times
    - Error rates
    - System health
    """

    def __init__(self, namespace: str = "rag"):
        self.namespace = namespace

        # Request metrics
        self.request_count = Counter(
            f"{namespace}_requests_total",
            "Total number of requests",
            ["endpoint", "method", "status"],
        )

        self.request_latency = Histogram(
            f"{namespace}_request_latency_seconds",
            "Request latency in seconds",
            ["endpoint", "method"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        # Query metrics
        self.query_count = Counter(
            f"{namespace}_queries_total",
            "Total number of queries",
            ["complexity", "strategy", "status"],
        )

        self.query_latency = Histogram(
            f"{namespace}_query_latency_seconds",
            "Query latency in seconds",
            ["complexity", "strategy"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
        )

        # Token metrics
        self.tokens_used = Counter(
            f"{namespace}_tokens_total",
            "Total tokens used",
            ["model", "operation"],
        )

        self.tokens_per_query = Histogram(
            f"{namespace}_tokens_per_query",
            "Tokens used per query",
            ["model"],
            buckets=[100, 500, 1000, 2000, 4000, 8000, 16000, 32000],
        )

        # Agent metrics
        self.agent_execution_count = Counter(
            f"{namespace}_agent_executions_total",
            "Total agent executions",
            ["agent", "status"],
        )

        self.agent_latency = Histogram(
            f"{namespace}_agent_latency_seconds",
            "Agent execution latency in seconds",
            ["agent"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        )

        # Retrieval metrics
        self.retrieval_count = Counter(
            f"{namespace}_retrievals_total",
            "Total retrieval operations",
            ["strategy", "status"],
        )

        self.retrieval_results = Histogram(
            f"{namespace}_retrieval_results_count",
            "Number of results per retrieval",
            ["strategy"],
            buckets=[0, 1, 5, 10, 20, 50, 100],
        )

        # Document metrics
        self.documents_ingested = Counter(
            f"{namespace}_documents_ingested_total",
            "Total documents ingested",
            ["status"],
        )

        self.chunks_created = Counter(
            f"{namespace}_chunks_created_total",
            "Total chunks created",
        )

        # Error metrics
        self.errors = Counter(
            f"{namespace}_errors_total",
            "Total errors",
            ["type", "component"],
        )

        # System metrics
        self.active_requests = Gauge(
            f"{namespace}_active_requests",
            "Number of active requests",
        )

        self.cache_hits = Counter(
            f"{namespace}_cache_hits_total",
            "Cache hit count",
            ["cache_type"],
        )

        self.cache_misses = Counter(
            f"{namespace}_cache_misses_total",
            "Cache miss count",
            ["cache_type"],
        )

        # Info metric
        self.info = Info(
            f"{namespace}_info",
            "RAG system information",
        )
        self._set_info()

    def _set_info(self) -> None:
        """Set system info metric."""
        settings = get_settings()
        self.info.info({
            "version": "1.0.0",
            "environment": settings.RAG_ENV,
        })

    def track_request(
        self,
        endpoint: str,
        method: str,
        status: int,
        latency: float,
    ) -> None:
        """Track an HTTP request."""
        self.request_count.labels(
            endpoint=endpoint,
            method=method,
            status=str(status),
        ).inc()
        self.request_latency.labels(
            endpoint=endpoint,
            method=method,
        ).observe(latency)

    def track_query(
        self,
        complexity: str,
        strategy: str,
        status: str,
        latency: float,
    ) -> None:
        """Track a query execution."""
        self.query_count.labels(
            complexity=complexity,
            strategy=strategy,
            status=status,
        ).inc()
        self.query_latency.labels(
            complexity=complexity,
            strategy=strategy,
        ).observe(latency)

    def track_tokens(
        self,
        model: str,
        operation: str,
        count: int,
    ) -> None:
        """Track token usage."""
        self.tokens_used.labels(
            model=model,
            operation=operation,
        ).inc(count)
        self.tokens_per_query.labels(model=model).observe(count)

    def track_agent(
        self,
        agent: str,
        status: str,
        latency: float,
    ) -> None:
        """Track agent execution."""
        self.agent_execution_count.labels(
            agent=agent,
            status=status,
        ).inc()
        self.agent_latency.labels(agent=agent).observe(latency)

    def track_retrieval(
        self,
        strategy: str,
        status: str,
        result_count: int,
    ) -> None:
        """Track retrieval operation."""
        self.retrieval_count.labels(
            strategy=strategy,
            status=status,
        ).inc()
        self.retrieval_results.labels(strategy=strategy).observe(result_count)

    def track_error(
        self,
        error_type: str,
        component: str,
    ) -> None:
        """Track an error."""
        self.errors.labels(
            type=error_type,
            component=component,
        ).inc()

    def track_cache(
        self,
        cache_type: str,
        hit: bool,
    ) -> None:
        """Track cache hit/miss."""
        if hit:
            self.cache_hits.labels(cache_type=cache_type).inc()
        else:
            self.cache_misses.labels(cache_type=cache_type).inc()

    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format."""
        return generate_latest(REGISTRY)


# Global metrics instance
_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get the metrics collector singleton."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


# Convenience functions
def track_request(endpoint: str, method: str, status: int, latency: float) -> None:
    """Track an HTTP request."""
    get_metrics().track_request(endpoint, method, status, latency)


def track_latency(component: str):
    """Decorator to track function latency."""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                get_metrics().track_agent(component, "success", time.time() - start)
                return result
            except Exception as e:
                get_metrics().track_agent(component, "error", time.time() - start)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                get_metrics().track_agent(component, "success", time.time() - start)
                return result
            except Exception as e:
                get_metrics().track_agent(component, "error", time.time() - start)
                raise

        if asyncio_iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def track_tokens(model: str, operation: str, count: int) -> None:
    """Track token usage."""
    get_metrics().track_tokens(model, operation, count)


def track_error(error_type: str, component: str) -> None:
    """Track an error."""
    get_metrics().track_error(error_type, component)


def asyncio_iscoroutinefunction(func):
    """Check if function is async."""
    import asyncio
    return asyncio.iscoroutinefunction(func)
