"""
Enterprise RAG System - Distributed Tracing (OpenTelemetry)
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Span, Status, StatusCode

from src.core.config import get_settings
from src.core.logging import get_logger


logger = get_logger(__name__)


class TracingManager:
    """
    OpenTelemetry tracing manager.

    Provides distributed tracing for the RAG system.
    """

    def __init__(
        self,
        service_name: str = "enterprise-rag",
        otlp_endpoint: Optional[str] = None,
        enable_console: bool = False,
    ):
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self.enable_console = enable_console
        self._tracer: Optional[trace.Tracer] = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the tracing system."""
        if self._initialized:
            return

        # Create resource
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": "1.0.0",
        })

        # Create provider
        provider = TracerProvider(resource=resource)

        # Add exporters
        if self.otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info("OTLP tracing enabled", endpoint=self.otlp_endpoint)

        if self.enable_console:
            console_exporter = ConsoleSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(console_exporter))

        # Set global provider
        trace.set_tracer_provider(provider)

        self._tracer = trace.get_tracer(self.service_name)
        self._initialized = True

        logger.info("Tracing initialized", service=self.service_name)

    def get_tracer(self) -> trace.Tracer:
        """Get the tracer instance."""
        if not self._initialized:
            self.initialize()
        return self._tracer

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a tracing span.

        Usage:
            with tracing.span("process_query", {"query": query}):
                # do work
        """
        tracer = self.get_tracer()
        with tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value) if not isinstance(value, (int, float, bool)) else value)
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def trace(self, name: Optional[str] = None):
        """
        Decorator to trace a function.

        Usage:
            @tracing.trace("my_function")
            async def my_function():
                ...
        """
        def decorator(func: Callable):
            span_name = name or func.__name__

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.span(span_name) as span:
                    # Add function arguments as attributes
                    for i, arg in enumerate(args):
                        span.set_attribute(f"arg_{i}", str(arg)[:100])
                    for key, value in kwargs.items():
                        span.set_attribute(f"kwarg_{key}", str(value)[:100])

                    result = await func(*args, **kwargs)
                    return result

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.span(span_name) as span:
                    for i, arg in enumerate(args):
                        span.set_attribute(f"arg_{i}", str(arg)[:100])
                    for key, value in kwargs.items():
                        span.set_attribute(f"kwarg_{key}", str(value)[:100])

                    result = func(*args, **kwargs)
                    return result

            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        return decorator


# Global tracing manager
_tracing: Optional[TracingManager] = None


def get_tracer() -> trace.Tracer:
    """Get the global tracer instance."""
    global _tracing
    if _tracing is None:
        settings = get_settings()
        _tracing = TracingManager(
            service_name="enterprise-rag",
            otlp_endpoint=getattr(settings, 'otlp_endpoint', None),
        )
    return _tracing.get_tracer()


@contextmanager
def create_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Create a tracing span.

    Usage:
        with create_span("process_query", {"query": query}):
            # do work
    """
    global _tracing
    if _tracing is None:
        _tracing = TracingManager()

    with _tracing.span(name, attributes) as span:
        yield span


def trace_function(name: Optional[str] = None):
    """
    Decorator to trace a function.

    Usage:
        @trace_function("my_function")
        async def my_function():
            ...
    """
    global _tracing
    if _tracing is None:
        _tracing = TracingManager()

    return _tracing.trace(name)


def add_span_attribute(key: str, value: Any) -> None:
    """Add an attribute to the current span."""
    span = trace.get_current_span()
    if span:
        span.set_attribute(key, str(value) if not isinstance(value, (int, float, bool)) else value)


def add_span_event(name: str, attributes: Optional[dict] = None) -> None:
    """Add an event to the current span."""
    span = trace.get_current_span()
    if span:
        span.add_event(name, attributes or {})


def set_span_error(error: Exception) -> None:
    """Mark the current span as errored."""
    span = trace.get_current_span()
    if span:
        span.set_status(Status(StatusCode.ERROR, str(error)))
        span.record_exception(error)
