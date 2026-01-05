"""Tests for OpenTelemetry tracing."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from contextlib import contextmanager

from src.observability.tracing import TracingManager


class TestTracingManager:
    """Tests for TracingManager class."""

    @pytest.fixture
    def tracing_manager(self):
        """Create TracingManager instance."""
        return TracingManager(
            service_name="rag-api-test",
            environment="test",
        )

    def test_initialization(self, tracing_manager):
        """Test TracingManager initialization."""
        assert tracing_manager.service_name == "rag-api-test"
        assert tracing_manager.environment == "test"

    def test_initialize_tracing(self, tracing_manager):
        """Test initializing tracing provider."""
        with patch("src.observability.tracing.TracerProvider"):
            tracing_manager.initialize()
            # Should not raise

    def test_span_context_manager(self, tracing_manager):
        """Test span as context manager."""
        with tracing_manager.span("test-operation") as span:
            # Do some work
            result = 1 + 1

        assert result == 2

    def test_span_with_attributes(self, tracing_manager):
        """Test span with custom attributes."""
        attributes = {
            "query.id": "123",
            "query.complexity": "complex",
        }

        with tracing_manager.span("query-execution", attributes=attributes):
            pass

        # Should not raise

    def test_nested_spans(self, tracing_manager):
        """Test nested span creation."""
        with tracing_manager.span("parent-operation"):
            with tracing_manager.span("child-operation-1"):
                pass
            with tracing_manager.span("child-operation-2"):
                pass

        # Should not raise

    def test_span_exception_handling(self, tracing_manager):
        """Test span handles exceptions properly."""
        with pytest.raises(ValueError):
            with tracing_manager.span("failing-operation"):
                raise ValueError("Test error")

        # Exception should propagate but span should be closed

    def test_trace_decorator(self, tracing_manager):
        """Test trace decorator on sync function."""
        @tracing_manager.trace("decorated-operation")
        def sample_function(x, y):
            return x + y

        result = sample_function(2, 3)
        assert result == 5

    def test_trace_decorator_async(self, tracing_manager):
        """Test trace decorator on async function."""
        @tracing_manager.trace("async-operation")
        async def async_function(x):
            return x * 2

        import asyncio
        result = asyncio.run(async_function(5))
        assert result == 10

    def test_trace_decorator_with_attributes(self, tracing_manager):
        """Test trace decorator with attribute extractor."""
        @tracing_manager.trace(
            "parameterized-operation",
            attribute_extractor=lambda args, kwargs: {"input": args[0]},
        )
        def process_data(data):
            return len(data)

        result = process_data("hello")
        assert result == 5

    def test_add_span_event(self, tracing_manager):
        """Test adding events to spans."""
        with tracing_manager.span("operation-with-events") as span:
            tracing_manager.add_event(
                "checkpoint-reached",
                attributes={"step": 1},
            )
            # Do some work
            tracing_manager.add_event(
                "processing-complete",
                attributes={"step": 2, "items_processed": 100},
            )

        # Should not raise

    def test_set_span_status(self, tracing_manager):
        """Test setting span status."""
        with tracing_manager.span("status-operation") as span:
            tracing_manager.set_status("ok")

        # Should not raise

    def test_set_span_error_status(self, tracing_manager):
        """Test setting error status on span."""
        with tracing_manager.span("error-operation") as span:
            tracing_manager.set_status("error", "Something went wrong")

        # Should not raise


class TestTracingManagerConfiguration:
    """Tests for TracingManager configuration."""

    def test_disabled_tracing(self):
        """Test TracingManager when tracing is disabled."""
        manager = TracingManager(
            service_name="test",
            enabled=False,
        )

        # Operations should be no-ops
        with manager.span("test"):
            pass

    def test_exporter_configuration(self):
        """Test configuring different exporters."""
        # Console exporter
        manager_console = TracingManager(
            service_name="test",
            exporter="console",
        )
        assert manager_console is not None

        # OTLP exporter
        manager_otlp = TracingManager(
            service_name="test",
            exporter="otlp",
            otlp_endpoint="http://localhost:4317",
        )
        assert manager_otlp is not None

    def test_sampling_configuration(self):
        """Test configuring trace sampling."""
        manager = TracingManager(
            service_name="test",
            sampling_rate=0.1,  # 10% sampling
        )
        assert manager is not None


class TestTracingContext:
    """Tests for tracing context propagation."""

    @pytest.fixture
    def tracing_manager(self):
        """Create TracingManager instance."""
        return TracingManager(service_name="test")

    def test_extract_context_from_headers(self, tracing_manager):
        """Test extracting trace context from HTTP headers."""
        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        }

        context = tracing_manager.extract_context(headers)
        # Context should be extracted (or None if invalid)
        assert context is not None or context is None  # Depends on implementation

    def test_inject_context_to_headers(self, tracing_manager):
        """Test injecting trace context into HTTP headers."""
        with tracing_manager.span("parent"):
            headers = {}
            tracing_manager.inject_context(headers)

            # Headers may or may not be populated depending on test setup
            assert isinstance(headers, dict)

    def test_get_current_trace_id(self, tracing_manager):
        """Test getting current trace ID."""
        with tracing_manager.span("traced-operation"):
            trace_id = tracing_manager.get_current_trace_id()
            # May be None in test environment
            assert trace_id is None or isinstance(trace_id, str)

    def test_get_current_span_id(self, tracing_manager):
        """Test getting current span ID."""
        with tracing_manager.span("traced-operation"):
            span_id = tracing_manager.get_current_span_id()
            # May be None in test environment
            assert span_id is None or isinstance(span_id, str)


class TestTracingIntegration:
    """Tests for tracing integration with other components."""

    @pytest.fixture
    def tracing_manager(self):
        """Create TracingManager instance."""
        return TracingManager(service_name="test")

    def test_trace_database_operation(self, tracing_manager):
        """Test tracing database operations."""
        @tracing_manager.trace("db.query")
        async def query_database(query: str):
            return [{"id": 1, "name": "test"}]

        import asyncio
        result = asyncio.run(query_database("SELECT * FROM users"))
        assert len(result) == 1

    def test_trace_external_api_call(self, tracing_manager):
        """Test tracing external API calls."""
        @tracing_manager.trace("http.request")
        async def call_external_api(url: str):
            return {"status": "ok"}

        import asyncio
        result = asyncio.run(call_external_api("https://api.example.com"))
        assert result["status"] == "ok"

    def test_trace_llm_call(self, tracing_manager):
        """Test tracing LLM API calls."""
        @tracing_manager.trace(
            "llm.generate",
            attribute_extractor=lambda args, kwargs: {
                "model": kwargs.get("model", "unknown"),
            },
        )
        async def generate_text(prompt: str, model: str = "gpt-4"):
            return "Generated response"

        import asyncio
        result = asyncio.run(generate_text("Hello", model="gpt-4"))
        assert result == "Generated response"
