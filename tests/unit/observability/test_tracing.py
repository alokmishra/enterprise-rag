"""Tests for OpenTelemetry tracing."""

import pytest
from unittest.mock import MagicMock, patch
import asyncio


class TestTracingManager:
    """Tests for TracingManager class."""

    @pytest.fixture
    def tracing_manager(self):
        """Create TracingManager instance."""
        from src.observability.tracing import TracingManager
        return TracingManager(
            service_name="rag-api-test",
            enable_console=False,
        )

    def test_initialization(self, tracing_manager):
        """Test TracingManager initialization."""
        assert tracing_manager.service_name == "rag-api-test"
        assert tracing_manager._initialized is False

    def test_initialize_tracing(self, tracing_manager):
        """Test initializing tracing provider."""
        tracing_manager.initialize()
        assert tracing_manager._initialized is True

    def test_initialize_is_idempotent(self, tracing_manager):
        """Test that initialize can be called multiple times safely."""
        tracing_manager.initialize()
        tracing_manager.initialize()
        assert tracing_manager._initialized is True

    def test_get_tracer(self, tracing_manager):
        """Test getting tracer instance."""
        tracer = tracing_manager.get_tracer()
        assert tracer is not None

    def test_span_context_manager(self, tracing_manager):
        """Test span as context manager."""
        with tracing_manager.span("test-operation") as span:
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

    def test_nested_spans(self, tracing_manager):
        """Test nested span creation."""
        with tracing_manager.span("parent-operation"):
            with tracing_manager.span("child-operation-1"):
                pass
            with tracing_manager.span("child-operation-2"):
                pass

    def test_span_exception_handling(self, tracing_manager):
        """Test span handles exceptions properly."""
        with pytest.raises(ValueError):
            with tracing_manager.span("failing-operation"):
                raise ValueError("Test error")

    def test_trace_decorator_sync(self, tracing_manager):
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

        result = asyncio.run(async_function(5))
        assert result == 10

    def test_trace_decorator_no_name(self, tracing_manager):
        """Test trace decorator uses function name when no name provided."""
        @tracing_manager.trace()
        def my_function():
            return "result"

        result = my_function()
        assert result == "result"


class TestTracingManagerConfiguration:
    """Tests for TracingManager configuration."""

    def test_with_otlp_endpoint(self):
        """Test TracingManager with OTLP endpoint."""
        from src.observability.tracing import TracingManager
        
        manager = TracingManager(
            service_name="test",
            otlp_endpoint="http://localhost:4317",
        )
        assert manager.otlp_endpoint == "http://localhost:4317"

    def test_with_console_enabled(self):
        """Test TracingManager with console exporter enabled."""
        from src.observability.tracing import TracingManager
        
        manager = TracingManager(
            service_name="test",
            enable_console=True,
        )
        assert manager.enable_console is True


class TestModuleFunctions:
    """Tests for module-level tracing functions."""

    def test_get_tracer(self):
        """Test get_tracer returns a tracer."""
        from src.observability.tracing import get_tracer
        
        tracer = get_tracer()
        assert tracer is not None

    def test_create_span(self):
        """Test create_span context manager."""
        from src.observability.tracing import create_span
        
        with create_span("test-span") as span:
            pass

    def test_create_span_with_attributes(self):
        """Test create_span with attributes."""
        from src.observability.tracing import create_span
        
        with create_span("test-span", {"key": "value"}) as span:
            pass

    def test_trace_function_decorator(self):
        """Test trace_function decorator."""
        from src.observability.tracing import trace_function
        
        @trace_function("my-operation")
        def my_func():
            return 42

        result = my_func()
        assert result == 42

    def test_trace_function_async(self):
        """Test trace_function decorator on async function."""
        from src.observability.tracing import trace_function
        
        @trace_function("async-operation")
        async def async_func():
            return 42

        result = asyncio.run(async_func())
        assert result == 42

    def test_add_span_attribute(self):
        """Test add_span_attribute function."""
        from src.observability.tracing import create_span, add_span_attribute
        
        with create_span("test-span"):
            add_span_attribute("test_key", "test_value")
            add_span_attribute("int_key", 42)
            add_span_attribute("float_key", 3.14)
            add_span_attribute("bool_key", True)

    def test_add_span_event(self):
        """Test add_span_event function."""
        from src.observability.tracing import create_span, add_span_event
        
        with create_span("test-span"):
            add_span_event("checkpoint", {"step": 1})

    def test_set_span_error(self):
        """Test set_span_error function."""
        from src.observability.tracing import create_span, set_span_error
        
        with create_span("test-span"):
            set_span_error(ValueError("test error"))


class TestTracingIntegration:
    """Tests for tracing integration scenarios."""

    @pytest.fixture
    def tracing_manager(self):
        """Create TracingManager instance."""
        from src.observability.tracing import TracingManager
        return TracingManager(service_name="test-integration")

    def test_trace_database_operation(self, tracing_manager):
        """Test tracing database operations."""
        @tracing_manager.trace("db.query")
        async def query_database(query: str):
            return [{"id": 1, "name": "test"}]

        result = asyncio.run(query_database("SELECT * FROM users"))
        assert len(result) == 1

    def test_trace_external_api_call(self, tracing_manager):
        """Test tracing external API calls."""
        @tracing_manager.trace("http.request")
        async def call_external_api(url: str):
            return {"status": "ok"}

        result = asyncio.run(call_external_api("https://api.example.com"))
        assert result["status"] == "ok"

    def test_trace_llm_call(self, tracing_manager):
        """Test tracing LLM API calls."""
        @tracing_manager.trace("llm.generate")
        async def generate_text(prompt: str, model: str = "gpt-4"):
            return "Generated response"

        result = asyncio.run(generate_text("Hello", model="gpt-4"))
        assert result == "Generated response"

    def test_nested_traced_functions(self, tracing_manager):
        """Test nested traced function calls."""
        @tracing_manager.trace("outer")
        def outer_func():
            return inner_func()

        @tracing_manager.trace("inner")
        def inner_func():
            return "inner result"

        result = outer_func()
        assert result == "inner result"
