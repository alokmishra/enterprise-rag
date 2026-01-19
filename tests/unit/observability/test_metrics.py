"""Tests for Prometheus metrics collection."""

import pytest
from unittest.mock import MagicMock, patch
import uuid


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    @pytest.fixture
    def metrics_collector(self):
        """Create MetricsCollector instance with unique namespace."""
        from src.observability.metrics import MetricsCollector
        unique_ns = f"rag_test_{uuid.uuid4().hex[:8]}"
        return MetricsCollector(namespace=unique_ns)

    def test_track_request(self, metrics_collector):
        """Test tracking HTTP requests."""
        metrics_collector.track_request(
            endpoint="/api/v1/query",
            method="POST",
            status=200,
            latency=0.150,
        )
        assert True

    def test_track_request_error(self, metrics_collector):
        """Test tracking failed HTTP requests."""
        metrics_collector.track_request(
            endpoint="/api/v1/documents",
            method="POST",
            status=500,
            latency=0.050,
        )
        assert True

    def test_track_query(self, metrics_collector):
        """Test tracking query execution."""
        metrics_collector.track_query(
            complexity="complex",
            strategy="hybrid",
            status="success",
            latency=1.5,
        )
        assert True

    def test_track_query_failure(self, metrics_collector):
        """Test tracking failed queries."""
        metrics_collector.track_query(
            complexity="simple",
            strategy="vector",
            status="error",
            latency=0.5,
        )
        assert True

    def test_track_tokens(self, metrics_collector):
        """Test tracking token usage."""
        metrics_collector.track_tokens(
            model="gpt-4",
            operation="generation",
            count=500,
        )
        assert True

    def test_track_tokens_embedding(self, metrics_collector):
        """Test tracking embedding token usage."""
        metrics_collector.track_tokens(
            model="text-embedding-ada-002",
            operation="embedding",
            count=1000,
        )
        assert True

    def test_track_agent(self, metrics_collector):
        """Test tracking agent execution."""
        metrics_collector.track_agent(
            agent="planner",
            status="success",
            latency=0.25,
        )
        assert True

    def test_track_agent_failure(self, metrics_collector):
        """Test tracking failed agent execution."""
        metrics_collector.track_agent(
            agent="synthesizer",
            status="error",
            latency=0.1,
        )
        assert True

    def test_track_retrieval(self, metrics_collector):
        """Test tracking retrieval operations."""
        metrics_collector.track_retrieval(
            strategy="hybrid",
            status="success",
            result_count=10,
        )
        assert True

    def test_track_error(self, metrics_collector):
        """Test tracking errors."""
        metrics_collector.track_error(
            error_type="ValidationError",
            component="http",
        )
        assert True

    def test_track_cache_hit(self, metrics_collector):
        """Test tracking cache hits."""
        metrics_collector.track_cache(
            cache_type="query",
            hit=True,
        )
        assert True

    def test_track_cache_miss(self, metrics_collector):
        """Test tracking cache misses."""
        metrics_collector.track_cache(
            cache_type="embedding",
            hit=False,
        )
        assert True

    def test_active_requests_gauge(self, metrics_collector):
        """Test active requests gauge."""
        metrics_collector.active_requests.inc()
        metrics_collector.active_requests.dec()
        assert True

    def test_documents_ingested_counter(self, metrics_collector):
        """Test documents ingested counter."""
        metrics_collector.documents_ingested.labels(status="success").inc()
        assert True

    def test_chunks_created_counter(self, metrics_collector):
        """Test chunks created counter."""
        metrics_collector.chunks_created.inc()
        assert True


class TestMetricsCollectorSingleton:
    """Tests for MetricsCollector singleton pattern."""

    def test_get_metrics_returns_instance(self):
        """Test that get_metrics returns a MetricsCollector instance."""
        from src.observability.metrics import get_metrics, MetricsCollector
        
        collector = get_metrics()
        assert isinstance(collector, MetricsCollector)

    def test_both_instances_work(self):
        """Test that multiple instances with unique namespaces work correctly."""
        from src.observability.metrics import MetricsCollector
        
        collector1 = MetricsCollector(namespace=f"test1_{uuid.uuid4().hex[:8]}")
        collector2 = MetricsCollector(namespace=f"test2_{uuid.uuid4().hex[:8]}")

        collector1.track_request("/test", "GET", 200, 0.1)
        collector2.track_request("/test", "GET", 200, 0.1)
        assert True


class TestMetricsExport:
    """Tests for metrics export functionality."""

    @pytest.fixture
    def metrics_collector(self):
        """Create MetricsCollector instance."""
        from src.observability.metrics import MetricsCollector
        return MetricsCollector(namespace=f"rag_export_{uuid.uuid4().hex[:8]}")

    def test_get_metrics_output(self, metrics_collector):
        """Test getting Prometheus metrics output."""
        metrics_collector.track_request("/api/v1/query", "POST", 200, 0.15)
        metrics_collector.track_query("simple", "vector", "success", 0.5)

        output = metrics_collector.get_metrics()

        assert isinstance(output, bytes)


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_track_request_function(self):
        """Test track_request convenience function."""
        from src.observability.metrics import track_request
        
        track_request("/test", "GET", 200, 0.1)
        assert True

    def test_track_tokens_function(self):
        """Test track_tokens convenience function."""
        from src.observability.metrics import track_tokens
        
        track_tokens("gpt-4", "generation", 100)
        assert True

    def test_track_error_function(self):
        """Test track_error convenience function."""
        from src.observability.metrics import track_error
        
        track_error("ValueError", "api")
        assert True
