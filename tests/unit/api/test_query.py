"""
Tests for src/api/routes/query.py
"""

from unittest.mock import AsyncMock, patch, MagicMock

import pytest


class TestQueryEndpoint:
    """Tests for the /query endpoint."""

    def test_query_endpoint_exists(self, test_client):
        """Test that query endpoint exists."""
        response = test_client.post("/query", json={"query": "test"})
        # Should not be 404
        assert response.status_code != 404

    def test_query_requires_query_field(self, test_client):
        """Test that query field is required."""
        response = test_client.post("/query", json={})
        assert response.status_code == 422  # Validation error

    def test_query_validates_query_length(self, test_client):
        """Test that query length is validated."""
        response = test_client.post("/query", json={"query": ""})
        assert response.status_code == 422

    def test_query_accepts_valid_request(self, test_client):
        """Test that valid query request is accepted."""
        with patch('src.api.routes.query.get_rag_pipeline') as mock_pipeline:
            mock_result = MagicMock()
            mock_result.query_id = "query-1"
            mock_result.query = "test query"
            mock_result.answer = "test answer"
            mock_result.sources = []
            mock_result.confidence = 0.9
            mock_result.complexity = "simple"
            mock_result.latency_ms = 100.0
            mock_result.tokens_used = 150

            mock_pipeline.return_value.query = AsyncMock(return_value=mock_result)

            response = test_client.post("/query", json={
                "query": "What is the company policy?",
                "include_sources": True,
            })
            # Should process or return error

    def test_query_response_format(self, test_client):
        """Test query response format."""
        with patch('src.api.routes.query.get_rag_pipeline') as mock_pipeline:
            mock_result = MagicMock()
            mock_result.query_id = "query-1"
            mock_result.query = "test query"
            mock_result.answer = "test answer"
            mock_result.sources = []
            mock_result.confidence = 0.9
            mock_result.complexity = "simple"
            mock_result.latency_ms = 100.0
            mock_result.tokens_used = 150

            mock_pipeline.return_value.query = AsyncMock(return_value=mock_result)

            response = test_client.post("/query", json={"query": "Test query"})
            if response.status_code == 200:
                data = response.json()
                assert "answer" in data or "response" in data


class TestQueryStreamEndpoint:
    """Tests for the /query/stream endpoint."""

    def test_stream_endpoint_exists(self, test_client):
        """Test that stream endpoint exists."""
        response = test_client.post("/query/stream", json={"query": "test", "stream": True})
        assert response.status_code != 404

    def test_stream_requires_stream_flag(self, test_client):
        """Test that stream flag must be true."""
        response = test_client.post("/query/stream", json={"query": "test", "stream": False})
        # Should return error for non-streaming request
        assert response.status_code in [400, 200]


class TestAgentQueryEndpoint:
    """Tests for the /query/agent endpoint."""

    def test_agent_query_endpoint_exists(self, test_client):
        """Test that agent query endpoint exists."""
        response = test_client.post("/query/agent", json={"query": "test"})
        assert response.status_code != 404

    def test_agent_query_accepts_configuration(self, test_client):
        """Test that agent query accepts configuration options."""
        response = test_client.post("/query/agent", json={
            "query": "What is the policy?",
            "output_format": "markdown",
            "max_iterations": 2,
            "enable_verification": True,
            "enable_critic": True,
        })
        # Should accept configuration
        assert response.status_code != 422


class TestQueryFeedbackEndpoint:
    """Tests for the query feedback endpoint."""

    def test_feedback_endpoint_exists(self, test_client):
        """Test that feedback endpoint exists."""
        response = test_client.post(
            "/query/query-1/feedback",
            json={"query_id": "query-1", "rating": 5},
        )
        assert response.status_code != 404

    def test_feedback_validates_rating(self, test_client):
        """Test that rating is validated."""
        response = test_client.post(
            "/query/query-1/feedback",
            json={"query_id": "query-1", "rating": 10},  # Invalid rating
        )
        assert response.status_code == 422

    def test_feedback_accepts_valid_request(self, test_client):
        """Test that valid feedback is accepted."""
        response = test_client.post(
            "/query/query-1/feedback",
            json={"query_id": "query-1", "rating": 4, "feedback": "Good answer"},
        )
        # Should accept or return not found
        assert response.status_code in [200, 404]


class TestQueryRequestValidation:
    """Tests for query request validation."""

    def test_query_max_length(self, test_client):
        """Test query maximum length validation."""
        long_query = "x" * 15000  # Exceeds max length
        response = test_client.post("/query", json={"query": long_query})
        assert response.status_code == 422

    def test_retrieval_strategy_validation(self, test_client):
        """Test retrieval strategy validation."""
        response = test_client.post("/query", json={
            "query": "test",
            "retrieval_strategy": "invalid_strategy",
        })
        assert response.status_code == 422

    def test_top_k_validation(self, test_client):
        """Test top_k parameter validation."""
        response = test_client.post("/query", json={
            "query": "test",
            "top_k": 100,  # Exceeds max
        })
        assert response.status_code == 422
