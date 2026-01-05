"""
Tests for src/core/exceptions.py
"""

import pytest


class TestRAGException:
    """Tests for the base RAGException."""

    def test_rag_exception_creation(self):
        """Test that RAGException can be created."""
        from src.core.exceptions import RAGException

        exc = RAGException("Test error")
        assert str(exc) == "Test error"

    def test_rag_exception_is_exception(self):
        """Test that RAGException is an Exception."""
        from src.core.exceptions import RAGException

        exc = RAGException("Test")
        assert isinstance(exc, Exception)

    def test_rag_exception_can_be_raised(self):
        """Test that RAGException can be raised and caught."""
        from src.core.exceptions import RAGException

        with pytest.raises(RAGException) as exc_info:
            raise RAGException("Test error message")

        assert "Test error message" in str(exc_info.value)


class TestSpecificExceptions:
    """Tests for specific exception types."""

    def test_retrieval_error(self):
        """Test RetrievalError exception."""
        from src.core.exceptions import RetrievalError

        exc = RetrievalError("Failed to retrieve documents")
        assert isinstance(exc, Exception)
        assert "retrieve" in str(exc).lower()

    def test_generation_error(self):
        """Test GenerationError exception."""
        from src.core.exceptions import GenerationError

        exc = GenerationError("LLM generation failed")
        assert isinstance(exc, Exception)

    def test_ingestion_error(self):
        """Test IngestionError exception."""
        from src.core.exceptions import IngestionError

        exc = IngestionError("Document parsing failed")
        assert isinstance(exc, Exception)

    def test_validation_error(self):
        """Test ValidationError exception."""
        from src.core.exceptions import ValidationError

        exc = ValidationError("Invalid input")
        assert isinstance(exc, Exception)

    def test_storage_error(self):
        """Test StorageError exception."""
        from src.core.exceptions import StorageError

        exc = StorageError("Database connection failed")
        assert isinstance(exc, Exception)

    def test_configuration_error(self):
        """Test ConfigurationError exception."""
        from src.core.exceptions import ConfigurationError

        exc = ConfigurationError("Missing required config")
        assert isinstance(exc, Exception)


class TestExceptionHierarchy:
    """Tests for exception inheritance hierarchy."""

    def test_specific_exceptions_inherit_from_rag_exception(self):
        """Test that specific exceptions inherit from RAGException."""
        from src.core.exceptions import (
            RAGException,
            RetrievalError,
            GenerationError,
            IngestionError,
        )

        assert issubclass(RetrievalError, RAGException)
        assert issubclass(GenerationError, RAGException)
        assert issubclass(IngestionError, RAGException)

    def test_catch_rag_exception_catches_specific(self):
        """Test that catching RAGException catches specific exceptions."""
        from src.core.exceptions import RAGException, RetrievalError

        with pytest.raises(RAGException):
            raise RetrievalError("Test")
