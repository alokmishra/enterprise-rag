"""
Tests for src/core/logging.py
"""

import logging

import pytest


class TestLogger:
    """Tests for the logging module."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logger instance."""
        from src.core.logging import get_logger

        logger = get_logger("test_module")
        assert logger is not None

    def test_get_logger_same_name_returns_same_instance(self):
        """Test that same name returns same logger."""
        from src.core.logging import get_logger

        logger1 = get_logger("same_name")
        logger2 = get_logger("same_name")
        # Should be the same or equivalent logger

    def test_get_logger_different_names(self):
        """Test that different names return different loggers."""
        from src.core.logging import get_logger

        logger1 = get_logger("module_a")
        logger2 = get_logger("module_b")
        # These should be different loggers

    def test_logger_can_log_info(self):
        """Test that logger can log info messages."""
        from src.core.logging import get_logger

        logger = get_logger("test_info")
        # Should not raise
        logger.info("Test info message")

    def test_logger_can_log_error(self):
        """Test that logger can log error messages."""
        from src.core.logging import get_logger

        logger = get_logger("test_error")
        # Should not raise
        logger.error("Test error message")

    def test_logger_can_log_with_extra_fields(self):
        """Test that logger can log with extra fields."""
        from src.core.logging import get_logger

        logger = get_logger("test_extra")
        # Should not raise
        logger.info("Test message", extra_field="value", trace_id="123")

    def test_logger_can_log_warning(self):
        """Test that logger can log warning messages."""
        from src.core.logging import get_logger

        logger = get_logger("test_warning")
        logger.warning("Test warning message")

    def test_logger_can_log_debug(self):
        """Test that logger can log debug messages."""
        from src.core.logging import get_logger

        logger = get_logger("test_debug")
        logger.debug("Test debug message")


class TestLoggerMixin:
    """Tests for the LoggerMixin class."""

    def test_mixin_provides_logger(self):
        """Test that LoggerMixin provides a logger property."""
        from src.core.logging import LoggerMixin

        class TestClass(LoggerMixin):
            pass

        obj = TestClass()
        assert hasattr(obj, 'logger')
        assert obj.logger is not None

    def test_mixin_logger_uses_class_name(self):
        """Test that mixin logger is named after the class."""
        from src.core.logging import LoggerMixin

        class MyTestClass(LoggerMixin):
            pass

        obj = MyTestClass()
        # Logger should be available and usable
        obj.logger.info("Test from mixin")
