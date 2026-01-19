"""Tests for audit logging."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.auth.audit import (
    AuditAction,
    AuditLogger,
    AuditLogEntry,
)


class TestAuditAction:
    """Tests for AuditAction enum."""

    def test_audit_action_values(self):
        """Test that all expected audit actions exist."""
        assert AuditAction.LOGIN
        assert AuditAction.LOGOUT
        assert AuditAction.LOGIN_FAILED
        assert AuditAction.DOCUMENT_CREATED
        assert AuditAction.DOCUMENT_UPDATED
        assert AuditAction.DOCUMENT_DELETED
        assert AuditAction.DOCUMENT_ACCESSED
        assert AuditAction.QUERY_EXECUTED
        assert AuditAction.API_KEY_CREATED
        assert AuditAction.API_KEY_REVOKED
        assert AuditAction.PERMISSION_DENIED
        assert AuditAction.SETTINGS_CHANGED


class TestAuditLogEntry:
    """Tests for AuditLogEntry model."""

    def test_create_audit_log_entry(self):
        """Test creating an audit log entry."""
        entry = AuditLogEntry(
            id=str(uuid4()),
            action=AuditAction.QUERY_EXECUTED,
            actor_id=str(uuid4()),
            tenant_id=str(uuid4()),
            resource_type="query",
            resource_id=str(uuid4()),
            details={"query": "What is our policy?"},
            ip_address="192.168.1.1",
            user_agent="TestClient/1.0",
            timestamp=datetime.utcnow(),
        )

        assert entry.action == AuditAction.QUERY_EXECUTED
        assert entry.resource_type == "query"
        assert "query" in entry.details

    def test_audit_log_entry_optional_fields(self):
        """Test audit log entry with optional fields."""
        entry = AuditLogEntry(
            id=str(uuid4()),
            action=AuditAction.LOGIN,
            actor_id=str(uuid4()),
            tenant_id=str(uuid4()),
            timestamp=datetime.utcnow(),
        )

        assert entry.resource_type is None
        assert entry.resource_id is None
        assert entry.details == {}


class TestAuditLogger:
    """Tests for AuditLogger class."""

    @pytest.fixture
    def mock_repository(self):
        """Create mock repository."""
        repo = AsyncMock()
        repo.create = AsyncMock()
        repo.create_batch = AsyncMock()
        repo.query = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def audit_logger(self, mock_repository):
        """Create AuditLogger instance."""
        return AuditLogger(repository=mock_repository)

    @pytest.fixture
    def audit_logger_no_repo(self):
        """Create AuditLogger instance without repository."""
        return AuditLogger()

    @pytest.fixture
    def sample_actor(self):
        """Sample actor data."""
        return {
            "actor_id": uuid4(),
            "tenant_id": uuid4(),
            "email": "test@example.com",
        }

    @pytest.mark.asyncio
    async def test_log_action(self, audit_logger, mock_repository, sample_actor):
        """Test logging an action."""
        entry = await audit_logger.log(
            action=AuditAction.DOCUMENT_CREATED,
            actor_id=sample_actor["actor_id"],
            tenant_id=sample_actor["tenant_id"],
            resource_type="document",
            resource_id=str(uuid4()),
            details={"filename": "report.pdf"},
        )

        assert entry is not None
        assert entry.action == AuditAction.DOCUMENT_CREATED
        mock_repository.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_login(self, audit_logger, sample_actor):
        """Test logging a login event."""
        entry = await audit_logger.log_login(
            user_id=sample_actor["actor_id"],
            email=sample_actor["email"],
            tenant_id=sample_actor["tenant_id"],
            ip_address="10.0.0.1",
            success=True,
        )

        assert entry.action == AuditAction.LOGIN

    @pytest.mark.asyncio
    async def test_log_failed_login(self, audit_logger, sample_actor):
        """Test logging a failed login event."""
        entry = await audit_logger.log_login(
            user_id=sample_actor["actor_id"],
            email=sample_actor["email"],
            tenant_id=sample_actor["tenant_id"],
            ip_address="10.0.0.1",
            success=False,
            error="Invalid credentials",
        )

        assert entry.action == AuditAction.LOGIN_FAILED

    @pytest.mark.asyncio
    async def test_log_query(self, audit_logger, sample_actor):
        """Test logging a query execution."""
        entry = await audit_logger.log_query(
            user_id=sample_actor["actor_id"],
            tenant_id=sample_actor["tenant_id"],
            query="What are the compliance requirements?",
            latency_ms=150,
        )

        assert entry.action == AuditAction.QUERY_EXECUTED
        assert entry.resource_type == "query"
        assert "query" in entry.details
        assert entry.details["latency_ms"] == 150

    @pytest.mark.asyncio
    async def test_log_document_access(self, audit_logger, sample_actor):
        """Test logging document access."""
        doc_id = str(uuid4())
        entry = await audit_logger.log_document_access(
            user_id=sample_actor["actor_id"],
            tenant_id=sample_actor["tenant_id"],
            document_id=doc_id,
        )

        assert entry.action == AuditAction.DOCUMENT_ACCESSED
        assert entry.resource_id == doc_id

    @pytest.mark.asyncio
    async def test_log_with_ip_and_user_agent(self, audit_logger, sample_actor):
        """Test logging with IP address and user agent."""
        entry = await audit_logger.log(
            action=AuditAction.SETTINGS_CHANGED,
            actor_id=sample_actor["actor_id"],
            tenant_id=sample_actor["tenant_id"],
            ip_address="192.168.1.100",
            user_agent="CustomApp/2.0",
            details={"setting": "notifications", "old_value": True, "new_value": False},
        )

        assert entry.ip_address == "192.168.1.100"
        assert entry.user_agent == "CustomApp/2.0"

    @pytest.mark.asyncio
    async def test_query_logs(self, audit_logger, mock_repository, sample_actor):
        """Test querying audit logs."""
        mock_entries = [
            AuditLogEntry(action=AuditAction.LOGIN, actor_id=sample_actor["actor_id"]),
            AuditLogEntry(action=AuditAction.QUERY_EXECUTED, actor_id=sample_actor["actor_id"]),
        ]
        mock_repository.query.return_value = mock_entries

        logs = await audit_logger.query(
            tenant_id=sample_actor["tenant_id"],
            limit=10,
        )

        assert len(logs) == 2

    @pytest.mark.asyncio
    async def test_query_logs_by_action(self, audit_logger, mock_repository, sample_actor):
        """Test querying audit logs filtered by action."""
        mock_entries = [
            AuditLogEntry(action=AuditAction.QUERY_EXECUTED, actor_id=sample_actor["actor_id"]),
        ]
        mock_repository.query.return_value = mock_entries

        logs = await audit_logger.query(
            tenant_id=sample_actor["tenant_id"],
            action=AuditAction.QUERY_EXECUTED,
            limit=100,
        )

        assert len(logs) == 1
        assert logs[0].action == AuditAction.QUERY_EXECUTED

    @pytest.mark.asyncio
    async def test_query_logs_by_actor(self, audit_logger, mock_repository, sample_actor):
        """Test querying audit logs filtered by actor."""
        mock_repository.query.return_value = []

        logs = await audit_logger.query(
            tenant_id=sample_actor["tenant_id"],
            actor_id=sample_actor["actor_id"],
        )

        assert logs == []

    @pytest.mark.asyncio
    async def test_query_logs_date_range(self, audit_logger, mock_repository, sample_actor):
        """Test querying audit logs with date range."""
        mock_repository.query.return_value = []

        from datetime import timedelta
        start = datetime.utcnow() - timedelta(days=7)
        end = datetime.utcnow()

        logs = await audit_logger.query(
            tenant_id=sample_actor["tenant_id"],
            start_time=start,
            end_time=end,
        )

        mock_repository.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_buffer_flush(self, audit_logger_no_repo):
        """Test that entries are buffered when no repository."""
        await audit_logger_no_repo.log(
            action=AuditAction.DOCUMENT_ACCESSED,
        )

        assert len(audit_logger_no_repo._buffer) == 1
