"""
Enterprise RAG System - Audit Logging
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from sqlalchemy import Column, String, DateTime, Text, Index
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB

from src.core.logging import get_logger
from src.storage.document.database import Base


logger = get_logger(__name__)


class AuditAction(str, Enum):
    """Audit action types."""
    # Authentication
    LOGIN = "auth.login"
    LOGOUT = "auth.logout"
    LOGIN_FAILED = "auth.login_failed"
    TOKEN_REFRESH = "auth.token_refresh"
    API_KEY_CREATED = "auth.api_key_created"
    API_KEY_REVOKED = "auth.api_key_revoked"

    # Documents
    DOCUMENT_CREATED = "document.created"
    DOCUMENT_UPDATED = "document.updated"
    DOCUMENT_DELETED = "document.deleted"
    DOCUMENT_ACCESSED = "document.accessed"

    # Queries
    QUERY_EXECUTED = "query.executed"
    QUERY_FAILED = "query.failed"

    # Users
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    USER_ROLE_CHANGED = "user.role_changed"

    # Admin
    SETTINGS_CHANGED = "admin.settings_changed"
    TENANT_CREATED = "tenant.created"
    TENANT_UPDATED = "tenant.updated"
    TENANT_DELETED = "tenant.deleted"

    # Security
    PERMISSION_DENIED = "security.permission_denied"
    SUSPICIOUS_ACTIVITY = "security.suspicious_activity"


class AuditLogEntry(BaseModel):
    """Audit log entry model."""
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    action: AuditAction
    actor_id: Optional[UUID] = None
    actor_email: Optional[str] = None
    actor_type: str = "user"  # user, api_key, system
    tenant_id: Optional[UUID] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    status: str = "success"  # success, failure
    error_message: Optional[str] = None


class AuditLogModel(Base):
    """SQLAlchemy model for audit logs."""
    __tablename__ = "audit_logs"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    action = Column(String(100), nullable=False, index=True)
    actor_id = Column(PGUUID(as_uuid=True), nullable=True, index=True)
    actor_email = Column(String(255), nullable=True)
    actor_type = Column(String(50), default="user")
    tenant_id = Column(PGUUID(as_uuid=True), nullable=True, index=True)
    resource_type = Column(String(100), nullable=True, index=True)
    resource_id = Column(String(255), nullable=True)
    details = Column(JSONB, default={})
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    status = Column(String(20), default="success")
    error_message = Column(Text, nullable=True)

    __table_args__ = (
        Index('ix_audit_logs_tenant_timestamp', 'tenant_id', 'timestamp'),
        Index('ix_audit_logs_actor_timestamp', 'actor_id', 'timestamp'),
    )


class AuditLogger:
    """
    Audit logging service.

    Records security-relevant events for compliance and debugging.
    """

    def __init__(self, repository=None):
        """
        Initialize the audit logger.

        Args:
            repository: Optional repository for persistent storage
        """
        self.repository = repository
        self._buffer: list[AuditLogEntry] = []
        self._buffer_size = 100

    async def log(
        self,
        action: AuditAction,
        actor_id: Optional[UUID] = None,
        actor_email: Optional[str] = None,
        actor_type: str = "user",
        tenant_id: Optional[UUID] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        status: str = "success",
        error_message: Optional[str] = None,
    ) -> AuditLogEntry:
        """
        Log an audit event.

        Args:
            action: The action being logged
            actor_id: ID of the user/key performing the action
            actor_email: Email of the actor
            actor_type: Type of actor (user, api_key, system)
            tenant_id: Tenant context
            resource_type: Type of resource affected
            resource_id: ID of the resource affected
            details: Additional details about the event
            ip_address: Client IP address
            user_agent: Client user agent
            status: success or failure
            error_message: Error message if failed

        Returns:
            The created audit log entry
        """
        entry = AuditLogEntry(
            action=action,
            actor_id=actor_id,
            actor_email=actor_email,
            actor_type=actor_type,
            tenant_id=tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            status=status,
            error_message=error_message,
        )

        # Log to structured logger
        logger.info(
            "Audit event",
            action=action.value,
            actor_id=str(actor_id) if actor_id else None,
            tenant_id=str(tenant_id) if tenant_id else None,
            resource_type=resource_type,
            resource_id=resource_id,
            status=status,
        )

        # Persist if repository available
        if self.repository:
            await self.repository.create(entry)
        else:
            # Buffer for batch writing
            self._buffer.append(entry)
            if len(self._buffer) >= self._buffer_size:
                await self._flush_buffer()

        return entry

    async def _flush_buffer(self) -> None:
        """Flush buffered entries to storage."""
        if self.repository and self._buffer:
            await self.repository.create_batch(self._buffer)
            self._buffer.clear()

    async def query(
        self,
        tenant_id: Optional[UUID] = None,
        actor_id: Optional[UUID] = None,
        action: Optional[AuditAction] = None,
        resource_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditLogEntry]:
        """
        Query audit logs.

        Args:
            tenant_id: Filter by tenant
            actor_id: Filter by actor
            action: Filter by action type
            resource_type: Filter by resource type
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of matching audit log entries
        """
        if self.repository:
            return await self.repository.query(
                tenant_id=tenant_id,
                actor_id=actor_id,
                action=action,
                resource_type=resource_type,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
                offset=offset,
            )
        return []

    # Convenience methods for common actions

    async def log_login(
        self,
        user_id: UUID,
        email: str,
        tenant_id: UUID,
        ip_address: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> AuditLogEntry:
        """Log a login attempt."""
        return await self.log(
            action=AuditAction.LOGIN if success else AuditAction.LOGIN_FAILED,
            actor_id=user_id,
            actor_email=email,
            tenant_id=tenant_id,
            ip_address=ip_address,
            status="success" if success else "failure",
            error_message=error,
        )

    async def log_query(
        self,
        user_id: UUID,
        tenant_id: UUID,
        query: str,
        latency_ms: float,
        success: bool = True,
        error: Optional[str] = None,
    ) -> AuditLogEntry:
        """Log a query execution."""
        return await self.log(
            action=AuditAction.QUERY_EXECUTED if success else AuditAction.QUERY_FAILED,
            actor_id=user_id,
            tenant_id=tenant_id,
            resource_type="query",
            details={
                "query": query[:500],  # Truncate long queries
                "latency_ms": latency_ms,
            },
            status="success" if success else "failure",
            error_message=error,
        )

    async def log_document_access(
        self,
        user_id: UUID,
        tenant_id: UUID,
        document_id: str,
        action: AuditAction = AuditAction.DOCUMENT_ACCESSED,
    ) -> AuditLogEntry:
        """Log document access."""
        return await self.log(
            action=action,
            actor_id=user_id,
            tenant_id=tenant_id,
            resource_type="document",
            resource_id=document_id,
        )


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get the audit logger singleton."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
