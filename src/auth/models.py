"""
Enterprise RAG System - Auth Models
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, EmailStr
from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID as PGUUID, ARRAY
from sqlalchemy.orm import relationship

from src.storage.document.database import Base


# =============================================================================
# Pydantic Models
# =============================================================================

class UserStatus(str, Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class User(BaseModel):
    """User model."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4)
    email: EmailStr
    name: str
    tenant_id: UUID
    role: str = "user"
    permissions: list[str] = Field(default_factory=list)
    status: UserStatus = UserStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None


class Tenant(BaseModel):
    """Tenant model for multi-tenancy."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4)
    name: str
    slug: str
    settings: dict = Field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Resource limits
    max_documents: int = 10000
    max_queries_per_day: int = 1000
    max_storage_gb: float = 10.0


class APIKey(BaseModel):
    """API Key model."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4)
    name: str
    key_hash: str
    key_prefix: str  # First 8 chars for identification
    tenant_id: UUID
    user_id: Optional[UUID] = None
    permissions: list[str] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None


# =============================================================================
# SQLAlchemy Models
# =============================================================================

class TenantModel(Base):
    """SQLAlchemy model for tenants."""
    __tablename__ = "tenants"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    settings = Column(Text, default="{}")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    # Resource limits
    max_documents = Column(String, default="10000")
    max_queries_per_day = Column(String, default="1000")
    max_storage_gb = Column(String, default="10.0")

    # Relationships
    users = relationship("UserModel", back_populates="tenant")
    api_keys = relationship("APIKeyModel", back_populates="tenant")


class UserModel(Base):
    """SQLAlchemy model for users."""
    __tablename__ = "users"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=True)
    tenant_id = Column(PGUUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    role = Column(String(50), default="user")
    permissions = Column(ARRAY(String), default=[])
    status = Column(SQLEnum(UserStatus), default=UserStatus.ACTIVE)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    tenant = relationship("TenantModel", back_populates="users")
    api_keys = relationship("APIKeyModel", back_populates="user")


class APIKeyModel(Base):
    """SQLAlchemy model for API keys."""
    __tablename__ = "api_keys"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    key_hash = Column(String(255), nullable=False, unique=True)
    key_prefix = Column(String(16), nullable=False, index=True)
    tenant_id = Column(PGUUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    permissions = Column(ARRAY(String), default=[])
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)

    # Relationships
    tenant = relationship("TenantModel", back_populates="api_keys")
    user = relationship("UserModel", back_populates="api_keys")
