"""
Enterprise RAG System - Database Connection Management
"""

import asyncio
from typing import Any, AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.config import settings
from src.core.exceptions import RAGException
from src.core.logging import LoggerMixin
from src.storage.document.models import Base


class DatabaseError(RAGException):
    """Database-related errors."""

    def __init__(self, message: str):
        super().__init__(message, code="DATABASE_ERROR")


class Database(LoggerMixin):
    """Async database connection manager."""

    def __init__(
        self,
        url: Optional[str] = None,
        pool_size: Optional[int] = None,
        max_overflow: Optional[int] = None,
    ):
        self._url = url or settings.DATABASE_URL
        self._pool_size = pool_size or settings.DATABASE_POOL_SIZE
        self._max_overflow = max_overflow or settings.DATABASE_MAX_OVERFLOW
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    @property
    def is_connected(self) -> bool:
        return self._engine is not None

    async def connect(self) -> None:
        """Create database engine and session factory."""
        if self._engine is not None:
            return

        try:
            self._engine = create_async_engine(
                self._url,
                pool_size=self._pool_size,
                max_overflow=self._max_overflow,
                pool_pre_ping=True,
                echo=settings.DEBUG,
            )

            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )

            self.logger.info("Database engine created", url=self._url.split("@")[-1])
        except Exception as e:
            raise DatabaseError(f"Failed to create database engine: {e}")

    async def disconnect(self) -> None:
        """Close database engine."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self.logger.info("Database engine disposed")

    async def health_check(self) -> dict[str, Any]:
        """Check database health."""
        if not self._engine:
            return {"status": "disconnected", "latency_ms": 0}

        try:
            start = asyncio.get_event_loop().time()
            async with self._engine.connect() as conn:
                await conn.execute("SELECT 1")
            latency = (asyncio.get_event_loop().time() - start) * 1000
            return {"status": "healthy", "latency_ms": round(latency, 2)}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "latency_ms": 0}

    async def create_tables(self) -> None:
        """Create all database tables."""
        if self._engine is None:
            raise DatabaseError("Database not connected")

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            self.logger.info("Database tables created")

    async def drop_tables(self) -> None:
        """Drop all database tables."""
        if self._engine is None:
            raise DatabaseError("Database not connected")

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            self.logger.info("Database tables dropped")

    def get_session(self) -> AsyncSession:
        """Get a new database session."""
        if self._session_factory is None:
            raise DatabaseError("Database not connected")
        return self._session_factory()

    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Async context manager for database sessions."""
        if self._session_factory is None:
            raise DatabaseError("Database not connected")

        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Singleton instance
_database: Optional[Database] = None


def get_database() -> Database:
    """Get the global database instance."""
    global _database
    if _database is None:
        _database = Database()
    return _database


async def init_database() -> Database:
    """Initialize and connect the database."""
    db = get_database()
    await db.connect()
    await db.create_tables()
    return db


async def close_database() -> None:
    """Close the database connection."""
    global _database
    if _database is not None:
        await _database.disconnect()
        _database = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database sessions."""
    db = get_database()
    async for session in db.session():
        yield session
