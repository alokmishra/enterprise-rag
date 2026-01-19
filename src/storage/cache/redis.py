"""
Enterprise RAG System - Redis Cache Implementation
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from src.core.config import settings
from src.core.exceptions import RAGException
from src.storage.base import CacheStore


class CacheError(RAGException):
    """Cache-related errors."""

    def __init__(self, message: str):
        super().__init__(message, code="CACHE_ERROR")


class RedisCache(CacheStore):
    """Redis implementation of cache store."""

    def __init__(
        self,
        url: Optional[str] = None,
        default_ttl: Optional[int] = None,
    ):
        self._url = url or settings.REDIS_URL
        self._default_ttl = default_ttl or settings.REDIS_CACHE_TTL
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[redis.Redis] = None

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    async def connect(self) -> None:
        """Connect to Redis server."""
        if self._client is not None:
            return

        try:
            self._pool = ConnectionPool.from_url(
                self._url,
                max_connections=20,
                decode_responses=True,
            )
            self._client = redis.Redis(connection_pool=self._pool)
            # Verify connection
            await self._client.ping()
            self.logger.info("Connected to Redis", url=self._url)
        except Exception as e:
            self._client = None
            self._pool = None
            raise CacheError(f"Failed to connect to Redis: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Redis server."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._pool is not None:
            await self._pool.disconnect()
            self._pool = None
        self.logger.info("Disconnected from Redis")

    async def health_check(self) -> dict[str, Any]:
        """Check Redis health."""
        if not self._client:
            return {"status": "disconnected", "latency_ms": 0}

        try:
            start = asyncio.get_event_loop().time()
            await self._client.ping()
            latency = (asyncio.get_event_loop().time() - start) * 1000
            return {"status": "healthy", "latency_ms": round(latency, 2)}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "latency_ms": 0}

    def _ensure_connected(self) -> redis.Redis:
        """Ensure client is connected."""
        if self._client is None:
            raise CacheError("Not connected to Redis")
        return self._client

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        client = self._ensure_connected()

        try:
            value = await client.get(key)
            if value is None:
                return None
            return json.loads(value)
        except json.JSONDecodeError:
            # Return raw string if not JSON
            return value
        except Exception as e:
            self.logger.warning("Cache get failed", key=key, error=str(e))
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set a value in cache."""
        client = self._ensure_connected()

        ttl = ttl or self._default_ttl

        try:
            serialized = json.dumps(value) if not isinstance(value, str) else value
            await client.set(key, serialized, ex=ttl)
        except Exception as e:
            self.logger.warning("Cache set failed", key=key, error=str(e))
            raise CacheError(f"Failed to set cache key: {e}")

    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        client = self._ensure_connected()

        try:
            result = await client.delete(key)
            return result > 0
        except Exception as e:
            self.logger.warning("Cache delete failed", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        client = self._ensure_connected()

        try:
            return await client.exists(key) > 0
        except Exception:
            return False

    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache, optionally by pattern."""
        client = self._ensure_connected()

        try:
            if pattern:
                keys = []
                async for key in client.scan_iter(match=pattern):
                    keys.append(key)
                if keys:
                    return await client.delete(*keys)
                return 0
            else:
                await client.flushdb()
                return -1  # Unknown count
        except Exception as e:
            self.logger.warning("Cache clear failed", pattern=pattern, error=str(e))
            return 0

    async def get_or_set(
        self,
        key: str,
        factory: callable,
        ttl: Optional[int] = None,
    ) -> Any:
        """Get from cache or compute and set."""
        value = await self.get(key)
        if value is not None:
            return value

        # Compute value
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()

        await self.set(key, value, ttl)
        return value

    async def incr(self, key: str, amount: int = 1) -> int:
        """Increment a counter."""
        client = self._ensure_connected()

        try:
            return await client.incrby(key, amount)
        except Exception as e:
            raise CacheError(f"Failed to increment: {e}")

    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL on existing key."""
        client = self._ensure_connected()

        try:
            return await client.expire(key, ttl)
        except Exception:
            return False


# Singleton instance
_cache: Optional[RedisCache] = None


def get_cache() -> RedisCache:
    """Get the global cache instance."""
    global _cache
    if _cache is None:
        _cache = RedisCache()
    return _cache


async def init_cache() -> RedisCache:
    """Initialize and connect the cache."""
    cache = get_cache()
    await cache.connect()
    return cache


async def close_cache() -> None:
    """Close the cache connection."""
    global _cache
    if _cache is not None:
        await _cache.disconnect()
        _cache = None
