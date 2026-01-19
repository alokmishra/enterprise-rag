"""
Tests for src/storage/cache/redis.py
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRedisCache:
    """Tests for the RedisCache class."""

    @pytest.fixture
    def mock_redis_client(self):
        """Create a mock Redis client."""
        client = AsyncMock()
        client.get = AsyncMock(return_value=None)
        client.set = AsyncMock(return_value=True)
        client.delete = AsyncMock(return_value=1)
        client.exists = AsyncMock(return_value=1)
        client.expire = AsyncMock(return_value=True)
        client.ping = AsyncMock(return_value=True)
        client.aclose = AsyncMock()
        client.incrby = AsyncMock(return_value=1)
        client.flushdb = AsyncMock()
        client.scan_iter = MagicMock(return_value=AsyncMock(__aiter__=lambda s: s, __anext__=AsyncMock(side_effect=StopAsyncIteration)))
        return client

    @pytest.fixture
    def mock_connection_pool(self):
        """Create a mock connection pool."""
        pool = MagicMock()
        pool.disconnect = AsyncMock()
        return pool

    def test_redis_cache_initialization(self):
        """Test RedisCache can be initialized."""
        from src.storage.cache.redis import RedisCache

        cache = RedisCache(url="redis://localhost:6379/0")
        assert cache is not None
        assert cache._url == "redis://localhost:6379/0"

    @pytest.mark.asyncio
    async def test_redis_cache_connect(self, mock_redis_client, mock_connection_pool):
        """Test RedisCache connect method."""
        from src.storage.cache.redis import RedisCache

        with patch('src.storage.cache.redis.ConnectionPool.from_url', return_value=mock_connection_pool):
            with patch('src.storage.cache.redis.redis.Redis', return_value=mock_redis_client):
                cache = RedisCache(url="redis://localhost:6379/0")
                await cache.connect()
                assert cache.is_connected
                mock_redis_client.ping.assert_called()

    @pytest.mark.asyncio
    async def test_redis_cache_get_returns_none_for_missing_key(self, mock_redis_client):
        """Test RedisCache get returns None for missing key."""
        from src.storage.cache.redis import RedisCache

        mock_redis_client.get = AsyncMock(return_value=None)

        cache = RedisCache(url="redis://localhost:6379/0")
        cache._client = mock_redis_client

        result = await cache.get("missing_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_redis_cache_get_returns_value(self, mock_redis_client):
        """Test RedisCache get returns cached value."""
        from src.storage.cache.redis import RedisCache

        cached_data = {"key": "value"}
        mock_redis_client.get = AsyncMock(return_value=json.dumps(cached_data))

        cache = RedisCache(url="redis://localhost:6379/0")
        cache._client = mock_redis_client

        result = await cache.get("existing_key")
        assert result == cached_data

    @pytest.mark.asyncio
    async def test_redis_cache_set(self, mock_redis_client):
        """Test RedisCache set method."""
        from src.storage.cache.redis import RedisCache

        cache = RedisCache(url="redis://localhost:6379/0")
        cache._client = mock_redis_client

        await cache.set("test_key", {"data": "value"}, ttl=3600)
        mock_redis_client.set.assert_called()

    @pytest.mark.asyncio
    async def test_redis_cache_set_with_ttl(self, mock_redis_client):
        """Test RedisCache set method with TTL."""
        from src.storage.cache.redis import RedisCache

        cache = RedisCache(url="redis://localhost:6379/0")
        cache._client = mock_redis_client

        await cache.set("test_key", "value", ttl=3600)
        mock_redis_client.set.assert_called_with("test_key", "value", ex=3600)

    @pytest.mark.asyncio
    async def test_redis_cache_delete(self, mock_redis_client):
        """Test RedisCache delete method."""
        from src.storage.cache.redis import RedisCache

        cache = RedisCache(url="redis://localhost:6379/0")
        cache._client = mock_redis_client

        result = await cache.delete("test_key")
        mock_redis_client.delete.assert_called_with("test_key")
        assert result is True

    @pytest.mark.asyncio
    async def test_redis_cache_exists(self, mock_redis_client):
        """Test RedisCache exists method."""
        from src.storage.cache.redis import RedisCache

        mock_redis_client.exists = AsyncMock(return_value=1)

        cache = RedisCache(url="redis://localhost:6379/0")
        cache._client = mock_redis_client

        result = await cache.exists("test_key")
        assert result is True

    @pytest.mark.asyncio
    async def test_redis_cache_health_check(self, mock_redis_client):
        """Test RedisCache health_check method."""
        from src.storage.cache.redis import RedisCache

        mock_redis_client.ping = AsyncMock(return_value=True)

        cache = RedisCache(url="redis://localhost:6379/0")
        cache._client = mock_redis_client

        result = await cache.health_check()
        assert result["status"] == "healthy"
        assert "latency_ms" in result

    @pytest.mark.asyncio
    async def test_redis_cache_health_check_disconnected(self):
        """Test RedisCache health_check when disconnected."""
        from src.storage.cache.redis import RedisCache

        cache = RedisCache(url="redis://localhost:6379/0")

        result = await cache.health_check()
        assert result["status"] == "disconnected"

    @pytest.mark.asyncio
    async def test_redis_cache_disconnect(self, mock_redis_client, mock_connection_pool):
        """Test RedisCache disconnect method."""
        from src.storage.cache.redis import RedisCache

        cache = RedisCache(url="redis://localhost:6379/0")
        cache._client = mock_redis_client
        cache._pool = mock_connection_pool

        await cache.disconnect()
        mock_redis_client.aclose.assert_called()
        mock_connection_pool.disconnect.assert_called()
        assert cache._client is None

    @pytest.mark.asyncio
    async def test_redis_cache_clear(self, mock_redis_client):
        """Test RedisCache clear method."""
        from src.storage.cache.redis import RedisCache

        cache = RedisCache(url="redis://localhost:6379/0")
        cache._client = mock_redis_client

        await cache.clear()
        mock_redis_client.flushdb.assert_called()

    @pytest.mark.asyncio
    async def test_redis_cache_incr(self, mock_redis_client):
        """Test RedisCache incr method."""
        from src.storage.cache.redis import RedisCache

        mock_redis_client.incrby = AsyncMock(return_value=5)

        cache = RedisCache(url="redis://localhost:6379/0")
        cache._client = mock_redis_client

        result = await cache.incr("counter", 1)
        assert result == 5


class TestRedisCacheFactory:
    """Tests for Redis cache factory functions."""

    def test_get_cache_returns_singleton(self):
        """Test that get_cache returns the same instance."""
        from src.storage.cache.redis import get_cache, RedisCache
        import src.storage.cache.redis as redis_module

        redis_module._cache = None

        cache1 = get_cache()
        cache2 = get_cache()
        assert cache1 is cache2
        assert isinstance(cache1, RedisCache)

        redis_module._cache = None
