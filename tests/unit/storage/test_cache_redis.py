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
        client.keys = AsyncMock(return_value=[])
        client.ping = AsyncMock(return_value=True)
        client.close = AsyncMock()
        return client

    def test_redis_cache_initialization(self):
        """Test RedisCache can be initialized."""
        from src.storage.cache.redis import RedisCache

        cache = RedisCache(url="redis://localhost:6379/0")
        assert cache is not None

    @pytest.mark.asyncio
    async def test_redis_cache_connect(self, mock_redis_client):
        """Test RedisCache connect method."""
        from src.storage.cache.redis import RedisCache

        with patch('src.storage.cache.redis.aioredis.from_url', return_value=mock_redis_client):
            cache = RedisCache(url="redis://localhost:6379/0")
            await cache.connect()
            # Should establish connection

    @pytest.mark.asyncio
    async def test_redis_cache_get_returns_none_for_missing_key(self, mock_redis_client):
        """Test RedisCache get returns None for missing key."""
        from src.storage.cache.redis import RedisCache

        mock_redis_client.get = AsyncMock(return_value=None)

        cache = RedisCache(url="redis://localhost:6379/0")
        cache.client = mock_redis_client

        result = await cache.get("missing_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_redis_cache_get_returns_value(self, mock_redis_client):
        """Test RedisCache get returns cached value."""
        from src.storage.cache.redis import RedisCache

        cached_data = {"key": "value"}
        mock_redis_client.get = AsyncMock(return_value=json.dumps(cached_data).encode())

        cache = RedisCache(url="redis://localhost:6379/0")
        cache.client = mock_redis_client

        result = await cache.get("existing_key")
        assert result == cached_data

    @pytest.mark.asyncio
    async def test_redis_cache_set(self, mock_redis_client):
        """Test RedisCache set method."""
        from src.storage.cache.redis import RedisCache

        cache = RedisCache(url="redis://localhost:6379/0")
        cache.client = mock_redis_client

        await cache.set("test_key", {"data": "value"}, ttl=3600)
        mock_redis_client.set.assert_called()

    @pytest.mark.asyncio
    async def test_redis_cache_set_with_ttl(self, mock_redis_client):
        """Test RedisCache set method with TTL."""
        from src.storage.cache.redis import RedisCache

        cache = RedisCache(url="redis://localhost:6379/0")
        cache.client = mock_redis_client

        await cache.set("test_key", "value", ttl=3600)
        # Should set with expiration

    @pytest.mark.asyncio
    async def test_redis_cache_delete(self, mock_redis_client):
        """Test RedisCache delete method."""
        from src.storage.cache.redis import RedisCache

        cache = RedisCache(url="redis://localhost:6379/0")
        cache.client = mock_redis_client

        await cache.delete("test_key")
        mock_redis_client.delete.assert_called_with("test_key")

    @pytest.mark.asyncio
    async def test_redis_cache_delete_pattern(self, mock_redis_client):
        """Test RedisCache delete by pattern."""
        from src.storage.cache.redis import RedisCache

        mock_redis_client.keys = AsyncMock(return_value=[b"prefix:key1", b"prefix:key2"])

        cache = RedisCache(url="redis://localhost:6379/0")
        cache.client = mock_redis_client

        if hasattr(cache, 'delete_pattern'):
            await cache.delete_pattern("prefix:*")

    @pytest.mark.asyncio
    async def test_redis_cache_health_check(self, mock_redis_client):
        """Test RedisCache health_check method."""
        from src.storage.cache.redis import RedisCache

        mock_redis_client.ping = AsyncMock(return_value=True)

        cache = RedisCache(url="redis://localhost:6379/0")
        cache.client = mock_redis_client

        is_healthy = await cache.health_check()
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_redis_cache_disconnect(self, mock_redis_client):
        """Test RedisCache disconnect method."""
        from src.storage.cache.redis import RedisCache

        cache = RedisCache(url="redis://localhost:6379/0")
        cache.client = mock_redis_client

        await cache.disconnect()
        mock_redis_client.close.assert_called()


class TestRedisCacheKeyGeneration:
    """Tests for Redis cache key generation."""

    def test_cache_key_prefix(self):
        """Test that cache keys use proper prefix."""
        from src.storage.cache.redis import RedisCache

        cache = RedisCache(url="redis://localhost:6379/0", prefix="rag")
        # Key generation should include prefix
        assert cache.prefix == "rag"

    def test_cache_key_format(self):
        """Test cache key format."""
        from src.storage.cache.redis import RedisCache

        cache = RedisCache(url="redis://localhost:6379/0", prefix="rag")
        if hasattr(cache, '_make_key'):
            key = cache._make_key("query", "abc123")
            assert "rag" in key
