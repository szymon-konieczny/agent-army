"""Redis async client wrapper for caching and session storage."""

import json
from datetime import datetime, date
from typing import Any, Optional

import structlog
from redis.asyncio import Redis, from_url

logger = structlog.get_logger(__name__)


class _DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        return super().default(obj)


class RedisStore:
    """Async Redis client wrapper for caching and session management.

    Provides:
    - Async Redis connection
    - JSON serialization/deserialization
    - Key expiration handling
    - Connection pooling
    - Health checks

    Attributes:
        url: Redis connection URL
        client: Redis async client
        default_ttl: Default key time-to-live in seconds
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        default_ttl: int = 3600,
    ) -> None:
        """Initialize Redis client.

        Args:
            url: Redis connection URL.
            default_ttl: Default TTL for keys in seconds.
        """
        self.url = url
        self.default_ttl = default_ttl
        self.client: Optional[Redis] = None
        self._is_connected = False

    async def connect(self) -> None:
        """Establish Redis connection.

        Raises:
            Exception: If connection fails.
        """
        try:
            self.client = await from_url(
                self.url,
                encoding="utf-8",
                decode_responses=True,
            )

            # Test connection
            await self.client.ping()
            self._is_connected = True

            await logger.ainfo(
                "redis_connected",
                url=self.url,
                default_ttl=self.default_ttl,
            )

        except Exception as exc:
            await logger.aerror(
                "redis_connection_failed",
                error=str(exc),
                url=self.url,
            )
            raise

    async def disconnect(self) -> None:
        """Close Redis connection.

        Raises:
            Exception: If disconnection fails.
        """
        if not self.client:
            return

        try:
            await self.client.close()
            self._is_connected = False

            await logger.ainfo(
                "redis_disconnected",
            )

        except Exception as exc:
            await logger.aerror(
                "redis_disconnection_failed",
                error=str(exc),
            )
            raise

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set a key-value pair in Redis.

        Args:
            key: Redis key.
            value: Value to store (will be JSON serialized).
            ttl: Time-to-live in seconds (uses default if None).

        Returns:
            True if successful, False otherwise.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.client:
            raise RuntimeError("Redis not connected. Call connect() first.")

        try:
            ttl = ttl or self.default_ttl
            serialized = json.dumps(value, cls=_DateTimeEncoder) if not isinstance(value, str) else value

            await self.client.setex(
                key,
                ttl,
                serialized,
            )

            await logger.adebug(
                "redis_set",
                key=key,
                ttl=ttl,
            )

            return True

        except Exception as exc:
            await logger.aerror(
                "redis_set_failed",
                key=key,
                error=str(exc),
            )
            return False

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from Redis.

        Args:
            key: Redis key.

        Returns:
            Deserialized value if found, None otherwise.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.client:
            raise RuntimeError("Redis not connected. Call connect() first.")

        try:
            value = await self.client.get(key)

            if value is None:
                return None

            # Try to deserialize as JSON, fall back to raw value
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value

        except Exception as exc:
            await logger.aerror(
                "redis_get_failed",
                key=key,
                error=str(exc),
            )
            return None

    async def delete(self, key: str) -> bool:
        """Delete a key from Redis.

        Args:
            key: Redis key.

        Returns:
            True if key was deleted, False if not found.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.client:
            raise RuntimeError("Redis not connected. Call connect() first.")

        try:
            result = await self.client.delete(key)
            return result > 0

        except Exception as exc:
            await logger.aerror(
                "redis_delete_failed",
                key=key,
                error=str(exc),
            )
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis.

        Args:
            key: Redis key.

        Returns:
            True if key exists, False otherwise.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.client:
            raise RuntimeError("Redis not connected. Call connect() first.")

        try:
            result = await self.client.exists(key)
            return result > 0

        except Exception as exc:
            await logger.aerror(
                "redis_exists_failed",
                key=key,
                error=str(exc),
            )
            return False

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a numeric value in Redis.

        Args:
            key: Redis key.
            amount: Amount to increment by.

        Returns:
            New value after increment.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.client:
            raise RuntimeError("Redis not connected. Call connect() first.")

        try:
            return await self.client.incrby(key, amount)

        except Exception as exc:
            await logger.aerror(
                "redis_increment_failed",
                key=key,
                error=str(exc),
            )
            raise

    async def append_list(
        self,
        key: str,
        value: Any,
        max_length: Optional[int] = None,
    ) -> int:
        """Append value to Redis list.

        Args:
            key: Redis list key.
            value: Value to append (will be JSON serialized).
            max_length: Maximum list length (trims if exceeded).

        Returns:
            New list length.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.client:
            raise RuntimeError("Redis not connected. Call connect() first.")

        try:
            serialized = json.dumps(value, cls=_DateTimeEncoder) if not isinstance(value, str) else value
            result = await self.client.rpush(key, serialized)

            if max_length and result > max_length:
                await self.client.ltrim(key, -max_length, -1)

            return result

        except Exception as exc:
            await logger.aerror(
                "redis_append_list_failed",
                key=key,
                error=str(exc),
            )
            raise

    async def get_list(self, key: str) -> list[Any]:
        """Get all values from Redis list.

        Args:
            key: Redis list key.

        Returns:
            List of deserialized values.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.client:
            raise RuntimeError("Redis not connected. Call connect() first.")

        try:
            values = await self.client.lrange(key, 0, -1)
            result = []

            for value in values:
                try:
                    result.append(json.loads(value))
                except (json.JSONDecodeError, TypeError):
                    result.append(value)

            return result

        except Exception as exc:
            await logger.aerror(
                "redis_get_list_failed",
                key=key,
                error=str(exc),
            )
            return []

    async def clear(self) -> bool:
        """Clear all keys from Redis database.

        Returns:
            True if successful, False otherwise.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.client:
            raise RuntimeError("Redis not connected. Call connect() first.")

        try:
            await self.client.flushdb()
            return True

        except Exception as exc:
            await logger.aerror(
                "redis_clear_failed",
                error=str(exc),
            )
            return False

    async def health_check(self) -> bool:
        """Check Redis connection health.

        Returns:
            True if connection is healthy, False otherwise.
        """
        if not self.client or not self._is_connected:
            return False

        try:
            await self.client.ping()
            return True
        except Exception:
            # Don't log every 5-second health check failure — too noisy
            self._is_connected = False
            return False

    def is_connected(self) -> bool:
        """Check if Redis is connected.

        Returns:
            True if connected, False otherwise.
        """
        return self._is_connected
