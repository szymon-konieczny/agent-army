"""In-memory cache store — zero-dependency drop-in replacement for Redis.

Used automatically in desktop / standalone mode so the app works
without ANY external services except PostgreSQL.

Same interface as RedisStore: set, get, delete, exists, health_check, etc.
Supports TTL-based expiration via a background cleanup task.
"""

import asyncio
import json
import time
from datetime import datetime, date
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


class _DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        return super().default(obj)


class InMemoryStore:
    """In-memory key-value store with TTL support.

    Drop-in replacement for RedisStore when Redis is not available.
    All data lives in the process — no external dependencies needed.

    Provides:
    - Async-compatible interface (same as RedisStore)
    - TTL-based key expiration
    - JSON serialization/deserialization
    - List operations
    - Atomic increment

    Attributes:
        default_ttl: Default time-to-live for keys in seconds.
    """

    def __init__(self, default_ttl: int = 3600) -> None:
        self.default_ttl = default_ttl
        self._data: dict[str, Any] = {}
        self._expiry: dict[str, float] = {}  # key → expiration timestamp
        self._lists: dict[str, list[str]] = {}
        self._is_connected = True
        self._cleanup_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """No-op — always connected. Starts background cleanup."""
        self._is_connected = True
        # Start background cleanup every 30s
        try:
            loop = asyncio.get_running_loop()
            self._cleanup_task = loop.create_task(self._cleanup_loop())
        except RuntimeError:
            pass  # No event loop — cleanup will happen on access
        await logger.ainfo("memory_store_connected", ttl=self.default_ttl)

    async def disconnect(self) -> None:
        """Cancel background cleanup."""
        self._is_connected = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None
        await logger.ainfo("memory_store_disconnected")

    async def _cleanup_loop(self) -> None:
        """Periodically remove expired keys."""
        try:
            while True:
                await asyncio.sleep(30)
                self._evict_expired()
        except asyncio.CancelledError:
            pass

    def _evict_expired(self) -> None:
        """Remove all expired keys."""
        now = time.monotonic()
        expired = [k for k, exp in self._expiry.items() if exp <= now]
        for k in expired:
            self._data.pop(k, None)
            self._expiry.pop(k, None)
            self._lists.pop(k, None)

    def _is_expired(self, key: str) -> bool:
        """Check if a key has expired."""
        if key not in self._expiry:
            return False
        if self._expiry[key] <= time.monotonic():
            # Lazy eviction
            self._data.pop(key, None)
            self._expiry.pop(key, None)
            self._lists.pop(key, None)
            return True
        return False

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Store a key-value pair with optional TTL."""
        try:
            ttl = ttl or self.default_ttl
            # Serialize the same way RedisStore does
            if not isinstance(value, str):
                serialized = json.dumps(value, cls=_DateTimeEncoder)
            else:
                serialized = value

            self._data[key] = serialized
            self._expiry[key] = time.monotonic() + ttl
            return True
        except Exception as exc:
            await logger.aerror("memory_set_failed", key=key, error=str(exc))
            return False

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        if self._is_expired(key):
            return None

        value = self._data.get(key)
        if value is None:
            return None

        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    async def delete(self, key: str) -> bool:
        """Delete a key."""
        existed = key in self._data
        self._data.pop(key, None)
        self._expiry.pop(key, None)
        self._lists.pop(key, None)
        return existed

    async def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        if self._is_expired(key):
            return False
        return key in self._data

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a numeric value."""
        if self._is_expired(key):
            self._data[key] = "0"
            self._expiry[key] = time.monotonic() + self.default_ttl

        current = self._data.get(key, "0")
        try:
            new_val = int(current) + amount
        except (ValueError, TypeError):
            new_val = amount

        self._data[key] = str(new_val)
        return new_val

    async def append_list(
        self,
        key: str,
        value: Any,
        max_length: Optional[int] = None,
    ) -> int:
        """Append a value to a list."""
        if key not in self._lists:
            self._lists[key] = []

        serialized = json.dumps(value, cls=_DateTimeEncoder) if not isinstance(value, str) else value
        self._lists[key].append(serialized)

        if max_length and len(self._lists[key]) > max_length:
            self._lists[key] = self._lists[key][-max_length:]

        return len(self._lists[key])

    async def get_list(self, key: str) -> list[Any]:
        """Get all values from a list."""
        items = self._lists.get(key, [])
        result = []
        for item in items:
            try:
                result.append(json.loads(item))
            except (json.JSONDecodeError, TypeError):
                result.append(item)
        return result

    async def clear(self) -> bool:
        """Clear all data."""
        self._data.clear()
        self._expiry.clear()
        self._lists.clear()
        return True

    async def health_check(self) -> bool:
        """Always healthy — it's in-memory."""
        return self._is_connected

    def is_connected(self) -> bool:
        """Always connected."""
        return self._is_connected
