"""PostgreSQL database abstraction layer with async connection pooling."""

from typing import Any, Optional

import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

logger = structlog.get_logger(__name__)


class Database:
    """Manages async PostgreSQL connections and session lifecycle.

    Provides:
    - Async connection pool with SQLAlchemy
    - Session management
    - Connection health checks
    - Proper lifecycle management

    Attributes:
        url: Database connection URL
        engine: SQLAlchemy async engine
        session_factory: Async session maker
        pool_size: Maximum pool size
        max_overflow: Maximum overflow connections
    """

    def __init__(
        self,
        url: str,
        pool_size: int = 20,
        max_overflow: int = 10,
        echo: bool = False,
    ) -> None:
        """Initialize database connection pool.

        Args:
            url: PostgreSQL connection URL (async dialect).
            pool_size: Maximum number of connections to keep in pool.
            max_overflow: Maximum overflow connections.
            echo: If True, log all SQL statements.
        """
        self.url = url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.echo = echo
        self.engine = None
        self.session_factory = None
        self._is_connected = False

    async def _ensure_database_exists(self) -> None:
        """Auto-create the target database if it doesn't exist.

        Connects to the default 'postgres' database and issues CREATE DATABASE.
        """
        import re

        # Parse the database name from the URL
        # URL format: postgresql+asyncpg://user:pass@host:port/dbname
        match = re.search(r"/([^/?]+)(\?|$)", self.url.rsplit("@", 1)[-1])
        if not match:
            return
        db_name = match.group(1)

        # Build URL pointing at the default 'postgres' database
        admin_url = self.url.rsplit("/", 1)[0] + "/postgres"

        try:
            admin_engine = create_async_engine(admin_url, poolclass=NullPool)
            async with admin_engine.connect() as conn:
                # Must run CREATE DATABASE outside a transaction
                await conn.execution_options(isolation_level="AUTOCOMMIT")
                # Check if the database already exists
                result = await conn.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :name"),
                    {"name": db_name},
                )
                if not result.scalar():
                    await conn.execute(text(f'CREATE DATABASE "{db_name}"'))
                    await logger.ainfo("database_auto_created", name=db_name)
            await admin_engine.dispose()
        except Exception as exc:
            await logger.awarning(
                "database_auto_create_failed",
                error=str(exc),
                db_name=db_name,
            )

    async def connect(self) -> None:
        """Establish database connection pool.

        Creates async engine and session factory.
        Automatically creates the database if it doesn't exist.

        Raises:
            Exception: If connection fails.
        """
        try:
            self.engine = create_async_engine(
                self.url,
                poolclass=None,  # Use default async pool
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                echo=self.echo,
            )

            self.session_factory = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

            # Test connection — auto-create database if missing
            try:
                async with self.engine.begin() as conn:
                    await conn.execute(text("SELECT 1"))
            except Exception:
                await logger.ainfo("database_not_found_attempting_create")
                await self.engine.dispose()
                await self._ensure_database_exists()
                # Recreate engine after creating the database
                self.engine = create_async_engine(
                    self.url,
                    poolclass=None,
                    pool_size=self.pool_size,
                    max_overflow=self.max_overflow,
                    echo=self.echo,
                )
                self.session_factory = sessionmaker(
                    self.engine,
                    class_=AsyncSession,
                    expire_on_commit=False,
                )
                async with self.engine.begin() as conn:
                    await conn.execute(text("SELECT 1"))

            self._is_connected = True

            await logger.ainfo(
                "database_connected",
                url=self.url,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
            )

        except Exception as exc:
            await logger.aerror(
                "database_connection_failed",
                error=str(exc),
                url=self.url,
            )
            raise

    async def disconnect(self) -> None:
        """Close all database connections.

        Raises:
            Exception: If disconnection fails.
        """
        if not self.engine:
            return

        try:
            await self.engine.dispose()
            self._is_connected = False

            await logger.ainfo(
                "database_disconnected",
            )

        except Exception as exc:
            await logger.aerror(
                "database_disconnection_failed",
                error=str(exc),
            )
            raise

    def get_session(self) -> AsyncSession:
        """Get a new async database session.

        Returns:
            AsyncSession for database operations.

        Raises:
            RuntimeError: If database not connected.
        """
        if not self.session_factory:
            raise RuntimeError("Database not connected. Call connect() first.")

        return self.session_factory()

    async def health_check(self) -> bool:
        """Check database connection health.

        Returns:
            True if connection is healthy, False otherwise.
        """
        if not self._is_connected or not self.engine:
            return False

        try:
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            await logger.awarning(
                "database_health_check_failed",
                error=str(exc),
            )
            return False

    async def execute_query(self, query: str) -> list[tuple[Any, ...]]:
        """Execute raw SQL query.

        Args:
            query: SQL query string.

        Returns:
            List of result tuples.

        Raises:
            Exception: If query execution fails.
        """
        session = self.get_session()
        try:
            result = await session.execute(query)
            return result.fetchall()
        finally:
            await session.close()

    async def execute_update(self, query: str) -> int:
        """Execute raw SQL update query.

        Args:
            query: SQL update query string.

        Returns:
            Number of affected rows.

        Raises:
            Exception: If query execution fails.
        """
        session = self.get_session()
        try:
            result = await session.execute(query)
            await session.commit()
            return result.rowcount
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    def is_connected(self) -> bool:
        """Check if database is connected.

        Returns:
            True if connected, False otherwise.
        """
        return self._is_connected

    async def get_pool_status(self) -> dict[str, Any]:
        """Get connection pool status.

        Returns:
            Dictionary with pool statistics.
        """
        if not self.engine:
            return {"status": "not_connected"}

        pool = self.engine.pool
        return {
            "status": "connected",
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "checked_out_connections": pool.checkedout(),  # type: ignore
        }
