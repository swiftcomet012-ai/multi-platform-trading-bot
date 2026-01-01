"""
Database connection and session management using SQLAlchemy.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from packages.shared.src.config import get_settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


class Database:
    """Database connection manager."""

    def __init__(self, url: str | None = None, echo: bool = False) -> None:
        """
        Initialize database connection.

        Args:
            url: Database URL (defaults to settings)
            echo: Echo SQL queries for debugging
        """
        settings = get_settings()
        self._url = url or settings.database.url

        # Convert sqlite:// to sqlite+aiosqlite://
        if self._url.startswith("sqlite://"):
            self._url = self._url.replace("sqlite://", "sqlite+aiosqlite://")

        self._echo = echo or settings.database.echo
        self._engine = create_async_engine(
            self._url,
            echo=self._echo,
            pool_pre_ping=True,
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

        # Enable foreign keys for SQLite
        if "sqlite" in self._url:
            @event.listens_for(self._engine.sync_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

    @property
    def engine(self):
        """Get the database engine."""
        return self._engine

    async def create_tables(self) -> None:
        """Create all tables in the database."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self) -> None:
        """Drop all tables in the database."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession]:
        """
        Get a database session.

        Usage:
            async with db.session() as session:
                # use session
        """
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def close(self) -> None:
        """Close the database connection."""
        await self._engine.dispose()


# Global database instance
_db: Database | None = None


def get_database() -> Database:
    """Get the global database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db


async def init_database(url: str | None = None) -> Database:
    """Initialize the database and create tables."""
    global _db
    _db = Database(url=url)
    await _db.create_tables()
    return _db


async def close_database() -> None:
    """Close the database connection."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None
