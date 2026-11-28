from contextlib import contextmanager
import logging
import os
import threading
from typing import Any, Dict, Iterator, Optional
from urllib.parse import urlsplit

from pgvector.psycopg import register_vector
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout

from ai_service.config import Settings, settings
from ai_service.persistence.exceptions import (
    DatabaseConfigurationError,
    DatabaseConnectionError,
    DatabaseMigrationError,
    translate_psycopg_error,
)
from ai_service.persistence.security import redact_dsn

logger = logging.getLogger("coverageiq.persistence.pool")

REQUIRED_SCHEMA_TABLES = (
    "policies",
    "policy_documents",
    "policy_chunks",
    "chunk_embeddings",
    "ingestion_jobs",
)


class PostgresPoolManager:
    """
    Thread-safe Psycopg 3 connection pool manager with pgvector type registration.

    - Credentials come strictly from runtime configuration / environment variables.
    - Connection strings and passwords are always redacted in logs and repr().
    - Connections are reused across queries; no connection is created per query.
    - Schema migrations are never executed by Python (owned by Node.js node-pg-migrate).
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        *,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        timeout: Optional[float] = None,
        app_settings: Optional[Settings] = None,
    ):
        cfg = app_settings or settings
        raw_dsn = dsn if dsn is not None else (os.getenv("DATABASE_URL") or cfg.DATABASE_URL)
        eff_min = int(min_size if min_size is not None else cfg.POSTGRES_POOL_MIN_SIZE)
        eff_max = int(max_size if max_size is not None else cfg.POSTGRES_POOL_MAX_SIZE)
        eff_timeout = float(timeout if timeout is not None else cfg.POSTGRES_POOL_TIMEOUT)

        if not isinstance(raw_dsn, str) or not raw_dsn.strip():
            raise DatabaseConfigurationError("DATABASE_URL must be configured as a non-empty PostgreSQL DSN.")
        parsed = urlsplit(raw_dsn.strip())
        if parsed.scheme not in ("postgresql", "postgres") or not parsed.hostname:
            raise DatabaseConfigurationError(
                f"Invalid PostgreSQL DSN scheme or host: {redact_dsn(raw_dsn)}"
            )
        if eff_min < 1:
            raise DatabaseConfigurationError("POSTGRES_POOL_MIN_SIZE must be at least 1.")
        if eff_max < eff_min:
            raise DatabaseConfigurationError("POSTGRES_POOL_MAX_SIZE cannot be less than POSTGRES_POOL_MIN_SIZE.")
        if eff_timeout <= 0.0:
            raise DatabaseConfigurationError("POSTGRES_POOL_TIMEOUT must be greater than 0.")

        self._dsn = raw_dsn.strip()
        self.min_size = eff_min
        self.max_size = eff_max
        self.timeout = eff_timeout
        self._lock = threading.Lock()
        self._pool: Optional[ConnectionPool] = None

    def __repr__(self) -> str:
        state = "open" if self.is_open else "closed"
        return (
            f"PostgresPoolManager(dsn={redact_dsn(self._dsn)!r}, "
            f"min_size={self.min_size}, max_size={self.max_size}, "
            f"timeout={self.timeout}, state={state!r})"
        )

    @property
    def redacted_dsn(self) -> str:
        return redact_dsn(self._dsn)

    @property
    def is_open(self) -> bool:
        pool = self._pool
        return pool is not None and not getattr(pool, "closed", False)

    @staticmethod
    def _configure_connection(conn: psycopg.Connection[Any]) -> None:
        prev_autocommit = conn.autocommit
        conn.autocommit = True
        try:
            conn.execute("SET TIME ZONE 'UTC'")
            register_vector(conn)
        except ValueError as exc:
            raise DatabaseMigrationError(
                "pgvector 'vector' extension is not registered in this database; run Node.js migrations first."
            ) from exc
        finally:
            conn.autocommit = prev_autocommit

    def open(self, *, wait: bool = True, timeout: Optional[float] = None) -> None:
        with self._lock:
            if self.is_open:
                return
            logger.info(
                "Opening PostgreSQL connection pool: dsn=%s min_size=%d max_size=%d",
                self.redacted_dsn,
                self.min_size,
                self.max_size,
            )
            try:
                pool = ConnectionPool(
                    conninfo=self._dsn,
                    min_size=self.min_size,
                    max_size=self.max_size,
                    timeout=self.timeout,
                    kwargs={"row_factory": dict_row},
                    configure=self._configure_connection,
                    open=True,
                )
                if wait:
                    pool.wait(timeout=timeout if timeout is not None else self.timeout)
                self._pool = pool
            except DatabaseMigrationError:
                if "pool" in locals():
                    pool.close()
                raise
            except PoolTimeout as exc:
                if "pool" in locals():
                    pool.close()
                raise DatabaseConnectionError(
                    f"Timed out initializing PostgreSQL pool for {self.redacted_dsn}."
                ) from exc
            except Exception as exc:
                if "pool" in locals():
                    pool.close()
                if isinstance(exc, ValueError) and "vector" in str(exc).lower():
                    raise DatabaseMigrationError(
                        "pgvector 'vector' extension is not installed in the target database."
                    ) from exc
                raise DatabaseConnectionError(
                    f"Failed to open PostgreSQL connection pool for {self.redacted_dsn}: {type(exc).__name__}"
                ) from exc

    def close(self, timeout: float = 5.0) -> None:
        with self._lock:
            pool = self._pool
            self._pool = None
            if pool is not None:
                logger.info("Closing PostgreSQL connection pool: dsn=%s", self.redacted_dsn)
                pool.close(timeout=timeout)

    def __enter__(self) -> "PostgresPoolManager":
        self.open(wait=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _require_pool(self) -> ConnectionPool:
        pool = self._pool
        if pool is None or getattr(pool, "closed", False):
            raise DatabaseConnectionError("PostgreSQL connection pool is closed or not initialized.")
        return pool

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection[Any]]:
        """Borrow a pooled connection with dict_row and registered pgvector codecs."""
        pool = self._require_pool()
        try:
            with pool.connection(timeout=self.timeout) as conn:
                yield conn
        except PoolTimeout as exc:
            raise DatabaseConnectionError("Timed out waiting for a connection from PostgreSQL pool.") from exc
        except Exception as exc:
            raise translate_psycopg_error(exc, context="PostgreSQL connection operation") from exc

    @contextmanager
    def transaction(self) -> Iterator[psycopg.Connection[Any]]:
        """
        Borrow a pooled connection inside an explicit ACID transaction.
        Commits on normal exit and rolls back cleanly on any raised exception.
        """
        pool = self._require_pool()
        try:
            with pool.connection(timeout=self.timeout) as conn:
                with conn.transaction():
                    yield conn
        except PoolTimeout as exc:
            raise DatabaseConnectionError("Timed out acquiring pooled connection for transaction.") from exc
        except Exception as exc:
            raise translate_psycopg_error(exc, context="PostgreSQL transaction") from exc

    def verify_schema(self) -> Dict[str, Any]:
        """Verify PostgreSQL connectivity, pgvector extension presence, and expected schema tables."""
        with self.connection() as conn:
            ver_row = conn.execute("SELECT current_setting('server_version_num')::int AS ver").fetchone()
            server_version_num = int(ver_row["ver"]) if ver_row else 0
            if server_version_num < 160000:
                raise DatabaseMigrationError(
                    f"PostgreSQL 16+ is required (detected server_version_num={server_version_num})."
                )

            ext_row = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            if not ext_row:
                raise DatabaseMigrationError("pgvector extension ('vector') is not installed.")

            rows = conn.execute(
                """
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public'
                  AND tablename = ANY(%(tables)s)
                """,
                {"tables": list(REQUIRED_SCHEMA_TABLES)},
            ).fetchall()
            present = {r["tablename"] for r in rows}
            missing = [t for t in REQUIRED_SCHEMA_TABLES if t not in present]
            if missing:
                raise DatabaseMigrationError(
                    f"Missing migrated schema table(s): {', '.join(missing)}. Run Node.js migrations."
                )

            return {
                "server_version_num": server_version_num,
                "pgvector_version": ext_row["extversion"],
                "tables": sorted(present),
            }

    def get_stats(self) -> Dict[str, Any]:
        if not self.is_open or self._pool is None:
            return {"is_open": False, "dsn": self.redacted_dsn}
        stats = dict(self._pool.get_stats())
        stats["is_open"] = True
        stats["dsn"] = self.redacted_dsn
        return stats


_db_pool_lock = threading.Lock()
_db_pool_manager: Optional[PostgresPoolManager] = None


def get_db_pool_manager(*, auto_open: bool = False) -> PostgresPoolManager:
    global _db_pool_manager
    with _db_pool_lock:
        if _db_pool_manager is None:
            _db_pool_manager = PostgresPoolManager()
        manager = _db_pool_manager
    if auto_open and not manager.is_open:
        manager.open(wait=True)
    return manager


def init_db_pool(*, verify_schema: bool = True, app_settings: Optional[Settings] = None) -> PostgresPoolManager:
    """Initialize and open the module-level PostgreSQL connection pool for FastAPI startup."""
    global _db_pool_manager
    with _db_pool_lock:
        if _db_pool_manager is None:
            _db_pool_manager = PostgresPoolManager(app_settings=app_settings)
        manager = _db_pool_manager
    manager.open(wait=True)
    if verify_schema:
        manager.verify_schema()
    return manager


def close_db_pool() -> None:
    """Close and reset the module-level PostgreSQL connection pool on application shutdown."""
    global _db_pool_manager
    with _db_pool_lock:
        manager = _db_pool_manager
        _db_pool_manager = None
    if manager is not None:
        manager.close()
