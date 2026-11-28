"""PostgreSQL connection management and database readiness inspection for M5."""

from collections.abc import Iterator
import os
from typing import Any

from fastapi import Request
from sqlalchemy import Connection, Engine, create_engine, text

from ai_service.recommendation.settings import Settings, get_settings

EXPECTED_TABLES = (
    "policies",
    "policy_documents",
    "organizations",
    "policy_attributes",
    "ranking_features",
)


def create_db_engine(settings: Settings | None = None) -> Engine:
    """Create a configured SQLAlchemy Core engine for PostgreSQL with explicit timeouts."""
    cfg = settings or get_settings()
    return create_engine(
        cfg.build_database_url(),
        pool_pre_ping=True,
        pool_size=cfg.db_pool_size,
        max_overflow=cfg.db_max_overflow,
        pool_timeout=cfg.db_pool_timeout_seconds,
        connect_args={
            "connect_timeout": cfg.db_connect_timeout_seconds,
            "options": f"-c statement_timeout={cfg.db_statement_timeout_ms}",
        },
    )


def check_database_connection(conn: Connection) -> dict[str, Any]:
    """Verify live PostgreSQL connectivity using a parameterized read-only query."""
    row = (
        conn.execute(
            text(
                "SELECT "
                "current_database() AS database, "
                "current_user AS username, "
                "current_timestamp AS checked_at"
            )
        )
        .mappings()
        .one()
    )
    return dict(row)


def inspect_database_readiness(conn: Connection) -> dict[str, Any]:
    """Inspect existing tables, pgvector extension, and catalog policy counts."""
    ext_rows = (
        conn.execute(
            text("SELECT extname FROM pg_extension WHERE extname = :extname"),
            {"extname": "vector"},
        )
        .scalars()
        .all()
    )

    table_rows = (
        conn.execute(
            text(
                "SELECT table_name "
                "FROM information_schema.tables "
                "WHERE table_schema = :schema_name "
                "AND table_name = ANY(:table_names)"
            ),
            {"schema_name": "public", "table_names": list(EXPECTED_TABLES)},
        )
        .scalars()
        .all()
    )

    present_set = set(table_rows)
    tables_present = {name: (name in present_set) for name in EXPECTED_TABLES}

    total_policy_count: int | None = None
    active_catalog_policy_count: int | None = None
    if tables_present["policies"]:
        counts = (
            conn.execute(
                text(
                    "SELECT "
                    "COUNT(*) AS total_count, "
                    "COUNT(*) FILTER ("
                    "WHERE visibility = 'catalog' "
                    "AND status = :active_status"
                    ") AS active_catalog_count "
                    "FROM policies"
                ),
                {"active_status": "active"},
            )
            .mappings()
            .one()
        )
        total_policy_count = int(counts["total_count"])
        active_catalog_policy_count = int(counts["active_catalog_count"])

    return {
        "pgvector_installed": len(ext_rows) > 0,
        "tables_present": tables_present,
        "recommendation_catalog_ready": bool(
            tables_present["policies"]
            and tables_present["policy_documents"]
            and tables_present["policy_attributes"]
        ),
        "total_policy_count": total_policy_count,
        "active_catalog_policy_count": active_catalog_policy_count,
    }


def get_db_connection(request: Request) -> Iterator[Connection]:
    """Yield a scoped SQLAlchemy Core connection from the application engine."""
    engine: Engine | None = getattr(request.app.state, "db_engine", None)
    desired_env_dsn = (os.getenv("DATABASE_URL") or "").strip()
    cached_dsn = getattr(request.app.state, "db_engine_dsn", None)
    if engine is None or (desired_env_dsn and cached_dsn != desired_env_dsn):
        if engine is not None:
            engine.dispose()
        cfg = getattr(request.app.state, "settings", None) or Settings()
        engine = create_db_engine(cfg)
        request.app.state.db_engine = engine
        request.app.state.db_engine_dsn = desired_env_dsn
    with engine.connect() as conn:
        yield conn
