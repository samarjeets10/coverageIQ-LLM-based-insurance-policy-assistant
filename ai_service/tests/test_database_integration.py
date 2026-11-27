"""Read-only PostgreSQL integration tests against existing CoverageIQ schema."""

import os

import pytest
from sqlalchemy.exc import SQLAlchemyError

from ai_service.common.tables import fetch_catalog_policies_metadata
from ai_service.recommendation.db import (
    check_database_connection,
    create_db_engine,
    inspect_database_readiness,
)
from ai_service.recommendation.settings import Settings


def _get_live_test_settings() -> Settings:
    """Load DB settings from environment or skip cleanly if credentials are unavailable."""
    cfg = Settings()
    if cfg.db_password is None and not os.getenv("DB_PASSWORD"):
        pytest.skip("DB_PASSWORD not configured; skipping live PostgreSQL read-only test")
    return cfg


def test_postgresql_read_only_integration() -> None:
    """Verify Python connects to PostgreSQL and reads existing policies metadata safely."""
    cfg = _get_live_test_settings()
    engine = create_db_engine(cfg)
    try:
        try:
            with engine.connect() as conn:
                conn_info = check_database_connection(conn)
                readiness = inspect_database_readiness(conn)
                catalog_policies = fetch_catalog_policies_metadata(conn)
        except SQLAlchemyError as exc:
            pytest.skip(f"PostgreSQL unreachable for read-only check: {exc}")

        assert conn_info["database"] == cfg.db_name
        assert readiness["tables_present"]["policies"] is True
        assert readiness["tables_present"]["policy_versions"] is True
        assert readiness["tables_present"]["organizations"] is True
        assert isinstance(catalog_policies, list)
    finally:
        engine.dispose()
