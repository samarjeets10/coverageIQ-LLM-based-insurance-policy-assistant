"""Verify SQLAlchemy Core definitions align with 001, 003 migration, and preserved drafts."""

from pathlib import Path

from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from ai_service.common.tables import (
    metadata,
    organizations,
    policies,
    policy_attributes,
    policy_versions,
    ranking_features,
)
from ai_service.recommendation.features import FEATURE_SCHEMA_VERSION

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parent
MIGRATION_003_PATH = REPO_ROOT / "server" / "migrations" / "003_recommendation_schema.sql"
DRAFT_SQL_V1_PATH = SERVICE_ROOT / "sql" / "proposed_003_recommendation_schema.sql"
DRAFT_SQL_V2_PATH = SERVICE_ROOT / "sql" / "proposed_003_recommendation_schema_v2.sql"
DRAFT_SQL_V3_PATH = SERVICE_ROOT / "sql" / "proposed_003_recommendation_schema_v3.sql"


def test_sqlalchemy_metadata_contains_expected_tables() -> None:
    """Ensure SQLAlchemy Core metadata reflects existing and Python-owned §2A tables."""
    expected = {
        "organizations",
        "policies",
        "policy_versions",
        "policy_attributes",
        "ranking_features",
    }
    assert expected.issubset(set(metadata.tables.keys()))
    assert "policy_type" in policies.c
    assert "current_version_id" in policies.c
    assert "version_number" in policy_versions.c
    assert "slug" in organizations.c


def test_preserved_sql_proposals_exist() -> None:
    """Ensure v1, v2, and v3 SQL proposal artifacts remain preserved."""
    assert DRAFT_SQL_V1_PATH.exists()
    assert DRAFT_SQL_V2_PATH.exists()
    assert DRAFT_SQL_V3_PATH.exists()


def test_migration_003_matches_sqlalchemy_columns_and_architecture() -> None:
    """Guard against drift between 003_recommendation_schema.sql and common/tables.py."""
    assert MIGRATION_003_PATH.exists()
    sql_text = MIGRATION_003_PATH.read_text(encoding="utf-8")

    executable_sql = "\n".join(
        line for line in sql_text.splitlines() if not line.lstrip().startswith("--")
    )

    # Architecture ownership: M5 owns policy_attributes & ranking_features ONLY
    assert "CREATE TABLE IF NOT EXISTS policy_attributes" in executable_sql
    assert "CREATE TABLE IF NOT EXISTS ranking_features" in executable_sql
    assert "CREATE TABLE IF NOT EXISTS chunks" not in executable_sql
    assert "CREATE TABLE IF NOT EXISTS sections" not in executable_sql
    assert "CREATE TABLE IF NOT EXISTS clauses" not in executable_sql
    assert "CREATE EXTENSION" not in executable_sql

    # Node migrate.js wraps each migration in BEGIN/COMMIT; SQL file must omit inner tx blocks
    assert "\nBEGIN;" not in executable_sql
    assert "\nCOMMIT;" not in executable_sql

    # Column parity across both Python-owned tables
    for col in policy_attributes.c:
        assert col.name in sql_text, f"Column {col.name} missing from 003 migration"
    for col in ranking_features.c:
        assert col.name in sql_text, f"Column {col.name} missing from 003 migration"

    # Fail-closed defaults and schema version alignment
    assert "DEFAULT 'individual'" in sql_text
    assert f"DEFAULT '{FEATURE_SCHEMA_VERSION}'" in sql_text
    assert "UNIQUE NULLS NOT DISTINCT (policy_id, policy_version_id)" in sql_text
    assert (
        "UNIQUE NULLS NOT DISTINCT (policy_id, policy_version_id, feature_schema_version)"
        in sql_text
    )

    # Verify SQLAlchemy DDL compiles cleanly under PostgreSQL dialect
    dialect = postgresql.dialect()
    pa_ddl = str(CreateTable(policy_attributes).compile(dialect=dialect))
    rf_ddl = str(CreateTable(ranking_features).compile(dialect=dialect))
    assert "UNIQUE NULLS NOT DISTINCT (policy_id, policy_version_id)" in pa_ddl
    assert "UNIQUE NULLS NOT DISTINCT (policy_id, policy_version_id, feature_schema_version)" in (
        rf_ddl
    )
