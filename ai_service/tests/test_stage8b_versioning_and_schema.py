"""Stage 8B tests: version-aware matching, duplicate prevention, ranking_features & DB gating."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import Connection, Engine, create_engine
from sqlalchemy.dialects import postgresql

from ai_service.common.tables import (
    organizations,
    policies,
    policy_attributes,
    policy_versions,
    ranking_features,
)
from ai_service.recommendation.db import inspect_database_readiness
from ai_service.recommendation.features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from ai_service.recommendation.filters import (
    PolicyAttributesTableUnavailableError,
    build_catalog_policy_attributes_query,
    filter_policies,
    load_catalog_policies_from_db,
)
from ai_service.recommendation.schemas import UserProfileRequest


def _profile() -> UserProfileRequest:
    return UserProfileRequest(
        policy_type="health",
        age=32,
        budget_max=25_000.0,
        min_sum_insured=500_000.0,
        required_benefits=["cashless_hospitalization"],
    )


def _catalog_row(
    *,
    policy_id: str | None = None,
    version_id: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "policy_id": policy_id or str(uuid.uuid4()),
        "organization_id": uuid.uuid4(),
        "policy_name": "Catalog Health Plan",
        "insurer_name": "Insurer A",
        "policy_type": "health",
        "description": "Catalog policy",
        "policy_status": "active",
        "current_version_id": version_id,
        "policy_version_id": version_id,
        "is_catalog_eligible": True,
        "min_entry_age": 18,
        "max_entry_age": 65,
        "max_renewal_age": 85,
        "lifelong_renewal": False,
        "coverage_mode": "both",
        "max_family_members": 4,
        "eligible_applicant_types": ["salaried", "self_employed", "student", "senior_citizen"],
        "annual_premium_min": 10_000.0,
        "annual_premium_max": 20_000.0,
        "sum_insured_min": 500_000.0,
        "sum_insured_max": 1_500_000.0,
        "deductible_amount": 0.0,
        "copay_pct": 0.0,
        "room_rent_limit_type": "no_limit",
        "room_rent_limit_value": None,
        "initial_waiting_period_days": 30,
        "ped_covered": True,
        "ped_waiting_period_months": 24,
        "specific_disease_waiting_months": 24,
        "maternity_waiting_months": None,
        "network_hospitals_count": 7_500,
        "cashless_available": True,
        "pan_india_coverage": True,
        "available_regions": [],
        "min_tenure_years": 1,
        "max_tenure_years": 3,
        "covered_benefits": ["cashless_hospitalization", "daycare_procedures"],
        "exclusions": [],
        "claim_settlement_ratio": 97.0,
        "no_claim_bonus_pct": 10.0,
        "restoration_benefit": True,
        "type_specific_metadata": {"family_floater_supported": True},
        "extraction_source": "verified",
    }
    row.update(overrides)
    return row


def _create_sqlite_mock_catalog_engine() -> Engine:
    """Create an ephemeral in-memory SQLite engine mirroring join/version columns.

    Allows exercising the actual SQLAlchemy query produced by
    `build_catalog_policy_attributes_query` across multiple versions (v0, v1,
    v2 pending, unversioned NULL) without touching PostgreSQL.
    """
    from sqlalchemy import Boolean, Column, Float, Integer, MetaData, String, Table

    m = MetaData()
    orgs = Table(
        "organizations",
        m,
        Column("id", String, primary_key=True),
    )
    pols = Table(
        "policies",
        m,
        Column("id", String, primary_key=True),
        Column("owner_user_id", String, nullable=True),
        Column("organization_id", String, nullable=True),
        Column("name", String, nullable=False),
        Column("insurer_name", String, nullable=True),
        Column("policy_type", String, nullable=True),
        Column("description", String, nullable=True),
        Column("status", String, nullable=False),
        Column("current_version_id", String, nullable=True),
        Column("created_at", Integer, nullable=False),
    )
    pa = Table(
        "policy_attributes",
        m,
        Column("id", String, primary_key=True),
        Column("policy_id", String, nullable=False),
        Column("policy_version_id", String, nullable=True),
        Column("is_catalog_eligible", Boolean, nullable=False),
        Column("min_entry_age", Integer, nullable=False),
        Column("max_entry_age", Integer, nullable=False),
        Column("max_renewal_age", Integer, nullable=True),
        Column("lifelong_renewal", Boolean, nullable=False),
        Column("coverage_mode", String, nullable=False),
        Column("max_family_members", Integer, nullable=False),
        Column("eligible_applicant_types", String, nullable=False),
        Column("annual_premium_min", Float, nullable=False),
        Column("annual_premium_max", Float, nullable=False),
        Column("sum_insured_min", Float, nullable=False),
        Column("sum_insured_max", Float, nullable=False),
        Column("deductible_amount", Float, nullable=False),
        Column("copay_pct", Float, nullable=False),
        Column("room_rent_limit_type", String, nullable=False),
        Column("room_rent_limit_value", Float, nullable=True),
        Column("initial_waiting_period_days", Integer, nullable=False),
        Column("ped_covered", Boolean, nullable=False),
        Column("ped_waiting_period_months", Integer, nullable=True),
        Column("specific_disease_waiting_months", Integer, nullable=False),
        Column("maternity_waiting_months", Integer, nullable=True),
        Column("network_hospitals_count", Integer, nullable=False),
        Column("cashless_available", Boolean, nullable=False),
        Column("pan_india_coverage", Boolean, nullable=False),
        Column("available_regions", String, nullable=False),
        Column("min_tenure_years", Integer, nullable=False),
        Column("max_tenure_years", Integer, nullable=False),
        Column("covered_benefits", String, nullable=False),
        Column("exclusions", String, nullable=False),
        Column("claim_settlement_ratio", Float, nullable=True),
        Column("no_claim_bonus_pct", Float, nullable=False),
        Column("restoration_benefit", Boolean, nullable=False),
        Column("type_specific_metadata", String, nullable=False),
        Column("extraction_source", String, nullable=False),
    )
    _ = orgs, pols, pa
    eng = create_engine("sqlite:///:memory:")
    m.create_all(eng)
    return eng


# ---------------------------------------------------------------------------
# 1. SQL Compilation & Version-Aware Join Verification
# ---------------------------------------------------------------------------


def test_compiled_catalog_query_enforces_current_version_matching() -> None:
    """Compiled SQL joins policy_attributes using current_version_id and NULL-safe equality."""
    stmt = build_catalog_policy_attributes_query(policy_type="health", active_only=True)
    sql_str = str(stmt.compile(dialect=postgresql.dialect()))
    assert "JOIN policy_attributes ON policies.id = policy_attributes.policy_id" in sql_str
    assert "policy_attributes.policy_version_id = policies.current_version_id" in sql_str
    assert (
        "policies.current_version_id IS NULL AND policy_attributes.policy_version_id IS NULL"
        in (sql_str)
    )
    assert "policy_attributes.is_catalog_eligible IS true" in sql_str


def test_version_aware_query_selects_current_excludes_stale_and_handles_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Behavioral SQL test verifying current v1 selection, v0/v2 exclusion, NULL & fail-closed."""
    from sqlalchemy import text

    org1 = str(uuid.uuid4())
    usr1 = str(uuid.uuid4())
    p1, p2, p3, p4 = (str(uuid.uuid4()) for _ in range(4))
    v0, v1, v2, v3, v4, v_unpromoted = (str(uuid.uuid4()) for _ in range(6))

    eng = _create_sqlite_mock_catalog_engine()
    try:
        with eng.begin() as conn:
            # P1: active policy on V1 with historical V0, live V1, pending V2, and stray NULL
            conn.execute(
                text(
                    "INSERT INTO policies (id, owner_user_id, organization_id, name, insurer_name, "
                    "policy_type, description, status, current_version_id, created_at) "
                    f"VALUES ('{p1}', NULL, '{org1}', 'Health V1 Active', 'Ins', 'health', '', "
                    f"'active', '{v1}', 1)"
                )
            )
            # P2: unversioned manual catalog policy (current_version_id IS NULL, row has NULL)
            conn.execute(
                text(
                    "INSERT INTO policies (id, owner_user_id, organization_id, name, insurer_name, "
                    "policy_type, description, status, current_version_id, created_at) "
                    f"VALUES ('{p2}', NULL, '{org1}', 'Manual Null Version', 'Ins', 'health', '', "
                    "'active', NULL, 2)"
                )
            )
            # P3: cut over to V3 before V3 attributes exist (only V1 exists -> MUST return 0 rows)
            conn.execute(
                text(
                    "INSERT INTO policies (id, owner_user_id, organization_id, name, insurer_name, "
                    "policy_type, description, status, current_version_id, created_at) "
                    f"VALUES ('{p3}', NULL, '{org1}', 'Cutover Missing V3', 'Ins', 'health', '', "
                    f"'active', '{v3}', 3)"
                )
            )
            # P4: private user upload with is_catalog_eligible=True -> MUST stay excluded
            conn.execute(
                text(
                    "INSERT INTO policies (id, owner_user_id, organization_id, name, insurer_name, "
                    "policy_type, description, status, current_version_id, created_at) "
                    f"VALUES ('{p4}', '{usr1}', NULL, 'Private Upload', 'Ins', 'health', '', "
                    f"'active', '{v4}', 4)"
                )
            )

            base_vals = (
                "1, 18, 65, 85, 0, 'individual', 1, '[\"salaried\"]', "
                "10000, 15000, 500000, 1000000, 0, 0, 'no_limit', NULL, "
                "30, 1, 24, 24, NULL, 5000, 1, 1, '[]', 1, 3, "
                "'[\"cashless_hospitalization\"]', '[]', 95.0, 10.0, 1, '{}', 'verified'"
            )
            for pol_id, ver_sql in [
                (p1, f"'{v0}'"),
                (p1, f"'{v1}'"),
                (p1, f"'{v2}'"),
                (p1, "NULL"),
                (p2, "NULL"),
                (p2, f"'{v_unpromoted}'"),
                (p3, f"'{v1}'"),
                (p4, f"'{v4}'"),
            ]:
                attr_id = str(uuid.uuid4())
                conn.execute(
                    text(
                        f"INSERT INTO policy_attributes VALUES ('{attr_id}', '{pol_id}', "
                        f"{ver_sql}, {base_vals})"
                    )
                )

        # Verify SQLAlchemy query against SQLite database (table exists)
        monkeypatch.setattr(
            "ai_service.recommendation.filters.check_policy_attributes_table_exists", lambda _c: True
        )
        with eng.connect() as conn:
            rows = load_catalog_policies_from_db(conn, active_only=True)

        by_id = {r["policy_id"]: r for r in rows}
        assert set(by_id.keys()) == {p1, p2}
        assert str(by_id[p1]["policy_version_id"]) == v1
        assert by_id[p2]["policy_version_id"] is None
    finally:
        eng.dispose()


# ---------------------------------------------------------------------------
# 2. Duplicate Prevention & Fail-Closed Attribute Behavior
# ---------------------------------------------------------------------------


def test_duplicate_policy_id_rows_rejected_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If an unconstrained DB ever returns duplicate policy_id rows, loader fails explicitly."""
    monkeypatch.setattr(
        "ai_service.recommendation.filters.check_policy_attributes_table_exists", lambda _c: True
    )
    dup_id = str(uuid.uuid4())
    dup_row = _catalog_row(policy_id=dup_id, version_id=str(uuid.uuid4()))

    mock_conn = MagicMock(spec=Connection)
    mock_conn.execute.return_value.mappings.return_value.all.return_value = [dup_row, dup_row]

    with pytest.raises(ValueError, match="Duplicate current-version policy_attributes rows"):
        load_catalog_policies_from_db(mock_conn)


def test_incomplete_or_fail_closed_attributes_rejected_by_hard_filter() -> None:
    """Verify missing metadata, ped_covered=False, cashless=False, is_catalog_eligible=False."""
    prof_strict = UserProfileRequest(
        policy_type="car",
        age=30,
        budget_max=20_000.0,
        min_sum_insured=500_000.0,
        vehicle_type="sedan",
        vehicle_age_years=3,
    )
    missing_car_meta = _catalog_row(
        policy_type="car",
        type_specific_metadata={},  # Missing vehicle_types & vehicle_age_limit_years
        is_catalog_eligible=True,
    )
    res_car = filter_policies(prof_strict, [missing_car_meta])
    assert res_car.status == "NO_MATCHING_POLICIES"
    failed_car = {f.rule_id for f in res_car.filter_report.policy_evaluations[0].failed_filters}
    assert {"CAR_VEHICLE_TYPE", "CAR_VEHICLE_AGE_MAX"}.issubset(failed_car)

    # Default fail-closed health attributes rejected when user asks for cashless/PED/eligible
    prof_health = UserProfileRequest(
        policy_type="health",
        age=32,
        budget_max=25_000.0,
        min_sum_insured=500_000.0,
        pre_existing_conditions=["diabetes"],
        require_cashless=True,
        location="pune",
    )
    default_fail_closed = _catalog_row(
        is_catalog_eligible=False,
        ped_covered=False,
        cashless_available=False,
        pan_india_coverage=False,
        available_regions=[],
    )
    res_health = filter_policies(prof_health, [default_fail_closed])
    assert res_health.status == "NO_MATCHING_POLICIES"
    failed_h = {f.rule_id for f in res_health.filter_report.policy_evaluations[0].failed_filters}
    assert {
        "CATALOG_ELIGIBLE",
        "PED_COVERAGE",
        "CASHLESS_REQUIRED",
        "GEOGRAPHIC_AVAILABILITY",
    }.issubset(failed_h)


# ---------------------------------------------------------------------------
# 3. ranking_features Compatibility, Feature Schema Stability & Readiness
# ---------------------------------------------------------------------------


def test_ranking_features_contract_and_feature_schema_preserved() -> None:
    """ranking_features stays aligned to schema 1.0.0 alongside 59 dynamic features."""
    assert len(FEATURE_COLUMNS) == 59
    assert FEATURE_SCHEMA_VERSION == "1.0.0"
    assert ranking_features.c.feature_schema_version.server_default.arg.text == "'1.0.0'"
    assert {
        "value_for_money_score",
        "coverage_breadth_score",
        "waiting_period_friendliness",
        "network_strength_score",
        "claim_reliability_score",
        "feature_vector",
        "policy_version_id",
    }.issubset(set(ranking_features.c.keys()))

    # Verify policy_attributes uses fail-closed defaults aligned with 003
    assert policy_attributes.c.is_catalog_eligible.server_default.arg.text == "FALSE"
    assert policy_attributes.c.coverage_mode.server_default.arg.text == "'individual'"
    assert policy_attributes.c.ped_covered.server_default.arg.text == "FALSE"
    assert policy_attributes.c.cashless_available.server_default.arg.text == "FALSE"
    assert policy_attributes.c.pan_india_coverage.server_default.arg.text == "FALSE"
    assert {c.name for c in (organizations, policies, policy_versions)}  # silence unused check


def test_inspect_database_readiness_reports_recommendation_catalog_ready() -> None:
    """inspect_database_readiness reports recommendation_catalog_ready accurately."""
    mock_conn = MagicMock()
    # ext_rows, table_rows
    ext_res = MagicMock()
    ext_res.scalars.return_value.all.return_value = ["vector"]
    tbl_res = MagicMock()
    tbl_res.scalars.return_value.all.return_value = [
        "policies",
        "policy_versions",
        "organizations",
    ]
    cnt_res = MagicMock()
    cnt_res.mappings.return_value.one.return_value = {
        "total_count": 0,
        "active_catalog_count": 0,
    }
    mock_conn.execute.side_effect = [ext_res, tbl_res, cnt_res]

    readiness = inspect_database_readiness(mock_conn)
    assert readiness["tables_present"]["policy_attributes"] is False
    assert readiness["tables_present"]["ranking_features"] is False
    assert readiness["recommendation_catalog_ready"] is False

    # Missing table still raises clean PolicyAttributesTableUnavailableError referencing 003
    mock_missing = MagicMock()
    mock_missing.execute.return_value.scalar_one_or_none.return_value = None
    with pytest.raises(
        PolicyAttributesTableUnavailableError,
        match="003_recommendation_schema.sql",
    ):
        load_catalog_policies_from_db(mock_missing)
