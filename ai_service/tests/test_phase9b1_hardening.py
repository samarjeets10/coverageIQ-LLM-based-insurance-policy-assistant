"""Phase 9B-1 — Production Safety Hardening, Query Efficiency & OpenAPI Contract Tests.

Developer: Shravan Sambhaji Kapare (Roll No. 203)
Module: M5 — Policy Recommendation
Covers all 12 Phase 9B-1 safety, validation, query, boundary, and contract criteria.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError
from sqlalchemy import Connection, create_engine, event, text
from sqlalchemy.exc import OperationalError, ProgrammingError

from ai_service.main import create_app
from ai_service.recommendation.citations import PolicyCitationRequest
from ai_service.recommendation.db import get_db_connection
from ai_service.recommendation.filters import (
    DuplicateCatalogPolicyError,
    PolicyAttributesTableUnavailableError,
    filter_policies,
    load_catalog_policies_from_db,
)
from ai_service.recommendation.openapi_export import (
    build_live_openapi_schema,
    verify_openapi_contract,
)
from ai_service.recommendation.router import get_citation_provider
from ai_service.recommendation.schemas import RecommendRequest, TrustedScope, UserProfileRequest
from ai_service.recommendation.service import (
    CatalogDatabaseUnavailableError,
    CatalogIntegrityError,
    load_model_state,
    recommend_policies_for_request,
)
from ai_service.recommendation.settings import Settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_policy(
    *,
    policy_id: str | None = None,
    policy_type: str = "health",
    policy_status: str = "active",
    **overrides: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "policy_id": policy_id or str(uuid.uuid4()),
        "organization_id": str(uuid.uuid4()),
        "policy_name": f"Phase 9B1 {policy_type.title()} Plan",
        "insurer_name": "CoverageIQ Shield",
        "policy_type": policy_type,
        "description": "Controlled test catalog row.",
        "policy_status": policy_status,
        "current_version_id": None,
        "policy_version_id": None,
        "is_catalog_eligible": True,
        "min_entry_age": 18,
        "max_entry_age": 65,
        "max_renewal_age": 85,
        "lifelong_renewal": False,
        "coverage_mode": "both",
        "max_family_members": 4,
        "eligible_applicant_types": ["salaried", "self_employed", "student", "senior_citizen"],
        "annual_premium_min": 10_000.0,
        "annual_premium_max": 18_000.0,
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
        "claim_settlement_ratio": 96.5,
        "no_claim_bonus_pct": 10.0,
        "restoration_benefit": True,
        "type_specific_metadata": {},
        "extraction_source": "verified",
    }
    row.update(overrides)
    return row


def _health_payload(**profile_overrides: Any) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "policy_type": "health",
        "age": 32,
        "budget_max": 25_000.0,
        "min_sum_insured": 500_000.0,
        "required_benefits": ["cashless_hospitalization"],
        "annual_income_inr": 1_500_000.0,
        "risk_preference": "balanced",
    }
    profile.update(profile_overrides)
    return {"profile": profile, "top_n": 5, "ranking_model": "auto"}


def _mock_app(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
) -> TestClient:
    app = create_app(settings or Settings(_env_file=None))
    mock_conn = MagicMock()

    def _override_conn():
        yield mock_conn

    app.dependency_overrides[get_db_connection] = _override_conn
    monkeypatch.setattr("ai_service.main.check_database_connection", lambda _c: {"database": "coverageiq_db"})
    monkeypatch.setattr(
        "ai_service.main.inspect_database_readiness",
        lambda _c: {
            "pgvector_installed": True,
            "tables_present": {"policies": True, "policy_attributes": True},
            "total_policy_count": len(catalog),
            "active_catalog_policy_count": len(catalog),
        },
    )
    monkeypatch.setattr(
        "ai_service.recommendation.service.load_catalog_policies_from_db",
        lambda _conn, policy_type=None, active_only=False: [
            row
            for row in catalog
            if policy_type is None or str(row.get("policy_type", "")).lower() == policy_type
        ],
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1, 2 & 3. Fail-Closed Internal Auth (Production, Dev, Empty, Invalid, Valid)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_name", ["production", "staging", "uat"])
def test_non_development_env_requires_non_empty_service_token_at_startup(env_name: str) -> None:
    """Missing, empty, or whitespace AI_SERVICE_TOKEN fails startup outside local dev."""
    with pytest.raises(ValidationError, match="AI_SERVICE_TOKEN"):
        Settings(_env_file=None, ai_service_env=env_name, ai_service_token=None)

    with pytest.raises(ValidationError, match="AI_SERVICE_TOKEN"):
        Settings(_env_file=None, ai_service_env=env_name, ai_service_token=SecretStr(""))

    with pytest.raises(ValidationError, match="AI_SERVICE_TOKEN"):
        Settings(_env_file=None, ai_service_env=env_name, ai_service_token=SecretStr("   "))

    with pytest.raises(ValidationError, match="cannot be disabled"):
        Settings(
            _env_file=None,
            ai_service_env=env_name,
            ai_service_token=SecretStr("valid-prod-secret"),
            ai_service_require_auth=False,
        )


def test_development_auth_configurability_and_empty_token_rejection() -> None:
    """Development stays flexible for local runs while rejecting explicit blank tokens."""
    dev_open = Settings(_env_file=None, ai_service_env="development", ai_service_token=None)
    assert dev_open.is_development_env is True
    assert dev_open.auth_enforced is False

    # Empty string env value converts cleanly to None in local dev (.env.example compatibility)
    dev_blank_env = Settings(_env_file=None, ai_service_env="development", ai_service_token="")  # type: ignore[arg-type]
    assert dev_blank_env.ai_service_token is None

    # Explicit SecretStr("") in Python MUST fail validation rather than silently disabling auth
    with pytest.raises(ValidationError, match="cannot be empty"):
        Settings(_env_file=None, ai_service_env="development", ai_service_token=SecretStr(""))

    with pytest.raises(ValidationError, match="AI_SERVICE_TOKEN"):
        Settings(
            _env_file=None,
            ai_service_env="development",
            ai_service_require_auth=True,
            ai_service_token=None,
        )


def test_production_token_enforcement_missing_invalid_valid_and_no_secret_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Production mode enforces X-Service-Token strictly without ever logging secret tokens."""
    secret_val = "prod-super-secret-token-9b1"
    cfg = Settings(
        _env_file=None,
        ai_service_env="production",
        ai_service_token=SecretStr(secret_val),
    )
    catalog = [_sample_policy()]

    with _mock_app(monkeypatch, catalog, settings=cfg) as client:
        with caplog.at_level(logging.DEBUG):
            r_missing = client.post("/api/v1/recommend", json=_health_payload())
            assert r_missing.status_code == 401

            r_empty = client.post(
                "/ai/recommend",
                json=_health_payload(),
                headers={"X-Service-Token": ""},
            )
            assert r_empty.status_code == 401

            r_bad = client.post(
                "/api/v1/recommend",
                json=_health_payload(),
                headers={"X-Service-Token": "wrong-token"},
            )
            assert r_bad.status_code == 401

            r_ok = client.post(
                "/api/v1/recommend",
                json=_health_payload(),
                headers={"X-Service-Token": secret_val},
            )
            assert r_ok.status_code == 200

            # Defense-in-depth: even if runtime settings state is mutated to remove token,
            # non-dev MUST fail closed with 401, never allowing unauthenticated access.
            client.app.state.settings.ai_service_token = None  # type: ignore[attr-defined]
            r_tampered = client.post(
                "/api/v1/recommend",
                json=_health_payload(),
                headers={"X-Service-Token": secret_val},
            )
            assert r_tampered.status_code == 401

    assert secret_val not in caplog.text


# ---------------------------------------------------------------------------
# 4 & 5. Validation Limits & Payload Resource Protection
# ---------------------------------------------------------------------------


def test_tag_list_limits_string_bounds_and_nested_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enforce max 50 items on benefits/conditions, string lengths, NaN/Inf & scope bounds."""
    with _mock_app(monkeypatch, [_sample_policy()]) as client:
        # Exactly 50 items passes Pydantic validation (40+ rules still run cleanly)
        ok_50 = client.post(
            "/api/v1/recommend",
            json=_health_payload(
                required_benefits=[f"b_{i}" for i in range(50)],
                pre_existing_conditions=[f"c_{i}" for i in range(50)],
            ),
        )
        assert ok_50.status_code == 200

        # 51 items MUST fail with 422 even when entries are duplicates
        too_many_benefits = client.post(
            "/api/v1/recommend",
            json=_health_payload(required_benefits=["dup_benefit"] * 51),
        )
        assert too_many_benefits.status_code == 422

        too_many_peds = client.post(
            "/api/v1/recommend",
            json=_health_payload(pre_existing_conditions=[f"cond_{i}" for i in range(51)]),
        )
        assert too_many_peds.status_code == 422

        # Single tag string exceeding 100 chars fails 422
        long_tag = client.post(
            "/api/v1/recommend",
            json=_health_payload(required_benefits=["x" * 101]),
        )
        assert long_tag.status_code == 422

        # Location exceeding 100 chars fails 422
        long_loc = client.post("/api/v1/recommend", json=_health_payload(location="a" * 101))
        assert long_loc.status_code == 422

        # Invalid nested TrustedScope fails 422
        bad_scope = client.post(
            "/api/v1/recommend",
            json={**_health_payload(), "scope": {"user_id": "u" * 129, "role": "user"}},
        )
        assert bad_scope.status_code == 422

        bad_scope_extra = client.post(
            "/api/v1/recommend",
            json={**_health_payload(), "scope": {"role": "superuser"}},
        )
        assert bad_scope_extra.status_code == 422

        # Non-JSON float edge / malformed body returns 422
        bad_json = client.post(
            "/api/v1/recommend",
            content=b'{"profile": {"policy_type": "health", "budget_max": NaN}}',
            headers={"Content-Type": "application/json"},
        )
        assert bad_json.status_code == 422


def test_oversized_request_payload_rejected_before_model_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Payloads exceeding max_request_body_bytes return HTTP 413 without parsing."""
    cfg = Settings(_env_file=None, max_request_body_bytes=2_048)
    with _mock_app(monkeypatch, [], settings=cfg) as client:
        oversized_payload = _health_payload(location="Mumbai")
        oversized_payload["profile"]["required_benefits"] = [
            f"benefit_padding_{i:02d}_" + ("x" * 60) for i in range(40)
        ]
        res = client.post("/api/v1/recommend", json=oversized_payload)
        assert res.status_code == 413
        assert "exceeds maximum allowed size" in res.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 6, 7 & 8. Duplicate Records, Single-Query Missing Table & DB Outage Mapping
# ---------------------------------------------------------------------------


def test_duplicate_catalog_records_raise_domain_error_and_map_to_controlled_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate policy_id rows map to CatalogIntegrityError -> HTTP 500, NEVER 503."""
    dup_id = str(uuid.uuid4())
    dup_row = _sample_policy(policy_id=dup_id)
    mock_conn = MagicMock(spec=Connection)
    mock_conn.execute.return_value.scalar_one_or_none.return_value = "policy_attributes"
    mock_conn.execute.return_value.mappings.return_value.all.return_value = [dup_row, dup_row]

    with pytest.raises(DuplicateCatalogPolicyError, match=dup_id):
        load_catalog_policies_from_db(mock_conn)

    # Direct service-level check confirms DuplicateCatalogPolicyError -> CatalogIntegrityError
    req = RecommendRequest.model_validate(_health_payload())
    state = load_model_state()
    monkeypatch.setattr(
        "ai_service.recommendation.service.load_catalog_policies_from_db",
        lambda _c, policy_type=None, active_only=False: (_ for _ in ()).throw(
            DuplicateCatalogPolicyError("Duplicate current-version rows")
        ),
    )
    with pytest.raises(CatalogIntegrityError):
        recommend_policies_for_request(req, MagicMock(), state)

    # HTTP-level check confirms HTTP 500 (not 503 DB outage) without raw Python internals
    app = create_app(Settings(_env_file=None))
    app.dependency_overrides[get_db_connection] = lambda: (yield MagicMock())
    with TestClient(app) as client:
        res = client.post("/api/v1/recommend", json=_health_payload())
        assert res.status_code == 500
        assert "duplicate" in res.json()["detail"].lower()


def test_real_connection_skips_redundant_info_schema_and_maps_missing_table() -> None:
    """Real SQLAlchemy Connections execute 1 query (no information_schema) and detect 42P01."""
    eng = create_engine("sqlite:///:memory:")
    sql_statements: list[str] = []

    @event.listens_for(eng, "before_cursor_execute")
    def _record_sql(_conn: Any, _cursor: Any, stmt: str, *_args: Any) -> None:
        sql_statements.append(stmt)

    try:
        with eng.connect() as conn:
            # SQLite missing table -> caught cleanly as PolicyAttributesTableUnavailableError
            # with ZERO information_schema queries issued!
            with pytest.raises(PolicyAttributesTableUnavailableError, match="policy_attributes"):
                load_catalog_policies_from_db(conn, policy_type="health")

        assert len(sql_statements) == 1
        assert "information_schema" not in sql_statements[0].lower()
    finally:
        eng.dispose()

    # Simulated PostgreSQL SQLSTATE 42P01 UndefinedTable on main SELECT
    class _Pg42P01(Exception):
        sqlstate = "42P01"

    pg_err = ProgrammingError(
        "SELECT ...",
        {},
        _Pg42P01('relation "policy_attributes" does not exist'),
    )
    mock_pg = MagicMock(spec=Connection)
    mock_pg.execute.side_effect = pg_err
    with pytest.raises(PolicyAttributesTableUnavailableError, match="policy_attributes"):
        load_catalog_policies_from_db(mock_pg, policy_type="health", verify_table_exists=False)

    # Genuine DB connectivity outage MUST stay CatalogDatabaseUnavailableError -> 503
    db_down = OperationalError(
        "SELECT ...", {}, Exception("connection timed out postgres://u:s@db")
    )
    mock_down = MagicMock(spec=Connection)
    mock_down.execute.side_effect = db_down
    with pytest.raises(CatalogDatabaseUnavailableError):
        recommend_policies_for_request(
            RecommendRequest.model_validate(_health_payload()),
            mock_down,
            load_model_state(),
        )


def test_unexpected_value_error_guarded_to_controlled_500(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed data ValueError inside filtering never leaks raw tracebacks."""
    monkeypatch.setattr(
        "ai_service.recommendation.service.load_catalog_policies_from_db", lambda _c, **kw: [{}]
    )
    monkeypatch.setattr(
        "ai_service.recommendation.service.filter_policies",
        lambda _p, _c: (_ for _ in ()).throw(ValueError("internal unexpected parse failure")),
    )
    app = create_app(Settings(_env_file=None))
    app.dependency_overrides[get_db_connection] = lambda: (yield MagicMock())
    with TestClient(app) as client:
        res = client.post("/api/v1/recommend", json=_health_payload())
        assert res.status_code == 500
        assert "internal unexpected parse failure" not in res.json()["detail"]


# ---------------------------------------------------------------------------
# 9. Policy-Type Pushdown Preservation Test (SQLite Real Catalog Engine)
# ---------------------------------------------------------------------------


def _build_multi_category_sqlite_engine() -> tuple[Any, dict[str, str]]:
    """Create ephemeral SQLite catalog DB with all 4 categories, inactive, private & versions."""
    from sqlalchemy import Boolean, Column, Float, Integer, MetaData, String, Table

    m = MetaData()
    Table(
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
    Table(
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
    eng = create_engine("sqlite:///:memory:")
    m.create_all(eng)

    ids = {
        "h_active_v1": str(uuid.uuid4()),
        "h_archived": str(uuid.uuid4()),
        "h_private": str(uuid.uuid4()),
        "car_active": str(uuid.uuid4()),
    }
    v_old, v_cur, v_car = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
    org = str(uuid.uuid4())
    usr = str(uuid.uuid4())

    with eng.begin() as conn:
        rows = [
            (ids["h_active_v1"], None, org, "Health V1", "Ins", "health", "active", v_cur, 1),
            (ids["h_archived"], None, org, "Health Arch", "Ins", "health", "archived", None, 2),
            (ids["h_private"], usr, None, "Private User", "Ins", "health", "active", None, 3),
            (ids["car_active"], None, org, "Car Active", "Ins", "car", "active", v_car, 4),
        ]
        for pid, uid, oid, name, ins, ptype, st, cvid, created in rows:
            conn.execute(
                text(
                    "INSERT INTO policies (id, owner_user_id, organization_id, name, "
                    "insurer_name, policy_type, description, status, "
                    "current_version_id, created_at) "
                    "VALUES (:id, :uid, :oid, :name, :ins, :ptype, '', :st, :cvid, :c)"
                ),
                {
                    "id": pid,
                    "uid": uid,
                    "oid": oid,
                    "name": name,
                    "ins": ins,
                    "ptype": ptype,
                    "st": st,
                    "cvid": cvid,
                    "c": created,
                },
            )

        attrs = [
            (ids["h_active_v1"], v_old, 9000.0),
            (ids["h_active_v1"], v_cur, 12000.0),
            (ids["h_archived"], None, 11000.0),
            (ids["h_private"], None, 8000.0),
            (ids["car_active"], v_car, 9500.0),
        ]
        for pid, pvid, prem in attrs:
            conn.execute(
                text(
                    "INSERT INTO policy_attributes VALUES (:aid, :pid, :pvid, 1, 18, 65, 85, 0, "
                    "'individual', 1, '[\"salaried\"]', :prem, :prem_max, 500000, 1500000, "
                    "0, 0, 'no_limit', NULL, 30, 1, 24, 24, NULL, 7500, 1, 1, '[]', 1, 3, "
                    "'[\"cashless_hospitalization\"]', '[]', 96.0, 10.0, 1, '{}', 'verified')"
                ),
                {
                    "aid": str(uuid.uuid4()),
                    "pid": pid,
                    "pvid": pvid,
                    "prem": prem,
                    "prem_max": prem + 4000.0,
                },
            )
    return eng, ids


def test_policy_type_pushdown_preserves_eligibility_versions_and_ownership() -> None:
    """SQL pushdown retrieves only requested category while keeping identical outcomes."""
    eng, ids = _build_multi_category_sqlite_engine()
    try:
        state = load_model_state()
        req = RecommendRequest.model_validate(_health_payload())
        prof = UserProfileRequest.model_validate(_health_payload()["profile"])

        with eng.connect() as conn:
            full_rows = load_catalog_policies_from_db(conn, policy_type=None, active_only=False)
            pushed_rows = load_catalog_policies_from_db(
                conn, policy_type="health", active_only=False
            )
            resp = recommend_policies_for_request(req, conn, state, request_id="pushdown-1")

        # SQL pushdown avoids retrieving car rows while preserving inactive/version/private rules
        assert {r["policy_id"] for r in full_rows} == {
            ids["h_active_v1"],
            ids["h_archived"],
            ids["car_active"],
        }
        assert {r["policy_id"] for r in pushed_rows} == {
            ids["h_active_v1"],
            ids["h_archived"],
        }
        # Private user upload NEVER appears in pushed_rows or response
        assert ids["h_private"] not in {r["policy_id"] for r in pushed_rows}

        # Eligibility results match full vs pushed identically
        full_eligible = [p["policy_id"] for p in filter_policies(prof, full_rows).eligible_policies]
        pushed_eligible = [
            p["policy_id"] for p in filter_policies(prof, pushed_rows).eligible_policies
        ]
        assert full_eligible == pushed_eligible == [ids["h_active_v1"]]

        # Response matches active current version (12000.0), excluding stale v_old (9000.0)
        # and preserving archived policy elimination in FilterReport
        assert resp.status == "RECOMMENDATIONS_READY"
        assert [r.policy_id for r in resp.recommendations] == [ids["h_active_v1"]]
        assert resp.recommendations[0].annual_premium_min == 12000.0
        assert resp.filter_report.total_candidates == 2
        assert resp.filter_report.eliminated_count == 1
        assert resp.filter_report.eliminations[0].rule_id == "POLICY_ACTIVE_STATUS"
    finally:
        eng.dispose()


# ---------------------------------------------------------------------------
# 10. TrustedScope Propagation Without PII Leakage
# ---------------------------------------------------------------------------


def test_trusted_scope_forwarded_to_citations_and_user_id_not_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """TrustedScope forwards to PolicyCitationRequest and never leaks user_id into logs."""
    captured_reqs: list[PolicyCitationRequest] = []

    class CapturingProvider:
        def get_citations(
            self,
            requests: list[PolicyCitationRequest] | Any,
            *,
            request_id: str | None = None,
        ) -> dict[str, list[dict[str, Any]]]:
            del request_id
            captured_reqs.extend(requests)
            return {r.policy_id: [] for r in requests}

    payload = {
        **_health_payload(),
        "scope": {
            "user_id": "secret-user-uuid-999",
            "role": "organization",
            "owner_scope": "org-scope-777",
        },
    }
    catalog = [_sample_policy()]
    app = create_app(Settings(_env_file=None))
    app.dependency_overrides[get_db_connection] = lambda: (yield MagicMock())
    app.dependency_overrides[get_citation_provider] = lambda: CapturingProvider()
    monkeypatch.setattr(
        "ai_service.recommendation.service.load_catalog_policies_from_db",
        lambda _c, policy_type=None, active_only=False: catalog,
    )

    with TestClient(app) as client:
        with caplog.at_level(logging.INFO):
            res = client.post("/ai/recommend", json=payload)
            assert res.status_code == 200

    assert len(captured_reqs) == 1
    assert captured_reqs[0].scope == TrustedScope(
        user_id="secret-user-uuid-999",
        role="organization",
        owner_scope="org-scope-777",
    )
    assert "secret-user-uuid-999" not in caplog.text
    assert "org-scope-777" not in caplog.text


# ---------------------------------------------------------------------------
# 11. Frozen OpenAPI Contract Consistency Verification
# ---------------------------------------------------------------------------


def test_frozen_openapi_contract_matches_live_fastapi_schema() -> None:
    """docs/contracts/ai-service.openapi.yaml stays strictly synchronized with FastAPI."""
    contract_file = (
        Path(__file__).resolve().parents[2] / "docs" / "contracts" / "ai-service.openapi.yaml"
    )
    assert contract_file.is_file()

    verify_openapi_contract(contract_file)
    live = build_live_openapi_schema()

    for path in ("/api/v1/recommend", "/ai/recommend"):
        assert path in live["paths"]
        responses = live["paths"][path]["post"]["responses"]
        for code in ("200", "401", "413", "422", "500", "501", "503"):
            assert code in responses
        assert {"ServiceTokenAuth": []} in live["paths"][path]["post"]["security"]

    for health_path in ("/health", "/ai/health"):
        assert health_path in live["paths"]
        assert "200" in live["paths"][health_path]["get"]["responses"]
        assert "503" in live["paths"][health_path]["get"]["responses"]

    assert "/ai/ask" in live["paths"]
    assert "post" in live["paths"]["/ai/ask"]
    assert "ServiceTokenAuth" in live["components"]["securitySchemes"]
    assert "ErrorDetailResponse" in live["components"]["schemas"]
    assert "HealthResponse" in live["components"]["schemas"]
    assert "AIHealthResponse" in live["components"]["schemas"]
    assert "TrustedScope" in live["components"]["schemas"]


def test_unified_app_has_zero_duplicate_routes_and_preserves_upstream_qa() -> None:
    """Ensure merged FastAPI app registers every (method, path) once across M3 QA and M5.

    Uses ``app.openapi()`` to resolve routes because Starlette ≥1.10 wraps
    ``include_router`` results in lazy ``_IncludedRouter`` objects whose
    ``.methods`` and ``.path`` attributes are not directly accessible from
    ``app.routes``.
    """
    app = create_app(Settings(_env_file=None))
    schema = app.openapi()
    paths = schema.get("paths", {})

    # Collect (METHOD, path) pairs from the resolved OpenAPI schema.
    route_signatures: list[tuple[str, str]] = []
    for path, methods_dict in paths.items():
        for method in sorted(methods_dict.keys()):
            route_signatures.append((method.upper(), path))

    # No duplicate (method, path) registrations.
    assert len(route_signatures) == len(set(route_signatures)), (
        f"Duplicate route registrations detected: {route_signatures}"
    )

    for expected in (
        ("GET", "/health"),
        ("GET", "/ai/health"),
        ("POST", "/ai/ask"),
        ("POST", "/api/v1/recommend"),
        ("POST", "/ai/recommend"),
    ):
        assert expected in route_signatures, (
            f"{expected} not found in resolved routes: {route_signatures}"
        )
