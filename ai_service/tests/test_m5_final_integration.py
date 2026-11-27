"""M5 Final Integration Verification Tests (Routes, Auth, DB, Features, Ranking & Citations).

Developer: Shravan Sambhaji Kapare (Roll No. 203)
Module: M5 — Policy Recommendation
Covers the complete 12-step M5 HTTP -> SQL/DB -> Filtering -> Stage 5 Features ->
LightGBM/Weighted Scoring -> Citation Provider -> Response pipeline with isolated
003-shaped SQLite catalogs and FastAPI TestClient.
"""

from __future__ import annotations

import json
import math
import uuid
from typing import Any
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import Boolean, Column, Float, Integer, MetaData, String, Table, create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import StaticPool

from ai_service.main import create_app
from ai_service.recommendation.citations import PolicyCitationRequest
from ai_service.recommendation.db import get_db_connection
from ai_service.recommendation.features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from ai_service.recommendation.openapi_export import verify_openapi_contract
from ai_service.recommendation.router import get_citation_provider
from ai_service.recommendation.settings import Settings
from ai_service.recommendation.train_lightgbm import MODEL_NAME, MODEL_VERSION
from ai_service.recommendation.weighted import WEIGHTED_SCORER_NAME


def _create_sqlite_catalog_db(
    *,
    include_policy_attributes: bool = True,
) -> tuple[Any, dict[str, str]]:
    """Build an isolated SQLite DB mirroring 001 + 003 catalog and version lineage."""
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
    if include_policy_attributes:
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

    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    m.create_all(eng)
    ids = {
        "org_id": str(uuid.uuid4()),
        "user_id": str(uuid.uuid4()),
        "tie_a": "00000000-0000-0000-0000-00000000000a",
        "tie_b": "00000000-0000-0000-0000-00000000000b",
        "tie_a_v_old": str(uuid.uuid4()),
        "tie_a_v_cur": str(uuid.uuid4()),
        "tie_b_v_cur": str(uuid.uuid4()),
        "car_id": str(uuid.uuid4()),
        "car_v_cur": str(uuid.uuid4()),
        "life_id": str(uuid.uuid4()),
        "life_v_cur": str(uuid.uuid4()),
        "term_id": str(uuid.uuid4()),
        "term_v_cur": str(uuid.uuid4()),
        "private_id": str(uuid.uuid4()),
        "private_v_cur": str(uuid.uuid4()),
    }
    return eng, ids


def _insert_policy(
    conn: Any,
    *,
    policy_id: str,
    owner_user_id: str | None,
    organization_id: str | None,
    name: str,
    policy_type: str,
    status: str,
    current_version_id: str | None,
    created_at: int,
) -> None:
    conn.execute(
        text(
            "INSERT INTO policies (id, owner_user_id, organization_id, name, insurer_name, "
            "policy_type, description, status, current_version_id, created_at) "
            "VALUES (:id, :uid, :oid, :name, 'CoverageIQ Insurer', :ptype, '', :st, :cvid, :c)"
        ),
        {
            "id": policy_id,
            "uid": owner_user_id,
            "oid": organization_id,
            "name": name,
            "ptype": policy_type,
            "st": status,
            "cvid": current_version_id,
            "c": created_at,
        },
    )


def _insert_attributes(conn: Any, **overrides: Any) -> None:
    row: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "policy_id": "",
        "policy_version_id": None,
        "is_catalog_eligible": 1,
        "min_entry_age": 18,
        "max_entry_age": 65,
        "max_renewal_age": 85,
        "lifelong_renewal": 0,
        "coverage_mode": "both",
        "max_family_members": 4,
        "eligible_applicant_types": '["salaried","self_employed"]',
        "annual_premium_min": 12000.0,
        "annual_premium_max": 18000.0,
        "sum_insured_min": 500000.0,
        "sum_insured_max": 1500000.0,
        "deductible_amount": 0.0,
        "copay_pct": 0.0,
        "room_rent_limit_type": "no_limit",
        "room_rent_limit_value": None,
        "initial_waiting_period_days": 30,
        "ped_covered": 1,
        "ped_waiting_period_months": 24,
        "specific_disease_waiting_months": 24,
        "maternity_waiting_months": None,
        "network_hospitals_count": 7500,
        "cashless_available": 1,
        "pan_india_coverage": 1,
        "available_regions": "[]",
        "min_tenure_years": 1,
        "max_tenure_years": 3,
        "covered_benefits": '["cashless_hospitalization","daycare_procedures"]',
        "exclusions": "[]",
        "claim_settlement_ratio": 96.5,
        "no_claim_bonus_pct": 10.0,
        "restoration_benefit": 1,
        "type_specific_metadata": "{}",
        "extraction_source": "verified",
    }
    row.update(overrides)
    conn.execute(
        text(
            "INSERT INTO policy_attributes VALUES (:id, :policy_id, :policy_version_id, "
            ":is_catalog_eligible, :min_entry_age, :max_entry_age, :max_renewal_age, "
            ":lifelong_renewal, :coverage_mode, :max_family_members, :eligible_applicant_types, "
            ":annual_premium_min, :annual_premium_max, :sum_insured_min, :sum_insured_max, "
            ":deductible_amount, :copay_pct, :room_rent_limit_type, :room_rent_limit_value, "
            ":initial_waiting_period_days, :ped_covered, :ped_waiting_period_months, "
            ":specific_disease_waiting_months, :maternity_waiting_months, "
            ":network_hospitals_count, :cashless_available, :pan_india_coverage, "
            ":available_regions, :min_tenure_years, :max_tenure_years, :covered_benefits, "
            ":exclusions, :claim_settlement_ratio, :no_claim_bonus_pct, :restoration_benefit, "
            ":type_specific_metadata, :extraction_source)"
        ),
        row,
    )


def test_e2e_authenticated_multi_category_lineage_and_tie_breaking_both_routes() -> None:
    """Verify full M5 HTTP -> SQL -> Filters -> Features -> LightGBM/Weighted -> Response flow."""
    eng, ids = _create_sqlite_catalog_db(include_policy_attributes=True)
    secret = "m5-final-prod-secret-token"
    cfg = Settings(
        _env_file=None,
        ai_service_env="production",
        ai_service_token=SecretStr(secret),
    )

    try:
        with eng.begin() as conn:
            _insert_policy(
                conn,
                policy_id=ids["tie_b"],
                owner_user_id=None,
                organization_id=ids["org_id"],
                name="Health Tie B",
                policy_type="health",
                status="active",
                current_version_id=ids["tie_b_v_cur"],
                created_at=1,
            )
            _insert_policy(
                conn,
                policy_id=ids["tie_a"],
                owner_user_id=None,
                organization_id=ids["org_id"],
                name="Health Tie A",
                policy_type="health",
                status="active",
                current_version_id=ids["tie_a_v_cur"],
                created_at=2,
            )
            # Private user policy must remain invisible even with TrustedScope
            _insert_policy(
                conn,
                policy_id=ids["private_id"],
                owner_user_id=ids["user_id"],
                organization_id=None,
                name="User Private Policy",
                policy_type="health",
                status="active",
                current_version_id=ids["private_v_cur"],
                created_at=3,
            )
            _insert_policy(
                conn,
                policy_id=ids["car_id"],
                owner_user_id=None,
                organization_id=ids["org_id"],
                name="Car Current",
                policy_type="car",
                status="active",
                current_version_id=ids["car_v_cur"],
                created_at=4,
            )
            _insert_policy(
                conn,
                policy_id=ids["life_id"],
                owner_user_id=None,
                organization_id=ids["org_id"],
                name="Life Current",
                policy_type="life",
                status="active",
                current_version_id=ids["life_v_cur"],
                created_at=5,
            )
            _insert_policy(
                conn,
                policy_id=ids["term_id"],
                owner_user_id=None,
                organization_id=ids["org_id"],
                name="Term Current",
                policy_type="term",
                status="active",
                current_version_id=ids["term_v_cur"],
                created_at=6,
            )

            # Stale version row on tie_a (cheaper old price must NEVER appear)
            _insert_attributes(
                conn,
                policy_id=ids["tie_a"],
                policy_version_id=ids["tie_a_v_old"],
                annual_premium_min=6000.0,
                annual_premium_max=9000.0,
            )
            # Identical current versions for deterministic tie-break (tie_a before tie_b)
            _insert_attributes(conn, policy_id=ids["tie_a"], policy_version_id=ids["tie_a_v_cur"])
            _insert_attributes(conn, policy_id=ids["tie_b"], policy_version_id=ids["tie_b_v_cur"])
            _insert_attributes(
                conn,
                policy_id=ids["private_id"],
                policy_version_id=ids["private_v_cur"],
                annual_premium_min=4000.0,
                annual_premium_max=7000.0,
            )
            _insert_attributes(
                conn,
                policy_id=ids["car_id"],
                policy_version_id=ids["car_v_cur"],
                coverage_mode="individual",
                max_family_members=1,
                covered_benefits='["own_damage","third_party_liability"]',
                type_specific_metadata=json.dumps(
                    {
                        "vehicle_types": ["sedan"],
                        "vehicle_age_limit_years": 10,
                        "own_damage_coverage": True,
                        "third_party_coverage": True,
                    }
                ),
            )
            _insert_attributes(
                conn,
                policy_id=ids["life_id"],
                policy_version_id=ids["life_v_cur"],
                coverage_mode="individual",
                max_family_members=1,
                sum_insured_max=3000000.0,
                max_tenure_years=25,
                covered_benefits='["death_benefit"]',
                type_specific_metadata=json.dumps(
                    {
                        "life_cover_min": 500000,
                        "life_cover_max": 3000000,
                        "policy_term_years_min": 5,
                        "policy_term_years_max": 25,
                        "maturity_benefit": True,
                    }
                ),
            )
            _insert_attributes(
                conn,
                policy_id=ids["term_id"],
                policy_version_id=ids["term_v_cur"],
                coverage_mode="individual",
                max_family_members=1,
                sum_insured_max=10000000.0,
                max_tenure_years=35,
                covered_benefits='["pure_term_life_cover"]',
                type_specific_metadata=json.dumps(
                    {
                        "life_cover_min": 2500000,
                        "life_cover_max": 10000000,
                        "policy_term_years_min": 10,
                        "policy_term_years_max": 35,
                        "smoking_status_eligibility": "all",
                    }
                ),
            )

        app = create_app(cfg)

        def _override_conn():
            with eng.connect() as conn:
                yield conn

        app.dependency_overrides[get_db_connection] = _override_conn

        headers = {"X-Service-Token": secret, "X-Request-ID": "m5-final-trace-01"}
        with TestClient(app) as client:
            for route in ("/api/v1/recommend", "/ai/recommend"):
                for mode, expected_model in (
                    ("auto", MODEL_NAME),
                    ("lightgbm", MODEL_NAME),
                    ("weighted", WEIGHTED_SCORER_NAME),
                ):
                    # Flat + nested health request checks tie-breaking and version lineage
                    payload: dict[str, Any] = (
                        {
                            "policy_type": "health",
                            "age": 34,
                            "budget_max": 25000,
                            "min_sum_insured": 500000,
                            "ranking_model": mode,
                            "scope": {"user_id": ids["user_id"], "role": "User"},
                        }
                        if route == "/api/v1/recommend"
                        else {
                            "profile": {
                                "policy_type": "health",
                                "age": 34,
                                "budget_max": 25000,
                                "min_sum_insured": 500000,
                            },
                            "ranking_model": mode,
                            "scope": {"user_id": ids["user_id"], "role": "Organization"},
                        }
                    )
                    res = client.post(route, json=payload, headers=headers)
                    assert res.status_code == 200
                    assert res.headers["X-Request-ID"] == "m5-final-trace-01"
                    body = res.json()
                    assert body["request_id"] == "m5-final-trace-01"
                    assert body["status"] == "RECOMMENDATIONS_READY"
                    assert body["model_used"] == expected_model
                    assert body["feature_schema_version"] == FEATURE_SCHEMA_VERSION
                    if mode != "weighted":
                        assert body["model_version"] == MODEL_VERSION
                    recs = body["recommendations"]
                    assert [r["policy_id"] for r in recs] == [ids["tie_a"], ids["tie_b"]]
                    assert [r["rank"] for r in recs] == [1, 2]
                    assert recs[0]["score"] == recs[1]["score"]
                    assert math.isfinite(recs[0]["score"])
                    assert recs[0]["annual_premium_min"] == 12000.0
                    assert recs[0]["policy_version_id"] == ids["tie_a_v_cur"]
                    assert recs[1]["policy_version_id"] == ids["tie_b_v_cur"]
                    assert recs[0]["citations"] == []
                    assert recs[1]["citations"] == []

            # Multi-category coverage check (car, life, term) via /ai/recommend
            for cat_profile, exp_pid, exp_vid in (
                (
                    {
                        "policy_type": "car",
                        "age": 30,
                        "budget_max": 25000,
                        "min_sum_insured": 500000,
                        "vehicle_type": "sedan",
                        "vehicle_age_years": 3,
                        "require_own_damage": True,
                    },
                    ids["car_id"],
                    ids["car_v_cur"],
                ),
                (
                    {
                        "policy_type": "life",
                        "age": 35,
                        "budget_max": 25000,
                        "min_sum_insured": 1500000,
                        "tenure_years": 20,
                        "require_maturity_benefit": True,
                    },
                    ids["life_id"],
                    ids["life_v_cur"],
                ),
                (
                    {
                        "policy_type": "term",
                        "age": 32,
                        "budget_max": 25000,
                        "min_sum_insured": 5000000,
                        "tenure_years": 25,
                    },
                    ids["term_id"],
                    ids["term_v_cur"],
                ),
            ):
                r_cat = client.post(
                    "/ai/recommend",
                    json={"profile": cat_profile, "ranking_model": "auto"},
                    headers=headers,
                )
                assert r_cat.status_code == 200
                cat_body = r_cat.json()
                assert [r["policy_id"] for r in cat_body["recommendations"]] == [exp_pid]
                assert cat_body["recommendations"][0]["policy_version_id"] == exp_vid
    finally:
        eng.dispose()


def test_e2e_auth_validation_db_and_uniform_request_id_propagation() -> None:
    """Verify uniform 401/413/422/500/501/503 responses and X-Request-ID echo on every error."""
    secret = "m5-error-trace-secret"
    cfg = Settings(
        _env_file=None,
        ai_service_env="production",
        ai_service_token=SecretStr(secret),
        max_request_body_bytes=1_024,
    )
    eng, _ = _create_sqlite_catalog_db(include_policy_attributes=False)
    try:
        app = create_app(cfg)

        def _override_conn():
            with eng.connect() as conn:
                yield conn

        app.dependency_overrides[get_db_connection] = _override_conn
        valid_req = {
            "policy_type": "health",
            "age": 30,
            "budget_max": 20000,
            "min_sum_insured": 500000,
        }

        with TestClient(app) as client:
            # 401 unauthorized across missing / invalid token
            r401 = client.post(
                "/ai/recommend",
                json=valid_req,
                headers={"X-Request-ID": "trace-401", "X-Service-Token": "wrong"},
            )
            assert r401.status_code == 401
            assert r401.headers["X-Request-ID"] == "trace-401"

            # 413 oversized payload before JSON parsing
            r413 = client.post(
                "/api/v1/recommend",
                json={**valid_req, "required_benefits": ["b_" + ("x" * 60) for _ in range(25)]},
                headers={"X-Request-ID": "trace-413", "X-Service-Token": secret},
            )
            assert r413.status_code == 413
            assert r413.headers["X-Request-ID"] == "trace-413"

            # 422 validation error (age out of range & conflicting individual family_size)
            r422 = client.post(
                "/api/v1/recommend",
                json={"policy_type": "health", "age": 150, "budget_max": 0, "min_sum_insured": 0},
                headers={"X-Request-ID": "trace-422", "X-Service-Token": secret},
            )
            assert r422.status_code == 422
            assert r422.headers["X-Request-ID"] == "trace-422"

            # 501 unimplemented ranking model
            r501 = client.post(
                "/ai/recommend",
                json={**valid_req, "ranking_model": "xgboost"},
                headers={"X-Request-ID": "trace-501", "X-Service-Token": secret},
            )
            assert r501.status_code == 501
            assert r501.headers["X-Request-ID"] == "trace-501"

            # 503 missing policy_attributes table fails closed cleanly without SQL leak
            r503_missing_tbl = client.post(
                "/api/v1/recommend",
                json=valid_req,
                headers={"X-Request-ID": "trace-503-tbl", "X-Service-Token": secret},
            )
            assert r503_missing_tbl.status_code == 503
            assert r503_missing_tbl.headers["X-Request-ID"] == "trace-503-tbl"
            assert "policy_attributes" in r503_missing_tbl.json()["detail"]

            # 503 connection-level SQLAlchemyError fails closed and echoes X-Request-ID
            app.dependency_overrides.clear()
            app.state.db_engine = MagicMock()
            app.state.db_engine.connect.side_effect = OperationalError(
                "CONNECT", {}, Exception("refused postgres://postgres:topsecret@localhost:5432")
            )
            r503_conn = client.post(
                "/ai/recommend",
                json=valid_req,
                headers={"X-Request-ID": "trace-503-conn", "X-Service-Token": secret},
            )
            assert r503_conn.status_code == 503
            assert r503_conn.headers["X-Request-ID"] == "trace-503-conn"
            assert "topsecret" not in r503_conn.text
    finally:
        eng.dispose()


def test_e2e_empty_catalog_and_mixed_malformed_sqlite_rows_fail_safely() -> None:
    """Empty catalog returns NO_MATCHING_POLICIES; malformed DB rows exclude only themselves."""
    eng, ids = _create_sqlite_catalog_db(include_policy_attributes=True)
    try:
        app = create_app(Settings(_env_file=None))

        def _override_conn():
            with eng.connect() as conn:
                yield conn

        app.dependency_overrides[get_db_connection] = _override_conn

        req = {
            "policy_type": "health",
            "age": 32,
            "budget_max": 25000,
            "min_sum_insured": 500000,
        }
        with TestClient(app) as client:
            # 1. Completely empty catalog -> valid 200 NO_MATCHING_POLICIES
            r_empty = client.post("/api/v1/recommend", json=req)
            assert r_empty.status_code == 200
            empty_body = r_empty.json()
            assert empty_body["status"] == "NO_MATCHING_POLICIES"
            assert empty_body["recommendations"] == []
            assert empty_body["filter_report"]["total_candidates"] == 0

            # 2. Add one malformed row + one valid row missing optional PED wait & CSR
            bad_id = str(uuid.uuid4())
            bad_v = str(uuid.uuid4())
            good_id = str(uuid.uuid4())
            good_v = str(uuid.uuid4())
            with eng.begin() as conn:
                _insert_policy(
                    conn,
                    policy_id=bad_id,
                    owner_user_id=None,
                    organization_id=ids["org_id"],
                    name="Corrupt Health",
                    policy_type="health",
                    status="active",
                    current_version_id=bad_v,
                    created_at=1,
                )
                _insert_attributes(
                    conn,
                    policy_id=bad_id,
                    policy_version_id=bad_v,
                    deductible_amount=-100.0,
                    type_specific_metadata='["not_an_object"]',
                )
                _insert_policy(
                    conn,
                    policy_id=good_id,
                    owner_user_id=None,
                    organization_id=ids["org_id"],
                    name="Sparse Valid Health",
                    policy_type="health",
                    status="active",
                    current_version_id=good_v,
                    created_at=2,
                )
                _insert_attributes(
                    conn,
                    policy_id=good_id,
                    policy_version_id=good_v,
                    ped_covered=1,
                    ped_waiting_period_months=None,
                    claim_settlement_ratio=None,
                )

            r_mixed = client.post("/api/v1/recommend", json=req)
            assert r_mixed.status_code == 200
            mixed_body = r_mixed.json()
            assert mixed_body["status"] == "RECOMMENDATIONS_READY"
            assert [r["policy_id"] for r in mixed_body["recommendations"]] == [good_id]
            assert mixed_body["recommendations"][0]["policy_version_id"] == good_v
            assert any("36-month (1080-day) fallback" in w for w in mixed_body["warnings"])
            highlights = mixed_body["recommendations"][0]["key_highlights"]
            assert "Pre-existing conditions covered" in highlights
            assert not any("36-month" in h for h in highlights)
            assert not any("Claim settlement ratio" in h for h in highlights)
            assert any(
                e["policy_id"] == bad_id and e["rule_id"] == "INVALID_POLICY_METADATA"
                for e in mixed_body["filter_report"]["eliminations"]
            )
    finally:
        eng.dispose()


def test_e2e_citation_provider_unavailable_empty_and_partial_no_fabrication() -> None:
    """Unavailable/partial citation responses keep [] cleanly without fabricating evidence."""
    eng, ids = _create_sqlite_catalog_db(include_policy_attributes=True)
    try:
        with eng.begin() as conn:
            for pid, pvid, name, created in (
                (ids["tie_a"], ids["tie_a_v_cur"], "Plan A", 1),
                (ids["tie_b"], ids["tie_b_v_cur"], "Plan B", 2),
            ):
                _insert_policy(
                    conn,
                    policy_id=pid,
                    owner_user_id=None,
                    organization_id=ids["org_id"],
                    name=name,
                    policy_type="health",
                    status="active",
                    current_version_id=pvid,
                    created_at=created,
                )
                _insert_attributes(conn, policy_id=pid, policy_version_id=pvid)

        class EmptyCitationProvider:
            def get_citations(
                self,
                requests: list[PolicyCitationRequest] | Any,
                *,
                request_id: str | None = None,
            ) -> dict[str, list[dict[str, Any]]]:
                del requests, request_id
                return {}

        class PartialCitationProvider:
            def get_citations(
                self,
                requests: list[PolicyCitationRequest] | Any,
                *,
                request_id: str | None = None,
            ) -> dict[str, list[dict[str, Any]]]:
                del request_id
                first = requests[0]
                return {
                    first.policy_id: [
                        {
                            "chunk_id": f"chk-{first.policy_version_id}",
                            "page": 3,
                            "clause": "Section 1.2 Coverage",
                        }
                    ]
                }

        app = create_app(Settings(_env_file=None))

        def _override_conn():
            with eng.connect() as conn:
                yield conn

        app.dependency_overrides[get_db_connection] = _override_conn
        payload = {
            "policy_type": "health",
            "age": 34,
            "budget_max": 25000,
            "min_sum_insured": 500000,
        }

        with TestClient(app) as client:
            app.dependency_overrides[get_citation_provider] = lambda: EmptyCitationProvider()
            r_empty_cit = client.post("/api/v1/recommend", json=payload)
            assert r_empty_cit.status_code == 200
            for rec in r_empty_cit.json()["recommendations"]:
                assert rec["citations"] == []

            app.dependency_overrides[get_citation_provider] = lambda: PartialCitationProvider()
            r_partial = client.post("/ai/recommend", json=payload)
            assert r_partial.status_code == 200
            recs = r_partial.json()["recommendations"]
            assert [r["policy_id"] for r in recs] == [ids["tie_a"], ids["tie_b"]]
            assert recs[0]["citations"] == [
                {
                    "chunk_id": f"chk-{ids['tie_a_v_cur']}",
                    "page": 3,
                    "clause": "Section 1.2 Coverage",
                }
            ]
            assert recs[1]["citations"] == []

        verify_openapi_contract()
        assert len(FEATURE_COLUMNS) == 59
    finally:
        eng.dispose()
