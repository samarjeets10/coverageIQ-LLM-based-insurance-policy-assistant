"""Stage 10A — Post-Integration Verification & M5 Completion Tests.

Developer: Shravan Sambhaji Kapare (Roll No. 203)
Module: M5 — Policy Recommendation
Covers:
1. Realistic 003-shaped multi-category catalog integration across Health, Car, Life, Term
   (including NULL claim_settlement_ratio / ped_waiting_period_months, DDL tenure defaults,
   policy_version_id propagation, private upload isolation, and current-version matching).
2. Real-catalog malformed metadata isolation (NaN strings, unparseable numbers, inverted bounds)
   failing closed per candidate without crashing valid catalog recommendations.
3. Duplicate policy_id protection in pure filter_policies() and policy_type SQL pushdown in
   filter_policies_from_db().
4. M3/M8 CitationItem Pydantic model interoperability, malformed citation rejection, and external
   TimeoutError / OSError isolation.
5. Security & traceability hardening: X-Request-ID echo on 401/500/501/503/200, TrustedScope role
   normalization, auth failure telemetry without secret leaks, and log scrubbing of scope/budget.
6. Ten-profile multi-category sanity and determinism verification across LightGBM and Weighted.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, SecretStr
from sqlalchemy import Boolean, Column, Float, Integer, MetaData, String, Table, create_engine, text
from sqlalchemy.pool import StaticPool

from ai_service.common.logging import sanitize_log_value
from ai_service.main import create_app
from ai_service.recommendation.citations import PolicyCitationRequest
from ai_service.recommendation.db import get_db_connection
from ai_service.recommendation.filters import (
    DuplicateCatalogPolicyError,
    filter_policies,
    filter_policies_from_db,
)
from ai_service.recommendation.router import get_citation_provider
from ai_service.recommendation.schemas import RecommendRequest, TrustedScope, UserProfileRequest
from ai_service.recommendation.settings import Settings


def _create_full_catalog_sqlite_db() -> tuple[Any, dict[str, str]]:
    """Create SQLite catalog DB matching 001 + 003 schema semantics across 4 categories."""
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
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    m.create_all(eng)

    org_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    ids = {
        "health_active": str(uuid.uuid4()),
        "health_v_old": str(uuid.uuid4()),
        "health_v_cur": str(uuid.uuid4()),
        "car_active": str(uuid.uuid4()),
        "car_v_cur": str(uuid.uuid4()),
        "life_active": str(uuid.uuid4()),
        "life_v_cur": str(uuid.uuid4()),
        "term_active": str(uuid.uuid4()),
        "term_v_cur": str(uuid.uuid4()),
        "private_user": str(uuid.uuid4()),
        "private_v_cur": str(uuid.uuid4()),
    }

    with eng.begin() as conn:
        policies_rows = [
            (
                ids["health_active"],
                None,
                org_id,
                "Real Health Plus",
                "Insurer H",
                "health",
                "active",
                ids["health_v_cur"],
                1,
            ),
            (
                ids["car_active"],
                None,
                org_id,
                "Real Motor Protect",
                "Insurer C",
                "car",
                "active",
                ids["car_v_cur"],
                2,
            ),
            (
                ids["life_active"],
                None,
                org_id,
                "Real Life Endowment",
                "Insurer L",
                "life",
                "active",
                ids["life_v_cur"],
                3,
            ),
            (
                ids["term_active"],
                None,
                org_id,
                "Real Term Shield",
                "Insurer T",
                "term",
                "active",
                ids["term_v_cur"],
                4,
            ),
            (
                ids["private_user"],
                user_id,
                None,
                "Private User Health PDF",
                "Insurer P",
                "health",
                "active",
                ids["private_v_cur"],
                5,
            ),
        ]
        for pid, uid, oid, name, ins, ptype, st, cvid, created in policies_rows:
            conn.execute(
                text(
                    "INSERT INTO policies (id, owner_user_id, organization_id, name, insurer_name, "
                    "policy_type, description, status, current_version_id, created_at) "
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

        attrs_rows = [
            # Stale health version (must never surface)
            (
                ids["health_active"],
                ids["health_v_old"],
                1,
                "both",
                4,
                25000.0,
                32000.0,
                500000.0,
                1500000.0,
                1,
                None,
                1,
                3,
                '["cashless_hospitalization"]',
                None,
                "{}",
            ),
            # Current health version with NULL CSR & NULL ped_waiting_period_months
            (
                ids["health_active"],
                ids["health_v_cur"],
                1,
                "both",
                4,
                12000.0,
                18000.0,
                500000.0,
                1500000.0,
                1,
                None,
                1,
                3,
                '["cashless_hospitalization", "daycare_procedures"]',
                None,
                '{"family_floater_supported": true}',
            ),
            # Car policy
            (
                ids["car_active"],
                ids["car_v_cur"],
                1,
                "individual",
                1,
                9000.0,
                14000.0,
                300000.0,
                1200000.0,
                0,
                None,
                1,
                3,
                '["third_party_liability", "own_damage"]',
                94.5,
                json.dumps(
                    {
                        "vehicle_types": ["hatchback", "sedan"],
                        "vehicle_age_limit_years": 10,
                        "own_damage_coverage": True,
                        "third_party_coverage": True,
                        "zero_depreciation": True,
                    }
                ),
            ),
            # Life policy with 003 DDL default min_tenure=1, max_tenure=3 while JSONB has 10..30!
            (
                ids["life_active"],
                ids["life_v_cur"],
                1,
                "individual",
                1,
                22000.0,
                30000.0,
                1000000.0,
                5000000.0,
                0,
                None,
                1,
                3,
                '["death_benefit", "maturity_benefit"]',
                98.2,
                json.dumps(
                    {
                        "life_cover_min": 1000000,
                        "life_cover_max": 5000000,
                        "policy_term_years_min": 10,
                        "policy_term_years_max": 30,
                        "premium_payment_term_years": 15,
                        "maturity_benefit": True,
                        "surrender_value_available": True,
                    }
                ),
            ),
            # Term policy with 003 DDL default min_tenure=1, max_tenure=3 while JSONB has 10..40!
            (
                ids["term_active"],
                ids["term_v_cur"],
                1,
                "individual",
                1,
                10000.0,
                16000.0,
                5000000.0,
                15000000.0,
                0,
                None,
                1,
                3,
                '["pure_term_life_cover", "critical_illness_rider"]',
                99.1,
                json.dumps(
                    {
                        "life_cover_min": 5000000,
                        "life_cover_max": 15000000,
                        "policy_term_years_min": 10,
                        "policy_term_years_max": 40,
                        "premium_payment_term_years": 20,
                        "smoking_status_eligibility": "all",
                        "critical_illness_rider": True,
                        "accidental_death_rider": True,
                    }
                ),
            ),
            # Private user policy (must ALWAYS stay excluded)
            (
                ids["private_user"],
                ids["private_v_cur"],
                1,
                "both",
                4,
                5000.0,
                8000.0,
                500000.0,
                2000000.0,
                1,
                12,
                1,
                3,
                '["cashless_hospitalization"]',
                99.9,
                "{}",
            ),
        ]
        for (
            pid,
            pvid,
            elig,
            mode,
            max_mem,
            pmin,
            pmax,
            smin,
            smax,
            ped_cov,
            ped_m,
            tmin,
            tmax,
            benefits,
            csr,
            meta,
        ) in attrs_rows:
            conn.execute(
                text(
                    "INSERT INTO policy_attributes VALUES (:aid, :pid, :pvid, :elig, 18, 65, "
                    'NULL, 1, :mode, :max_mem, \'["salaried","self_employed"]\', '
                    ":pmin, :pmax, :smin, :smax, 0, 0, 'no_limit', NULL, 30, :ped_cov, "
                    ":ped_m, 24, NULL, 7500, 1, 1, '[]', :tmin, :tmax, :benefits, '[]', "
                    ":csr, 10.0, 1, :meta, 'ingestion_llm')"
                ),
                {
                    "aid": str(uuid.uuid4()),
                    "pid": pid,
                    "pvid": pvid,
                    "elig": elig,
                    "mode": mode,
                    "max_mem": max_mem,
                    "pmin": pmin,
                    "pmax": pmax,
                    "smin": smin,
                    "smax": smax,
                    "ped_cov": ped_cov,
                    "ped_m": ped_m,
                    "tmin": tmin,
                    "tmax": tmax,
                    "benefits": benefits,
                    "csr": csr,
                    "meta": meta,
                },
            )
    return eng, ids


def _catalog_dict(
    *,
    policy_id: str | None = None,
    policy_type: str = "health",
    **overrides: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "policy_id": policy_id or str(uuid.uuid4()),
        "organization_id": uuid.uuid4(),
        "policy_name": f"Stage10A {policy_type.title()} Policy",
        "insurer_name": "CoverageIQ Shield",
        "policy_type": policy_type,
        "description": "Realistic fixture row.",
        "policy_status": "active",
        "current_version_id": uuid.uuid4(),
        "policy_version_id": uuid.uuid4(),
        "is_catalog_eligible": True,
        "min_entry_age": 18,
        "max_entry_age": 65,
        "max_renewal_age": None,
        "lifelong_renewal": True,
        "coverage_mode": "both",
        "max_family_members": 4,
        "eligible_applicant_types": ["salaried", "self_employed", "student", "senior_citizen"],
        "annual_premium_min": Decimal("11000.00"),
        "annual_premium_max": Decimal("17000.00"),
        "sum_insured_min": Decimal("500000.00"),
        "sum_insured_max": Decimal("1500000.00"),
        "deductible_amount": Decimal("0.00"),
        "copay_pct": Decimal("0.00"),
        "room_rent_limit_type": "no_limit",
        "room_rent_limit_value": None,
        "initial_waiting_period_days": 30,
        "ped_covered": True,
        "ped_waiting_period_months": 24,
        "specific_disease_waiting_months": 24,
        "maternity_waiting_months": None,
        "network_hospitals_count": 7500,
        "cashless_available": True,
        "pan_india_coverage": True,
        "available_regions": [],
        "min_tenure_years": 1,
        "max_tenure_years": 3,
        "covered_benefits": ["cashless_hospitalization", "daycare_procedures"],
        "exclusions": ["cosmetic_surgery"],
        "claim_settlement_ratio": Decimal("96.50"),
        "no_claim_bonus_pct": Decimal("20.00"),
        "restoration_benefit": True,
        "type_specific_metadata": {},
        "extraction_source": "verified",
    }
    row.update(overrides)
    return row


def test_real_003_shaped_multi_category_catalog_roundtrip_and_version_id() -> None:
    """Real 003-shaped DB rows across 4 categories work end-to-end with policy_version_id."""
    eng, ids = _create_full_catalog_sqlite_db()
    try:
        app = create_app(Settings(_env_file=None))

        def _override_conn():
            with eng.connect() as conn:
                yield conn

        app.dependency_overrides[get_db_connection] = _override_conn

        queries = [
            (
                "health",
                {
                    "policy_type": "health",
                    "age": 34,
                    "budget_max": 25000,
                    "min_sum_insured": 500000,
                    "pre_existing_conditions": ["diabetes"],
                    "tenure_years": 2,
                },
                ids["health_active"],
                ids["health_v_cur"],
            ),
            (
                "car",
                {
                    "policy_type": "car",
                    "age": 29,
                    "budget_max": 18000,
                    "min_sum_insured": 500000,
                    "vehicle_type": "sedan",
                    "vehicle_age_years": 3,
                    "require_own_damage": True,
                },
                ids["car_active"],
                ids["car_v_cur"],
            ),
            (
                # Validates 003 DDL tenure default (1..3) is cleanly overridden by JSONB 10..30!
                "life",
                {
                    "policy_type": "life",
                    "age": 35,
                    "budget_max": 35000,
                    "min_sum_insured": 2500000,
                    "tenure_years": 20,
                    "require_maturity_benefit": True,
                },
                ids["life_active"],
                ids["life_v_cur"],
            ),
            (
                # Validates 003 DDL tenure default (1..3) is cleanly overridden by JSONB 10..40!
                "term",
                {
                    "policy_type": "term",
                    "age": 31,
                    "budget_max": 20000,
                    "min_sum_insured": 10000000,
                    "tenure_years": 25,
                    "smoking_status": "smoker",
                    "require_critical_illness_rider": True,
                },
                ids["term_active"],
                ids["term_v_cur"],
            ),
        ]

        with TestClient(app) as client:
            for backend in ("auto", "weighted"):
                for _cat, profile_data, expected_pid, expected_vid in queries:
                    req = {
                        "profile": profile_data,
                        "ranking_model": backend,
                        "top_n": 5,
                    }
                    res = client.post(
                        "/api/v1/recommend",
                        json=req,
                        headers={"X-Request-ID": f"stage10a-{_cat}-{backend}"},
                    )
                    assert res.status_code == 200
                    body = res.json()
                    assert body["status"] == "RECOMMENDATIONS_READY"
                    rec_ids = [r["policy_id"] for r in body["recommendations"]]
                    assert rec_ids == [expected_pid]
                    assert body["recommendations"][0]["policy_version_id"] == expected_vid
                    assert ids["private_user"] not in rec_ids
    finally:
        eng.dispose()


def test_malformed_catalog_row_isolated_fail_closed_while_valid_policies_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One corrupt catalog row fails closed cleanly while remaining valid rows rank normally."""
    valid_health = _catalog_dict(policy_id="valid-health-1")
    corrupt_nan_prem = _catalog_dict(
        policy_id="corrupt-nan-prem",
        annual_premium_min="NaN",
    )
    corrupt_str_age = _catalog_dict(
        policy_id="corrupt-str-age",
        min_entry_age="unknown",
    )
    corrupt_inverted_bounds = _catalog_dict(
        policy_id="corrupt-inverted",
        min_entry_age=60,
        max_entry_age=25,
        annual_premium_min=30000.0,
        annual_premium_max=10000.0,
    )
    corrupt_negative_sum = _catalog_dict(
        policy_id="corrupt-neg-sum",
        sum_insured_max=-500000.0,
    )
    catalog = [
        corrupt_nan_prem,
        corrupt_str_age,
        corrupt_inverted_bounds,
        corrupt_negative_sum,
        valid_health,
    ]

    app = create_app(Settings(_env_file=None))
    app.dependency_overrides[get_db_connection] = lambda: (yield MagicMock())
    monkeypatch.setattr(
        "ai_service.recommendation.service.load_catalog_policies_from_db",
        lambda _c, policy_type=None, active_only=False: catalog,
    )

    payload = {
        "profile": {
            "policy_type": "health",
            "age": 32,
            "budget_max": 25000,
            "min_sum_insured": 500000,
        }
    }
    with TestClient(app) as client:
        res = client.post("/api/v1/recommend", json=payload)
        assert res.status_code == 200
        body = res.json()
        assert [r["policy_id"] for r in body["recommendations"]] == ["valid-health-1"]
        assert body["filter_report"]["total_candidates"] == 5
        assert body["filter_report"]["passed_count"] == 1
        assert body["filter_report"]["eliminated_count"] == 4

    # Corrupt car metadata string inside JSONB also fails closed per policy without 500 crash
    car_corrupt_meta = _catalog_dict(
        policy_id="car-bad-meta",
        policy_type="car",
        type_specific_metadata={"vehicle_types": ["sedan"], "vehicle_age_limit_years": "ten"},
    )
    car_valid = _catalog_dict(
        policy_id="car-good-meta",
        policy_type="car",
        type_specific_metadata={"vehicle_types": ["sedan"], "vehicle_age_limit_years": 10},
    )
    car_prof = UserProfileRequest(
        policy_type="car",
        age=30,
        budget_max=25000,
        min_sum_insured=500000,
        vehicle_type="sedan",
        vehicle_age_years=4,
    )
    car_filter = filter_policies(car_prof, [car_corrupt_meta, car_valid])
    assert [p["policy_id"] for p in car_filter.eligible_policies] == ["car-good-meta"]
    assert car_filter.filter_report.eliminations[0].rule_id == "CAR_VEHICLE_AGE_MAX"


def test_filter_policies_duplicate_id_guard_and_db_pushdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pure filter_policies rejects duplicate policy_id; filter_policies_from_db pushes type."""
    prof = UserProfileRequest(
        policy_type="health",
        age=32,
        budget_max=25000,
        min_sum_insured=500000,
    )
    dup_pol = _catalog_dict(policy_id="dup-10a")
    with pytest.raises(DuplicateCatalogPolicyError, match="dup-10a"):
        filter_policies(prof, [dup_pol, dict(dup_pol)])

    # HTTP service handles duplicate passed via monkeypatched loader as controlled 500 + request_id
    app = create_app(Settings(_env_file=None))
    app.dependency_overrides[get_db_connection] = lambda: (yield MagicMock())
    monkeypatch.setattr(
        "ai_service.recommendation.service.load_catalog_policies_from_db",
        lambda _c, policy_type=None, active_only=False: [dup_pol, dict(dup_pol)],
    )
    with TestClient(app) as client:
        res = client.post(
            "/api/v1/recommend",
            json={
                "policy_type": "health",
                "age": 32,
                "budget_max": 25000,
                "min_sum_insured": 500000,
            },
            headers={"X-Request-ID": "req-dup-check"},
        )
        assert res.status_code == 500
        assert res.headers["X-Request-ID"] == "req-dup-check"

    # filter_policies_from_db passes policy_type down to load_catalog_policies_from_db
    seen_kwargs: dict[str, Any] = {}

    def _spy_loader(_conn: Any, **kwargs: Any) -> list[dict[str, Any]]:
        seen_kwargs.update(kwargs)
        return []

    monkeypatch.setattr(
        "ai_service.recommendation.filters.load_catalog_policies_from_db", _spy_loader
    )
    out = filter_policies_from_db(MagicMock(), prof)
    assert out.status == "NO_MATCHING_POLICIES"
    assert seen_kwargs["policy_type"] == "health"
    assert seen_kwargs["active_only"] is False


def test_citation_pydantic_interop_validation_and_timeout_isolation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pydantic CitationItem models normalize cleanly; invalid citations or timeouts return 500."""

    class FakeCitationItem(BaseModel):
        chunk_id: str
        page: int
        clause: str
        category: str = "Coverage"

    class PydanticProvider:
        def get_citations(
            self,
            requests: list[PolicyCitationRequest] | Any,
            *,
            request_id: str | None = None,
        ) -> dict[str, Any]:
            del request_id
            return {
                r.policy_id: [FakeCitationItem(chunk_id="chk-10a", page=7, clause="Sec 2.3")]
                for r in requests
            }

    catalog = [_catalog_dict(policy_id="cit-pol-1")]
    app = create_app(Settings(_env_file=None))
    app.dependency_overrides[get_db_connection] = lambda: (yield MagicMock())
    app.dependency_overrides[get_citation_provider] = lambda: PydanticProvider()
    monkeypatch.setattr(
        "ai_service.recommendation.service.load_catalog_policies_from_db",
        lambda _c, policy_type=None, active_only=False: catalog,
    )

    payload = {
        "profile": {
            "policy_type": "health",
            "age": 32,
            "budget_max": 25000,
            "min_sum_insured": 500000,
        }
    }

    with TestClient(app) as client:
        ok = client.post("/api/v1/recommend", json=payload, headers={"X-Request-ID": "cit-10a-ok"})
        assert ok.status_code == 200
        assert ok.json()["recommendations"][0]["citations"] == [
            {"chunk_id": "chk-10a", "page": 7, "clause": "Sec 2.3", "category": "Coverage"}
        ]

        # Ungrounded dict missing clause/page -> 500 + X-Request-ID
        class IncompleteCitationProvider:
            def get_citations(self, requests: Any, *, request_id: str | None = None) -> Any:
                del request_id
                return {requests[0].policy_id: [{"chunk_id": "", "page": 0, "clause": ""}]}

        app.dependency_overrides[get_citation_provider] = lambda: IncompleteCitationProvider()
        bad = client.post(
            "/api/v1/recommend", json=payload, headers={"X-Request-ID": "cit-10a-bad"}
        )
        assert bad.status_code == 500
        assert bad.headers["X-Request-ID"] == "cit-10a-bad"

        # TimeoutError (OSError subclass) logged cleanly as recommendation.citation_error
        class TimeoutCitationProvider:
            def get_citations(self, requests: Any, *, request_id: str | None = None) -> Any:
                del requests, request_id
                raise TimeoutError("M3 service timed out")

        app.dependency_overrides[get_citation_provider] = lambda: TimeoutCitationProvider()
        with caplog.at_level(logging.ERROR, logger="ai_service.recommendation.service"):
            timed = client.post(
                "/api/v1/recommend",
                json=payload,
                headers={"X-Request-ID": "cit-10a-timeout"},
            )
            assert timed.status_code == 500
            assert timed.headers["X-Request-ID"] == "cit-10a-timeout"
        assert any(
            getattr(rec, "event", None) == "recommendation.citation_error" for rec in caplog.records
        )


def test_security_request_id_echo_scope_normalization_and_scrubbing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify X-Request-ID echoed on 401/501/503, scope case-normalized, and scope scrubbed."""
    cfg = Settings(
        _env_file=None,
        ai_service_env="production",
        ai_service_token=SecretStr("prod-secret-token-10a"),
    )
    app = create_app(cfg)
    app.dependency_overrides[get_db_connection] = lambda: (yield MagicMock())
    monkeypatch.setattr(
        "ai_service.recommendation.service.load_catalog_policies_from_db",
        lambda _c, policy_type=None, active_only=False: [],
    )

    scrubbed = sanitize_log_value(
        {
            "scope": {"user_id": "u-1", "role": "user"},
            "monthly_budget_inr": 2500,
            "monthly_budget": 2500,
            "scope_present": True,
            "scope_role": "user",
        }
    )
    assert scrubbed["scope"] == "[REDACTED]"
    assert scrubbed["monthly_budget_inr"] == "[REDACTED]"
    assert scrubbed["monthly_budget"] == "[REDACTED]"
    assert scrubbed["scope_present"] is True
    assert scrubbed["scope_role"] == "user"

    # TrustedScope normalizes role case while rejecting invalid roles
    scope_norm = TrustedScope(user_id=" u-1 ", role=" Organization ")  # type: ignore[arg-type]
    assert scope_norm.role == "organization"
    assert scope_norm.user_id == "u-1"

    payload = {
        "profile": {
            "policy_type": "health",
            "age": 32,
            "budget_max": 25000,
            "min_sum_insured": 500000,
        },
        "scope": {"user_id": "u-1", "role": "Organization"},
        "ranking_model": "xgboost",
    }
    with TestClient(app) as client:
        with caplog.at_level(logging.WARNING, logger="ai_service.recommendation.router"):
            unauth = client.post(
                "/api/v1/recommend",
                json=payload,
                headers={"X-Request-ID": "req-401-trace", "X-Service-Token": "wrong"},
            )
            assert unauth.status_code == 401
            assert unauth.headers["X-Request-ID"] == "req-401-trace"
            assert "wrong" not in caplog.text
            assert "prod-secret-token-10a" not in caplog.text

        unimpl = client.post(
            "/api/v1/recommend",
            json=payload,
            headers={
                "X-Request-ID": "req-501-trace",
                "X-Service-Token": "prod-secret-token-10a",
            },
        )
        assert unimpl.status_code == 501
        assert unimpl.headers["X-Request-ID"] == "req-501-trace"


def test_ten_diverse_profiles_sanity_and_batch_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate 10 diverse realistic profiles across 4 categories against 50-policy catalog."""
    catalog: list[dict[str, Any]] = []
    for i in range(15):
        catalog.append(
            _catalog_dict(
                policy_id=f"hlt-{i:02d}",
                policy_type="health",
                annual_premium_min=Decimal(str(8000 + i * 1500)),
                annual_premium_max=Decimal(str(12000 + i * 1500)),
                sum_insured_max=Decimal(str(500000 + (i % 5) * 500000)),
                coverage_mode="individual" if i % 4 == 0 else "both",
                max_family_members=1 if i % 4 == 0 else 5,
                ped_covered=(i % 3 != 0),
                ped_waiting_period_months=24 if i % 3 != 0 else None,
            )
        )
    for i in range(12):
        catalog.append(
            _catalog_dict(
                policy_id=f"car-{i:02d}",
                policy_type="car",
                annual_premium_min=Decimal(str(6000 + i * 1200)),
                annual_premium_max=Decimal(str(10000 + i * 1200)),
                sum_insured_max=Decimal(str(400000 + (i % 4) * 300000)),
                coverage_mode="individual",
                max_family_members=1,
                type_specific_metadata={
                    "vehicle_types": ["two_wheeler"] if i == 0 else ["hatchback", "sedan", "suv"],
                    "vehicle_age_limit_years": 5 + (i % 3) * 5,
                    "own_damage_coverage": True,
                    "third_party_coverage": True,
                    "zero_depreciation": i % 2 == 0,
                },
            )
        )
    for i in range(12):
        catalog.append(
            _catalog_dict(
                policy_id=f"lif-{i:02d}",
                policy_type="life",
                annual_premium_min=Decimal(str(15000 + i * 2500)),
                annual_premium_max=Decimal(str(22000 + i * 2500)),
                sum_insured_max=Decimal(str(1000000 + (i % 4) * 1500000)),
                coverage_mode="individual",
                max_family_members=1,
                type_specific_metadata={
                    "life_cover_min": 500000,
                    "life_cover_max": 1000000 + (i % 4) * 1500000,
                    "policy_term_years_min": 5,
                    "policy_term_years_max": 30,
                    "premium_payment_term_years": 10 + (i % 3) * 5,
                    "maturity_benefit": i % 2 == 0,
                    "surrender_value_available": True,
                },
            )
        )
    for i in range(11):
        catalog.append(
            _catalog_dict(
                policy_id=f"trm-{i:02d}",
                policy_type="term",
                annual_premium_min=Decimal(str(7000 + i * 1000)),
                annual_premium_max=Decimal(str(11000 + i * 1000)),
                sum_insured_max=Decimal(str(5000000 + (i % 4) * 5000000)),
                coverage_mode="individual",
                max_family_members=1,
                type_specific_metadata={
                    "life_cover_min": 2500000,
                    "life_cover_max": 5000000 + (i % 4) * 5000000,
                    "policy_term_years_min": 10,
                    "policy_term_years_max": 40,
                    "premium_payment_term_years": 20,
                    "smoking_status_eligibility": "non_smoker_only" if i % 3 == 0 else "all",
                    "critical_illness_rider": i % 2 == 0,
                    "accidental_death_rider": True,
                },
            )
        )

    app = create_app(Settings(_env_file=None))
    app.dependency_overrides[get_db_connection] = lambda: (yield MagicMock())
    monkeypatch.setattr(
        "ai_service.recommendation.service.load_catalog_policies_from_db",
        lambda _c, policy_type=None, active_only=False: [
            row
            for row in catalog
            if policy_type is None or str(row["policy_type"]).lower() == policy_type
        ],
    )

    ten_profiles = [
        {
            "policy_type": "health",
            "age": 30,
            "budget_max": 22000,
            "min_sum_insured": 500000,
            "risk_preference": "balanced",
            "annual_income_inr": 1200000,
        },
        {
            "policy_type": "health",
            "age": 38,
            "budget_max": 30000,
            "min_sum_insured": 1000000,
            "coverage_mode": "family_floater",
            "family_size": 4,
            "pre_existing_conditions": ["diabetes"],
            "risk_preference": "comprehensive_coverage",
        },
        {
            "policy_type": "health",
            "age": 62,
            "budget_max": 35000,
            "min_sum_insured": 500000,
            "applicant_type": "senior_citizen",
        },
        {
            "policy_type": "car",
            "age": 28,
            "budget_max": 18000,
            "min_sum_insured": 400000,
            "vehicle_type": "sedan",
            "vehicle_age_years": 2,
            "require_zero_depreciation": True,
        },
        {
            "policy_type": "car",
            "age": 24,
            "budget_max": 12000,
            "min_sum_insured": 400000,
            "vehicle_type": "two_wheeler",
            "vehicle_age_years": 4,
        },
        {
            "policy_type": "life",
            "age": 36,
            "budget_max": 45000,
            "min_sum_insured": 1500000,
            "tenure_years": 20,
            "require_maturity_benefit": True,
        },
        {
            "policy_type": "life",
            "age": 42,
            "budget_max": 50000,
            "min_sum_insured": 2500000,
            "tenure_years": 15,
            "max_premium_payment_term_years": 15,
        },
        {
            "policy_type": "term",
            "age": 31,
            "budget_max": 20000,
            "min_sum_insured": 10000000,
            "tenure_years": 30,
            "smoking_status": "smoker",
        },
        {
            "policy_type": "term",
            "age": 29,
            "budget_max": 25000,
            "min_sum_insured": 10000000,
            "tenure_years": 25,
            "smoking_status": "non_smoker",
            "require_critical_illness_rider": True,
        },
        # Strict zero-match case across 50-policy catalog
        {"policy_type": "health", "age": 35, "budget_max": 500, "min_sum_insured": 50000000},
    ]

    with TestClient(app) as client:
        for backend in ("lightgbm", "weighted"):
            t_start = time.perf_counter()
            for idx, prof in enumerate(ten_profiles):
                req_model = RecommendRequest.model_validate(
                    {"profile": prof, "ranking_model": backend, "top_n": 3}
                )
                r1 = client.post("/api/v1/recommend", json=req_model.model_dump())
                r2 = client.post("/api/v1/recommend", json=req_model.model_dump())
                assert r1.status_code == 200
                b1, b2 = r1.json(), r2.json()
                assert b1["recommendations"] == b2["recommendations"]
                if idx < 9:
                    assert b1["status"] == "RECOMMENDATIONS_READY"
                    assert 1 <= len(b1["recommendations"]) <= 3
                    scores = [rec["score"] for rec in b1["recommendations"]]
                    assert scores == sorted(scores, reverse=True)
                else:
                    assert b1["status"] == "NO_MATCHING_POLICIES"
                    assert b1["recommendations"] == []
                    assert b1["filter_report"]["eliminated_count"] == 15
            elapsed = time.perf_counter() - t_start
            assert elapsed < 5.0
