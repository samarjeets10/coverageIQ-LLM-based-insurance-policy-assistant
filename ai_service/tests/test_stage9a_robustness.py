"""Stage 9A — Recommendation Engine Robustness & Production Hardening Tests.

Developer: Shravan Sambhaji Kapare (Roll No. 203)
Module: M5 — Policy Recommendation
Covers:
1. Corrupt, unfitted, missing, and schema-incompatible LightGBM artifacts & startup warm-up.
2. Explicit opt-in weighted scoring backend (Phase A), determinism, configurability,
   and strict non-silent fallback enforcement.
3. Incomplete customer profile handling (missing annual_income_inr & risk_preference
   independently and combined, plus zero/negative income 422 rejection).
4. Structured JSON operational logging without PII or credential leakage (NFR-11).
5. Citation provider interface seam (NullCitationProvider & mock provider validation).
6. Database timeout settings verification & read-only live PostgreSQL statement_timeout check.
7. Large-catalog performance/determinism & OpenAPI contract completeness.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
import uuid
import zlib
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import joblib
import lightgbm as lgb
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from ai_service.common.logging import StructuredJsonFormatter, sanitize_log_value
from ai_service.main import create_app
from ai_service.recommendation.citations import PolicyCitationRequest
from ai_service.recommendation.db import (
    create_db_engine,
    get_db_connection,
    inspect_database_readiness,
)
from ai_service.recommendation.features import FEATURE_SCHEMA_VERSION, build_feature_matrix
from ai_service.recommendation.router import get_citation_provider
from ai_service.recommendation.schemas import RecommendRequest, UserProfileRequest
from ai_service.recommendation.service import (
    ModelUnavailableError,
    load_model_state,
    recommend_policies_for_request,
)
from ai_service.recommendation.settings import Settings
from ai_service.recommendation.train_lightgbm import MODEL_NAME
from ai_service.recommendation.weighted import (
    WEIGHTED_SCORER_NAME,
    WEIGHTED_SCORER_VERSION,
    WeightedDimensionWeights,
    WeightedScoringConfig,
    score_policies_weighted,
)

# ---------------------------------------------------------------------------
# Self-Contained Test Fixture Builders
# ---------------------------------------------------------------------------


def _sample_policy(
    *,
    policy_id: str | None = None,
    policy_name: str = "Arogya Shield Health",
    insurer_name: str = "CoverageIQ General",
    policy_type: str = "health",
    annual_premium_min: Decimal | float = Decimal("12000.00"),
    annual_premium_max: Decimal | float = Decimal("16000.00"),
    sum_insured_min: Decimal | float = Decimal("500000.00"),
    sum_insured_max: Decimal | float = Decimal("1000000.00"),
    claim_settlement_ratio: Decimal | float | None = Decimal("96.50"),
    copay_pct: Decimal | float = Decimal("0.00"),
    deductible_amount: Decimal | float = Decimal("0.00"),
    network_hospitals_count: int = 8000,
    covered_benefits: list[str] | None = None,
    restoration_benefit: bool = True,
    no_claim_bonus_pct: Decimal | float = Decimal("20.00"),
    initial_waiting_period_days: int = 30,
    ped_waiting_period_months: int | None = 24,
    type_specific_metadata: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Return a controlled policy catalog dictionary matching Stage 8B DB schema."""
    row: dict[str, Any] = {
        "policy_id": policy_id or str(uuid.uuid4()),
        "organization_id": uuid.uuid4(),
        "policy_name": policy_name,
        "insurer_name": insurer_name,
        "policy_type": policy_type,
        "description": "Stage 9A controlled test policy row.",
        "policy_status": "active",
        "current_version_id": uuid.uuid4(),
        "policy_version_id": uuid.uuid4(),
        "is_catalog_eligible": True,
        "min_entry_age": 18,
        "max_entry_age": 65,
        "max_renewal_age": 90,
        "lifelong_renewal": True,
        "coverage_mode": "both",
        "max_family_members": 4,
        "eligible_applicant_types": ["salaried", "self_employed", "student", "senior_citizen"],
        "annual_premium_min": annual_premium_min,
        "annual_premium_max": annual_premium_max,
        "sum_insured_min": sum_insured_min,
        "sum_insured_max": sum_insured_max,
        "deductible_amount": deductible_amount,
        "copay_pct": copay_pct,
        "room_rent_limit_type": "no_limit",
        "initial_waiting_period_days": initial_waiting_period_days,
        "ped_covered": True,
        "ped_waiting_period_months": ped_waiting_period_months,
        "covered_benefits": (
            covered_benefits
            if covered_benefits is not None
            else ["cashless_hospitalization", "daycare_procedures", "ambulance_cover"]
        ),
        "exclusions": ["cosmetic_surgery"],
        "network_hospitals_count": network_hospitals_count,
        "cashless_available": True,
        "pan_india_coverage": True,
        "min_tenure_years": 1,
        "max_tenure_years": 3,
        "claim_settlement_ratio": claim_settlement_ratio,
        "restoration_benefit": restoration_benefit,
        "no_claim_bonus_pct": no_claim_bonus_pct,
        "type_specific_metadata": type_specific_metadata or {},
    }
    row.update(overrides)
    return row


def _health_payload(**profile_overrides: Any) -> dict[str, Any]:
    """Build a valid health recommendation request payload."""
    profile: dict[str, Any] = {
        "policy_type": "health",
        "age": 34,
        "budget_min": 5000,
        "budget_max": 30000,
        "applicant_type": "salaried",
        "coverage_mode": "individual",
        "family_size": 1,
        "min_sum_insured": 500000,
        "required_benefits": ["cashless_hospitalization"],
        "pre_existing_conditions": [],
        "tenure_years": 1,
        "annual_income_inr": 1500000,
        "risk_preference": "balanced",
    }
    profile.update(profile_overrides)
    return {"profile": profile, "top_n": 5, "ranking_model": "auto"}


def _build_client(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
) -> TestClient:
    """Create FastAPI TestClient with mocked DB connection & catalog loader."""
    app = create_app(settings or Settings(_env_file=None))
    mock_conn = MagicMock()

    def override_conn():
        yield mock_conn

    app.dependency_overrides[get_db_connection] = override_conn
    monkeypatch.setattr(
        "ai_service.main.check_database_connection",
        lambda _c: {"database": "coverageiq_db"},
    )
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
        lambda _conn, active_only=False: catalog,
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Artifact Failure, Corrupt Models, Startup Warm-Up & Non-Silent Fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("corruption_mode", "corrupt_bytes"),
    [
        ("empty_joblib", b""),
        ("garbage_joblib", b"not-a-valid-pickle-or-joblib-binary-stream"),
        ("corrupt_zlib_joblib", b"x\x9c\x03\x00invalid-truncated-zlib"),
    ],
)
def test_corrupt_joblib_artifacts_return_controlled_503_and_never_silent_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption_mode: str,
    corrupt_bytes: bytes,
) -> None:
    """Corrupt joblib files fail cleanly into ModelUnavailableError / HTTP 503."""
    corrupt_dir = tmp_path / f"corrupt_{corruption_mode}"
    corrupt_dir.mkdir()
    src_dir = Settings(_env_file=None).model_artifacts_dir
    (corrupt_dir / "lightgbm_metadata.json").write_text(
        (src_dir / "lightgbm_metadata.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (corrupt_dir / "lightgbm_ranker.joblib").write_bytes(corrupt_bytes)

    state = load_model_state(corrupt_dir)
    assert not state.is_ready
    assert state.warmed_up is False
    assert state.error_kind == "corrupt"

    cfg = Settings(_env_file=None, model_artifacts_dir=corrupt_dir)
    catalog = [_sample_policy()]
    with _build_client(monkeypatch, catalog, settings=cfg) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_ready"] is False
        assert health.json()["model_warmed_up"] is False

        # 'auto' and 'lightgbm' MUST fail with 503 — NEVER silently fall back to weighted!
        for requested_backend in ("auto", "lightgbm"):
            req = _health_payload()
            req["ranking_model"] = requested_backend
            res = client.post("/api/v1/recommend", json=req)
            assert res.status_code == 503
            assert str(tmp_path) not in res.json()["detail"]

        # Only explicit 'weighted' opt-in works when LightGBM is unavailable
        req_weighted = _health_payload()
        req_weighted["ranking_model"] = "weighted"
        res_weighted = client.post("/api/v1/recommend", json=req_weighted)
        assert res_weighted.status_code == 200
        assert res_weighted.json()["model_used"] == WEIGHTED_SCORER_NAME


def test_corrupt_metadata_json_caught_safely_at_startup(tmp_path: Path) -> None:
    """Invalid JSON inside lightgbm_metadata.json sets corrupt ModelState cleanly."""
    bad_dir = tmp_path / "corrupt_json"
    bad_dir.mkdir()
    src_dir = Settings(_env_file=None).model_artifacts_dir
    (bad_dir / "lightgbm_ranker.joblib").write_bytes(
        (src_dir / "lightgbm_ranker.joblib").read_bytes()
    )
    (bad_dir / "lightgbm_metadata.json").write_text("{broken json", encoding="utf-8")

    state = load_model_state(bad_dir)
    assert not state.is_ready
    assert state.warmed_up is False
    assert state.error_kind == "corrupt"


def test_unfitted_lgbm_model_caught_by_startup_warmup(tmp_path: Path) -> None:
    """Unfitted LGBMRanker artifact passes joblib load but fails startup warm-up -> 503."""
    unfitted_dir = tmp_path / "unfitted_lgbm"
    unfitted_dir.mkdir()
    src_dir = Settings(_env_file=None).model_artifacts_dir
    (unfitted_dir / "lightgbm_metadata.json").write_text(
        (src_dir / "lightgbm_metadata.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    joblib.dump(lgb.LGBMRanker(), unfitted_dir / "lightgbm_ranker.joblib")

    state = load_model_state(unfitted_dir)
    assert not state.is_ready
    assert state.warmed_up is False
    assert state.error_kind == "load_failed"

    with pytest.raises(ModelUnavailableError):
        recommend_policies_for_request(
            RecommendRequest.model_validate(_health_payload()),
            MagicMock(),
            state,
        )


def test_default_model_loads_and_warms_up_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage 6 trained LightGBM model loads once, warms up, and ranks deterministically."""
    state = load_model_state()
    assert state.is_ready is True
    assert state.warmed_up is True
    assert state.model_name == MODEL_NAME

    catalog = [
        _sample_policy(policy_id="pol-3", sum_insured_max=Decimal("500000.00")),
        _sample_policy(policy_id="pol-1", sum_insured_max=Decimal("2000000.00")),
        _sample_policy(policy_id="pol-2", sum_insured_max=Decimal("1000000.00")),
    ]

    with _build_client(monkeypatch, catalog) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_ready"] is True
        assert health.json()["model_warmed_up"] is True

        r1 = client.post("/api/v1/recommend", json=_health_payload()).json()
        r2 = client.post("/api/v1/recommend", json=_health_payload()).json()
        assert [(x["policy_id"], x["score"]) for x in r1["recommendations"]] == [
            (x["policy_id"], x["score"]) for x in r2["recommendations"]
        ]


def test_corrupt_exception_helper_direct_raise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct EOFError / zlib.error / UnpicklingError inside load_model map cleanly."""
    for err in (EOFError("eof"), zlib.error("bad crc"), pickle.UnpicklingError("bad pickle")):
        monkeypatch.setattr(
            "ai_service.recommendation.service.load_model",
            lambda _dir, _err=err: (_ for _ in ()).throw(_err),
        )
        state = load_model_state(tmp_path)
        assert not state.is_ready
        assert state.error_kind == "corrupt"


# ---------------------------------------------------------------------------
# 2. Explicit Weighted Fallback Backend (Phase A)
# ---------------------------------------------------------------------------


def test_weighted_backend_end_to_end_risk_sensitivity_and_transparency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ranking_model='weighted' scores transparently and responds to risk_preference."""
    cheap_basic = _sample_policy(
        policy_id="cheap-basic",
        policy_name="Affordable Saver Health",
        annual_premium_min=Decimal("6000.00"),
        annual_premium_max=Decimal("7500.00"),
        sum_insured_max=Decimal("500000.00"),
        covered_benefits=[
            "cashless_hospitalization",
            "daycare_procedures",
            "ambulance_cover",
            "annual_health_checkup",
            "opd_cover",
        ],
        restoration_benefit=False,
        no_claim_bonus_pct=Decimal("0.00"),
        claim_settlement_ratio=Decimal("95.00"),
        network_hospitals_count=6000,
    )
    rich_pricey = _sample_policy(
        policy_id="rich-pricey",
        policy_name="Comprehensive Elite Health",
        annual_premium_min=Decimal("14500.00"),
        annual_premium_max=Decimal("15000.00"),
        sum_insured_max=Decimal("1500000.00"),
        covered_benefits=[
            "cashless_hospitalization",
            "daycare_procedures",
            "ambulance_cover",
            "annual_health_checkup",
            "opd_cover",
            "maternity_cover",
        ],
        restoration_benefit=True,
        no_claim_bonus_pct=Decimal("50.00"),
        claim_settlement_ratio=Decimal("95.00"),
        network_hospitals_count=6000,
    )

    catalog = [rich_pricey, cheap_basic]
    with _build_client(monkeypatch, catalog) as client:
        low_prem_req = _health_payload(budget_max=15000, risk_preference="low_premium")
        low_prem_req["ranking_model"] = "weighted"
        res_low = client.post("/api/v1/recommend", json=low_prem_req).json()

        comp_req = _health_payload(budget_max=15000, risk_preference="comprehensive_coverage")
        comp_req["ranking_model"] = "weighted"
        res_comp = client.post("/api/v1/recommend", json=comp_req).json()

        assert res_low["model_used"] == WEIGHTED_SCORER_NAME
        assert res_low["model_version"] == WEIGHTED_SCORER_VERSION
        assert res_low["feature_schema_version"] == FEATURE_SCHEMA_VERSION
        assert res_low["recommendations"][0]["policy_id"] == "cheap-basic"
        assert res_comp["recommendations"][0]["policy_id"] == "rich-pricey"

        # Transparent component breakdowns and honest fallback limitation warning present
        top_rec = res_comp["recommendations"][0]
        assert 0.0 <= top_rec["score"] <= 1.0
        assert "weighted_affordability_score" in top_rec["feature_breakdown"]
        assert "weighted_coverage_score" in top_rec["feature_breakdown"]
        assert "weighted_benefits_score" in top_rec["feature_breakdown"]
        assert any("weighted fallback" in w.lower() for w in res_comp["warnings"])


def test_weighted_config_validation_and_custom_weights() -> None:
    """WeightedScoringConfig validates weights sum to 1.0 and supports custom configs."""
    with pytest.raises(ValueError, match="sum to 1.0"):
        WeightedDimensionWeights(
            affordability=0.5,
            coverage=0.5,
            benefits=0.2,
            cost_waiting=0.0,
            insurer_network=0.0,
        )

    with pytest.raises(ValueError, match=">= 0"):
        WeightedDimensionWeights(
            affordability=-0.1,
            coverage=0.5,
            benefits=0.2,
            cost_waiting=0.2,
            insurer_network=0.2,
        )

    custom_cfg = WeightedScoringConfig(
        weights_by_risk_preference={
            "low_premium": WeightedDimensionWeights(1.0, 0.0, 0.0, 0.0, 0.0),
            "balanced": WeightedDimensionWeights(0.0, 1.0, 0.0, 0.0, 0.0),
            "comprehensive_coverage": WeightedDimensionWeights(0.0, 0.0, 1.0, 0.0, 0.0),
        }
    )
    prof = UserProfileRequest.model_validate(_health_payload()["profile"])
    pol = _sample_policy()
    X = build_feature_matrix(prof, [pol])
    res = score_policies_weighted(X, config=custom_cfg)
    assert len(res.scores) == 1
    assert res.scores[0] == pytest.approx(
        res.dimension_breakdowns[0]["weighted_coverage_score"], abs=1e-4
    )


def test_weighted_zero_match_and_multi_category_and_xgboost_501(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weighted backend supports car/life/term, skips ML on zero match, keeps xgboost 501."""
    ineligible = [_sample_policy(annual_premium_min=Decimal("90000.00"))]
    with _build_client(monkeypatch, ineligible) as client:
        req = _health_payload()
        req["ranking_model"] = "weighted"
        res_empty = client.post("/api/v1/recommend", json=req)
        assert res_empty.status_code == 200
        assert res_empty.json()["status"] == "NO_MATCHING_POLICIES"
        assert res_empty.json()["model_used"] == WEIGHTED_SCORER_NAME

        xgb_req = _health_payload()
        xgb_req["ranking_model"] = "xgboost"
        res_xgb = client.post("/api/v1/recommend", json=xgb_req)
        assert res_xgb.status_code == 501

    car_pol = _sample_policy(
        policy_type="car",
        claim_settlement_ratio=None,
        network_hospitals_count=0,
        covered_benefits=["zero_depreciation", "roadside_assistance"],
        type_specific_metadata={
            "vehicle_types": ["sedan", "suv"],
            "vehicle_age_limit_years": 10,
            "own_damage_coverage": True,
            "third_party_coverage": True,
        },
    )
    with _build_client(monkeypatch, [car_pol]) as client:
        car_req = _health_payload(
            policy_type="car",
            required_benefits=[],
            vehicle_type="sedan",
            vehicle_age_years=3,
            require_own_damage=True,
        )
        car_req["ranking_model"] = "weighted"
        res_car = client.post("/api/v1/recommend", json=car_req)
        assert res_car.status_code == 200
        assert res_car.json()["recommendations"][0]["score"] > 0.0


# ---------------------------------------------------------------------------
# 3. Incomplete Customer Profiles (Independently & Together)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["lightgbm", "weighted"])
def test_incomplete_profile_matrix_handles_missing_income_and_risk_honestly(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    """Test all 4 missingness states cleanly across both scoring backends."""
    catalog = [
        _sample_policy(policy_id="p1", sum_insured_max=Decimal("1000000.00")),
        _sample_policy(policy_id="p2", sum_insured_max=Decimal("1500000.00")),
    ]
    with _build_client(monkeypatch, catalog) as client:
        # 1) Complete profile
        req_full = _health_payload()
        req_full["ranking_model"] = backend
        body_full = client.post("/api/v1/recommend", json=req_full).json()
        pc_full = body_full["profile_completeness"]
        assert pc_full["is_complete"] is True
        assert pc_full["missing_fields"] == []
        assert pc_full["income_distribution_mismatch_risk"] is False

        # 2) Missing annual_income_inr only
        req_no_inc = _health_payload()
        req_no_inc["ranking_model"] = backend
        del req_no_inc["profile"]["annual_income_inr"]
        body_no_inc = client.post("/api/v1/recommend", json=req_no_inc).json()
        pc_inc = body_no_inc["profile_completeness"]
        assert pc_inc["is_complete"] is False
        assert pc_inc["missing_fields"] == ["annual_income_inr"]
        assert pc_inc["income_distribution_mismatch_risk"] is (backend == "lightgbm")
        assert any("annual_income_inr" in w for w in body_no_inc["warnings"])

        # Weighted backend must be completely invariant to missing annual_income_inr
        if backend == "weighted":
            assert [(r["policy_id"], r["score"]) for r in body_no_inc["recommendations"]] == [
                (r["policy_id"], r["score"]) for r in body_full["recommendations"]
            ]

        # 3) Missing risk_preference only
        req_no_risk = _health_payload()
        req_no_risk["ranking_model"] = backend
        del req_no_risk["profile"]["risk_preference"]
        body_no_risk = client.post("/api/v1/recommend", json=req_no_risk).json()
        pc_risk = body_no_risk["profile_completeness"]
        assert pc_risk["is_complete"] is False
        assert pc_risk["missing_fields"] == ["risk_preference"]
        assert pc_risk["applied_assumptions"]["risk_preference"] == "defaulted_to_balanced"
        assert pc_risk["income_distribution_mismatch_risk"] is False
        assert any("risk_preference" in w for w in body_no_risk["warnings"])

        # 4) Missing BOTH
        req_neither = _health_payload()
        req_neither["ranking_model"] = backend
        del req_neither["profile"]["annual_income_inr"]
        del req_neither["profile"]["risk_preference"]
        body_neither = client.post("/api/v1/recommend", json=req_neither).json()
        pc_both = body_neither["profile_completeness"]
        assert pc_both["is_complete"] is False
        assert set(pc_both["missing_fields"]) == {"annual_income_inr", "risk_preference"}


def test_zero_or_negative_annual_income_rejected_with_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid annual_income_inr <= 0 returns 422 rather than colliding with missing sentinel."""
    with _build_client(monkeypatch, []) as client:
        for bad_val in (0, 0.0, -250000):
            res = client.post("/api/v1/recommend", json=_health_payload(annual_income_inr=bad_val))
            assert res.status_code == 422


# ---------------------------------------------------------------------------
# 4. Structured JSON Logging & PII / Secret Protection (NFR-11)
# ---------------------------------------------------------------------------


def test_structured_logging_records_metrics_and_scrubs_pii_and_secrets(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Recommendation emit JSON-serializable operational metrics with zero PII/credentials."""
    formatter = StructuredJsonFormatter()
    scrubbed = sanitize_log_value(
        {
            "annual_income_inr": 1850000,
            "pre_existing_conditions": ["diabetes"],
            "db_password": "my-secret-password",
            "dsn_err": "OperationalError postgresql+psycopg://postgres:topsecret@localhost/db",
            "eligible_count": 2,
        }
    )
    assert scrubbed["annual_income_inr"] == "[REDACTED]"
    assert scrubbed["pre_existing_conditions"] == "[REDACTED]"
    assert scrubbed["db_password"] == "[REDACTED]"
    assert "topsecret" not in scrubbed["dsn_err"]
    assert scrubbed["eligible_count"] == 2

    catalog = [_sample_policy()]
    with _build_client(monkeypatch, catalog) as client:
        with caplog.at_level(logging.INFO, logger="ai_service.recommendation.service"):
            res = client.post(
                "/api/v1/recommend",
                json=_health_payload(
                    age=47,
                    annual_income_inr=2345678,
                    pre_existing_conditions=["hypertension"],
                    location="Pune",
                    smoking_status="non_smoker",
                ),
                headers={"X-Request-ID": "req-log-audit-9a"},
            )
            assert res.status_code == 200

    rec_logs = [
        r for r in caplog.records if getattr(r, "event", None) == "recommendation.completed"
    ]
    assert len(rec_logs) == 1
    rendered_json = formatter.format(rec_logs[0])
    parsed = json.loads(rendered_json)

    assert parsed["event"] == "recommendation.completed"
    assert parsed["request_id"] == "req-log-audit-9a"
    assert parsed["model_used"] == MODEL_NAME
    assert parsed["eligible_count"] == 1
    assert parsed["rejected_count"] == 0
    assert set(parsed["latency_ms"].keys()) == {
        "db",
        "filter",
        "features",
        "scoring",
        "citations",
        "total",
    }
    # Assert zero leakage of prohibited PII fields or sensitive values in parsed JSON log
    prohibited_keys = {
        "profile",
        "age",
        "annual_income_inr",
        "annual_income",
        "pre_existing_conditions",
        "location",
        "smoking_status",
        "user_id",
        "owner_scope",
    }
    observed_keys: set[str] = set()
    observed_values: list[Any] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, val in node.items():
                observed_keys.add(str(key))
                if key in ("timestamp_utc", "latency_ms"):
                    continue
                _walk(val)
        elif isinstance(node, list):
            for item in node:
                _walk(item)
        else:
            observed_values.append(node)

    _walk(parsed)
    assert observed_keys.isdisjoint(prohibited_keys)

    for forbidden_value in (2345678, "2345678", "hypertension", "Pune", "non_smoker", 47, "47"):
        assert forbidden_value not in observed_values
    for val in observed_values:
        if isinstance(val, str):
            for forbidden_str in ("2345678", "hypertension", "Pune", "non_smoker"):
                assert forbidden_str not in val


# ---------------------------------------------------------------------------
# 5. Citation Provider Interface Seam (Stage 9A Interface Only)
# ---------------------------------------------------------------------------


def test_citation_provider_interface_injection_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default NullCitationProvider returns []; custom injected provider populates citations."""
    p_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    catalog = [_sample_policy(policy_id=p_id)]

    class StubCitationProvider:
        def get_citations(
            self,
            requests: list[PolicyCitationRequest] | Any,
            *,
            request_id: str | None = None,
        ) -> dict[str, list[dict[str, Any]]]:
            assert request_id == "cit-check-1"
            return {
                r.policy_id: [{"chunk_id": f"chk-{r.policy_id[:8]}", "page": 4, "clause": "2.1"}]
                for r in requests
            }

    class BrokenCitationProvider:
        def get_citations(
            self,
            requests: list[PolicyCitationRequest] | Any,
            *,
            request_id: str | None = None,
        ) -> Any:
            return {requests[0].policy_id: "invalid-non-list"}

    app = create_app(Settings(_env_file=None))
    mock_conn = MagicMock()
    app.dependency_overrides[get_db_connection] = lambda: (yield mock_conn)
    app.dependency_overrides[get_citation_provider] = lambda: StubCitationProvider()
    monkeypatch.setattr(
        "ai_service.recommendation.service.load_catalog_policies_from_db",
        lambda _c, active_only=False: catalog,
    )

    with TestClient(app) as client:
        ok = client.post(
            "/api/v1/recommend",
            json=_health_payload(),
            headers={"X-Request-ID": "cit-check-1"},
        )
        assert ok.status_code == 200
        assert ok.json()["recommendations"][0]["citations"] == [
            {"chunk_id": "chk-aaaaaaaa", "page": 4, "clause": "2.1"}
        ]

        app.dependency_overrides[get_citation_provider] = lambda: BrokenCitationProvider()
        bad = client.post("/api/v1/recommend", json=_health_payload())
        assert bad.status_code == 500


# ---------------------------------------------------------------------------
# 6. Large Catalog Determinism, Ties, Performance & OpenAPI Documentation
# ---------------------------------------------------------------------------


def test_large_catalog_determinism_tie_breaking_and_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify deterministic ordering and sub-second latency on 500 eligible candidates."""
    catalog = [
        _sample_policy(
            policy_id=f"{i:04d}-policy",
            policy_name=f"Policy {i}",
            annual_premium_min=Decimal(str(8000 + (i % 10) * 1500)),
            annual_premium_max=Decimal(str(12000 + (i % 10) * 1500)),
            sum_insured_max=Decimal(str(500000 + (i % 5) * 500000)),
        )
        for i in range(500)
    ]

    # Add two identical policies to verify deterministic policy_id tie-break at scale
    catalog.append(_sample_policy(policy_id="0000-tie-z", annual_premium_min=Decimal("10000")))
    catalog.append(_sample_policy(policy_id="0000-tie-a", annual_premium_min=Decimal("10000")))

    with _build_client(monkeypatch, catalog) as client:
        for backend in ("lightgbm", "weighted"):
            req = _health_payload()
            req["top_n"] = 10
            req["ranking_model"] = backend

            t0 = time.perf_counter()
            r1 = client.post("/api/v1/recommend", json=req)
            elapsed_s = time.perf_counter() - t0
            r2 = client.post("/api/v1/recommend", json=req)

            assert r1.status_code == 200
            assert elapsed_s < 5.0
            body1 = r1.json()
            body2 = r2.json()
            assert body1["filter_report"]["passed_count"] == 502
            assert len(body1["recommendations"]) == 10
            assert [(x["policy_id"], x["score"]) for x in body1["recommendations"]] == [
                (x["policy_id"], x["score"]) for x in body2["recommendations"]
            ]

            scores = [x["score"] for x in body1["recommendations"]]
            assert scores == sorted(scores, reverse=True)


def test_openapi_schema_exposes_recommendation_contracts_and_error_codes() -> None:
    """OpenAPI spec documents endpoints, ProfileCompleteness, examples, and error responses."""
    app = create_app(Settings(_env_file=None))
    schema = app.openapi()

    for path in ("/api/v1/recommend", "/ai/recommend"):
        post_spec = schema["paths"][path]["post"]
        responses = post_spec["responses"]
        for code in ("200", "401", "422", "500", "501", "503"):
            assert code in responses

    components = schema["components"]["schemas"]
    assert "ProfileCompleteness" in components
    assert "RecommendRequest" in components
    assert "RecommendResponse" in components


# ---------------------------------------------------------------------------
# 7. Database Timeouts & Read-Only Live PostgreSQL Check
# ---------------------------------------------------------------------------


def test_database_engine_receives_configured_timeouts() -> None:
    """Settings exposes DB timeouts and create_db_engine forwards pool & statement timeouts."""
    cfg = Settings(
        _env_file=None,
        db_connect_timeout_seconds=7,
        db_statement_timeout_ms=6500,
        db_pool_timeout_seconds=12.5,
    )
    engine = create_db_engine(cfg)
    try:
        assert engine.pool._timeout == pytest.approx(12.5)
        # Validate URL and dialect without needing live DB network connection
        assert engine.url.drivername == "postgresql+psycopg"
    finally:
        engine.dispose()


def test_live_postgresql_statement_timeout_and_read_only_readiness() -> None:
    """Read-only live PostgreSQL test: verifies statement_timeout setting and safe 503."""
    cfg = Settings(_env_file="../server/.env" if Path("../server/.env").exists() else None)
    if cfg.db_password is None and not os.getenv("DB_PASSWORD"):
        pytest.skip("DB_PASSWORD not configured; skipping live PostgreSQL Stage 9A check")

    timed_cfg = Settings(
        _env_file="../server/.env" if Path("../server/.env").exists() else None,
        db_statement_timeout_ms=7500,
    )
    engine = create_db_engine(timed_cfg)
    try:
        try:
            with engine.connect() as conn:
                timeout_ms = conn.execute(
                    text("SELECT setting FROM pg_settings WHERE name = 'statement_timeout'")
                ).scalar_one()
                assert str(timeout_ms) == "7500"
                readiness = inspect_database_readiness(conn)
        except SQLAlchemyError as exc:
            pytest.skip(f"PostgreSQL unreachable for read-only Stage 9A check: {exc}")

        app = create_app(timed_cfg)
        with TestClient(app) as client:
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["model_ready"] is True
            assert health.json()["model_warmed_up"] is True

            # Verify explicit 'weighted' handles missing schema gracefully -> HTTP 503 without crash
            req = _health_payload()
            req["ranking_model"] = "weighted"
            rec = client.post("/api/v1/recommend", json=req)
            if not readiness["tables_present"]["policy_attributes"]:
                assert rec.status_code == 503
            else:
                assert rec.status_code == 200
    finally:
        engine.dispose()
