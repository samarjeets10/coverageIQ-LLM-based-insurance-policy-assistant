"""Stage 7 — Tests for End-to-End Recommendation Inference Integration.

Developer: Shravan Sambhaji Kapare (Roll No. 203)
Module: M5 — Policy Recommendation
Covers all inference scenarios with isolated database fixtures/mocks and
read-only verification against live PostgreSQL when credentials exist.
"""

from __future__ import annotations

import json
import os
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from ai_service.main import create_app
from ai_service.recommendation.db import (
    create_db_engine,
    get_db_connection,
    inspect_database_readiness,
)
from ai_service.recommendation.features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from ai_service.recommendation.filters import filter_policies as real_filter_policies
from ai_service.recommendation.schemas import RecommendRequest
from ai_service.recommendation.service import (
    CatalogDatabaseUnavailableError,
    InferenceProcessingError,
    ModelUnavailableError,
    UnsupportedRankingModelError,
    load_model_state,
    recommend_policies_for_request,
)
from ai_service.recommendation.settings import Settings
from ai_service.recommendation.train_lightgbm import MODEL_NAME, MODEL_VERSION

# ---------------------------------------------------------------------------
# Controlled Fixture Helpers
# ---------------------------------------------------------------------------


def _db_policy_row(
    *,
    policy_id: str | None = None,
    policy_name: str = "ArogyaCare Prime Health",
    insurer_name: str = "ArogyaShield General Insurance",
    policy_type: str = "health",
    policy_status: str = "active",
    is_catalog_eligible: bool = True,
    annual_premium_min: Decimal | float = Decimal("12000.00"),
    annual_premium_max: Decimal | float = Decimal("18000.00"),
    sum_insured_min: Decimal | float = Decimal("500000.00"),
    sum_insured_max: Decimal | float = Decimal("1500000.00"),
    claim_settlement_ratio: Decimal | float = Decimal("97.25"),
    copay_pct: Decimal | float = Decimal("0.00"),
    deductible_amount: Decimal | float = Decimal("0.00"),
    network_hospitals_count: int = 8500,
    covered_benefits: list[str] | None = None,
    exclusions: list[str] | None = None,
    type_specific_metadata: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a schema-faithful candidate policy row mirroring DB layer-D output."""
    pid = policy_id or str(uuid.uuid4())
    row: dict[str, Any] = {
        "policy_id": pid,
        "organization_id": uuid.uuid4(),
        "policy_name": policy_name,
        "insurer_name": insurer_name,
        "policy_type": policy_type,
        "description": "Controlled catalog policy fixture for inference verification.",
        "policy_status": policy_status,
        "current_version_id": uuid.uuid4(),
        "policy_version_id": uuid.uuid4(),
        "is_catalog_eligible": is_catalog_eligible,
        "min_entry_age": 18,
        "max_entry_age": 65,
        "max_renewal_age": 85,
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
        "room_rent_limit_value": None,
        "initial_waiting_period_days": 30,
        "ped_covered": True,
        "ped_waiting_period_months": 24,
        "specific_disease_waiting_months": 24,
        "maternity_waiting_months": 24,
        "network_hospitals_count": network_hospitals_count,
        "cashless_available": True,
        "pan_india_coverage": True,
        "available_regions": ["mumbai", "pune", "bengaluru"],
        "min_tenure_years": 1,
        "max_tenure_years": 3,
        "covered_benefits": covered_benefits
        if covered_benefits is not None
        else [
            "cashless_hospitalization",
            "daycare_procedures",
            "ambulance_cover",
            "annual_health_checkup",
        ],
        "exclusions": exclusions if exclusions is not None else ["cosmetic_surgery"],
        "claim_settlement_ratio": claim_settlement_ratio,
        "no_claim_bonus_pct": Decimal("25.00"),
        "restoration_benefit": True,
        "type_specific_metadata": type_specific_metadata or {"family_floater_supported": True},
        "extraction_source": "verified",
    }
    row.update(overrides)
    return row


def _health_payload(**profile_overrides: Any) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "policy_type": "health",
        "age": 32,
        "budget_min": 5000.0,
        "budget_max": 25000.0,
        "applicant_type": "salaried",
        "coverage_mode": "individual",
        "family_size": 1,
        "min_sum_insured": 500000.0,
        "required_benefits": ["cashless_hospitalization"],
        "tenure_years": 1,
        "annual_income_inr": 1_200_000.0,
        "risk_preference": "balanced",
    }
    profile.update(profile_overrides)
    return {"profile": profile, "top_n": 5, "ranking_model": "auto"}


def _build_mixed_health_catalog() -> list[dict[str, Any]]:
    """Return 4 policies: 3 eligible with distinct qualities + 1 ineligible."""
    p_best = _db_policy_row(
        policy_id="11111111-1111-1111-1111-111111111111",
        policy_name="Elite Health Max",
        sum_insured_max=Decimal("2500000.00"),
        annual_premium_min=Decimal("14000.00"),
        annual_premium_max=Decimal("19000.00"),
        claim_settlement_ratio=Decimal("99.10"),
        network_hospitals_count=12000,
        covered_benefits=[
            "cashless_hospitalization",
            "daycare_procedures",
            "ambulance_cover",
            "annual_health_checkup",
            "maternity_cover",
            "opd_cover",
        ],
    )
    p_mid = _db_policy_row(
        policy_id="22222222-2222-2222-2222-222222222222",
        policy_name="Standard Health Select",
        sum_insured_max=Decimal("1000000.00"),
        annual_premium_min=Decimal("10000.00"),
        annual_premium_max=Decimal("15000.00"),
        claim_settlement_ratio=Decimal("94.00"),
        network_hospitals_count=5500,
        covered_benefits=["cashless_hospitalization", "daycare_procedures"],
    )
    p_basic = _db_policy_row(
        policy_id="33333333-3333-3333-3333-333333333333",
        policy_name="Basic Health Starter",
        sum_insured_max=Decimal("500000.00"),
        annual_premium_min=Decimal("19000.00"),
        annual_premium_max=Decimal("24000.00"),
        claim_settlement_ratio=Decimal("87.00"),
        copay_pct=Decimal("25.00"),
        deductible_amount=Decimal("10000.00"),
        restoration_benefit=False,
        no_claim_bonus_pct=Decimal("0.00"),
        network_hospitals_count=1800,
        covered_benefits=["cashless_hospitalization"],
    )
    p_ineligible_budget = _db_policy_row(
        policy_id="44444444-4444-4444-4444-444444444444",
        policy_name="Ultra Luxury Unaffordable",
        annual_premium_min=Decimal("45000.00"),
        annual_premium_max=Decimal("60000.00"),
        sum_insured_max=Decimal("5000000.00"),
    )
    return [p_basic, p_ineligible_budget, p_mid, p_best]


def _app_with_mock_catalog(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[dict[str, Any]] | Exception,
    *,
    settings: Settings | None = None,
) -> TestClient:
    """Create a FastAPI TestClient overriding only DB dependency/catalog loader."""
    app = create_app(settings or Settings(_env_file=None))
    mock_conn = MagicMock()

    def override_conn():
        yield mock_conn

    app.dependency_overrides[get_db_connection] = override_conn

    if isinstance(catalog, Exception):

        def _raise(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
            raise catalog

        monkeypatch.setattr("ai_service.recommendation.service.load_catalog_policies_from_db", _raise)
    else:
        monkeypatch.setattr(
            "ai_service.recommendation.service.load_catalog_policies_from_db",
            lambda _conn, active_only=False: catalog,
        )

    return TestClient(app)


# ---------------------------------------------------------------------------
# 1, 4, 12. Successful Inference, Descending Ordering & Response Structure
# ---------------------------------------------------------------------------


def test_successful_recommendation_both_routes_and_response_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests 1, 4 & 12: Ranked recommendations, descending scores, Decimal DB compatibility."""
    catalog = _build_mixed_health_catalog()
    payload = _health_payload()
    payload["top_n"] = 2

    with _app_with_mock_catalog(monkeypatch, catalog) as client:
        for route in ("/api/v1/recommend", "/ai/recommend"):
            res = client.post(
                route,
                json=payload,
                headers={"X-Request-ID": "req-stage7-test-123"},
            )
            assert res.status_code == 200
            assert res.headers["X-Request-ID"] == "req-stage7-test-123"

            body = res.json()
            assert body["request_id"] == "req-stage7-test-123"
            assert body["status"] == "RECOMMENDATIONS_READY"
            assert body["model_used"] == MODEL_NAME
            assert body["model_version"] == MODEL_VERSION
            assert body["feature_schema_version"] == FEATURE_SCHEMA_VERSION

            report = body["filter_report"]
            assert report["status"] == "ELIGIBLE_CANDIDATES_FOUND"
            assert report["total_candidates"] == 4
            assert report["passed_count"] == 3
            assert report["eliminated_count"] == 1
            assert report["eliminations"][0]["policy_id"] == (
                "44444444-4444-4444-4444-444444444444"
            )

            recs = body["recommendations"]
            assert len(recs) == 2
            assert [r["rank"] for r in recs] == [1, 2]
            assert recs[0]["score"] >= recs[1]["score"]
            for rec in recs:
                assert rec["policy_id"] != "44444444-4444-4444-4444-444444444444"
                assert rec["citations"] == []
                assert rec["feature_breakdown"]
                assert isinstance(rec["key_highlights"], list)
                assert rec["model_used"] == MODEL_NAME

            # Synthetic disclaimer must always remain present in warnings
            assert any("synthetic" in w.lower() for w in body["warnings"])
            # When optional fields are provided, omission warnings must NOT appear
            assert not any("annual_income_inr" in w for w in body["warnings"])
            assert not any("risk_preference" in w for w in body["warnings"])


def test_descending_score_sorting_and_deterministic_tie_breaking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure eligible policies always sort descending by score, then ascending policy_id."""
    catalog = [
        _db_policy_row(policy_id="pol-b", policy_name="Policy B"),
        _db_policy_row(policy_id="pol-a", policy_name="Policy A"),
        _db_policy_row(policy_id="pol-c", policy_name="Policy C"),
    ]

    # Force a controlled tie + one strictly higher score
    monkeypatch.setattr(
        "ai_service.recommendation.service.predict_ranking_scores",
        lambda _model, X: np.array([1.25, 1.25, 3.75], dtype=np.float64),
    )

    state = load_model_state()
    req = RecommendRequest.model_validate(_health_payload())

    monkeypatch.setattr(
        "ai_service.recommendation.service.load_catalog_policies_from_db",
        lambda _conn, active_only=False: catalog,
    )
    resp = recommend_policies_for_request(req, MagicMock(), state, request_id="tie-test")
    assert [r.policy_id for r in resp.recommendations] == ["pol-c", "pol-a", "pol-b"]
    assert [r.score for r in resp.recommendations] == [3.75, 1.25, 1.25]


# ---------------------------------------------------------------------------
# 2 & 3. Execution Order & Strict Exclusion of Ineligible Policies
# ---------------------------------------------------------------------------


def test_execution_order_and_ineligible_policies_never_reach_ml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests 2 & 3: Hard filtering runs first and blocks ineligible policies from Stage 5/6."""
    catalog = _build_mixed_health_catalog()
    call_order: list[str] = []
    seen_feature_policy_ids: list[str] = []
    scored_row_counts: list[int] = []

    from ai_service.recommendation import features as real_features
    from ai_service.recommendation import train_lightgbm as real_lgb

    def wrapped_filter(profile: Any, policies: Any) -> Any:
        call_order.append("filter")
        return real_filter_policies(profile, policies)

    def wrapped_features(profile: Any, eligible: Any) -> pd.DataFrame:
        call_order.append("features")
        seen_feature_policy_ids.extend(str(p["policy_id"]) for p in eligible)
        return real_features.build_feature_matrix(profile, eligible)

    def wrapped_predict(model: Any, X: pd.DataFrame) -> np.ndarray:
        call_order.append("predict")
        scored_row_counts.append(len(X))
        assert tuple(X.columns) == FEATURE_COLUMNS
        return real_lgb.predict_ranking_scores(model, X)

    monkeypatch.setattr("ai_service.recommendation.service.filter_policies", wrapped_filter)
    monkeypatch.setattr("ai_service.recommendation.service.build_feature_matrix", wrapped_features)
    monkeypatch.setattr("ai_service.recommendation.service.predict_ranking_scores", wrapped_predict)

    with _app_with_mock_catalog(monkeypatch, catalog) as client:
        res = client.post("/api/v1/recommend", json=_health_payload())
        assert res.status_code == 200

    assert call_order == ["filter", "features", "predict"]
    assert "44444444-4444-4444-4444-444444444444" not in seen_feature_policy_ids
    assert len(seen_feature_policy_ids) == 3
    assert scored_row_counts == [3]


# ---------------------------------------------------------------------------
# 5 & 11. Zero Eligible Policies & Empty Catalog Handling
# ---------------------------------------------------------------------------


def test_zero_eligible_policies_skips_feature_engineering_and_ml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 5: Ineligible-only catalog yields NO_MATCHING_POLICIES without ML invocation."""
    only_ineligible = [
        _db_policy_row(
            policy_id="99999999-9999-9999-9999-999999999999",
            annual_premium_min=Decimal("90000.00"),
            annual_premium_max=Decimal("120000.00"),
        )
    ]

    def _must_not_run(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("ML pipeline must NOT run when zero policies pass filtering")

    monkeypatch.setattr("ai_service.recommendation.service.build_feature_matrix", _must_not_run)
    monkeypatch.setattr("ai_service.recommendation.service.predict_ranking_scores", _must_not_run)

    with _app_with_mock_catalog(monkeypatch, only_ineligible) as client:
        res = client.post("/api/v1/recommend", json=_health_payload())
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "NO_MATCHING_POLICIES"
        assert body["recommendations"] == []
        assert body["filter_report"]["total_candidates"] == 1
        assert body["filter_report"]["passed_count"] == 0
        assert body["filter_report"]["eliminated_count"] == 1
        assert len(body["filter_report"]["eliminations"]) == 1


def test_empty_candidate_catalog_returns_explicit_no_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 11: Empty DB catalog returns valid NO_MATCHING_POLICIES without ML calls."""

    def _must_not_run(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Feature engineering / ML must NOT run on empty catalog")

    monkeypatch.setattr("ai_service.recommendation.service.build_feature_matrix", _must_not_run)
    monkeypatch.setattr("ai_service.recommendation.service.predict_ranking_scores", _must_not_run)

    with _app_with_mock_catalog(monkeypatch, []) as client:
        res = client.post("/api/v1/recommend", json=_health_payload())
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "NO_MATCHING_POLICIES"
        assert body["recommendations"] == []
        assert body["filter_report"]["total_candidates"] == 0
        assert body["filter_report"]["passed_count"] == 0


# ---------------------------------------------------------------------------
# 6. Invalid Customer Profile & Unsupported Model Options
# ---------------------------------------------------------------------------


def test_invalid_profile_returns_422_and_unsupported_model_returns_501(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 6: Reject invalid profiles with 422 and unimplemented ranking_model with 501."""
    with _app_with_mock_catalog(monkeypatch, []) as client:
        invalid_age = _health_payload(age=999)
        r_age = client.post("/api/v1/recommend", json=invalid_age)
        assert r_age.status_code == 422

        extra_field = _health_payload(suitability_score=0.99)
        r_extra = client.post("/api/v1/recommend", json=extra_field)
        assert r_extra.status_code == 422

        xgb_req = _health_payload()
        xgb_req["ranking_model"] = "xgboost"
        r_xgb = client.post("/api/v1/recommend", json=xgb_req)
        assert r_xgb.status_code == 501
        assert "xgboost" in r_xgb.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 7 & 8. Missing Artifacts, Incompatible Schema & Operational /health
# ---------------------------------------------------------------------------


def test_missing_model_artifacts_returns_503_while_health_stays_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 7: Missing artifacts return 503 on recommend without leaking paths."""
    missing_dir = tmp_path / "nonexistent_models_secret_dir"
    cfg = Settings(_env_file=None, model_artifacts_dir=missing_dir)
    app = create_app(cfg)
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
            "tables_present": {"policies": True},
            "total_policy_count": 0,
            "active_catalog_policy_count": 0,
        },
    )

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_ready"] is False
        assert health.json()["services"]["model"] == "unavailable"

        rec = client.post("/api/v1/recommend", json=_health_payload())
        assert rec.status_code == 503
        detail = rec.json()["detail"]
        assert "unavailable" in detail.lower()
        assert str(tmp_path) not in detail


def test_incompatible_model_schema_returns_503(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 8: Model metadata schema version or column mismatch triggers controlled HTTP 503."""
    bad_dir = tmp_path / "bad_schema_models"
    bad_dir.mkdir()
    src_dir = Settings(_env_file=None).model_artifacts_dir
    (bad_dir / "lightgbm_ranker.joblib").write_bytes(
        (src_dir / "lightgbm_ranker.joblib").read_bytes()
    )
    meta = json.loads((src_dir / "lightgbm_metadata.json").read_text(encoding="utf-8"))
    meta["feature_schema_version"] = "99.0.0"
    (bad_dir / "lightgbm_metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    cfg = Settings(_env_file=None, model_artifacts_dir=bad_dir)
    with _app_with_mock_catalog(monkeypatch, [], settings=cfg) as client:
        rec = client.post("/api/v1/recommend", json=_health_payload())
        assert rec.status_code == 503
        assert "incompatible" in rec.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 9. Invalid / Non-Finite Features or Model Predictions
# ---------------------------------------------------------------------------


def test_non_finite_features_and_scores_raise_controlled_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 9: Non-finite features or predictions fail with 500, never fallback scores."""
    catalog = [_db_policy_row()]

    def _nan_features(profile: Any, eligible: Any) -> pd.DataFrame:
        df = pd.DataFrame(
            np.zeros((len(eligible), len(FEATURE_COLUMNS)), dtype=np.float64),
            columns=list(FEATURE_COLUMNS),
        )
        df.iloc[0, 0] = np.nan
        return df

    monkeypatch.setattr("ai_service.recommendation.service.build_feature_matrix", _nan_features)

    with _app_with_mock_catalog(monkeypatch, catalog) as client:
        res = client.post("/api/v1/recommend", json=_health_payload())
        assert res.status_code == 500
        assert "non-finite" in res.json()["detail"].lower()

    # Restore valid feature builder, inject NaN predictions from scorer
    from ai_service.recommendation import features as real_features

    monkeypatch.setattr(
        "ai_service.recommendation.service.build_feature_matrix",
        real_features.build_feature_matrix,
    )
    monkeypatch.setattr(
        "ai_service.recommendation.service.predict_ranking_scores",
        lambda _m, X: np.array([np.inf] * len(X), dtype=np.float64),
    )

    with _app_with_mock_catalog(monkeypatch, catalog) as client:
        res2 = client.post("/api/v1/recommend", json=_health_payload())
        assert res2.status_code == 500
        assert "non-finite" in res2.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 10. Database Retrieval Failures
# ---------------------------------------------------------------------------


def test_database_failures_return_503_without_leaking_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 10: DB OperationalError and missing policy_attributes table both return HTTP 503."""
    from ai_service.recommendation.filters import PolicyAttributesTableUnavailableError

    db_err = OperationalError(
        "SELECT * FROM secret_table",
        {},
        Exception("password authentication failed for postgres://user:supersecret@db"),
    )
    with _app_with_mock_catalog(monkeypatch, db_err) as client:
        res = client.post("/api/v1/recommend", json=_health_payload())
        assert res.status_code == 503
        detail = res.json()["detail"]
        assert "supersecret" not in detail
        assert "postgres://" not in detail

    missing_table_err = PolicyAttributesTableUnavailableError("Table 'policy_attributes' missing")
    with _app_with_mock_catalog(monkeypatch, missing_table_err) as client:
        res_missing = client.post("/api/v1/recommend", json=_health_payload())
        assert res_missing.status_code == 503
        assert "policy_attributes" in res_missing.json()["detail"]


def test_db_connect_dependency_failure_returns_503_on_health_and_recommend() -> None:
    """Connection-time OperationalError returns HTTP 503 without uncaught crashes."""
    app = create_app(Settings(_env_file=None))

    with TestClient(app) as client:
        app.state.db_engine = MagicMock()
        app.state.db_engine.connect.side_effect = OperationalError(
            "CONNECT",
            {},
            Exception("connection refused postgres://postgres:topsecret@localhost:5432"),
        )
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json()["services"]["database"] == "down"
        assert "topsecret" not in health.text

        rec = client.post("/api/v1/recommend", json=_health_payload())
        assert rec.status_code == 503
        assert "unavailable" in rec.json()["detail"].lower()
        assert "topsecret" not in rec.text


# ---------------------------------------------------------------------------
# 13. Single-Load Model Verification (Loaded Once During Lifespan)
# ---------------------------------------------------------------------------


def test_model_loaded_once_across_multiple_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 13: Model artifacts load once during startup lifespan, NOT per request."""
    from ai_service.recommendation import train_lightgbm as real_lgb

    load_calls = 0

    def counting_load_model(model_dir: Any = None) -> Any:
        nonlocal load_calls
        load_calls += 1
        return real_lgb.load_model(model_dir)

    monkeypatch.setattr("ai_service.recommendation.service.load_model", counting_load_model)
    catalog = [_db_policy_row()]

    with _app_with_mock_catalog(monkeypatch, catalog) as client:
        assert load_calls == 1
        for i in range(5):
            res = client.post("/api/v1/recommend", json=_health_payload(age=30 + i))
            assert res.status_code == 200
        assert load_calls == 1


# ---------------------------------------------------------------------------
# Additional Domain Checks: Optional Field Warnings & Multi-Category Coverage
# ---------------------------------------------------------------------------


def test_omitted_income_and_risk_preference_surface_explicit_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing annual_income_inr or risk_preference must emit clear warnings."""
    catalog = [_db_policy_row()]
    payload = _health_payload()
    del payload["profile"]["annual_income_inr"]
    del payload["profile"]["risk_preference"]

    with _app_with_mock_catalog(monkeypatch, catalog) as client:
        res = client.post("/api/v1/recommend", json=payload)
        assert res.status_code == 200
        warnings = res.json()["warnings"]
        assert any("annual_income_inr" in w for w in warnings)
        assert any("risk_preference" in w for w in warnings)
        assert any("synthetic" in w.lower() for w in warnings)


@pytest.mark.parametrize(
    ("category", "profile_extra", "meta"),
    [
        (
            "car",
            {"vehicle_type": "sedan", "vehicle_age_years": 3, "require_own_damage": True},
            {
                "vehicle_types": ["sedan", "suv"],
                "vehicle_age_limit_years": 10,
                "own_damage_coverage": True,
                "third_party_coverage": True,
            },
        ),
        (
            "life",
            {"tenure_years": 15, "require_maturity_benefit": True},
            {
                "life_cover_min": 500000,
                "life_cover_max": 5000000,
                "policy_term_years_min": 10,
                "policy_term_years_max": 30,
                "premium_payment_term_years": 15,
                "maturity_benefit": True,
                "surrender_value_available": True,
            },
        ),
        (
            "term",
            {"tenure_years": 20, "smoking_status": "non_smoker"},
            {
                "life_cover_min": 1000000,
                "life_cover_max": 10000000,
                "policy_term_years_min": 10,
                "policy_term_years_max": 40,
                "premium_payment_term_years": 20,
                "smoking_status_eligibility": "all",
                "critical_illness_rider": True,
                "accidental_death_rider": True,
            },
        ),
    ],
)
def test_car_life_term_inference_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    category: str,
    profile_extra: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    """Verify Stage 7 inference works cleanly across all supported categories."""
    pol = _db_policy_row(
        policy_type=category,
        policy_name=f"Sample {category.title()} Plan",
        min_tenure_years=1,
        max_tenure_years=40,
        type_specific_metadata=meta,
    )
    payload = _health_payload(policy_type=category, **profile_extra)
    with _app_with_mock_catalog(monkeypatch, [pol]) as client:
        res = client.post("/api/v1/recommend", json=payload)
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "RECOMMENDATIONS_READY"
        assert body["recommendations"][0]["policy_type"] == category


def test_direct_service_exceptions_unit_checks() -> None:
    """Direct checks on service-level exception types and zero-fallback behavior."""
    bad_state = load_model_state(Path("nonexistent_stage7_dir"))
    assert not bad_state.is_ready
    assert bad_state.error_kind == "missing"

    req_bad_model = RecommendRequest.model_validate(
        {**_health_payload(), "ranking_model": "xgboost"}
    )
    with pytest.raises(UnsupportedRankingModelError):
        recommend_policies_for_request(req_bad_model, MagicMock(), load_model_state())

    req_ok = RecommendRequest.model_validate(_health_payload())
    with pytest.raises(ModelUnavailableError):
        recommend_policies_for_request(req_ok, MagicMock(), bad_state)

    # Catalog DB exception propagation
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = OperationalError("SELECT 1", {}, Exception("down"))
    with pytest.raises(CatalogDatabaseUnavailableError):
        recommend_policies_for_request(req_ok, mock_conn, load_model_state())

    # InferenceProcessingError never exposed silently
    assert issubclass(InferenceProcessingError, RuntimeError)


# ---------------------------------------------------------------------------
# Read-Only Live Database Check (When DB Credentials Exist)
# ---------------------------------------------------------------------------


def test_live_postgresql_read_only_recommendation_readiness() -> None:
    """Read-only live PostgreSQL check: /health stays 200, recommend handles unmigrated DB."""
    cfg = Settings()
    if cfg.db_password is None and not os.getenv("DB_PASSWORD"):
        pytest.skip("DB_PASSWORD not configured; skipping live PostgreSQL Stage 7 check")

    engine = create_db_engine(cfg)
    try:
        try:
            with engine.connect() as conn:
                readiness = inspect_database_readiness(conn)
        except SQLAlchemyError as exc:
            pytest.skip(f"PostgreSQL unreachable for read-only Stage 7 check: {exc}")

        app = create_app(cfg)
        with TestClient(app) as client:
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["model_ready"] is True

            rec = client.post("/api/v1/recommend", json=_health_payload())
            if not readiness["tables_present"]["policy_attributes"]:
                assert rec.status_code == 503
                assert "policy_attributes" in rec.json()["detail"]
            else:
                assert rec.status_code == 200
    finally:
        engine.dispose()
