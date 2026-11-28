"""M5.I1: Offline Missingness, Determinism, Profile-Sensitivity & Scale Test Suite.

Developer: Shravan Sambhaji Kapare (Roll No. 203)
Module: M5 — Policy Recommendation
Scope (Zero shared DB/migration/contract changes):
- Section A: Missing optional profile attributes & schema edge inputs.
- Section B: Missing optional policy attributes & 59-column schema preservation.
- Section C: Deterministic weighted/LightGBM scoring & equal-score tie-breaking.
- Section D: Controlled profile-attribute sensitivity & feature isolation.
- Section E: 520-policy synthetic multi-category catalog validation & diagnostics.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from ai_service.recommendation.features import (
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURE_COLUMNS,
    build_feature_matrix,
    extract_policy_features,
)
from ai_service.recommendation.filters import DuplicateCatalogPolicyError, filter_policies
from ai_service.recommendation.schemas import RecommendRequest, UserProfileRequest
from ai_service.recommendation.service import (
    CatalogIntegrityError,
    ModelState,
    load_model_state,
    recommend_policies_for_request,
)
from ai_service.recommendation.train_lightgbm import predict_ranking_scores
from ai_service.recommendation.weighted import score_policies_weighted

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def loaded_model_state() -> ModelState:
    """Load the tracked LightGBM artifact once per module without HTTP or DB I/O."""
    state = load_model_state()
    assert state.is_ready is True
    assert state.warmed_up is True
    assert state.model is not None
    return state


def _make_catalog_policy(
    *,
    policy_id: str | None = None,
    policy_type: str = "health",
    policy_status: str = "active",
    is_catalog_eligible: bool = True,
    annual_premium_min: Decimal | float = Decimal("10000.00"),
    annual_premium_max: Decimal | float = Decimal("14000.00"),
    sum_insured_min: Decimal | float = Decimal("500000.00"),
    sum_insured_max: Decimal | float = Decimal("1000000.00"),
    coverage_mode: str = "both",
    max_family_members: int = 4,
    deductible_amount: Decimal | float | None = Decimal("0.00"),
    copay_pct: Decimal | float | None = Decimal("0.00"),
    initial_waiting_period_days: int | None = 30,
    ped_covered: bool = True,
    ped_waiting_period_months: int | None = 24,
    claim_settlement_ratio: Decimal | float | None = Decimal("96.00"),
    network_hospitals_count: int = 7500,
    covered_benefits: list[str] | None = None,
    exclusions: list[str] | None = None,
    type_specific_metadata: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a deterministic controlled catalog row matching 003 DB structure."""
    row: dict[str, Any] = {
        "policy_id": policy_id or str(uuid.uuid4()),
        "organization_id": uuid.uuid4(),
        "policy_name": f"Controlled {policy_type.title()} Plan",
        "insurer_name": "CoverageIQ Test Assurance",
        "policy_type": policy_type,
        "description": "Controlled offline test fixture.",
        "policy_status": policy_status,
        "current_version_id": uuid.uuid4(),
        "policy_version_id": uuid.uuid4(),
        "is_catalog_eligible": is_catalog_eligible,
        "min_entry_age": 18,
        "max_entry_age": 65,
        "max_renewal_age": 85,
        "lifelong_renewal": False,
        "coverage_mode": coverage_mode,
        "max_family_members": max_family_members,
        "eligible_applicant_types": ["salaried", "self_employed", "student", "senior_citizen"],
        "annual_premium_min": annual_premium_min,
        "annual_premium_max": annual_premium_max,
        "sum_insured_min": sum_insured_min,
        "sum_insured_max": sum_insured_max,
        "deductible_amount": deductible_amount,
        "copay_pct": copay_pct,
        "room_rent_limit_type": "no_limit",
        "room_rent_limit_value": None,
        "initial_waiting_period_days": initial_waiting_period_days,
        "ped_covered": ped_covered,
        "ped_waiting_period_months": ped_waiting_period_months,
        "specific_disease_waiting_months": 24,
        "maternity_waiting_months": None,
        "network_hospitals_count": network_hospitals_count,
        "cashless_available": True,
        "pan_india_coverage": True,
        "available_regions": ["mumbai", "pune", "bengaluru"],
        "min_tenure_years": 1,
        "max_tenure_years": 3,
        "covered_benefits": (
            covered_benefits
            if covered_benefits is not None
            else ["cashless_hospitalization", "daycare_procedures", "ambulance_cover"]
        ),
        "exclusions": exclusions if exclusions is not None else ["cosmetic_surgery"],
        "claim_settlement_ratio": claim_settlement_ratio,
        "no_claim_bonus_pct": Decimal("20.00"),
        "restoration_benefit": True,
        "type_specific_metadata": type_specific_metadata or {},
        "extraction_source": "verified",
    }
    row.update(overrides)
    return row


def _health_profile(**overrides: Any) -> dict[str, Any]:
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
        "pre_existing_conditions": [],
        "tenure_years": 1,
        "annual_income_inr": 1_200_000.0,
        "risk_preference": "balanced",
    }
    profile.update(overrides)
    return profile


def _patch_catalog(
    monkeypatch: pytest.MonkeyPatch,
    catalog: list[dict[str, Any]],
    *,
    filter_by_type: bool = True,
) -> None:
    """Mock catalog DB loading without modifying recommendation service logic."""

    def _loader(
        _conn: Any,
        policy_type: str | None = None,
        active_only: bool = False,
    ) -> list[dict[str, Any]]:
        del active_only
        if filter_by_type and policy_type is not None:
            norm = policy_type.strip().lower()
            return [row for row in catalog if str(row.get("policy_type", "")).lower() == norm]
        return list(catalog)

    monkeypatch.setattr("ai_service.recommendation.service.load_catalog_policies_from_db", _loader)


# ---------------------------------------------------------------------------
# A. Missing Optional Profile Fields & Normalization Edge Cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["weighted", "lightgbm"])
def test_a1_missing_optional_profile_combinations_income_levels_and_warnings(
    monkeypatch: pytest.MonkeyPatch,
    loaded_model_state: ModelState,
    backend: str,
) -> None:
    """Verify full/partial/missing income & risk combinations across both ranking backends."""
    catalog = [
        _make_catalog_policy(
            policy_id="pol-budget",
            annual_premium_min=Decimal("8000"),
            annual_premium_max=Decimal("10000"),
            sum_insured_max=Decimal("500000"),
            claim_settlement_ratio=Decimal("93.0"),
        ),
        _make_catalog_policy(
            policy_id="pol-balanced",
            annual_premium_min=Decimal("12000"),
            annual_premium_max=Decimal("16000"),
            sum_insured_max=Decimal("1500000"),
            claim_settlement_ratio=Decimal("97.5"),
        ),
    ]
    _patch_catalog(monkeypatch, catalog)
    conn = MagicMock()

    res_full = recommend_policies_for_request(
        RecommendRequest.model_validate({"profile": _health_profile(), "ranking_model": backend}),
        conn,
        loaded_model_state,
    )
    res_no_inc = recommend_policies_for_request(
        RecommendRequest.model_validate(
            {"profile": _health_profile(annual_income_inr=None), "ranking_model": backend}
        ),
        conn,
        loaded_model_state,
    )
    res_no_risk = recommend_policies_for_request(
        RecommendRequest.model_validate(
            {"profile": _health_profile(risk_preference=None), "ranking_model": backend}
        ),
        conn,
        loaded_model_state,
    )
    res_both_missing = recommend_policies_for_request(
        RecommendRequest.model_validate(
            {
                "profile": _health_profile(annual_income_inr=None, risk_preference=None),
                "ranking_model": backend,
            }
        ),
        conn,
        loaded_model_state,
    )

    for res in (res_full, res_no_inc, res_no_risk, res_both_missing):
        assert res.status == "RECOMMENDATIONS_READY"
        assert len(res.recommendations) == 2
        assert all(math.isfinite(rec.score) for rec in res.recommendations)

    # Full vs missing completeness & warnings
    assert res_full.profile_completeness is not None
    assert res_full.profile_completeness.is_complete is True
    assert res_full.profile_completeness.missing_fields == []
    assert res_full.profile_completeness.income_distribution_mismatch_risk is False

    assert res_no_inc.profile_completeness is not None
    assert res_no_inc.profile_completeness.is_complete is False
    assert res_no_inc.profile_completeness.missing_fields == ["annual_income_inr"]
    assert res_no_inc.profile_completeness.income_distribution_mismatch_risk is (
        backend == "lightgbm"
    )
    assert any("annual_income_inr" in w for w in res_no_inc.warnings)

    assert res_no_risk.profile_completeness is not None
    assert res_no_risk.profile_completeness.missing_fields == ["risk_preference"]
    assert (
        res_no_risk.profile_completeness.applied_assumptions["risk_preference"]
        == "defaulted_to_balanced"
    )
    assert any("risk_preference" in w for w in res_no_risk.warnings)

    assert res_both_missing.profile_completeness is not None
    assert set(res_both_missing.profile_completeness.missing_fields) == {
        "annual_income_inr",
        "risk_preference",
    }

    # Missing annual_income_inr must NEVER confer any favorable change in weighted backend
    weighted_scores_by_inc = []
    for inc in (None, 300_000.0, 1_200_000.0, 3_000_000.0):
        resp_inc = recommend_policies_for_request(
            RecommendRequest.model_validate(
                {"profile": _health_profile(annual_income_inr=inc), "ranking_model": backend}
            ),
            conn,
            loaded_model_state,
        )
        assert all(math.isfinite(r.score) for r in resp_inc.recommendations)
        weighted_scores_by_inc.append([(r.policy_id, r.score) for r in resp_inc.recommendations])

    if backend == "weighted":
        assert all(s == weighted_scores_by_inc[0] for s in weighted_scores_by_inc[1:])
        assert [(r.policy_id, r.score) for r in res_no_risk.recommendations] == [
            (r.policy_id, r.score) for r in res_full.recommendations
        ]


def test_a2_permitted_whitespace_and_rejected_malformed_optional_and_mandatory_fields() -> None:
    """Whitespace optional strings normalize cleanly; invalid optional/mandatory raise 422."""
    # Schema permits whitespace-only strings on string fields (normalizes cleanly to None)
    normalized = UserProfileRequest.model_validate(
        _health_profile(
            risk_preference="   ",
            location="  ",
            smoking_status=" ",
            vehicle_type="  ",
        )
    )
    assert normalized.risk_preference is None
    assert normalized.location is None
    assert normalized.smoking_status is None
    assert normalized.vehicle_type is None

    # Encoded features remain finite and match defaulted 'balanced' + empty income indicator
    feats = extract_policy_features(normalized, _make_catalog_policy())
    assert feats["risk_preference_encoded"] == 1.0
    assert math.isfinite(feats["annual_income"])

    # Malformed optional numeric/enum/bounds must fail ValidationError loudly (never NaN/Inf)
    bad_optional_cases = [
        {"annual_income_inr": float("nan")},
        {"annual_income_inr": float("inf")},
        {"annual_income_inr": -5000.0},
        {"risk_preference": "ultra_risky"},
        {"smoking_status": "occasional"},
        {"max_copay_pct": -1.0},
        {"max_copay_pct": 105.0},
        {"max_ped_waiting_months": -3},
        {"max_deductible": -100.0},
    ]
    for bad_override in bad_optional_cases:
        with pytest.raises(ValidationError):
            UserProfileRequest.model_validate(_health_profile(**bad_override))

    # Mandatory validation remains strictly enforced
    for mandatory_drop in ("policy_type", "age", "budget_max", "min_sum_insured"):
        payload = _health_profile()
        del payload[mandatory_drop]
        with pytest.raises(ValidationError):
            UserProfileRequest.model_validate(payload)


# ---------------------------------------------------------------------------
# B. Missing Optional Policy Attributes & 59-Feature Schema Integrity
# ---------------------------------------------------------------------------


def test_b1_missing_policy_attributes_preserve_59_schema_and_scoring_semantics(
    loaded_model_state: ModelState,
) -> None:
    """Verify CSR=None conservative handling, PED=None encoding, and strict 59-column parity."""
    prof = UserProfileRequest.model_validate(_health_profile())
    pol_full = _make_catalog_policy(
        policy_id="pol-full",
        claim_settlement_ratio=Decimal("96.00"),
        ped_waiting_period_months=36,
    )
    pol_no_csr = _make_catalog_policy(
        policy_id="pol-no-csr",
        claim_settlement_ratio=None,
        ped_waiting_period_months=36,
    )
    pol_no_ped_wait = _make_catalog_policy(
        policy_id="pol-no-ped",
        claim_settlement_ratio=Decimal("96.00"),
        ped_waiting_period_months=None,
    )
    pol_sparse_opt = _make_catalog_policy(
        policy_id="pol-sparse",
        claim_settlement_ratio=None,
        ped_waiting_period_months=None,
        max_renewal_age=None,
        lifelong_renewal=True,
        deductible_amount=None,
        copay_pct=None,
        initial_waiting_period_days=None,
        no_claim_bonus_pct=None,
    )

    matrix = build_feature_matrix(prof, [pol_full, pol_no_csr, pol_no_ped_wait, pol_sparse_opt])
    assert matrix.shape == (4, 59)
    assert tuple(matrix.columns) == FEATURE_COLUMNS
    assert set(matrix.columns).isdisjoint(FORBIDDEN_FEATURE_COLUMNS)
    assert all(dtype == np.float64 for dtype in matrix.dtypes)
    assert np.isfinite(matrix.to_numpy()).all()

    # Exact feature missingness indicators and conservative CSR handling
    assert list(matrix["has_claim_settlement_ratio"]) == [1.0, 0.0, 1.0, 0.0]
    assert list(matrix["claim_settlement_ratio"]) == [96.0, 0.0, 96.0, 0.0]
    assert list(matrix.loc[[0, 1], "ped_waiting_period_days"]) == [1080.0, 1080.0]
    for row_idx in (2, 3):
        ped_days = float(matrix.loc[row_idx, "ped_waiting_period_days"])
        assert math.isfinite(ped_days) and ped_days >= 0.0
    # Sparse lifelong_renewal=True policy with age=32 yields remaining_eligible_years=68.0
    assert matrix.loc[3, "remaining_eligible_years"] == 68.0
    assert matrix.loc[3, "deductible_amount"] == 0.0
    assert matrix.loc[3, "co_payment_percentage"] == 0.0

    # Weighted CSR=None uses conservative default_missing_csr_pct (90% < 96%) -> lower score
    w_res = score_policies_weighted(matrix)
    assert np.isfinite(w_res.scores).all()
    assert w_res.scores[1] < w_res.scores[0]
    assert (
        w_res.dimension_breakdowns[1]["weighted_insurer_network_score"]
        < w_res.dimension_breakdowns[0]["weighted_insurer_network_score"]
    )
    # Unrecorded ped_waiting_period_months asserts neutral finite/bounded behavior only (M5.I2)
    for row_idx in (2, 3):
        assert 0.0 <= float(w_res.scores[row_idx]) <= 1.0
        assert 0.0 <= w_res.dimension_breakdowns[row_idx]["weighted_cost_waiting_score"] <= 1.0

    # LightGBM inference over combined profile/policy missingness remains finite & error-free
    prof_missing = UserProfileRequest.model_validate(
        _health_profile(annual_income_inr=None, risk_preference=None)
    )
    matrix_both_missing = build_feature_matrix(prof_missing, [pol_sparse_opt, pol_full])
    assert list(matrix_both_missing["has_annual_income"]) == [0.0, 0.0]
    assert list(matrix_both_missing["has_claim_settlement_ratio"]) == [0.0, 1.0]
    lgb_scores = predict_ranking_scores(loaded_model_state.model, matrix_both_missing)  # type: ignore[arg-type]
    assert np.isfinite(lgb_scores).all()


def test_b2_optional_policy_missingness_across_car_life_term(
    loaded_model_state: ModelState,
) -> None:
    """Missing optional attributes in car/life/term stay isolated and zero-free of leaks."""
    assert loaded_model_state.model is not None

    car_prof = UserProfileRequest.model_validate(
        _health_profile(policy_type="car", vehicle_type="sedan", vehicle_age_years=None)
    )
    car_pol = _make_catalog_policy(
        policy_id="car-opt-null",
        policy_type="car",
        claim_settlement_ratio=None,
        ped_waiting_period_months=None,
        deductible_amount=None,
        type_specific_metadata={"vehicle_types": ["sedan"], "vehicle_age_limit_years": None},
    )
    car_df = build_feature_matrix(car_prof, [car_pol])
    assert car_df.shape == (1, 59)
    assert car_df.loc[0, "is_car"] == 1.0
    assert car_df.loc[0, "has_vehicle_age"] == 0.0
    assert car_df.loc[0, "has_claim_settlement_ratio"] == 0.0
    assert car_df.loc[0, "ped_waiting_period_days"] == 0.0
    assert car_df.loc[0, "life_cover"] == 0.0

    life_prof = UserProfileRequest.model_validate(
        _health_profile(
            policy_type="life",
            min_sum_insured=1_000_000.0,
            tenure_years=20,
            required_benefits=["death_benefit"],
            annual_income_inr=None,
            risk_preference=None,
        )
    )
    life_pol_empty = _make_catalog_policy(
        policy_id="life-opt-empty-meta",
        policy_type="life",
        coverage_mode="individual",
        max_family_members=1,
        annual_premium_min=Decimal("18000.00"),
        annual_premium_max=Decimal("22000.00"),
        sum_insured_max=Decimal("2500000.00"),
        max_tenure_years=25,
        max_renewal_age=None,
        deductible_amount=None,
        copay_pct=None,
        initial_waiting_period_days=None,
        ped_waiting_period_months=None,
        claim_settlement_ratio=None,
        no_claim_bonus_pct=None,
        covered_benefits=["death_benefit"],
        type_specific_metadata={},
    )
    life_pol_partial = _make_catalog_policy(
        policy_id="life-opt-partial-meta",
        policy_type="life",
        coverage_mode="individual",
        max_family_members=1,
        annual_premium_min=Decimal("19000.00"),
        annual_premium_max=Decimal("24000.00"),
        sum_insured_max=Decimal("2500000.00"),
        max_tenure_years=25,
        max_renewal_age=None,
        deductible_amount=None,
        copay_pct=None,
        initial_waiting_period_days=None,
        ped_waiting_period_months=None,
        claim_settlement_ratio=None,
        no_claim_bonus_pct=None,
        covered_benefits=["death_benefit"],
        type_specific_metadata={
            "life_cover_max": 4_000_000,
            "policy_term_years_max": None,
            "maturity_benefit": None,
            "surrender_value_available": True,
            "premium_payment_term_years": None,
        },
    )
    life_filtered = filter_policies(life_prof, [life_pol_empty, life_pol_partial])
    assert life_filtered.status == "ELIGIBLE_CANDIDATES_FOUND"
    assert len(life_filtered.eligible_policies) == 2

    life_df = build_feature_matrix(life_prof, life_filtered.eligible_policies)
    assert life_df.shape == (2, 59)
    assert tuple(life_df.columns) == FEATURE_COLUMNS
    assert set(life_df.columns).isdisjoint(FORBIDDEN_FEATURE_COLUMNS)
    assert all(dtype == np.float64 for dtype in life_df.dtypes)
    assert np.isfinite(life_df.to_numpy()).all()

    assert list(life_df["policy_type_encoded"]) == [2.0, 2.0]
    assert list(life_df["is_life"]) == [1.0, 1.0]
    assert list(life_df["is_health"]) == [0.0, 0.0]
    assert list(life_df["is_car"]) == [0.0, 0.0]
    assert list(life_df["is_term"]) == [0.0, 0.0]
    assert list(life_df["has_annual_income"]) == [0.0, 0.0]
    assert list(life_df["has_claim_settlement_ratio"]) == [0.0, 0.0]
    assert list(life_df["life_cover"]) == [2_500_000.0, 4_000_000.0]
    assert list(life_df["requested_life_cover"]) == [1_000_000.0, 1_000_000.0]
    assert list(life_df["life_cover_ratio"]) == pytest.approx([2.5, 4.0])
    assert list(life_df["policy_term_years"]) == [25.0, 25.0]
    assert list(life_df["premium_payment_term_years"]) == [0.0, 0.0]
    assert list(life_df["maturity_benefit_indicator"]) == [0.0, 0.0]
    assert list(life_df["surrender_value_indicator"]) == [0.0, 1.0]

    # Unrelated category-specific features must stay strictly isolated for life policies
    isolated_zero_cols = (
        "hospital_network_count",
        "ped_coverage_indicator",
        "ped_waiting_period_days",
        "initial_waiting_period_days",
        "co_payment_percentage",
        "has_vehicle_age",
        "vehicle_age",
        "vehicle_age_limit",
        "own_damage_indicator",
        "smoking_eligibility_compatibility",
        "critical_illness_rider_indicator",
        "accidental_death_rider_indicator",
    )
    for iso_col in isolated_zero_cols:
        assert list(life_df[iso_col]) == [0.0, 0.0]

    life_lgb_scores = predict_ranking_scores(loaded_model_state.model, life_df)
    life_weighted_res = score_policies_weighted(life_df)
    assert len(life_lgb_scores) == 2 and np.isfinite(life_lgb_scores).all()
    assert len(life_weighted_res.scores) == 2 and np.isfinite(life_weighted_res.scores).all()

    term_prof = UserProfileRequest.model_validate(
        _health_profile(
            policy_type="term",
            min_sum_insured=5_000_000.0,
            tenure_years=20,
            smoking_status=None,
            annual_income_inr=None,
        )
    )
    term_pol = _make_catalog_policy(
        policy_id="term-opt-null",
        policy_type="term",
        sum_insured_max=Decimal("10000000.00"),
        max_tenure_years=30,
        claim_settlement_ratio=None,
        ped_waiting_period_months=None,
        type_specific_metadata={},
    )
    term_df = build_feature_matrix(term_prof, [term_pol])
    assert term_df.loc[0, "is_term"] == 1.0
    assert term_df.loc[0, "life_cover"] == 10_000_000.0
    assert term_df.loc[0, "policy_term_years"] == 30.0
    assert term_df.loc[0, "premium_payment_term_years"] == 0.0
    assert term_df.loc[0, "hospital_network_count"] == 0.0
    assert np.isfinite(predict_ranking_scores(loaded_model_state.model, term_df)).all()
    assert np.isfinite(score_policies_weighted(term_df).scores).all()


# ---------------------------------------------------------------------------
# C. Ranking Determinism, Equal-Score Tie-Breaking & Uniqueness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["weighted", "lightgbm"])
def test_c1_ranking_determinism_permutation_invariant_tie_breaking(
    monkeypatch: pytest.MonkeyPatch,
    loaded_model_state: ModelState,
    backend: str,
) -> None:
    """Identical inputs & permuted equal-score ties always yield identical deterministic order."""
    shared_attrs: dict[str, Any] = {
        "policy_type": "health",
        "annual_premium_min": Decimal("11000.00"),
        "annual_premium_max": Decimal("15000.00"),
        "sum_insured_max": Decimal("1000000.00"),
        "claim_settlement_ratio": Decimal("97.00"),
        "network_hospitals_count": 7000,
    }
    p_c = _make_catalog_policy(policy_id="tie-policy-c", **shared_attrs)
    p_a = _make_catalog_policy(policy_id="tie-policy-a", **shared_attrs)
    p_b = _make_catalog_policy(policy_id="tie-policy-b", **shared_attrs)
    p_better = _make_catalog_policy(
        policy_id="top-winner",
        annual_premium_min=Decimal("9000.00"),
        annual_premium_max=Decimal("10000.00"),
        sum_insured_max=Decimal("2000000.00"),
        claim_settlement_ratio=Decimal("99.00"),
        network_hospitals_count=11000,
    )

    orderings = [
        [p_c, p_a, p_b, p_better],
        [p_better, p_b, p_c, p_a],
    ]
    req = RecommendRequest.model_validate(
        {"profile": _health_profile(), "top_n": 10, "ranking_model": backend}
    )
    snapshots: list[list[tuple[str, int, float]]] = []
    for catalog_order in orderings:
        _patch_catalog(monkeypatch, catalog_order)
        res = recommend_policies_for_request(req, MagicMock(), loaded_model_state)
        assert res.status == "RECOMMENDATIONS_READY"
        rec_ids = [r.policy_id for r in res.recommendations]
        assert len(rec_ids) == len(set(rec_ids)) == 4
        scores = [r.score for r in res.recommendations]
        assert all(math.isfinite(s) for s in scores)
        assert scores == sorted(scores, reverse=True)
        # Equal-score tie group must break strictly by ascending policy_id regardless of input order
        tie_ids_in_result = [pid for pid in rec_ids if pid.startswith("tie-policy-")]
        assert tie_ids_in_result == ["tie-policy-a", "tie-policy-b", "tie-policy-c"]
        snapshots.append([(r.policy_id, r.rank, r.score) for r in res.recommendations])

    assert snapshots[0] == snapshots[1]


# ---------------------------------------------------------------------------
# D. Controlled Profile-Attribute Sensitivity & Feature Isolation
# ---------------------------------------------------------------------------


def test_d1_profile_perturbations_isolate_expected_features_and_preserve_legitimate_ties(
    monkeypatch: pytest.MonkeyPatch,
    loaded_model_state: ModelState,
) -> None:
    """Controlled profile changes mutate intended features cleanly while leaving others fixed."""
    pol_cheap = _make_catalog_policy(
        policy_id="pol-cheap",
        annual_premium_min=Decimal("4000"),
        annual_premium_max=Decimal("5000"),
        sum_insured_max=Decimal("1000000"),
        covered_benefits=["cashless_hospitalization", "daycare_procedures", "ambulance_cover"],
        restoration_benefit=False,
        no_claim_bonus_pct=Decimal("10"),
    )
    pol_rich = _make_catalog_policy(
        policy_id="pol-rich",
        annual_premium_min=Decimal("22000"),
        annual_premium_max=Decimal("24000"),
        sum_insured_max=Decimal("2500000"),
        covered_benefits=[
            "cashless_hospitalization",
            "daycare_procedures",
            "ambulance_cover",
            "annual_health_checkup",
            "maternity_cover",
        ],
        restoration_benefit=True,
        no_claim_bonus_pct=Decimal("50"),
    )

    # 1. budget_max sensitivity -> changes budget/affordability features only
    base_prof = UserProfileRequest.model_validate(_health_profile(budget_max=25000.0))
    high_budget_prof = UserProfileRequest.model_validate(_health_profile(budget_max=45000.0))
    f_base = extract_policy_features(base_prof, pol_rich)
    f_high = extract_policy_features(high_budget_prof, pol_rich)
    budget_cols = {
        "monthly_budget",
        "premium_to_budget_ratio",
        "affordability_margin",
        "affordability_score",
    }
    assert f_high["monthly_budget"] > f_base["monthly_budget"]
    assert f_high["premium_to_budget_ratio"] < f_base["premium_to_budget_ratio"]
    assert f_high["affordability_margin"] > f_base["affordability_margin"]
    for col in FEATURE_COLUMNS:
        if col not in budget_cols:
            assert f_base[col] == f_high[col]

    # 2. min_sum_insured sensitivity -> changes coverage demand & headroom features only
    high_cov_prof = UserProfileRequest.model_validate(_health_profile(min_sum_insured=1_000_000.0))
    f_cov = extract_policy_features(high_cov_prof, pol_rich)
    cov_cols = {"requested_sum_insured", "coverage_to_requirement_ratio", "coverage_headroom"}
    assert f_cov["requested_sum_insured"] == 1_000_000.0
    assert f_cov["coverage_to_requirement_ratio"] < f_base["coverage_to_requirement_ratio"]
    for col in FEATURE_COLUMNS:
        if col not in cov_cols:
            assert f_base[col] == f_cov[col]

    # 3. risk_preference sensitivity -> alters risk_preference_encoded only & flips weighted rank
    low_prem_prof = UserProfileRequest.model_validate(
        _health_profile(risk_preference="low_premium")
    )
    comp_prof = UserProfileRequest.model_validate(
        _health_profile(risk_preference="comprehensive_coverage")
    )
    f_lp = extract_policy_features(low_prem_prof, pol_rich)
    f_cp = extract_policy_features(comp_prof, pol_rich)
    assert f_lp["risk_preference_encoded"] == 0.0
    assert f_cp["risk_preference_encoded"] == 2.0
    for col in FEATURE_COLUMNS:
        if col != "risk_preference_encoded":
            assert f_lp[col] == f_cp[col]

    _patch_catalog(monkeypatch, [pol_rich, pol_cheap])
    res_lp = recommend_policies_for_request(
        RecommendRequest(profile=low_prem_prof, ranking_model="weighted"),
        MagicMock(),
        loaded_model_state,
    )
    res_cp = recommend_policies_for_request(
        RecommendRequest(profile=comp_prof, ranking_model="weighted"),
        MagicMock(),
        loaded_model_state,
    )
    assert res_lp.recommendations[0].policy_id == "pol-cheap"
    assert res_cp.recommendations[0].policy_id == "pol-rich"

    # 4. required_benefits sensitivity -> modifies benefit match ratio only
    more_req_prof = UserProfileRequest.model_validate(
        _health_profile(required_benefits=["cashless_hospitalization", "maternity_cover"])
    )
    f_req = extract_policy_features(more_req_prof, pol_rich)
    benefit_cols = {"matched_required_benefits", "required_benefit_match_ratio"}
    assert f_req["matched_required_benefits"] == 2.0
    assert f_req["required_benefit_match_ratio"] == 1.0
    for col in FEATURE_COLUMNS:
        if col not in benefit_cols:
            assert f_req[col] == f_base[col]

    # 5. age perturbation on identical policies -> updates age features cleanly, tie preserved
    pol_tie_1 = _make_catalog_policy(policy_id="age-tie-1")
    pol_tie_2 = _make_catalog_policy(policy_id="age-tie-2")
    _patch_catalog(monkeypatch, [pol_tie_2, pol_tie_1])
    young_prof = UserProfileRequest.model_validate(_health_profile(age=25))
    older_prof = UserProfileRequest.model_validate(_health_profile(age=50))
    f_young = extract_policy_features(young_prof, pol_tie_1)
    f_older = extract_policy_features(older_prof, pol_tie_1)
    age_cols = {"applicant_age", "age_eligibility_margin", "remaining_eligible_years"}
    assert f_older["age_eligibility_margin"] < f_young["age_eligibility_margin"]
    for col in FEATURE_COLUMNS:
        if col not in age_cols:
            assert f_older[col] == f_young[col]

    r_young = recommend_policies_for_request(
        RecommendRequest(profile=young_prof, ranking_model="weighted"),
        MagicMock(),
        loaded_model_state,
    )
    r_older = recommend_policies_for_request(
        RecommendRequest(profile=older_prof, ranking_model="weighted"),
        MagicMock(),
        loaded_model_state,
    )
    assert [r.policy_id for r in r_young.recommendations] == ["age-tie-1", "age-tie-2"]
    assert [r.policy_id for r in r_older.recommendations] == ["age-tie-1", "age-tie-2"]


# ---------------------------------------------------------------------------
# E. Larger Synthetic Catalog (>= 500 Policies Across All 4 Categories)
# ---------------------------------------------------------------------------


def _build_520_policy_multi_category_catalog() -> list[dict[str, Any]]:
    """Generate 520 deterministic catalog rows (130 per category) mixing valid & invalid rows."""
    catalog: list[dict[str, Any]] = []
    for idx in range(130):
        # Mix active valid rows, draft/ineligible, malformed NaN, inverted age bounds
        invalid_status = "draft" if idx % 13 == 0 else "active"
        eligible_flag = False if idx % 17 == 0 else True
        prem_min: Any = Decimal(str(7000 + (idx % 20) * 900))
        prem_max: Any = Decimal(str(11000 + (idx % 20) * 900))
        min_age = 18
        max_age = 65
        if idx % 29 == 0:
            prem_min = "NaN"  # corrupt non-finite row
        if idx % 31 == 0:
            min_age, max_age = 60, 20  # inverted bounds

        catalog.append(
            _make_catalog_policy(
                policy_id=f"hlt-{idx:03d}",
                policy_type="health",
                policy_status=invalid_status,
                is_catalog_eligible=eligible_flag,
                min_entry_age=min_age,
                max_entry_age=max_age,
                annual_premium_min=prem_min,
                annual_premium_max=prem_max,
                sum_insured_max=Decimal(str(500000 + (idx % 6) * 500000)),
                claim_settlement_ratio=None if idx % 5 == 0 else Decimal("96.00"),
                ped_covered=(idx % 4 != 0),
                ped_waiting_period_months=None if idx % 4 == 0 else 24,
            )
        )

    for idx in range(130):
        catalog.append(
            _make_catalog_policy(
                policy_id=f"car-{idx:03d}",
                policy_type="car",
                policy_status="archived" if idx % 19 == 0 else "active",
                is_catalog_eligible=(idx % 23 != 0),
                coverage_mode="individual",
                max_family_members=1,
                annual_premium_min=Decimal(str(5000 + (idx % 15) * 1000)),
                annual_premium_max=Decimal(str(9000 + (idx % 15) * 1000)),
                sum_insured_max=Decimal(str(300000 + (idx % 5) * 300000)),
                claim_settlement_ratio=None if idx % 6 == 0 else Decimal("94.50"),
                type_specific_metadata={
                    "vehicle_types": ["two_wheeler"] if idx % 10 == 0 else ["hatchback", "sedan"],
                    "vehicle_age_limit_years": 2 if idx % 12 == 0 else 12,
                    "own_damage_coverage": True,
                    "third_party_coverage": True,
                    "zero_depreciation": idx % 2 == 0,
                },
            )
        )

    for idx in range(130):
        catalog.append(
            _make_catalog_policy(
                policy_id=f"lif-{idx:03d}",
                policy_type="life",
                policy_status="draft" if idx % 17 == 0 else "active",
                is_catalog_eligible=(idx % 21 != 0),
                coverage_mode="individual",
                max_family_members=1,
                annual_premium_min=Decimal(str(15000 + (idx % 12) * 2000)),
                annual_premium_max=Decimal(str(22000 + (idx % 12) * 2000)),
                sum_insured_max=Decimal(str(1000000 + (idx % 4) * 1500000)),
                covered_benefits=["death_benefit", "maturity_benefit"],
                type_specific_metadata={
                    "life_cover_min": 500000,
                    "life_cover_max": 1000000 + (idx % 4) * 1500000,
                    "policy_term_years_min": 5,
                    "policy_term_years_max": 30,
                    "premium_payment_term_years": 15,
                    "maturity_benefit": idx % 3 != 0,
                    "surrender_value_available": True,
                },
            )
        )

    for idx in range(130):
        catalog.append(
            _make_catalog_policy(
                policy_id=f"trm-{idx:03d}",
                policy_type="term",
                policy_status="active" if idx % 15 != 0 else "draft",
                is_catalog_eligible=(idx % 19 != 0),
                coverage_mode="individual",
                max_family_members=1,
                annual_premium_min=Decimal(str(6000 + (idx % 14) * 1000)),
                annual_premium_max=Decimal(str(10000 + (idx % 14) * 1000)),
                sum_insured_max=Decimal(str(5000000 + (idx % 3) * 5000000)),
                covered_benefits=["pure_term_life_cover"],
                type_specific_metadata={
                    "life_cover_min": 2500000,
                    "life_cover_max": 5000000 + (idx % 3) * 5000000,
                    "policy_term_years_min": 10,
                    "policy_term_years_max": 40,
                    "premium_payment_term_years": 20,
                    "smoking_status_eligibility": "non_smoker_only" if idx % 5 == 0 else "all",
                    "critical_illness_rider": idx % 2 == 0,
                    "accidental_death_rider": True,
                },
            )
        )
    return catalog


def test_e1_large_520_policy_multi_category_catalog_determinism_and_isolation(
    monkeypatch: pytest.MonkeyPatch,
    loaded_model_state: ModelState,
) -> None:
    """Validate 520-policy catalog filtering, isolation, determinism & duplicate rejection."""
    catalog = _build_520_policy_multi_category_catalog()
    assert len(catalog) == 520
    by_id = {row["policy_id"]: row for row in catalog}

    # Unfiltered call against full 520-row pool ensures cross-category isolation in pure filtering
    health_prof = UserProfileRequest.model_validate(
        _health_profile(pre_existing_conditions=["diabetes"], annual_income_inr=None)
    )
    pure_health_res = filter_policies(health_prof, catalog)
    assert pure_health_res.filter_report.total_candidates == 520
    assert pure_health_res.filter_report.passed_count > 0
    for ep in pure_health_res.eligible_policies:
        assert ep["policy_type"] == "health"
        assert ep["policy_status"] == "active"
        assert ep["is_catalog_eligible"] is True
        assert ep["ped_covered"] is True
        assert ep["annual_premium_min"] != "NaN"
        assert ep["min_entry_age"] <= ep["max_entry_age"]

    # End-to-end service calls per category across both weighted and LightGBM backends
    _patch_catalog(monkeypatch, catalog, filter_by_type=False)
    category_profiles = {
        "health": _health_profile(pre_existing_conditions=["diabetes"], annual_income_inr=None),
        "car": {
            "policy_type": "car",
            "age": 30,
            "budget_max": 20000,
            "min_sum_insured": 300000,
            "vehicle_type": "sedan",
            "vehicle_age_years": 5,
            "require_own_damage": True,
        },
        "life": {
            "policy_type": "life",
            "age": 35,
            "budget_max": 40000,
            "min_sum_insured": 1000000,
            "tenure_years": 20,
            "require_maturity_benefit": True,
        },
        "term": {
            "policy_type": "term",
            "age": 32,
            "budget_max": 20000,
            "min_sum_insured": 5000000,
            "tenure_years": 25,
            "smoking_status": "smoker",
            "require_critical_illness_rider": True,
        },
    }

    for backend in ("weighted", "lightgbm"):
        for cat, prof_data in category_profiles.items():
            req = RecommendRequest.model_validate(
                {"profile": prof_data, "top_n": 10, "ranking_model": backend}
            )
            t0 = time.perf_counter()
            r1 = recommend_policies_for_request(req, MagicMock(), loaded_model_state)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            r2 = recommend_policies_for_request(req, MagicMock(), loaded_model_state)

            logger.info(
                "520-catalog diagnostic | category=%s backend=%s elapsed_ms=%.2f passed=%d",
                cat,
                backend,
                elapsed_ms,
                r1.filter_report.passed_count,
            )

            assert r1.status == "RECOMMENDATIONS_READY"
            assert [(x.policy_id, x.score) for x in r1.recommendations] == [
                (x.policy_id, x.score) for x in r2.recommendations
            ]
            rec_ids = [x.policy_id for x in r1.recommendations]
            assert len(rec_ids) == len(set(rec_ids))
            scores = [x.score for x in r1.recommendations]
            assert all(math.isfinite(s) for s in scores)
            assert scores == sorted(scores, reverse=True)

            # Assert zero invalid/cross-category leakage
            for pid in rec_ids:
                src = by_id[pid]
                assert src["policy_type"] == cat
                assert src["policy_status"] == "active"
                assert src["is_catalog_eligible"] is True

    # Duplicate policy_id rejection on 520+ catalog
    dup_catalog = catalog + [dict(catalog[0])]
    with pytest.raises(DuplicateCatalogPolicyError, match="hlt-000"):
        filter_policies(health_prof, dup_catalog)

    _patch_catalog(monkeypatch, dup_catalog, filter_by_type=False)
    with pytest.raises(CatalogIntegrityError, match="duplicate policy_id"):
        recommend_policies_for_request(
            RecommendRequest.model_validate({"profile": category_profiles["health"]}),
            MagicMock(),
            loaded_model_state,
        )
