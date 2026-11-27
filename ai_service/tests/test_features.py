"""Comprehensive tests for Stage 5 Feature Engineering Pipeline."""

import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ai_service.recommendation.dataset import load_synthetic_dataset
from ai_service.recommendation.features import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_METADATA,
    FEATURE_SCHEMA_VERSION,
    FORBIDDEN_FEATURE_COLUMNS,
    MissingMandatoryFeatureInputError,
    build_feature_matrix,
    build_training_feature_dataset,
    extract_policy_features,
)
from ai_service.recommendation.schemas import UserProfileRequest


def _sample_policy(policy_type: str = "health", **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "policy_id": "POL-TEST-1",
        "policy_code": "POL-HLT-999",
        "policy_name": "Sample Plan",
        "policy_type": policy_type,
        "min_entry_age": 18,
        "max_entry_age": 65,
        "max_renewal_age": 80,
        "lifelong_renewal": False,
        "coverage_mode": "both",
        "annual_premium_min": 10_000.0,
        "annual_premium_max": 20_000.0,
        "sum_insured_min": 500_000.0,
        "sum_insured_max": 2_000_000.0,
        "deductible_amount": 2_000.0,
        "copay_pct": 10.0,
        "room_rent_limit_type": "no_limit",
        "initial_waiting_period_days": 30,
        "ped_covered": True,
        "ped_waiting_period_months": 24,
        "network_hospitals_count": 7_500,
        "cashless_available": True,
        "pan_india_coverage": True,
        "max_tenure_years": 3,
        "covered_benefits": ["cashless_hospitalization", "daycare_procedures", "ambulance_cover"],
        "exclusions": ["cosmetic_surgery"],
        "claim_settlement_ratio": 96.5,
        "no_claim_bonus_pct": 20.0,
        "restoration_benefit": True,
        "type_specific_metadata": {},
    }
    base.update(overrides)
    return base


def _sample_profile(policy_type: str = "health", **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "profile_id": "PROF-TEST-1",
        "preferred_policy_type": policy_type,
        "age": 35,
        "annual_income_inr": 1_200_000.0,
        "monthly_budget_inr": 2_500.0,
        "budget_min": 8_000.0,
        "budget_max": 30_000.0,
        "family_size": 2,
        "coverage_mode": "family_floater",
        "min_sum_insured": 1_000_000.0,
        "required_benefits": ["cashless_hospitalization", "daycare_procedures"],
        "pre_existing_conditions": [],
        "tenure_years": 1,
        "risk_preference": "balanced",
        "smoking_status": "non_smoker",
    }
    base.update(overrides)
    return base


def test_health_car_life_term_feature_extraction_and_category_isolation() -> None:
    """Tests 1-4 & 10: Extract features per category without cross-category contamination."""
    # Health
    health_feats = extract_policy_features(_sample_profile("health"), _sample_policy("health"))
    assert health_feats["policy_type_encoded"] == 0.0
    assert health_feats["is_health"] == 1.0
    assert health_feats["is_car"] == 0.0
    assert health_feats["ped_coverage_indicator"] == 1.0
    assert health_feats["ped_waiting_period_days"] == 720.0
    assert health_feats["hospital_network_count"] == 7_500.0
    assert health_feats["family_floater_indicator"] == 1.0
    assert health_feats["own_damage_indicator"] == 0.0
    assert health_feats["life_cover"] == 0.0

    # Car (network_hospitals_count on car row must NOT populate hospital_network_count)
    car_policy = _sample_policy(
        "car",
        network_hospitals_count=5_000,
        type_specific_metadata={
            "vehicle_types": ["sedan"],
            "vehicle_age_limit_years": 10,
            "own_damage_coverage": True,
            "third_party_coverage": True,
            "zero_depreciation": True,
            "roadside_assistance": False,
            "engine_protection": True,
        },
    )
    car_feats = extract_policy_features(
        _sample_profile("car", vehicle_age_years=3, family_size=1, coverage_mode="individual"),
        car_policy,
    )
    assert car_feats["policy_type_encoded"] == 1.0
    assert car_feats["is_car"] == 1.0
    assert car_feats["has_vehicle_age"] == 1.0
    assert car_feats["vehicle_age"] == 3.0
    assert car_feats["vehicle_age_limit"] == 10.0
    assert car_feats["vehicle_age_eligibility_margin"] == 7.0
    assert car_feats["own_damage_indicator"] == 1.0
    assert car_feats["zero_depreciation_indicator"] == 1.0
    assert car_feats["roadside_assistance_indicator"] == 0.0
    assert car_feats["hospital_network_count"] == 0.0

    # Life
    life_policy = _sample_policy(
        "life",
        sum_insured_max=5_000_000.0,
        max_tenure_years=25,
        type_specific_metadata={
            "life_cover_max": 5_000_000.0,
            "policy_term_years_max": 25,
            "premium_payment_term_years": 10,
            "maturity_benefit": True,
            "surrender_value_available": True,
        },
    )
    life_feats = extract_policy_features(
        _sample_profile("life", min_sum_insured=2_500_000.0, family_size=1),
        life_policy,
    )
    assert life_feats["policy_type_encoded"] == 2.0
    assert life_feats["is_life"] == 1.0
    assert life_feats["life_cover"] == 5_000_000.0
    assert life_feats["requested_life_cover"] == 2_500_000.0
    assert life_feats["life_cover_ratio"] == pytest.approx(2.0)
    assert life_feats["policy_term_years"] == 25.0
    assert life_feats["premium_payment_term_years"] == 10.0
    assert life_feats["maturity_benefit_indicator"] == 1.0
    assert life_feats["surrender_value_indicator"] == 1.0
    assert life_feats["smoking_eligibility_compatibility"] == 0.0

    # Term
    term_policy = _sample_policy(
        "term",
        sum_insured_max=10_000_000.0,
        max_tenure_years=35,
        type_specific_metadata={
            "life_cover_max": 10_000_000.0,
            "policy_term_years_max": 35,
            "premium_payment_term_years": 20,
            "smoking_status_eligibility": "smoker_surcharge",
            "critical_illness_rider": True,
            "accidental_death_rider": True,
        },
    )
    term_feats = extract_policy_features(
        _sample_profile("term", smoking_status="smoker", min_sum_insured=5_000_000.0),
        term_policy,
    )
    assert term_feats["policy_type_encoded"] == 3.0
    assert term_feats["is_term"] == 1.0
    assert term_feats["smoking_eligibility_compatibility"] == 0.5
    assert term_feats["critical_illness_rider_indicator"] == 1.0
    assert term_feats["accidental_death_rider_indicator"] == 1.0
    assert term_feats["maturity_benefit_indicator"] == 0.0


def test_fixed_schema_ordering_and_mathematical_formulas() -> None:
    """Tests 5-9: Verify fixed column order and exact affordability/coverage/age/benefit math."""
    assert len(FEATURE_COLUMNS) == 59
    assert len(set(FEATURE_COLUMNS)) == 59
    assert list(FEATURE_SCHEMA_METADATA.keys()) == list(FEATURE_COLUMNS)
    assert FEATURE_SCHEMA_VERSION == "1.0.0"

    profile = _sample_profile(
        age=40,
        budget_max=40_000.0,
        min_sum_insured=1_000_000.0,
        required_benefits=["cashless_hospitalization", "maternity_cover"],
        pre_existing_conditions=["cosmetic_surgery"],
        risk_preference="comprehensive_coverage",
    )
    policy = _sample_policy(
        annual_premium_min=12_000.0,
        annual_premium_max=28_000.0,
        sum_insured_max=2_500_000.0,
        max_entry_age=65,
        max_renewal_age=85,
        covered_benefits=["cashless_hospitalization", "daycare_procedures"],
        exclusions=["cosmetic_surgery"],
    )
    feats = extract_policy_features(profile, policy)
    assert tuple(feats.keys()) == FEATURE_COLUMNS
    assert feats["risk_preference_encoded"] == 2.0
    assert feats["annual_premium"] == pytest.approx(20_000.0)
    assert feats["monthly_premium"] == pytest.approx(20_000.0 / 12.0)
    assert feats["premium_to_budget_ratio"] == pytest.approx(0.5)
    assert feats["affordability_margin"] == pytest.approx(20_000.0)
    assert feats["affordability_score"] == pytest.approx(0.7)
    assert feats["coverage_to_requirement_ratio"] == pytest.approx(2.5)
    assert feats["coverage_headroom"] == pytest.approx(1_500_000.0)
    assert feats["age_eligibility_margin"] == pytest.approx(25.0)
    assert feats["remaining_eligible_years"] == pytest.approx(45.0)
    assert feats["total_benefit_count"] == pytest.approx(2.0)
    assert feats["matched_required_benefits"] == pytest.approx(1.0)
    assert feats["required_benefit_match_ratio"] == pytest.approx(0.5)
    assert feats["exclusion_conflict_count"] == pytest.approx(1.0)


def test_missing_optional_mandatory_zero_denom_and_finite_guarantees() -> None:
    """Tests 11-15: Optional defaults, mandatory errors, safe division, zero NaN/inf."""
    # Pydantic UserProfileRequest has no annual_income_inr/risk_preference -> safe defaults
    pydantic_prof = UserProfileRequest(
        policy_type="health",
        age=30,
        budget_max=25_000.0,
        min_sum_insured=500_000.0,
        required_benefits=[],
    )
    minimal_policy = _sample_policy(
        claim_settlement_ratio=None,
        ped_waiting_period_months=None,
        max_renewal_age=None,
        lifelong_renewal=True,
        deductible_amount=None,
        copay_pct=None,
    )
    feats = extract_policy_features(pydantic_prof, minimal_policy)
    assert feats["has_annual_income"] == 0.0
    assert feats["annual_income"] == 0.0
    assert feats["has_claim_settlement_ratio"] == 0.0
    assert feats["claim_settlement_ratio"] == 0.0
    assert feats["required_benefit_match_ratio"] == 1.0
    assert feats["remaining_eligible_years"] == 70.0
    assert all(np.isfinite(list(feats.values())))

    # Missing mandatory profile fields -> raises MissingMandatoryFeatureInputError
    with pytest.raises(MissingMandatoryFeatureInputError, match="age"):
        extract_policy_features(
            {"policy_type": "health", "budget_max": 20_000, "min_sum_insured": 500_000},
            minimal_policy,
        )

    # Zero denominator / non-positive budget -> explicit validation error
    with pytest.raises(MissingMandatoryFeatureInputError, match="strictly positive"):
        extract_policy_features(
            {"policy_type": "health", "age": 30, "budget_max": 0.0, "min_sum_insured": 500_000},
            minimal_policy,
        )

    # Missing mandatory policy fields -> raises MissingMandatoryFeatureInputError
    with pytest.raises(MissingMandatoryFeatureInputError, match="sum_insured_max"):
        extract_policy_features(
            pydantic_prof,
            {"policy_type": "health", "annual_premium_min": 5000, "annual_premium_max": 9000},
        )


def test_leakage_prevention_determinism_and_no_input_mutation() -> None:
    """Tests 16-19: Identifiers & targets cannot leak into features; inputs are never mutated."""
    assert set(FEATURE_COLUMNS).isdisjoint(FORBIDDEN_FEATURE_COLUMNS)

    profile_a = _sample_profile(profile_id="PROF-1", suitability_score=0.99, relevance_label=3)
    profile_b = _sample_profile(profile_id="PROF-999", suitability_score=0.01, relevance_label=0)
    policy_a = _sample_policy(policy_id="ID-1", policy_code="CODE-1", is_eligible=True)
    policy_b = _sample_policy(
        policy_id="ID-999",
        policy_code="CODE-999",
        is_eligible=False,
        failed_filters=["AGE_MAX"],
    )

    prof_snap = copy.deepcopy(profile_a)
    pol_snap = copy.deepcopy(policy_a)

    feats_a = extract_policy_features(profile_a, policy_a)
    feats_a_repeat = extract_policy_features(profile_a, policy_a)
    feats_b = extract_policy_features(profile_b, policy_b)

    assert feats_a == feats_a_repeat == feats_b
    assert profile_a == prof_snap
    assert policy_a == pol_snap


def test_batch_feature_matrix_empty_and_mixed_categories() -> None:
    """Tests 20, 21 & 24: Batch matrix ordering, empty handling, and mixed categories."""
    profile = _sample_profile("health")

    empty_df = build_feature_matrix(profile, [])
    assert empty_df.empty
    assert tuple(empty_df.columns) == FEATURE_COLUMNS
    assert all(dtype == np.float64 for dtype in empty_df.dtypes)

    mixed_policies = [
        _sample_policy(
            "term", policy_id="P-T", type_specific_metadata={"critical_illness_rider": True}
        ),
        _sample_policy("health", policy_id="P-H"),
        _sample_policy("car", policy_id="P-C", type_specific_metadata={"zero_depreciation": True}),
        _sample_policy("life", policy_id="P-L", type_specific_metadata={"maturity_benefit": True}),
    ]
    matrix = build_feature_matrix(profile, mixed_policies)
    assert matrix.shape == (4, len(FEATURE_COLUMNS))
    assert tuple(matrix.columns) == FEATURE_COLUMNS
    assert np.isfinite(matrix.to_numpy()).all()
    assert list(matrix["policy_type_encoded"]) == [3.0, 0.0, 1.0, 2.0]


def test_build_training_feature_dataset_separation_and_synthetic_coverage() -> None:
    """Tests 22 & 23: Training dataset builds leakage-free X, y, groups, metadata."""
    policies_df, profiles_df, relevance_df = load_synthetic_dataset(validate=True)
    dataset = build_training_feature_dataset(
        profiles=profiles_df,
        policies=policies_df,
        relevance=relevance_df,
        eligible_only=True,
    )

    expected_eligible_count = int(relevance_df["is_eligible"].sum())
    assert dataset.X.shape == (expected_eligible_count, len(FEATURE_COLUMNS))
    assert tuple(dataset.X.columns) == FEATURE_COLUMNS
    assert set(dataset.X.columns).isdisjoint(FORBIDDEN_FEATURE_COLUMNS)
    assert np.isfinite(dataset.X.to_numpy()).all()
    assert not dataset.X.isna().any().any()

    assert len(dataset.y) == expected_eligible_count
    assert dataset.y.dtype == np.int64
    assert set(dataset.y.unique()).issubset({0, 1, 2, 3})

    assert len(dataset.groups) == expected_eligible_count
    assert dataset.groups.name == "profile_id"
    assert dataset.groups.nunique() == 56  # 60 profiles minus 4 strict zero-match profiles

    assert len(dataset.metadata) == expected_eligible_count
    assert set(dataset.metadata["policy_type"]) == {"health", "car", "life", "term"}
    assert bool(dataset.metadata["is_eligible"].all()) is True

    # Mutating target labels in relevance copy must never affect X
    perturbed_rel = relevance_df.copy()
    perturbed_rel.loc[perturbed_rel["is_eligible"], "suitability_score"] = 0.1234
    perturbed_rel.loc[perturbed_rel["is_eligible"], "relevance_label"] = 1
    perturbed_ds = build_training_feature_dataset(
        profiles=profiles_df,
        policies=policies_df,
        relevance=perturbed_rel,
        eligible_only=True,
    )
    pd.testing.assert_frame_equal(dataset.X, perturbed_ds.X)
