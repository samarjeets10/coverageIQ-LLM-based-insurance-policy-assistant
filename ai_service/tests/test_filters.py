"""Comprehensive tests for Stage 4 Deterministic Hard Filtering Engine."""

import copy
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from ai_service.recommendation.dataset import load_synthetic_dataset
from ai_service.recommendation.filters import (
    MissingMandatoryProfileError,
    PolicyAttributesTableUnavailableError,
    build_catalog_policy_attributes_query,
    evaluate_policy_eligibility,
    filter_policies,
    filter_policies_from_db,
    load_catalog_policies_from_db,
)
from ai_service.recommendation.schemas import UserProfileRequest


def _base_health_profile(**overrides: Any) -> UserProfileRequest:
    data: dict[str, Any] = {
        "policy_type": "health",
        "age": 32,
        "budget_min": 5_000.0,
        "budget_max": 25_000.0,
        "applicant_type": "salaried",
        "coverage_mode": "individual",
        "family_size": 1,
        "min_sum_insured": 500_000.0,
        "required_benefits": ["cashless_hospitalization"],
        "tenure_years": 1,
    }
    data.update(overrides)
    return UserProfileRequest(**data)


def _base_health_policy(**overrides: Any) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "policy_id": "POL-HLT-BASE",
        "policy_name": "ArogyaShield Base Health",
        "policy_type": "health",
        "policy_status": "active",
        "is_catalog_eligible": True,
        "min_entry_age": 18,
        "max_entry_age": 65,
        "max_renewal_age": 85,
        "lifelong_renewal": False,
        "coverage_mode": "both",
        "max_family_members": 4,
        "eligible_applicant_types": ["salaried", "self_employed", "student", "senior_citizen"],
        "annual_premium_min": 8_000.0,
        "annual_premium_max": 20_000.0,
        "sum_insured_min": 300_000.0,
        "sum_insured_max": 2_000_000.0,
        "deductible_amount": 0.0,
        "copay_pct": 10.0,
        "room_rent_limit_type": "no_limit",
        "initial_waiting_period_days": 30,
        "ped_covered": True,
        "ped_waiting_period_months": 24,
        "network_hospitals_count": 7_500,
        "cashless_available": True,
        "pan_india_coverage": True,
        "available_regions": ["mumbai", "pune", "bengaluru"],
        "min_tenure_years": 1,
        "max_tenure_years": 3,
        "covered_benefits": ["cashless_hospitalization", "daycare_procedures"],
        "exclusions": ["cosmetic_surgery"],
        "type_specific_metadata": {"family_floater_supported": True},
    }
    policy.update(overrides)
    return policy


def test_active_policy_passes_and_inactive_fails() -> None:
    """Tests 1 & 2: Active catalog policy passes; archived/non-catalog policy fails."""
    profile = _base_health_profile()
    active_result = filter_policies(profile, [_base_health_policy()])
    assert active_result.status == "ELIGIBLE_CANDIDATES_FOUND"
    assert active_result.filter_report.passed_count == 1
    assert active_result.filter_report.eliminated_count == 0
    assert active_result.filter_report.policy_evaluations[0].is_eligible is True

    inactive = _base_health_policy(policy_status="archived", is_catalog_eligible=False)
    inactive_result = filter_policies(profile, [inactive])
    assert inactive_result.status == "NO_MATCHING_POLICIES"
    assert inactive_result.eligible_policies == []
    failed_ids = {
        f.rule_id for f in inactive_result.filter_report.policy_evaluations[0].failed_filters
    }
    assert {"POLICY_ACTIVE_STATUS", "CATALOG_ELIGIBLE"}.issubset(failed_ids)


def test_wrong_policy_type_and_age_bounds_fail() -> None:
    """Tests 3, 4 & 5: Policy type mismatch, age < min, and age > max fail with details."""
    profile = _base_health_profile(policy_type="health", age=17)
    wrong_type = _base_health_policy(policy_type="car", min_entry_age=18, max_entry_age=60)
    ev_low = evaluate_policy_eligibility(profile, wrong_type)
    assert not ev_low.is_eligible
    failed_map = {f.rule_id: f for f in ev_low.failed_filters}
    assert "POLICY_TYPE_MATCH" in failed_map
    assert "AGE_MIN" in failed_map
    assert failed_map["AGE_MIN"].actual == 17

    senior_profile = _base_health_profile(age=68)
    ev_high = evaluate_policy_eligibility(senior_profile, _base_health_policy(max_entry_age=60))
    assert not ev_high.is_eligible
    high_map = {f.rule_id: f for f in ev_high.failed_filters}
    assert "AGE_MAX" in high_map
    assert high_map["AGE_MAX"].actual == 68
    assert high_map["AGE_MAX"].expected == "<= 60"


def test_budget_sum_insured_and_geography_failures() -> None:
    """Tests 6, 7 & 8: Budget, sum-insured, and geography violations fail."""
    profile = _base_health_profile(
        budget_max=10_000.0,
        min_sum_insured=1_500_000.0,
        location="pune",
    )
    policy = _base_health_policy(
        annual_premium_min=15_000.0,
        annual_premium_max=30_000.0,
        sum_insured_max=1_000_000.0,
        pan_india_coverage=False,
        available_regions=["delhi", "kolkata"],
    )
    ev = evaluate_policy_eligibility(profile, policy)
    assert not ev.is_eligible
    failed_ids = {f.rule_id for f in ev.failed_filters}
    assert {"BUDGET_MAX", "SUM_INSURED_MIN", "GEOGRAPHIC_AVAILABILITY"}.issubset(failed_ids)


def test_health_ped_waiting_copay_and_network_failures() -> None:
    """Tests 9, 10, 11 & 12: Health PED, waiting period, co-pay, and network rules fail."""
    profile = _base_health_profile(
        pre_existing_conditions=["diabetes"],
        max_ped_waiting_months=12,
        max_waiting_period_days=15,
        max_copay_pct=10.0,
        min_network_hospitals=10_000,
    )
    policy = _base_health_policy(
        ped_covered=False,
        ped_waiting_period_months=36,
        initial_waiting_period_days=30,
        copay_pct=25.0,
        network_hospitals_count=4_500,
    )
    ev = evaluate_policy_eligibility(profile, policy)
    assert not ev.is_eligible
    failed_ids = {f.rule_id for f in ev.failed_filters}
    assert {
        "PED_COVERAGE",
        "INITIAL_WAITING_PERIOD",
        "COPAY_MAX",
        "HOSPITAL_NETWORK_MIN",
    }.issubset(failed_ids)

    # When PED is covered but waiting period exceeds limit
    ped_wait_ev = evaluate_policy_eligibility(
        profile, _base_health_policy(ped_covered=True, ped_waiting_period_months=36)
    )
    assert any(f.rule_id == "PED_WAITING_PERIOD" for f in ped_wait_ev.failed_filters)


def test_car_vehicle_type_age_and_addon_requirements() -> None:
    """Tests 13 & 14: Car vehicle type, vehicle age, and Car metadata flags enforced."""
    car_policy = _base_health_policy(
        policy_id="POL-CAR-1",
        policy_name="Bharosa Motor Plus",
        policy_type="car",
        coverage_mode="individual",
        max_family_members=1,
        covered_benefits=["third_party_liability", "own_damage"],
        type_specific_metadata={
            "vehicle_types": ["hatchback", "sedan"],
            "vehicle_age_limit_years": 7,
            "own_damage_coverage": True,
            "third_party_coverage": True,
            "zero_depreciation": False,
            "roadside_assistance": False,
            "engine_protection": False,
        },
    )
    bad_car_profile = UserProfileRequest(
        policy_type="car",
        age=30,
        budget_max=25_000.0,
        min_sum_insured=500_000.0,
        vehicle_type="suv",
        vehicle_age_years=9,
        require_zero_depreciation=True,
        require_roadside_assistance=True,
        require_engine_protection=True,
    )
    ev = evaluate_policy_eligibility(bad_car_profile, car_policy)
    assert not ev.is_eligible
    failed_ids = {f.rule_id for f in ev.failed_filters}
    assert {
        "CAR_VEHICLE_TYPE",
        "CAR_VEHICLE_AGE_MAX",
        "CAR_ZERO_DEPRECIATION",
        "CAR_ROADSIDE_ASSISTANCE",
        "CAR_ENGINE_PROTECTION",
    }.issubset(failed_ids)

    good_car_profile = UserProfileRequest(
        policy_type="car",
        age=30,
        budget_max=25_000.0,
        min_sum_insured=500_000.0,
        vehicle_type="sedan",
        vehicle_age_years=4,
        require_own_damage=True,
        require_third_party=True,
    )
    assert evaluate_policy_eligibility(good_car_profile, car_policy).is_eligible is True


def test_life_cover_maturity_ppt_and_term_smoking_riders() -> None:
    """Tests 15 & 16: Life cover/maturity/PPT rules and Term smoking/rider rules."""
    life_policy = _base_health_policy(
        policy_id="POL-LIF-1",
        policy_name="SurakshaLife Savings",
        policy_type="life",
        coverage_mode="individual",
        max_family_members=1,
        min_tenure_years=10,
        max_tenure_years=25,
        sum_insured_max=2_000_000.0,
        type_specific_metadata={
            "life_cover_min": 500_000.0,
            "life_cover_max": 2_000_000.0,
            "policy_term_years_min": 10,
            "policy_term_years_max": 25,
            "maturity_benefit": False,
            "premium_payment_term_years": 20,
            "surrender_value_available": False,
        },
    )
    life_profile = UserProfileRequest(
        policy_type="life",
        age=35,
        budget_max=30_000.0,
        min_sum_insured=5_000_000.0,
        tenure_years=15,
        require_maturity_benefit=True,
        require_surrender_value=True,
        max_premium_payment_term_years=10,
    )
    ev_life = evaluate_policy_eligibility(life_profile, life_policy)
    assert not ev_life.is_eligible
    life_failed = {f.rule_id for f in ev_life.failed_filters}
    assert {
        "SUM_INSURED_MIN",
        "LIFE_COVER_MIN",
        "LIFE_MATURITY_BENEFIT",
        "LIFE_SURRENDER_VALUE",
        "LIFE_PREMIUM_PAYMENT_TERM",
    }.issubset(life_failed)

    term_policy = _base_health_policy(
        policy_id="POL-TRM-1",
        policy_name="Kavach Non-Smoker Term",
        policy_type="term",
        coverage_mode="individual",
        max_family_members=1,
        min_tenure_years=10,
        max_tenure_years=35,
        sum_insured_max=20_000_000.0,
        type_specific_metadata={
            "life_cover_min": 2_500_000.0,
            "life_cover_max": 20_000_000.0,
            "policy_term_years_min": 10,
            "policy_term_years_max": 35,
            "premium_payment_term_years": 20,
            "smoking_status_eligibility": "non_smoker_only",
            "critical_illness_rider": False,
            "accidental_death_rider": False,
        },
    )
    term_profile = UserProfileRequest(
        policy_type="term",
        age=30,
        budget_max=30_000.0,
        min_sum_insured=10_000_000.0,
        tenure_years=20,
        smoking_status="smoker",
        require_critical_illness_rider=True,
        require_accidental_death_rider=True,
    )
    ev_term = evaluate_policy_eligibility(term_profile, term_policy)
    assert not ev_term.is_eligible
    term_failed = {f.rule_id for f in ev_term.failed_filters}
    assert {
        "TERM_SMOKING_ELIGIBILITY",
        "TERM_CRITICAL_ILLNESS_RIDER",
        "TERM_ACCIDENTAL_DEATH_RIDER",
    }.issubset(term_failed)


def test_missing_optional_vs_mandatory_information_handling() -> None:
    """Tests 17 & 18: Missing optional prefs pass; missing mandatory/unverified info fails."""
    # Optional profile preferences omitted -> policy should pass without false rejection
    minimal_profile = _base_health_profile(
        location=None,
        max_copay_pct=None,
        max_deductible=None,
        max_waiting_period_days=None,
        max_ped_waiting_months=None,
        required_benefits=[],
        pre_existing_conditions=[],
    )
    assert evaluate_policy_eligibility(minimal_profile, _base_health_policy()).is_eligible is True

    # Requested smoker check against policy missing smoking metadata must NOT pass
    term_missing_meta = _base_health_policy(
        policy_type="term",
        coverage_mode="individual",
        min_tenure_years=10,
        max_tenure_years=30,
        sum_insured_max=10_000_000.0,
        type_specific_metadata={},
    )
    smoker_prof = UserProfileRequest(
        policy_type="term",
        age=30,
        budget_max=25_000.0,
        min_sum_insured=5_000_000.0,
        tenure_years=20,
        smoking_status="smoker",
    )
    ev_unverified = evaluate_policy_eligibility(smoker_prof, term_missing_meta)
    assert not ev_unverified.is_eligible
    assert any(f.rule_id == "TERM_SMOKING_ELIGIBILITY" for f in ev_unverified.failed_filters)

    # Missing mandatory profile fields -> raises MissingMandatoryProfileError
    with pytest.raises(MissingMandatoryProfileError, match="age"):
        filter_policies(
            {"policy_type": "health", "budget_max": 20000, "min_sum_insured": 500000}, []
        )

    # Missing mandatory policy fields -> structured failure, no crash
    broken_policy = {"policy_id": "POL-BAD", "policy_name": "Incomplete", "policy_type": "health"}
    ev_broken = evaluate_policy_eligibility(minimal_profile, broken_policy)
    assert not ev_broken.is_eligible
    assert ev_broken.failed_filters[0].rule_id == "MISSING_MANDATORY_POLICY_DATA"


def test_multiple_failures_determinism_and_no_input_mutation() -> None:
    """Tests 19, 22 & 23: Multiple failures reported, inputs unchanged, deterministic output."""
    profile = _base_health_profile(
        age=75,
        budget_min=1_000.0,
        budget_max=4_000.0,
        min_sum_insured=5_000_000.0,
    )
    policy = _base_health_policy(
        policy_status="draft",
        min_entry_age=18,
        max_entry_age=60,
        annual_premium_min=15_000.0,
        sum_insured_max=1_000_000.0,
    )
    profile_snapshot = profile.model_dump()
    policy_snapshot = copy.deepcopy(policy)

    res_a = filter_policies(profile, [policy])
    res_b = filter_policies(profile, [policy])

    assert res_a.model_dump() == res_b.model_dump()
    assert len(res_a.filter_report.policy_evaluations[0].failed_filters) >= 4
    assert profile.model_dump() == profile_snapshot
    assert policy == policy_snapshot


def test_synthetic_dataset_categories_zero_match_and_parity() -> None:
    """Tests 20, 21 & dataset parity across 4 categories and 4,800 synthetic pairs."""
    policies_df, profiles_df, relevance_df = load_synthetic_dataset(validate=True)
    policies_list = policies_df.to_dict(orient="records")
    profiles_map = {r["profile_id"]: r for r in profiles_df.to_dict(orient="records")}
    policies_map = {r["policy_id"]: r for r in policies_list}

    # Verify all 4 zero-match profiles strictly return 0 eligible policies & full audit trail
    for zero_id in ("PROF-HEA-015", "PROF-CAR-015", "PROF-LIF-015", "PROF-TER-015"):
        res = filter_policies(profiles_map[zero_id], policies_list)
        assert res.status == "NO_MATCHING_POLICIES"
        assert res.eligible_policies == []
        assert res.filter_report.status == "NO_MATCHING_POLICIES"
        assert res.filter_report.total_candidates == len(policies_list)
        assert res.filter_report.passed_count == 0
        assert res.filter_report.eliminated_count == len(policies_list)
        assert len(res.filter_report.eliminations) >= len(policies_list)

    # Verify positive matches exist across all 4 categories
    for cat in ("health", "car", "life", "term"):
        cat_profile = next(
            p
            for p in profiles_map.values()
            if p["preferred_policy_type"] == cat and not p["profile_id"].endswith("015")
        )
        res = filter_policies(cat_profile, policies_list)
        assert res.status == "ELIGIBLE_CANDIDATES_FOUND"
        assert res.filter_report.passed_count >= 1
        assert all(p["policy_type"] == cat for p in res.eligible_policies)

    # Full 4,800-pair parity verification against dataset.py labels
    for rel_row in relevance_df.to_dict(orient="records"):
        ev = evaluate_policy_eligibility(
            profiles_map[rel_row["profile_id"]],
            policies_map[rel_row["policy_id"]],
        )
        assert ev.is_eligible is bool(rel_row["is_eligible"])


def test_database_adapter_query_compilation_and_missing_table_error() -> None:
    """Test 24: SQLAlchemy Core query builds cleanly; missing table fails explicitly."""
    stmt = build_catalog_policy_attributes_query(
        policy_type="Health",
        active_only=True,
        max_budget=25000.0,
        min_sum_insured=500000.0,
        age=32,
    )
    compiled = stmt.compile(dialect=postgresql.dialect())
    sql_str = str(compiled)
    assert "JOIN policy_attributes" in sql_str
    assert "policies.organization_id IS NOT NULL" in sql_str
    assert "policies.owner_user_id IS NULL" in sql_str
    assert "25000" not in sql_str  # must be bound parameter, never raw interpolated literal
    assert compiled.params["policy_type_1"] == "health"
    assert compiled.params["annual_premium_min_1"] == 25000.0
    assert compiled.params["sum_insured_max_1"] == 500000.0

    # Mocked connection where policy_attributes does not exist -> explicit error
    mock_conn = MagicMock()
    mock_conn.execute.return_value.scalar_one_or_none.return_value = None
    with pytest.raises(PolicyAttributesTableUnavailableError, match="policy_attributes"):
        load_catalog_policies_from_db(mock_conn)

    with pytest.raises(PolicyAttributesTableUnavailableError, match="policy_attributes"):
        filter_policies_from_db(mock_conn, _base_health_profile())
