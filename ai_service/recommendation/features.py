"""Deterministic, leakage-free feature engineering pipeline for Module 6A.

Transforms validated (user_profile, eligible_policy) pairs into a stable
numerical feature matrix for downstream LightGBM and XGBoost ranking models.
This module is pure Python/NumPy/pandas and requires no database, FastAPI,
or ML model dependencies.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_service.recommendation.coercion import (
    OPTIONAL_FLOAT_METADATA_BOUNDS,
    OPTIONAL_INT_METADATA_MINS,
    TYPE_METADATA_NUMERIC_BOUNDS,
    InvalidNumericMetadataError,
    find_invalid_mandatory_policy_bounds,
    is_missing_or_blank,
    parse_bool,
    parse_bounded_optional_float,
    parse_bounded_optional_int,
    parse_strict_metadata_mapping,
    parse_strict_optional_float,
    validate_policy_optional_numeric_fields,
)
from ai_service.recommendation.dataset import load_synthetic_dataset
from ai_service.recommendation.schemas import UserProfileRequest

FEATURE_SCHEMA_VERSION = "1.0.0"
# Training Health PED waiting periods (`dataset.py`) span {12, 24, 36, 48} months,
# and `lightgbm_ranker.joblib` splits once at 1260.0 days (`tree_index=11`) with
# higher leaf scores above 1260.0. Using 36 months (1080.0 days) prevents missing
# health PED metadata from being treated as 0 days while preserving the 59-feature
# schema and avoiding the accidental >1260-day LightGBM boost. Full distinction vs.
# explicit 36-month waits requires schema v2.0.0 (`has_ped_waiting_period`) + retrain.
CONSERVATIVE_MISSING_HEALTH_PED_WAIT_MONTHS: int = 36
CONSERVATIVE_MISSING_HEALTH_PED_WAIT_DAYS: float = float(
    CONSERVATIVE_MISSING_HEALTH_PED_WAIT_MONTHS * 30
)

POLICY_TYPE_ENCODING: dict[str, float] = {
    "health": 0.0,
    "car": 1.0,
    "life": 2.0,
    "term": 3.0,
}

RISK_PREFERENCE_ENCODING: dict[str, float] = {
    "low_premium": 0.0,
    "balanced": 1.0,
    "comprehensive_coverage": 2.0,
}

FORBIDDEN_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "suitability_score",
        "relevance_label",
        "is_eligible",
        "profile_id",
        "policy_id",
        "policy_code",
        "failed_filters",
        "passed_filters",
        "elimination_reasons",
    }
)

FEATURE_COLUMNS: tuple[str, ...] = (
    # Category & availability indicators (8)
    "policy_type_encoded",
    "is_health",
    "is_car",
    "is_life",
    "is_term",
    "has_claim_settlement_ratio",
    "has_annual_income",
    "has_vehicle_age",
    # Common user features (5)
    "applicant_age",
    "annual_income",
    "monthly_budget",
    "family_size",
    "risk_preference_encoded",
    # Common affordability features (5)
    "annual_premium",
    "monthly_premium",
    "premium_to_budget_ratio",
    "affordability_margin",
    "affordability_score",
    # Common coverage & age features (6)
    "sum_insured",
    "requested_sum_insured",
    "coverage_to_requirement_ratio",
    "coverage_headroom",
    "age_eligibility_margin",
    "remaining_eligible_years",
    # Common benefit & exclusion features (4)
    "total_benefit_count",
    "matched_required_benefits",
    "required_benefit_match_ratio",
    "exclusion_conflict_count",
    # Common policy quality features (5)
    "claim_settlement_ratio",
    "deductible_amount",
    "no_claim_bonus_pct",
    "restoration_benefit_indicator",
    "pan_india_coverage_indicator",
    # Health-specific features (8)
    "ped_coverage_indicator",
    "ped_waiting_period_days",
    "initial_waiting_period_days",
    "room_rent_restriction_indicator",
    "cashless_availability_indicator",
    "hospital_network_count",
    "co_payment_percentage",
    "family_floater_indicator",
    # Car-specific features (8)
    "vehicle_age",
    "vehicle_age_limit",
    "vehicle_age_eligibility_margin",
    "own_damage_indicator",
    "third_party_indicator",
    "zero_depreciation_indicator",
    "roadside_assistance_indicator",
    "engine_protection_indicator",
    # Life & Term-specific features (10)
    "life_cover",
    "requested_life_cover",
    "life_cover_ratio",
    "policy_term_years",
    "premium_payment_term_years",
    "maturity_benefit_indicator",
    "surrender_value_indicator",
    "smoking_eligibility_compatibility",
    "critical_illness_rider_indicator",
    "accidental_death_rider_indicator",
)


def _build_feature_schema_metadata() -> dict[str, dict[str, str]]:
    """Build structured metadata mapping aligned 1-to-1 with FEATURE_COLUMNS."""
    groups_by_prefix: tuple[tuple[set[str], str, str], ...] = (
        (
            {
                "policy_type_encoded",
                "is_health",
                "is_car",
                "is_life",
                "is_term",
                "has_claim_settlement_ratio",
                "has_annual_income",
                "has_vehicle_age",
            },
            "category_and_availability",
            "all",
        ),
        (
            {
                "applicant_age",
                "annual_income",
                "monthly_budget",
                "family_size",
                "risk_preference_encoded",
            },
            "user_profile",
            "all",
        ),
        (
            {
                "annual_premium",
                "monthly_premium",
                "premium_to_budget_ratio",
                "affordability_margin",
                "affordability_score",
            },
            "affordability",
            "all",
        ),
        (
            {
                "sum_insured",
                "requested_sum_insured",
                "coverage_to_requirement_ratio",
                "coverage_headroom",
                "age_eligibility_margin",
                "remaining_eligible_years",
            },
            "coverage_and_age",
            "all",
        ),
        (
            {
                "total_benefit_count",
                "matched_required_benefits",
                "required_benefit_match_ratio",
                "exclusion_conflict_count",
            },
            "benefits_and_exclusions",
            "all",
        ),
        (
            {
                "claim_settlement_ratio",
                "deductible_amount",
                "no_claim_bonus_pct",
                "restoration_benefit_indicator",
                "pan_india_coverage_indicator",
            },
            "policy_quality",
            "all",
        ),
        (
            {
                "ped_coverage_indicator",
                "ped_waiting_period_days",
                "initial_waiting_period_days",
                "room_rent_restriction_indicator",
                "cashless_availability_indicator",
                "hospital_network_count",
                "co_payment_percentage",
                "family_floater_indicator",
            },
            "health_specific",
            "health",
        ),
        (
            {
                "vehicle_age",
                "vehicle_age_limit",
                "vehicle_age_eligibility_margin",
                "own_damage_indicator",
                "third_party_indicator",
                "zero_depreciation_indicator",
                "roadside_assistance_indicator",
                "engine_protection_indicator",
            },
            "car_specific",
            "car",
        ),
        (
            {
                "life_cover",
                "requested_life_cover",
                "life_cover_ratio",
                "policy_term_years",
                "premium_payment_term_years",
            },
            "life_and_term_shared",
            "life,term",
        ),
        (
            {"maturity_benefit_indicator", "surrender_value_indicator"},
            "life_specific",
            "life",
        ),
        (
            {
                "smoking_eligibility_compatibility",
                "critical_illness_rider_indicator",
                "accidental_death_rider_indicator",
            },
            "term_specific",
            "term",
        ),
    )

    spec: dict[str, dict[str, str]] = {}
    for col in FEATURE_COLUMNS:
        for cols, group_name, categories in groups_by_prefix:
            if col in cols:
                spec[col] = {
                    "group": group_name,
                    "dtype": "float64",
                    "applicable_categories": categories,
                    "missing_strategy": (
                        "0.0 sentinel + availability indicator"
                        if categories == "all"
                        else "0.0 when category inapplicable or missing"
                    ),
                }
                break
    return spec


FEATURE_SCHEMA_METADATA: dict[str, dict[str, str]] = _build_feature_schema_metadata()


class MissingMandatoryFeatureInputError(ValueError):
    """Raised when mandatory profile or policy fields required for features are missing."""


@dataclass(frozen=True)
class TrainingFeatureDataset:
    """Leakage-free container separating feature matrix X from targets and group keys."""

    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    metadata: pd.DataFrame
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS


def _is_missing(value: Any) -> bool:
    return is_missing_or_blank(value)


def _to_optional_float(value: Any, *, field_name: str = "value") -> float | None:
    return parse_strict_optional_float(value, field_name=field_name)


def _to_bool(value: Any, default: bool = False) -> bool:
    return parse_bool(value, default=default)


def _to_bounded_policy_float(value: Any, *, field_name: str) -> float | None:
    min_val, max_val = OPTIONAL_FLOAT_METADATA_BOUNDS.get(field_name, (None, None))
    return parse_bounded_optional_float(
        value, field_name=field_name, min_value=min_val, max_value=max_val
    )


def _to_bounded_policy_int(value: Any, *, field_name: str) -> int | None:
    min_val = OPTIONAL_INT_METADATA_MINS.get(field_name)
    return parse_bounded_optional_int(value, field_name=field_name, min_value=min_val)


def _to_bounded_type_meta_float(value: Any, *, key: str) -> float | None:
    min_val, max_val = TYPE_METADATA_NUMERIC_BOUNDS.get(key, (0.0, None))
    return parse_bounded_optional_float(
        value,
        field_name=f"type_specific_metadata.{key}",
        min_value=min_val,
        max_value=max_val,
    )


def _parse_str_list(raw: Any) -> list[str]:
    if _is_missing(raw):
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(item).strip().lower() for item in raw if str(item).strip()]
    try:
        decoded = json.loads(str(raw))
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(decoded, list):
        return []
    return [str(item).strip().lower() for item in decoded if str(item).strip()]


def _parse_metadata(raw: Any) -> dict[str, Any]:
    return parse_strict_metadata_mapping(raw, field_name="type_specific_metadata")


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator <= 0.0 or not math.isfinite(denominator):
        return default
    ratio = numerator / denominator
    return ratio if math.isfinite(ratio) else default


def _normalize_profile_fields(profile: UserProfileRequest | Mapping[str, Any]) -> dict[str, Any]:
    """Extract only observable user profile fields without accessing IDs or targets."""
    if isinstance(profile, UserProfileRequest):
        raw: Mapping[str, Any] = profile.model_dump()
    elif isinstance(profile, Mapping):
        raw = profile
    else:
        raise MissingMandatoryFeatureInputError("Profile must be a UserProfileRequest or mapping")

    policy_type_raw = raw.get("policy_type") or raw.get("preferred_policy_type")
    if _is_missing(policy_type_raw):
        raise MissingMandatoryFeatureInputError("Missing mandatory profile field: policy_type")
    policy_type = str(policy_type_raw).strip().lower()
    if policy_type not in POLICY_TYPE_ENCODING:
        raise MissingMandatoryFeatureInputError(f"Unsupported policy_type: {policy_type}")

    age = _to_optional_float(raw.get("age"), field_name="profile.age")
    budget_max = _to_optional_float(raw.get("budget_max"), field_name="profile.budget_max")
    min_sum_insured = _to_optional_float(
        raw.get("min_sum_insured"), field_name="profile.min_sum_insured"
    )
    if age is None or budget_max is None or min_sum_insured is None:
        missing = [
            name
            for name, val in (
                ("age", age),
                ("budget_max", budget_max),
                ("min_sum_insured", min_sum_insured),
            )
            if val is None
        ]
        raise MissingMandatoryFeatureInputError(
            f"Missing mandatory profile field(s): {', '.join(missing)}"
        )

    if budget_max <= 0.0 or min_sum_insured <= 0.0:
        raise MissingMandatoryFeatureInputError(
            "budget_max and min_sum_insured must be strictly positive"
        )

    annual_income = _to_optional_float(
        raw.get("annual_income_inr") if "annual_income_inr" in raw else raw.get("annual_income"),
        field_name="profile.annual_income_inr",
    )
    monthly_budget = _to_optional_float(
        raw.get("monthly_budget_inr") if "monthly_budget_inr" in raw else raw.get("monthly_budget"),
        field_name="profile.monthly_budget_inr",
    )
    if monthly_budget is None:
        monthly_budget = budget_max / 12.0

    family_size = _to_optional_float(raw.get("family_size"), field_name="profile.family_size")
    risk_pref_raw = str(raw.get("risk_preference") or "balanced").strip().lower()
    risk_encoded = RISK_PREFERENCE_ENCODING.get(risk_pref_raw, 1.0)

    smoking_raw = raw.get("smoking_status")
    smoking_status = None if _is_missing(smoking_raw) else str(smoking_raw).strip().lower()

    return {
        "policy_type": policy_type,
        "age": age,
        "annual_income": annual_income if annual_income is not None else 0.0,
        "has_annual_income": 1.0 if (annual_income is not None and annual_income > 0.0) else 0.0,
        "monthly_budget": monthly_budget,
        "budget_max": budget_max,
        "family_size": family_size if family_size is not None else 1.0,
        "min_sum_insured": min_sum_insured,
        "required_benefits": set(_parse_str_list(raw.get("required_benefits"))),
        "pre_existing_conditions": set(_parse_str_list(raw.get("pre_existing_conditions"))),
        "tenure_years": _to_optional_float(
            raw.get("tenure_years"), field_name="profile.tenure_years"
        )
        or 1.0,
        "risk_preference_encoded": risk_encoded,
        "smoking_status": smoking_status,
        "vehicle_age_years": _to_optional_float(
            raw.get("vehicle_age_years"), field_name="profile.vehicle_age_years"
        ),
    }


def _normalize_policy_fields(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Extract observable policy catalog fields while enforcing Stage 4 validation bounds."""
    if not isinstance(policy, Mapping):
        raise MissingMandatoryFeatureInputError("Policy must be a mapping")

    policy_type_raw = policy.get("policy_type")
    if _is_missing(policy_type_raw):
        raise MissingMandatoryFeatureInputError("Missing mandatory policy field: policy_type")
    policy_type = str(policy_type_raw).strip().lower()
    if policy_type not in POLICY_TYPE_ENCODING:
        raise MissingMandatoryFeatureInputError(f"Unsupported policy_type: {policy_type}")

    invalid_optional = validate_policy_optional_numeric_fields(policy)
    if invalid_optional:
        raise InvalidNumericMetadataError(
            "Policy contains malformed, boolean, non-finite, or out-of-range optional "
            f"numeric metadata: {', '.join(invalid_optional)}"
        )

    prem_single = _to_bounded_policy_float(
        policy.get("annual_premium"), field_name="annual_premium"
    )
    prem_min = parse_bounded_optional_float(
        policy.get("annual_premium_min"), field_name="annual_premium_min", min_value=0.0
    )
    prem_max = parse_bounded_optional_float(
        policy.get("annual_premium_max"), field_name="annual_premium_max", min_value=0.0
    )
    if prem_min is None and prem_single is not None:
        prem_min = prem_single
    if prem_max is None and prem_single is not None:
        prem_max = prem_single

    sum_max = _to_optional_float(policy.get("sum_insured_max"), field_name="sum_insured_max")
    if sum_max is None:
        sum_max = _to_bounded_policy_float(policy.get("sum_insured"), field_name="sum_insured")
    min_age = parse_bounded_optional_int(
        policy.get("min_entry_age"), field_name="min_entry_age", min_value=0, max_value=120
    )
    max_age = parse_bounded_optional_int(
        policy.get("max_entry_age"), field_name="max_entry_age", min_value=0, max_value=120
    )

    missing_fields = [
        name
        for name, val in (
            ("annual_premium_min", prem_min),
            ("annual_premium_max", prem_max),
            ("sum_insured_max", sum_max),
            ("min_entry_age", min_age),
            ("max_entry_age", max_age),
        )
        if val is None
    ]
    if missing_fields:
        raise MissingMandatoryFeatureInputError(
            f"Missing mandatory policy field(s): {', '.join(missing_fields)}"
        )

    assert prem_min is not None and prem_max is not None
    assert sum_max is not None and min_age is not None and max_age is not None
    invalid_bounds = find_invalid_mandatory_policy_bounds(
        min_age=min_age,
        max_age=max_age,
        prem_min=prem_min,
        prem_max=prem_max,
        sum_max=sum_max,
    )
    if invalid_bounds:
        raise InvalidNumericMetadataError(
            f"Policy failed numeric domain integrity check(s): {', '.join(invalid_bounds)}"
        )

    max_renewal_age_int = _to_bounded_policy_int(
        policy.get("max_renewal_age"), field_name="max_renewal_age"
    )
    max_renewal_age = float(max_renewal_age_int) if max_renewal_age_int is not None else None
    lifelong = _to_bool(policy.get("lifelong_renewal"), default=False)
    csr = _to_bounded_policy_float(
        policy.get("claim_settlement_ratio"), field_name="claim_settlement_ratio"
    )
    ped_covered = _to_bool(policy.get("ped_covered"), default=False)
    ped_wait_months = _to_bounded_policy_int(
        policy.get("ped_waiting_period_months"),
        field_name="ped_waiting_period_months",
    )
    if policy_type == "health" and ped_covered:
        ped_wait_days = (
            float(ped_wait_months) * 30.0
            if ped_wait_months is not None
            else CONSERVATIVE_MISSING_HEALTH_PED_WAIT_DAYS
        )
    else:
        ped_wait_days = 0.0

    deductible = _to_bounded_policy_float(
        policy.get("deductible_amount"), field_name="deductible_amount"
    )
    copay = _to_bounded_policy_float(policy.get("copay_pct"), field_name="copay_pct")
    init_wait = _to_bounded_policy_int(
        policy.get("initial_waiting_period_days"), field_name="initial_waiting_period_days"
    )
    net_count = _to_bounded_policy_int(
        policy.get("network_hospitals_count"), field_name="network_hospitals_count"
    )
    max_tenure = _to_bounded_policy_int(
        policy.get("max_tenure_years"), field_name="max_tenure_years"
    )
    ncb = _to_bounded_policy_float(
        policy.get("no_claim_bonus_pct"), field_name="no_claim_bonus_pct"
    )

    return {
        "policy_type": policy_type,
        "min_entry_age": float(min_age),
        "max_entry_age": float(max_age),
        "max_renewal_age": max_renewal_age,
        "lifelong_renewal": lifelong,
        "coverage_mode": str(policy.get("coverage_mode") or "individual").strip().lower(),
        "annual_premium_min": prem_min,
        "annual_premium_max": prem_max,
        "annual_premium": 0.5 * (prem_min + prem_max),
        "sum_insured": sum_max,
        "deductible_amount": deductible if deductible is not None else 0.0,
        "copay_pct": copay if copay is not None else 0.0,
        "room_rent_limit_type": str(policy.get("room_rent_limit_type") or "no_limit")
        .strip()
        .lower(),
        "initial_waiting_period_days": float(init_wait) if init_wait is not None else 0.0,
        "ped_covered": ped_covered,
        "ped_waiting_period_days": ped_wait_days,
        "network_hospitals_count": float(net_count) if net_count is not None else 0.0,
        "cashless_available": _to_bool(policy.get("cashless_available"), default=False),
        "pan_india_coverage": _to_bool(policy.get("pan_india_coverage"), default=False),
        "max_tenure_years": float(max_tenure) if max_tenure is not None else 1.0,
        "covered_benefits": set(_parse_str_list(policy.get("covered_benefits"))),
        "exclusions": set(_parse_str_list(policy.get("exclusions"))),
        "claim_settlement_ratio": csr if csr is not None else 0.0,
        "has_claim_settlement_ratio": 1.0 if csr is not None else 0.0,
        "no_claim_bonus_pct": ncb if ncb is not None else 0.0,
        "restoration_benefit": _to_bool(policy.get("restoration_benefit"), default=False),
        "meta": _parse_metadata(policy.get("type_specific_metadata")),
    }


def _compute_smoking_compatibility(
    smoking_status: str | None,
    meta: Mapping[str, Any],
) -> float:
    elig = str(meta.get("smoking_status_eligibility") or "all").strip().lower()
    if smoking_status == "smoker":
        if elig == "all":
            return 1.0
        if elig == "smoker_surcharge":
            return 0.5
        return 0.0
    return 1.0


def _extract_from_normalized(
    prof: Mapping[str, Any],
    pol: Mapping[str, Any],
) -> dict[str, float]:
    """Compute the fixed ordered feature dictionary from normalized profile and policy views."""
    pol_type = pol["policy_type"]
    is_health = 1.0 if pol_type == "health" else 0.0
    is_car = 1.0 if pol_type == "car" else 0.0
    is_life = 1.0 if pol_type == "life" else 0.0
    is_term = 1.0 if pol_type == "term" else 0.0

    budget_max = prof["budget_max"]
    annual_premium = pol["annual_premium"]
    prem_min = pol["annual_premium_min"]
    prem_to_budget = _safe_divide(annual_premium, budget_max)
    afford_margin = budget_max - annual_premium
    afford_score = max(0.0, min(1.0, _safe_divide(budget_max - prem_min, budget_max)))

    sum_insured = pol["sum_insured"]
    req_sum = prof["min_sum_insured"]
    coverage_ratio = _safe_divide(sum_insured, req_sum)
    coverage_headroom = sum_insured - req_sum

    age = prof["age"]
    age_margin = pol["max_entry_age"] - age
    if pol["max_renewal_age"] is not None:
        max_horizon = pol["max_renewal_age"]
    elif pol["lifelong_renewal"]:
        max_horizon = 100.0
    else:
        max_horizon = pol["max_entry_age"]
    remaining_years = max(0.0, max_horizon - age)

    req_benefits = prof["required_benefits"]
    pol_benefits = pol["covered_benefits"]
    matched_benefits = float(len(req_benefits & pol_benefits))
    benefit_match_ratio = (
        1.0 if not req_benefits else _safe_divide(matched_benefits, float(len(req_benefits)))
    )
    exclusion_conflicts = float(
        len((req_benefits | prof["pre_existing_conditions"]) & pol["exclusions"])
    )

    meta = pol["meta"]

    # Car-only features
    veh_age = prof["vehicle_age_years"]
    veh_limit = _to_bounded_type_meta_float(
        meta.get("vehicle_age_limit_years"),
        key="vehicle_age_limit_years",
    )
    has_veh_age = 1.0 if (is_car == 1.0 and veh_age is not None) else 0.0
    veh_age_val = veh_age if (is_car == 1.0 and veh_age is not None) else 0.0
    veh_limit_val = veh_limit if (is_car == 1.0 and veh_limit is not None) else 0.0
    veh_margin = (
        (veh_limit - veh_age)
        if (is_car == 1.0 and veh_limit is not None and veh_age is not None)
        else 0.0
    )

    # Life & Term shared features (zeroed for Health & Car)
    is_life_or_term = is_life == 1.0 or is_term == 1.0
    raw_life_cover = _to_bounded_type_meta_float(
        meta.get("life_cover_max"),
        key="life_cover_max",
    )
    life_cover = (
        (raw_life_cover if raw_life_cover is not None else sum_insured) if is_life_or_term else 0.0
    )
    req_life_cover = req_sum if is_life_or_term else 0.0
    life_cover_ratio = _safe_divide(life_cover, req_life_cover) if is_life_or_term else 0.0
    raw_policy_term = _to_bounded_type_meta_float(
        meta.get("policy_term_years_max"),
        key="policy_term_years_max",
    )
    policy_term = (
        (raw_policy_term if raw_policy_term is not None else pol["max_tenure_years"])
        if is_life_or_term
        else 0.0
    )
    raw_ppt_years = _to_bounded_type_meta_float(
        meta.get("premium_payment_term_years"),
        key="premium_payment_term_years",
    )
    ppt_years = (raw_ppt_years if raw_ppt_years is not None else 0.0) if is_life_or_term else 0.0

    features: dict[str, float] = {
        # Category & availability indicators
        "policy_type_encoded": POLICY_TYPE_ENCODING[pol_type],
        "is_health": is_health,
        "is_car": is_car,
        "is_life": is_life,
        "is_term": is_term,
        "has_claim_settlement_ratio": pol["has_claim_settlement_ratio"],
        "has_annual_income": prof["has_annual_income"],
        "has_vehicle_age": has_veh_age,
        # Common user features
        "applicant_age": float(age),
        "annual_income": float(prof["annual_income"]),
        "monthly_budget": float(prof["monthly_budget"]),
        "family_size": float(prof["family_size"]),
        "risk_preference_encoded": float(prof["risk_preference_encoded"]),
        # Common affordability features
        "annual_premium": float(annual_premium),
        "monthly_premium": float(annual_premium / 12.0),
        "premium_to_budget_ratio": float(prem_to_budget),
        "affordability_margin": float(afford_margin),
        "affordability_score": float(afford_score),
        # Common coverage & age features
        "sum_insured": float(sum_insured),
        "requested_sum_insured": float(req_sum),
        "coverage_to_requirement_ratio": float(coverage_ratio),
        "coverage_headroom": float(coverage_headroom),
        "age_eligibility_margin": float(age_margin),
        "remaining_eligible_years": float(remaining_years),
        # Common benefit & exclusion features
        "total_benefit_count": float(len(pol_benefits)),
        "matched_required_benefits": matched_benefits,
        "required_benefit_match_ratio": float(benefit_match_ratio),
        "exclusion_conflict_count": exclusion_conflicts,
        # Common policy quality features
        "claim_settlement_ratio": float(pol["claim_settlement_ratio"]),
        "deductible_amount": float(pol["deductible_amount"]),
        "no_claim_bonus_pct": float(pol["no_claim_bonus_pct"]),
        "restoration_benefit_indicator": 1.0 if pol["restoration_benefit"] else 0.0,
        "pan_india_coverage_indicator": 1.0 if pol["pan_india_coverage"] else 0.0,
        # Health-specific features
        "ped_coverage_indicator": 1.0 if (is_health == 1.0 and pol["ped_covered"]) else 0.0,
        "ped_waiting_period_days": float(pol["ped_waiting_period_days"])
        if is_health == 1.0
        else 0.0,
        "initial_waiting_period_days": (
            float(pol["initial_waiting_period_days"]) if is_health == 1.0 else 0.0
        ),
        "room_rent_restriction_indicator": (
            1.0 if (is_health == 1.0 and pol["room_rent_limit_type"] != "no_limit") else 0.0
        ),
        "cashless_availability_indicator": (
            1.0 if (is_health == 1.0 and pol["cashless_available"]) else 0.0
        ),
        "hospital_network_count": (
            float(pol["network_hospitals_count"]) if is_health == 1.0 else 0.0
        ),
        "co_payment_percentage": float(pol["copay_pct"]) if is_health == 1.0 else 0.0,
        "family_floater_indicator": (
            1.0
            if (is_health == 1.0 and pol["coverage_mode"] in ("family_floater", "both"))
            else 0.0
        ),
        # Car-specific features
        "vehicle_age": float(veh_age_val),
        "vehicle_age_limit": float(veh_limit_val),
        "vehicle_age_eligibility_margin": float(veh_margin),
        "own_damage_indicator": (
            1.0 if (is_car == 1.0 and _to_bool(meta.get("own_damage_coverage"))) else 0.0
        ),
        "third_party_indicator": (
            1.0 if (is_car == 1.0 and _to_bool(meta.get("third_party_coverage"))) else 0.0
        ),
        "zero_depreciation_indicator": (
            1.0 if (is_car == 1.0 and _to_bool(meta.get("zero_depreciation"))) else 0.0
        ),
        "roadside_assistance_indicator": (
            1.0 if (is_car == 1.0 and _to_bool(meta.get("roadside_assistance"))) else 0.0
        ),
        "engine_protection_indicator": (
            1.0 if (is_car == 1.0 and _to_bool(meta.get("engine_protection"))) else 0.0
        ),
        # Life & Term-specific features
        "life_cover": float(life_cover),
        "requested_life_cover": float(req_life_cover),
        "life_cover_ratio": float(life_cover_ratio),
        "policy_term_years": float(policy_term),
        "premium_payment_term_years": float(ppt_years),
        "maturity_benefit_indicator": (
            1.0 if (is_life == 1.0 and _to_bool(meta.get("maturity_benefit"))) else 0.0
        ),
        "surrender_value_indicator": (
            1.0 if (is_life == 1.0 and _to_bool(meta.get("surrender_value_available"))) else 0.0
        ),
        "smoking_eligibility_compatibility": (
            _compute_smoking_compatibility(prof["smoking_status"], meta) if is_term == 1.0 else 0.0
        ),
        "critical_illness_rider_indicator": (
            1.0 if (is_term == 1.0 and _to_bool(meta.get("critical_illness_rider"))) else 0.0
        ),
        "accidental_death_rider_indicator": (
            1.0 if (is_term == 1.0 and _to_bool(meta.get("accidental_death_rider"))) else 0.0
        ),
    }

    for col in FEATURE_COLUMNS:
        val = features[col]
        if not math.isfinite(val):
            raise ValueError(f"Non-finite feature value computed for '{col}': {val}")
    return features


def extract_policy_features(
    profile: UserProfileRequest | Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, float]:
    """Extract deterministic numerical features for one (profile, policy) pair.

    Enforces Stage 4 numeric parsing, integer constraints, and domain bounds
    directly on policy attributes, raising ``InvalidNumericMetadataError`` for
    malformed or out-of-range inputs and ``MissingMandatoryFeatureInputError``
    when required inputs are absent.
    """
    prof = _normalize_profile_fields(profile)
    pol = _normalize_policy_fields(policy)
    return _extract_from_normalized(prof, pol)


def _validate_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce exact column ordering, float64 dtype, zero NaN/inf, and zero leakage."""
    if tuple(df.columns) != FEATURE_COLUMNS:
        raise ValueError("Feature matrix columns do not match FEATURE_COLUMNS schema")
    if set(df.columns) & FORBIDDEN_FEATURE_COLUMNS:
        raise ValueError("Forbidden target or identifier column detected in feature matrix")
    if not df.empty:
        values = df.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Feature matrix contains NaN or infinite values")
    return df


def build_feature_matrix(
    profile: UserProfileRequest | Mapping[str, Any],
    eligible_policies: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Build a fixed-schema float64 feature matrix for a batch of eligible policies.

    Enforces the same Stage 4 strict numeric/integer/domain-bound validation on
    each policy record via ``extract_policy_features`` / ``_normalize_policy_fields``.
    """
    prof = _normalize_profile_fields(profile)
    if not eligible_policies:
        empty = pd.DataFrame(columns=list(FEATURE_COLUMNS)).astype(np.float64)
        return _validate_feature_matrix(empty)

    rows = [
        _extract_from_normalized(prof, _normalize_policy_fields(policy))
        for policy in eligible_policies
    ]
    matrix = pd.DataFrame.from_records(rows, columns=list(FEATURE_COLUMNS)).astype(np.float64)
    return _validate_feature_matrix(matrix)


def _coerce_dataframe(data: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(list(data))


def build_training_feature_dataset(
    profiles: pd.DataFrame | Sequence[Mapping[str, Any]] | None = None,
    policies: pd.DataFrame | Sequence[Mapping[str, Any]] | None = None,
    relevance: pd.DataFrame | Sequence[Mapping[str, Any]] | None = None,
    *,
    eligible_only: bool = True,
    data_dir: Path | None = None,
) -> TrainingFeatureDataset:
    """Build leakage-free (X, y, groups, metadata) ranking training dataset.

    Pre-normalizes unique profiles and policies before pair feature assembly,
    keeps targets/identifiers out of X, and preserves profile_id grouping for
    group-aware validation (GroupKFold / GroupShuffleSplit).
    """
    if profiles is None or policies is None or relevance is None:
        loaded_policies, loaded_profiles, loaded_relevance = load_synthetic_dataset(
            data_dir=data_dir, validate=True
        )
        policies_df = loaded_policies if policies is None else _coerce_dataframe(policies)
        profiles_df = loaded_profiles if profiles is None else _coerce_dataframe(profiles)
        relevance_df = loaded_relevance if relevance is None else _coerce_dataframe(relevance)
    else:
        policies_df = _coerce_dataframe(policies)
        profiles_df = _coerce_dataframe(profiles)
        relevance_df = _coerce_dataframe(relevance)

    working_rel = (
        relevance_df[relevance_df["is_eligible"].astype(bool)].reset_index(drop=True)
        if eligible_only
        else relevance_df.reset_index(drop=True)
    )

    profile_records = {str(row["profile_id"]): row for row in profiles_df.to_dict(orient="records")}
    policy_records = {str(row["policy_id"]): row for row in policies_df.to_dict(orient="records")}

    normalized_profiles = {
        pid: _normalize_profile_fields(row) for pid, row in profile_records.items()
    }
    normalized_policies = {
        pid: _normalize_policy_fields(row) for pid, row in policy_records.items()
    }

    feature_rows: list[dict[str, float]] = []
    meta_rows: list[dict[str, Any]] = []
    targets: list[int] = []
    groups: list[str] = []

    for rel_row in working_rel.to_dict(orient="records"):
        prof_id = str(rel_row["profile_id"])
        pol_id = str(rel_row["policy_id"])
        if prof_id not in normalized_profiles:
            raise MissingMandatoryFeatureInputError(f"Orphan profile_id in relevance: {prof_id}")
        if pol_id not in normalized_policies:
            raise MissingMandatoryFeatureInputError(f"Orphan policy_id in relevance: {pol_id}")

        feature_rows.append(
            _extract_from_normalized(normalized_profiles[prof_id], normalized_policies[pol_id])
        )
        targets.append(int(rel_row["relevance_label"]))
        groups.append(prof_id)
        meta_rows.append(
            {
                "profile_id": prof_id,
                "policy_id": pol_id,
                "policy_code": str(policy_records[pol_id].get("policy_code", "")),
                "policy_type": str(policy_records[pol_id]["policy_type"]).strip().lower(),
                "suitability_score": float(rel_row["suitability_score"]),
                "is_eligible": bool(rel_row["is_eligible"]),
            }
        )

    if feature_rows:
        X = pd.DataFrame.from_records(feature_rows, columns=list(FEATURE_COLUMNS)).astype(
            np.float64
        )
    else:
        X = pd.DataFrame(columns=list(FEATURE_COLUMNS)).astype(np.float64)
    X = _validate_feature_matrix(X)

    y = pd.Series(targets, name="relevance_label", dtype=np.int64)
    group_series = pd.Series(groups, name="profile_id", dtype="object")
    meta_df = pd.DataFrame(
        meta_rows,
        columns=[
            "profile_id",
            "policy_id",
            "policy_code",
            "policy_type",
            "suitability_score",
            "is_eligible",
        ],
    )

    return TrainingFeatureDataset(
        X=X,
        y=y,
        groups=group_series,
        metadata=meta_df,
    )
