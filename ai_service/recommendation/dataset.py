"""Synthetic dataset generation, validation, and CSV loading for Module 6A.

IMPORTANT:
This module generates synthetic development records solely for testing hard
filtering, feature engineering, ranking pipelines, and controlled experiments.
It does not represent real-world insurer products or customer preferences.
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_service.recommendation.schemas import UserProfileRequest

DEFAULT_SEED = 42
POLICY_CATEGORIES: tuple[str, ...] = ("health", "car", "life", "term")
_NAMESPACE_POLICY = uuid.uuid5(uuid.NAMESPACE_DNS, "coverageiq.synthetic.policy")

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

POLICY_COLUMNS: tuple[str, ...] = (
    "policy_id",
    "policy_code",
    "policy_name",
    "insurer_name",
    "policy_type",
    "description",
    "policy_status",
    "is_catalog_eligible",
    "min_entry_age",
    "max_entry_age",
    "max_renewal_age",
    "lifelong_renewal",
    "coverage_mode",
    "max_family_members",
    "eligible_applicant_types",
    "annual_premium_min",
    "annual_premium_max",
    "sum_insured_min",
    "sum_insured_max",
    "deductible_amount",
    "copay_pct",
    "room_rent_limit_type",
    "room_rent_limit_value",
    "initial_waiting_period_days",
    "ped_covered",
    "ped_waiting_period_months",
    "specific_disease_waiting_months",
    "maternity_waiting_months",
    "network_hospitals_count",
    "cashless_available",
    "pan_india_coverage",
    "available_regions",
    "min_tenure_years",
    "max_tenure_years",
    "covered_benefits",
    "exclusions",
    "claim_settlement_ratio",
    "no_claim_bonus_pct",
    "restoration_benefit",
    "type_specific_metadata",
    "extraction_source",
)

PROFILE_COLUMNS: tuple[str, ...] = (
    "profile_id",
    "preferred_policy_type",
    "age",
    "annual_income_inr",
    "monthly_budget_inr",
    "budget_min",
    "budget_max",
    "applicant_type",
    "occupation",
    "coverage_mode",
    "family_size",
    "min_sum_insured",
    "required_benefits",
    "pre_existing_conditions",
    "max_ped_waiting_months",
    "max_waiting_period_days",
    "max_copay_pct",
    "max_deductible",
    "room_rent_preference",
    "require_cashless",
    "min_network_hospitals",
    "geographic_location",
    "tenure_years",
    "require_lifelong_renewal",
    "smoking_status",
    "vehicle_type",
    "vehicle_age_years",
    "risk_preference",
)

RELEVANCE_COLUMNS: tuple[str, ...] = (
    "profile_id",
    "policy_id",
    "policy_type_match",
    "is_eligible",
    "failed_rules",
    "suitability_score",
    "relevance_label",
)

_FICTIONAL_INSURERS: dict[str, tuple[str, ...]] = {
    "health": (
        "ArogyaShield General Insurance",
        "SwasthyaCare Mutual",
        "Nirmaan Health Direct",
        "JeevanRaksha Health Assurance",
        "PranaVeda General Insurance",
    ),
    "car": (
        "Bharosa Motor Protect",
        "VahanSuraksha General",
        "GatiShield Auto Insurance",
        "MargRaksha General Insurance",
        "ChakraAssure Mobility",
    ),
    "life": (
        "SurakshaLife Assurance",
        "KalpaVriddhi Life Corp",
        "Samriddhi Future Life",
        "Aashray Life Assurance",
        "DhanSanchay Life Partners",
    ),
    "term": (
        "Kavach Term Care",
        "Sankalp Pure Protection",
        "AadharShield Term Life",
        "Nishchint Term Assurance",
        "Rakshak Term Direct",
    ),
}

_INDIAN_CITIES: tuple[str, ...] = (
    "mumbai",
    "pune",
    "bengaluru",
    "delhi",
    "hyderabad",
    "chennai",
    "ahmedabad",
    "kolkata",
    "jaipur",
    "surat",
)

_APPLICANT_POOL: tuple[str, ...] = (
    "salaried",
    "self_employed",
    "student",
    "senior_citizen",
)

_OCCUPATION_BY_APPLICANT: dict[str, tuple[str, ...]] = {
    "salaried": ("software_engineer", "bank_analyst", "teacher", "operations_lead"),
    "self_employed": ("retail_owner", "freelance_architect", "clinic_owner", "consultant"),
    "student": ("postgraduate_student", "medical_intern"),
    "senior_citizen": ("retired_engineer", "pensioner", "retired_professor"),
}

_HEALTH_BENEFITS = (
    "cashless_hospitalization",
    "daycare_procedures",
    "pre_post_hospitalization",
    "annual_health_checkup",
    "maternity_cover",
    "ambulance_cover",
    "domiciliary_treatment",
    "ayush_treatment",
)
_CAR_BENEFITS = (
    "own_damage",
    "third_party_liability",
    "zero_depreciation",
    "roadside_assistance",
    "engine_protection",
    "ncb_protection",
    "consumables_cover",
)
_LIFE_BENEFITS = (
    "death_benefit",
    "maturity_benefit",
    "guaranteed_additions",
    "accidental_rider",
    "tax_saving_80c",
    "loan_against_policy",
)
_TERM_BENEFITS = (
    "pure_term_life_cover",
    "critical_illness_rider",
    "accidental_death_rider",
    "waiver_of_premium",
    "terminal_illness_payout",
    "tax_saving_80c",
)

_HEALTH_EXCLUSIONS = ("cosmetic_surgery", "adventure_sports", "substance_abuse", "diabetes")
_CAR_EXCLUSIONS = ("drunk_driving", "unlicensed_driver", "commercial_use", "electrical_failure")
_LIFE_EXCLUSIONS = ("suicide_year_1", "hazardous_aviation", "war_perils")
_TERM_EXCLUSIONS = ("suicide_year_1", "narcotics_overdose", "undisclosed_smoking")

_PLAN_TIERS: tuple[str, ...] = ("Essential", "Plus", "Prime", "Select", "Advantage")


class DatasetValidationError(ValueError):
    """Raised when synthetic dataset validation detects schema or integrity errors."""


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return bool(pd.isna(value)) if not isinstance(value, (list, dict)) else False


def _optional_int(value: Any) -> int | None:
    return None if _is_missing(value) else int(value)


def _optional_float(value: Any) -> float | None:
    return None if _is_missing(value) else float(value)


def _optional_str(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _parse_json_list(raw: Any, field_name: str) -> list[str]:
    if isinstance(raw, list):
        return [str(x).strip().lower() for x in raw if str(x).strip()]
    if _is_missing(raw):
        raise DatasetValidationError(f"Field '{field_name}' cannot be null/NaN")
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"Field '{field_name}' is not valid JSON: {exc}") from exc
    if not isinstance(parsed, list):
        raise DatasetValidationError(f"Field '{field_name}' must decode to a JSON array")
    return [str(x).strip().lower() for x in parsed if str(x).strip()]


def _parse_json_dict(raw: Any, field_name: str) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if _is_missing(raw):
        raise DatasetValidationError(f"Field '{field_name}' cannot be null/NaN")
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"Field '{field_name}' is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise DatasetValidationError(f"Field '{field_name}' must decode to a JSON object")
    return parsed


def _build_health_policy(idx: int, rng: np.random.Generator) -> dict[str, Any]:
    tier = _PLAN_TIERS[idx % len(_PLAN_TIERS)]
    insurer = _FICTIONAL_INSURERS["health"][idx % len(_FICTIONAL_INSURERS["health"])]
    is_senior_focused = idx in (16, 17)
    min_age = 60 if is_senior_focused else int(rng.choice([0, 18, 21]))
    max_age = 80 if is_senior_focused else int(rng.choice([60, 65, 70, 75]))
    max_renewal = None if idx % 3 != 0 else max(max_age + 10, 85)
    lifelong = max_renewal is None

    sum_min = float(rng.choice([300_000, 500_000]))
    sum_max = float(rng.choice([1_000_000, 2_000_000, 5_000_000, 10_000_000]))
    prem_min = float(round(5_500 + (sum_min / 100_000) * 1_100 + rng.uniform(400, 3_500), -2))
    prem_max = float(round(prem_min * float(rng.uniform(2.2, 3.6)), -2))

    coverage_mode = (
        "individual" if idx in (4, 14) else ("family_floater" if idx in (5, 15) else "both")
    )
    max_members = 1 if coverage_mode == "individual" else int(rng.choice([4, 6]))

    copay = float(rng.choice([0.0, 0.0, 10.0, 20.0]))
    if is_senior_focused:
        copay = float(rng.choice([10.0, 20.0, 30.0]))

    room_type = str(rng.choice(["no_limit", "no_limit", "single_private", "shared"]))
    room_val = (
        5_000.0
        if room_type == "fixed_per_day"
        else (1.0 if room_type == "pct_sum_insured" else None)
    )

    ped_covered = idx != 18
    ped_wait = int(rng.choice([12, 24, 36, 48])) if ped_covered else None
    exclusions = list(_HEALTH_EXCLUSIONS[:3])
    if not ped_covered:
        exclusions.append("diabetes")

    benefits = ["cashless_hospitalization", "daycare_procedures", "pre_post_hospitalization"]
    for b in _HEALTH_BENEFITS[3:]:
        if rng.random() < 0.62:
            benefits.append(b)
    benefits = sorted(set(benefits))

    pan_india = idx not in (12, 13)
    regions = (
        list(_INDIAN_CITIES)
        if pan_india
        else list(rng.choice(_INDIAN_CITIES, size=4, replace=False))
    )
    applicants = ["senior_citizen"] if is_senior_focused else list(_APPLICANT_POOL)

    net_hospitals = int(rng.integers(4_500, 14_500))
    status = "archived" if idx == 19 else "active"
    code = f"POL-HLT-{idx + 1:03d}"

    return {
        "policy_id": str(uuid.uuid5(_NAMESPACE_POLICY, code)),
        "policy_code": code,
        "policy_name": f"{insurer.split()[0]} Health {tier} Plan {idx + 1}",
        "insurer_name": insurer,
        "policy_type": "health",
        "description": (
            f"Synthetic {coverage_mode.replace('_', ' ')} health insurance plan with "
            f"network hospitals={net_hospitals} and PED wait={ped_wait or 'N/A'} months."
        ),
        "policy_status": status,
        "is_catalog_eligible": status == "active",
        "min_entry_age": min_age,
        "max_entry_age": max_age,
        "max_renewal_age": max_renewal,
        "lifelong_renewal": lifelong,
        "coverage_mode": coverage_mode,
        "max_family_members": max_members,
        "eligible_applicant_types": json.dumps(applicants),
        "annual_premium_min": prem_min,
        "annual_premium_max": prem_max,
        "sum_insured_min": sum_min,
        "sum_insured_max": sum_max,
        "deductible_amount": float(rng.choice([0.0, 0.0, 5_000.0, 10_000.0])),
        "copay_pct": copay,
        "room_rent_limit_type": room_type,
        "room_rent_limit_value": room_val,
        "initial_waiting_period_days": int(rng.choice([15, 30, 30, 60])),
        "ped_covered": ped_covered,
        "ped_waiting_period_months": ped_wait,
        "specific_disease_waiting_months": int(rng.choice([12, 24])),
        "maternity_waiting_months": 24 if "maternity_cover" in benefits else None,
        "network_hospitals_count": net_hospitals,
        "cashless_available": idx != 14,
        "pan_india_coverage": pan_india,
        "available_regions": json.dumps(sorted(regions)),
        "min_tenure_years": 1,
        "max_tenure_years": int(rng.choice([2, 3])),
        "covered_benefits": json.dumps(benefits),
        "exclusions": json.dumps(sorted(exclusions)),
        "claim_settlement_ratio": float(round(float(rng.uniform(88.5, 99.2)), 2)),
        "no_claim_bonus_pct": float(rng.choice([10.0, 20.0, 50.0])),
        "restoration_benefit": bool(idx % 2 == 0),
        "type_specific_metadata": json.dumps(
            {
                "family_floater_supported": coverage_mode in ("family_floater", "both"),
                "icu_limit_type": "no_limit" if room_type == "no_limit" else "capped",
            }
        ),
        "extraction_source": "synthetic",
    }


def _build_car_policy(idx: int, rng: np.random.Generator) -> dict[str, Any]:
    tier = _PLAN_TIERS[idx % len(_PLAN_TIERS)]
    insurer = _FICTIONAL_INSURERS["car"][idx % len(_FICTIONAL_INSURERS["car"])]
    min_age = 18
    max_age = int(rng.choice([70, 75, 80]))
    sum_min = float(rng.choice([200_000, 350_000, 500_000]))
    sum_max = float(rng.choice([1_200_000, 2_500_000, 4_000_000]))
    prem_min = float(round(4_500 + rng.uniform(500, 6_000), -2))
    prem_max = float(round(prem_min * float(rng.uniform(2.5, 4.0)), -2))

    vehicle_types = (
        ["two_wheeler"]
        if idx in (16, 17)
        else (["hatchback", "sedan"] if idx in (14, 15) else ["hatchback", "sedan", "suv"])
    )
    vehicle_age_limit = int(rng.choice([5, 7, 10, 15]))
    zero_dep = vehicle_age_limit >= 5 and idx % 3 != 1
    rsa = idx % 2 == 0
    own_damage = idx != 18
    third_party = True
    engine_prot = idx % 4 == 0

    benefits = ["third_party_liability"]
    if own_damage:
        benefits.append("own_damage")
    if zero_dep:
        benefits.append("zero_depreciation")
    if rsa:
        benefits.append("roadside_assistance")
    if engine_prot:
        benefits.append("engine_protection")
    if rng.random() < 0.5:
        benefits.append("consumables_cover")

    pan_india = idx not in (12, 13)
    regions = (
        list(_INDIAN_CITIES)
        if pan_india
        else list(rng.choice(_INDIAN_CITIES, size=5, replace=False))
    )
    status = "draft" if idx == 19 else "active"
    code = f"POL-CAR-{idx + 1:03d}"

    return {
        "policy_id": str(uuid.uuid5(_NAMESPACE_POLICY, code)),
        "policy_code": code,
        "policy_name": f"{insurer.split()[0]} Motor {tier} Cover {idx + 1}",
        "insurer_name": insurer,
        "policy_type": "car",
        "description": (
            f"Synthetic motor insurance plan covering {', '.join(vehicle_types)} "
            f"up to {vehicle_age_limit} years vehicle age."
        ),
        "policy_status": status,
        "is_catalog_eligible": status == "active",
        "min_entry_age": min_age,
        "max_entry_age": max_age,
        "max_renewal_age": 85,
        "lifelong_renewal": False,
        "coverage_mode": "individual",
        "max_family_members": 1,
        "eligible_applicant_types": json.dumps(
            ["salaried", "self_employed", "student", "senior_citizen"]
        ),
        "annual_premium_min": prem_min,
        "annual_premium_max": prem_max,
        "sum_insured_min": sum_min,
        "sum_insured_max": sum_max,
        "deductible_amount": float(rng.choice([1_000.0, 2_000.0, 5_000.0])),
        "copay_pct": 0.0,
        "room_rent_limit_type": "no_limit",
        "room_rent_limit_value": None,
        "initial_waiting_period_days": 0,
        "ped_covered": True,
        "ped_waiting_period_months": None,
        "specific_disease_waiting_months": 0,
        "maternity_waiting_months": None,
        "network_hospitals_count": int(
            rng.integers(2_000, 8_500)
        ),  # cashless garages count stored in network count
        "cashless_available": idx != 18,
        "pan_india_coverage": pan_india,
        "available_regions": json.dumps(sorted(regions)),
        "min_tenure_years": 1,
        "max_tenure_years": 3,
        "covered_benefits": json.dumps(sorted(set(benefits))),
        "exclusions": json.dumps(list(_CAR_EXCLUSIONS[:3])),
        "claim_settlement_ratio": float(round(float(rng.uniform(86.0, 98.5)), 2)),
        "no_claim_bonus_pct": float(rng.choice([20.0, 35.0, 50.0])),
        "restoration_benefit": False,
        "type_specific_metadata": json.dumps(
            {
                "vehicle_types": vehicle_types,
                "vehicle_age_limit_years": vehicle_age_limit,
                "own_damage_coverage": own_damage,
                "third_party_coverage": third_party,
                "zero_depreciation": zero_dep,
                "roadside_assistance": rsa,
                "engine_protection": engine_prot,
            }
        ),
        "extraction_source": "synthetic",
    }


def _build_life_policy(idx: int, rng: np.random.Generator) -> dict[str, Any]:
    tier = _PLAN_TIERS[idx % len(_PLAN_TIERS)]
    insurer = _FICTIONAL_INSURERS["life"][idx % len(_FICTIONAL_INSURERS["life"])]
    min_age = int(rng.choice([18, 21]))
    max_age = int(rng.choice([55, 60, 65]))
    sum_min = float(rng.choice([500_000, 1_000_000]))
    sum_max = float(rng.choice([5_000_000, 10_000_000, 20_000_000]))
    prem_min = float(round(12_000 + rng.uniform(2_000, 18_000), -2))
    prem_max = float(round(prem_min * float(rng.uniform(2.5, 4.5)), -2))

    min_term = int(rng.choice([5, 10]))
    max_term = int(rng.choice([20, 25, 30, 35]))
    ppt = int(rng.choice([5, 10, 15, 20]))
    maturity_benefit = idx != 18

    benefits = ["death_benefit", "tax_saving_80c"]
    if maturity_benefit:
        benefits.append("maturity_benefit")
    for b in _LIFE_BENEFITS[2:]:
        if b not in benefits and rng.random() < 0.55:
            benefits.append(b)

    pan_india = idx not in (14, 15)
    regions = (
        list(_INDIAN_CITIES)
        if pan_india
        else list(rng.choice(_INDIAN_CITIES, size=5, replace=False))
    )
    status = "archived" if idx == 19 else "active"
    code = f"POL-LIF-{idx + 1:03d}"

    return {
        "policy_id": str(uuid.uuid5(_NAMESPACE_POLICY, code)),
        "policy_code": code,
        "policy_name": f"{insurer.split()[0]} Jeevan {tier} Savings {idx + 1}",
        "insurer_name": insurer,
        "policy_type": "life",
        "description": (
            f"Synthetic endowment/savings life policy for {min_term}-{max_term} year policy terms "
            f"with maturity_benefit={maturity_benefit}."
        ),
        "policy_status": status,
        "is_catalog_eligible": status == "active",
        "min_entry_age": min_age,
        "max_entry_age": max_age,
        "max_renewal_age": max_age + 15,
        "lifelong_renewal": False,
        "coverage_mode": "individual",
        "max_family_members": 1,
        "eligible_applicant_types": json.dumps(["salaried", "self_employed", "student"]),
        "annual_premium_min": prem_min,
        "annual_premium_max": prem_max,
        "sum_insured_min": sum_min,
        "sum_insured_max": sum_max,
        "deductible_amount": 0.0,
        "copay_pct": 0.0,
        "room_rent_limit_type": "no_limit",
        "room_rent_limit_value": None,
        "initial_waiting_period_days": 0,
        "ped_covered": True,
        "ped_waiting_period_months": None,
        "specific_disease_waiting_months": 0,
        "maternity_waiting_months": None,
        "network_hospitals_count": 0,
        "cashless_available": False,
        "pan_india_coverage": pan_india,
        "available_regions": json.dumps(sorted(regions)),
        "min_tenure_years": min_term,
        "max_tenure_years": max_term,
        "covered_benefits": json.dumps(sorted(set(benefits))),
        "exclusions": json.dumps(list(_LIFE_EXCLUSIONS)),
        "claim_settlement_ratio": float(round(float(rng.uniform(94.0, 99.6)), 2)),
        "no_claim_bonus_pct": 0.0,
        "restoration_benefit": False,
        "type_specific_metadata": json.dumps(
            {
                "life_cover_min": sum_min,
                "life_cover_max": sum_max,
                "policy_term_years_min": min_term,
                "policy_term_years_max": max_term,
                "maturity_benefit": maturity_benefit,
                "premium_payment_term_years": ppt,
                "surrender_value_available": True,
            }
        ),
        "extraction_source": "synthetic",
    }


def _build_term_policy(idx: int, rng: np.random.Generator) -> dict[str, Any]:
    tier = _PLAN_TIERS[idx % len(_PLAN_TIERS)]
    insurer = _FICTIONAL_INSURERS["term"][idx % len(_FICTIONAL_INSURERS["term"])]
    min_age = int(rng.choice([18, 21]))
    max_age = int(rng.choice([60, 65]))
    sum_min = float(rng.choice([2_500_000, 5_000_000]))
    sum_max = float(rng.choice([20_000_000, 50_000_000, 100_000_000]))
    prem_min = float(round(6_500 + rng.uniform(1_000, 8_000), -2))
    prem_max = float(round(prem_min * float(rng.uniform(2.8, 5.0)), -2))

    min_term = int(rng.choice([5, 10]))
    max_term = int(rng.choice([30, 40, 45]))
    ppt = int(rng.choice([10, 15, 20, 30]))
    smoking_elig = (
        "non_smoker_only" if idx in (16, 17) else ("smoker_surcharge" if idx % 2 == 0 else "all")
    )

    benefits = ["pure_term_life_cover", "tax_saving_80c"]
    for b in _TERM_BENEFITS[1:]:
        if b not in benefits and rng.random() < 0.65:
            benefits.append(b)

    pan_india = idx not in (14, 15)
    regions = (
        list(_INDIAN_CITIES)
        if pan_india
        else list(rng.choice(_INDIAN_CITIES, size=5, replace=False))
    )
    status = "draft" if idx == 19 else "active"
    code = f"POL-TRM-{idx + 1:03d}"

    return {
        "policy_id": str(uuid.uuid5(_NAMESPACE_POLICY, code)),
        "policy_code": code,
        "policy_name": f"{insurer.split()[0]} Pure Term {tier} {idx + 1}",
        "insurer_name": insurer,
        "policy_type": "term",
        "description": (
            f"Synthetic pure term protection plan for {min_term}-{max_term} years "
            f"with smoking_eligibility={smoking_elig}."
        ),
        "policy_status": status,
        "is_catalog_eligible": status == "active",
        "min_entry_age": min_age,
        "max_entry_age": max_age,
        "max_renewal_age": 85,
        "lifelong_renewal": False,
        "coverage_mode": "individual",
        "max_family_members": 1,
        "eligible_applicant_types": json.dumps(["salaried", "self_employed"]),
        "annual_premium_min": prem_min,
        "annual_premium_max": prem_max,
        "sum_insured_min": sum_min,
        "sum_insured_max": sum_max,
        "deductible_amount": 0.0,
        "copay_pct": 0.0,
        "room_rent_limit_type": "no_limit",
        "room_rent_limit_value": None,
        "initial_waiting_period_days": 0,
        "ped_covered": True,
        "ped_waiting_period_months": None,
        "specific_disease_waiting_months": 0,
        "maternity_waiting_months": None,
        "network_hospitals_count": 0,
        "cashless_available": False,
        "pan_india_coverage": pan_india,
        "available_regions": json.dumps(sorted(regions)),
        "min_tenure_years": min_term,
        "max_tenure_years": max_term,
        "covered_benefits": json.dumps(sorted(set(benefits))),
        "exclusions": json.dumps(list(_TERM_EXCLUSIONS)),
        "claim_settlement_ratio": float(round(float(rng.uniform(96.0, 99.8)), 2)),
        "no_claim_bonus_pct": 0.0,
        "restoration_benefit": False,
        "type_specific_metadata": json.dumps(
            {
                "life_cover_min": sum_min,
                "life_cover_max": sum_max,
                "policy_term_years_min": min_term,
                "policy_term_years_max": max_term,
                "premium_payment_term_years": ppt,
                "smoking_status_eligibility": smoking_elig,
                "critical_illness_rider": "critical_illness_rider" in benefits,
                "accidental_death_rider": "accidental_death_rider" in benefits,
            }
        ),
        "extraction_source": "synthetic",
    }


def generate_synthetic_policies(
    seed: int = DEFAULT_SEED,
    per_category: int = 20,
) -> pd.DataFrame:
    """Generate deterministic synthetic insurance policies across all four categories."""
    if per_category < 13:
        raise ValueError("per_category must be >= 13 so at least 50 total policies are generated")

    rng = np.random.default_rng(seed)
    builders = {
        "health": _build_health_policy,
        "car": _build_car_policy,
        "life": _build_life_policy,
        "term": _build_term_policy,
    }
    rows: list[dict[str, Any]] = []
    for category in POLICY_CATEGORIES:
        builder = builders[category]
        for idx in range(per_category):
            rows.append(builder(idx, rng))

    return pd.DataFrame(rows, columns=list(POLICY_COLUMNS))


def generate_synthetic_profiles(
    seed: int = DEFAULT_SEED,
    per_category: int = 15,
) -> pd.DataFrame:
    """Generate deterministic fictional Indian user profiles compatible with UserProfileRequest."""
    if per_category < 5:
        raise ValueError("per_category must be >= 5 so at least 20 total profiles are generated")

    rng = np.random.default_rng(seed + 101)
    rows: list[dict[str, Any]] = []

    for category in POLICY_CATEGORIES:
        for idx in range(per_category):
            is_strict_edge = idx == per_category - 1
            risk_pref = str(rng.choice(["low_premium", "balanced", "comprehensive_coverage"]))
            city = str(rng.choice(_INDIAN_CITIES))

            if category == "health":
                if idx == per_category - 2:
                    applicant_type = "senior_citizen"
                    age = int(rng.integers(62, 73))
                else:
                    applicant_type = str(rng.choice(["salaried", "self_employed", "student"]))
                    age = int(rng.integers(22, 56))
                coverage_mode = (
                    "family_floater"
                    if (idx % 2 == 1 and applicant_type != "student")
                    else "individual"
                )
                family_size = int(rng.choice([2, 3, 4])) if coverage_mode == "family_floater" else 1
                annual_income = float(round(rng.uniform(450_000, 2_800_000), -3))
                budget_max = (
                    3_000.0 if is_strict_edge else float(round(rng.uniform(14_000, 48_000), -2))
                )
                budget_min = float(round(budget_max * 0.35, -2))
                min_sum = float(rng.choice([500_000, 1_000_000, 1_500_000]))
                req_benefits = ["cashless_hospitalization"]
                if idx % 3 == 0:
                    req_benefits.append("daycare_procedures")
                peds = ["diabetes"] if idx in (2, 7, 13) else (["hypertension"] if idx == 9 else [])
                max_ped_wait = 36 if peds else None
                max_wait_days = 30 if idx % 2 == 0 else None
                max_copay = 20.0 if applicant_type != "senior_citizen" else 30.0
                max_deductible = 10_000.0
                room_pref = "no_limit" if risk_pref == "comprehensive_coverage" else "any"
                req_cashless = True
                min_hosp = 4_000 if not is_strict_edge else 50_000
                tenure = int(rng.choice([1, 2]))
                req_lifelong = idx % 4 == 0
                smoking = "non_smoker"
                veh_type = None
                veh_age = None

            elif category == "car":
                applicant_type = str(rng.choice(["salaried", "self_employed", "student"]))
                age = int(rng.integers(21, 60))
                coverage_mode = "individual"
                family_size = 1
                annual_income = float(round(rng.uniform(500_000, 3_000_000), -3))
                budget_max = (
                    1_500.0 if is_strict_edge else float(round(rng.uniform(10_000, 32_000), -2))
                )
                budget_min = float(round(budget_max * 0.30, -2))
                min_sum = float(rng.choice([400_000, 700_000, 1_000_000]))
                req_benefits = ["third_party_liability"]
                if idx % 2 == 0:
                    req_benefits.append("own_damage")
                if idx % 4 == 0:
                    req_benefits.append("zero_depreciation")
                peds = []
                max_ped_wait = None
                max_wait_days = None
                max_copay = None
                max_deductible = 5_000.0
                room_pref = "any"
                req_cashless = False
                min_hosp = 0
                tenure = 1
                req_lifelong = False
                smoking = "non_smoker"
                veh_type = (
                    "two_wheeler" if idx == 12 else str(rng.choice(["hatchback", "sedan", "suv"]))
                )
                veh_age = 18 if is_strict_edge else int(rng.integers(1, 7))

            elif category == "life":
                applicant_type = str(rng.choice(["salaried", "self_employed"]))
                age = int(rng.integers(24, 50))
                coverage_mode = "individual"
                family_size = 1
                annual_income = float(round(rng.uniform(600_000, 3_600_000), -3))
                budget_max = (
                    5_000.0 if is_strict_edge else float(round(rng.uniform(25_000, 95_000), -2))
                )
                budget_min = float(round(budget_max * 0.35, -2))
                min_sum = float(rng.choice([1_000_000, 2_500_000, 5_000_000]))
                req_benefits = ["death_benefit"]
                if idx % 2 == 0:
                    req_benefits.append("maturity_benefit")
                peds = []
                max_ped_wait = None
                max_wait_days = None
                max_copay = None
                max_deductible = None
                room_pref = "any"
                req_cashless = False
                min_hosp = 0
                tenure = int(rng.choice([10, 15, 20]))
                req_lifelong = False
                smoking = "non_smoker"
                veh_type = None
                veh_age = None

            else:  # term
                applicant_type = str(rng.choice(["salaried", "self_employed"]))
                age = 68 if is_strict_edge else int(rng.integers(23, 52))
                coverage_mode = "individual"
                family_size = 1
                annual_income = float(round(rng.uniform(700_000, 4_200_000), -3))
                budget_max = float(round(rng.uniform(15_000, 48_000), -2))
                budget_min = float(round(budget_max * 0.30, -2))
                min_sum = float(rng.choice([5_000_000, 10_000_000, 20_000_000]))
                req_benefits = ["pure_term_life_cover"]
                if idx % 3 == 0:
                    req_benefits.append("critical_illness_rider")
                peds = []
                max_ped_wait = None
                max_wait_days = None
                max_copay = None
                max_deductible = None
                room_pref = "any"
                req_cashless = False
                min_hosp = 0
                tenure = int(rng.choice([15, 20, 30]))
                req_lifelong = False
                smoking = "smoker" if idx in (5, 11) else "non_smoker"
                veh_type = None
                veh_age = None

            occupation = str(rng.choice(_OCCUPATION_BY_APPLICANT[applicant_type]))
            monthly_budget = float(round(budget_max / 12.0, 2))
            profile_id = f"PROF-{category[:3].upper()}-{idx + 1:03d}"

            row = {
                "profile_id": profile_id,
                "preferred_policy_type": category,
                "age": age,
                "annual_income_inr": annual_income,
                "monthly_budget_inr": monthly_budget,
                "budget_min": budget_min,
                "budget_max": budget_max,
                "applicant_type": applicant_type,
                "occupation": occupation,
                "coverage_mode": coverage_mode,
                "family_size": family_size,
                "min_sum_insured": min_sum,
                "required_benefits": json.dumps(req_benefits),
                "pre_existing_conditions": json.dumps(peds),
                "max_ped_waiting_months": max_ped_wait,
                "max_waiting_period_days": max_wait_days,
                "max_copay_pct": max_copay,
                "max_deductible": max_deductible,
                "room_rent_preference": room_pref,
                "require_cashless": req_cashless,
                "min_network_hospitals": min_hosp,
                "geographic_location": city,
                "tenure_years": tenure,
                "require_lifelong_renewal": req_lifelong,
                "smoking_status": smoking,
                "vehicle_type": veh_type,
                "vehicle_age_years": veh_age,
                "risk_preference": risk_pref,
            }
            profile_row_to_request(row)
            rows.append(row)

    return pd.DataFrame(rows, columns=list(PROFILE_COLUMNS))


def profile_row_to_request(row: dict[str, Any] | pd.Series) -> UserProfileRequest:
    """Convert a synthetic profile row into a validated UserProfileRequest schema."""
    data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    return UserProfileRequest(
        policy_type=str(data["preferred_policy_type"]).strip().lower(),
        age=int(data["age"]),
        budget_min=float(data["budget_min"]),
        budget_max=float(data["budget_max"]),
        applicant_type=str(data["applicant_type"]).strip().lower(),
        coverage_mode=str(data["coverage_mode"]).strip().lower(),
        family_size=int(data["family_size"]),
        min_sum_insured=float(data["min_sum_insured"]),
        required_benefits=_parse_json_list(data["required_benefits"], "required_benefits"),
        pre_existing_conditions=_parse_json_list(
            data["pre_existing_conditions"], "pre_existing_conditions"
        ),
        max_ped_waiting_months=_optional_int(data.get("max_ped_waiting_months")),
        max_waiting_period_days=_optional_int(data.get("max_waiting_period_days")),
        max_copay_pct=_optional_float(data.get("max_copay_pct")),
        max_deductible=_optional_float(data.get("max_deductible")),
        room_rent_preference=str(data.get("room_rent_preference", "any")).strip().lower(),
        require_cashless=bool(data.get("require_cashless", False)),
        min_network_hospitals=int(data.get("min_network_hospitals", 0)),
        location=_optional_str(data.get("geographic_location")),
        tenure_years=int(data.get("tenure_years", 1)),
        require_lifelong_renewal=bool(data.get("require_lifelong_renewal", False)),
    )


def evaluate_eligibility(
    profile: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Apply deterministic mandatory eligibility checks between a profile and policy."""
    failed: list[str] = []

    if str(policy["policy_status"]).lower() != "active" or not bool(policy["is_catalog_eligible"]):
        failed.append("inactive_or_non_catalog_policy")

    if str(policy["policy_type"]).lower() != str(profile["preferred_policy_type"]).lower():
        failed.append("policy_type_mismatch")

    age = int(profile["age"])
    if not (int(policy["min_entry_age"]) <= age <= int(policy["max_entry_age"])):
        failed.append("age_out_of_range")

    eligible_applicants = set(
        _parse_json_list(policy["eligible_applicant_types"], "eligible_applicant_types")
    )
    if str(profile["applicant_type"]).strip().lower() not in eligible_applicants:
        failed.append("applicant_type_ineligible")

    prof_mode = str(profile["coverage_mode"]).strip().lower()
    pol_mode = str(policy["coverage_mode"]).strip().lower()
    if prof_mode == "family_floater":
        if pol_mode not in ("family_floater", "both"):
            failed.append("family_floater_unsupported")
        elif int(profile["family_size"]) > int(policy["max_family_members"]):
            failed.append("family_size_exceeded")
    elif pol_mode not in ("individual", "both"):
        failed.append("individual_mode_unsupported")

    if float(policy["annual_premium_min"]) > float(profile["budget_max"]):
        failed.append("exceeds_max_budget")
    if float(policy["annual_premium_max"]) < float(profile["budget_min"]):
        failed.append("below_min_budget")

    if float(policy["sum_insured_max"]) < float(profile["min_sum_insured"]):
        failed.append("insufficient_sum_insured")

    loc = _optional_str(profile.get("geographic_location"))
    if loc and not bool(policy["pan_india_coverage"]):
        regions = set(_parse_json_list(policy["available_regions"], "available_regions"))
        if loc.lower() not in regions:
            failed.append("region_unavailable")

    tenure = int(profile["tenure_years"])
    if not (int(policy["min_tenure_years"]) <= tenure <= int(policy["max_tenure_years"])):
        failed.append("tenure_out_of_range")

    if bool(profile["require_lifelong_renewal"]) and not bool(policy["lifelong_renewal"]):
        failed.append("lifelong_renewal_required")

    max_copay = _optional_float(profile.get("max_copay_pct"))
    if max_copay is not None and float(policy["copay_pct"]) > max_copay:
        failed.append("copay_exceeds_limit")

    max_deduct = _optional_float(profile.get("max_deductible"))
    if max_deduct is not None and float(policy["deductible_amount"]) > max_deduct:
        failed.append("deductible_exceeds_limit")

    max_wait_days = _optional_int(profile.get("max_waiting_period_days"))
    if max_wait_days is not None and int(policy["initial_waiting_period_days"]) > max_wait_days:
        failed.append("initial_waiting_period_exceeded")

    peds = set(_parse_json_list(profile["pre_existing_conditions"], "pre_existing_conditions"))
    exclusions = set(_parse_json_list(policy["exclusions"], "exclusions"))
    if peds:
        if not bool(policy["ped_covered"]) or (peds & exclusions):
            failed.append("pre_existing_condition_excluded")
        else:
            max_ped_wait = _optional_int(profile.get("max_ped_waiting_months"))
            pol_ped_wait = _optional_int(policy.get("ped_waiting_period_months"))
            if max_ped_wait is not None and (pol_ped_wait is None or pol_ped_wait > max_ped_wait):
                failed.append("ped_waiting_period_exceeded")

    room_pref = str(profile.get("room_rent_preference", "any")).strip().lower()
    pol_room = str(policy["room_rent_limit_type"]).strip().lower()
    if room_pref == "no_limit" and pol_room != "no_limit":
        failed.append("room_rent_limit_not_unrestricted")
    elif room_pref == "single_private" and pol_room not in ("no_limit", "single_private"):
        failed.append("room_rent_limit_below_single_private")

    if bool(profile["require_cashless"]) and not bool(policy["cashless_available"]):
        failed.append("cashless_unavailable")

    if int(profile["min_network_hospitals"]) > int(policy["network_hospitals_count"]):
        failed.append("network_hospitals_below_minimum")

    req_benefits = set(_parse_json_list(profile["required_benefits"], "required_benefits"))
    pol_benefits = set(_parse_json_list(policy["covered_benefits"], "covered_benefits"))
    if not req_benefits.issubset(pol_benefits):
        failed.append("missing_required_benefits")

    meta = _parse_json_dict(policy["type_specific_metadata"], "type_specific_metadata")
    pol_type = str(policy["policy_type"]).lower()
    if pol_type == "car":
        v_type = _optional_str(profile.get("vehicle_type"))
        v_age = _optional_int(profile.get("vehicle_age_years"))
        if v_type and v_type.lower() not in [str(x).lower() for x in meta.get("vehicle_types", [])]:
            failed.append("vehicle_type_unsupported")
        if v_age is not None and v_age > int(meta.get("vehicle_age_limit_years", 0)):
            failed.append("vehicle_age_exceeds_limit")
    elif pol_type == "term":
        smoking = _optional_str(profile.get("smoking_status"))
        if smoking == "smoker" and meta.get("smoking_status_eligibility") == "non_smoker_only":
            failed.append("smoker_ineligible")

    return len(failed) == 0, failed


def compute_suitability_score(
    profile: dict[str, Any],
    policy: dict[str, Any],
    is_eligible: bool,
) -> tuple[float, int]:
    """Compute a leakage-free graded suitability score [0, 1] and label {0,1,2,3}."""
    if not is_eligible:
        return 0.0, 0

    budget_max = max(float(profile["budget_max"]), 1.0)
    prem_mid = 0.5 * (float(policy["annual_premium_min"]) + float(policy["annual_premium_max"]))
    effective_prem = (
        min(prem_mid, budget_max) if float(policy["annual_premium_min"]) <= budget_max else prem_mid
    )
    affordability = max(
        0.0, min(1.0, (budget_max - float(policy["annual_premium_min"])) / budget_max)
    )
    midpoint_fit = max(0.0, min(1.0, 1.0 - abs(effective_prem - 0.65 * budget_max) / budget_max))
    affordability_score = 0.65 * affordability + 0.35 * midpoint_fit

    min_sum = max(float(profile["min_sum_insured"]), 1.0)
    sum_ratio = float(policy["sum_insured_max"]) / min_sum
    coverage_score = min(1.0, math.log1p(max(0.0, sum_ratio - 1.0)) / math.log1p(4.0))

    pol_benefits = _parse_json_list(policy["covered_benefits"], "covered_benefits")
    benefit_richness = min(1.0, len(pol_benefits) / 6.0)
    if bool(policy["restoration_benefit"]):
        benefit_richness = min(1.0, benefit_richness + 0.12)
    if float(policy["no_claim_bonus_pct"]) > 0:
        benefit_richness = min(1.0, benefit_richness + 0.08)

    copay_score = 1.0 - min(1.0, float(policy["copay_pct"]) / 40.0)
    deduct_score = 1.0 - min(1.0, float(policy["deductible_amount"]) / 25_000.0)
    wait_score = 1.0 - min(1.0, float(policy["initial_waiting_period_days"]) / 90.0)
    ped_wait = _optional_int(policy.get("ped_waiting_period_months"))
    ped_score = 1.0 if ped_wait is None else max(0.0, 1.0 - ped_wait / 60.0)
    cost_waiting_score = (
        0.35 * copay_score + 0.25 * deduct_score + 0.20 * wait_score + 0.20 * ped_score
    )

    csr = _optional_float(policy.get("claim_settlement_ratio")) or 90.0
    csr_norm = max(0.0, min(1.0, (csr - 85.0) / 15.0))
    net_norm = min(1.0, float(policy["network_hospitals_count"]) / 12_000.0)
    insurer_network_score = 0.70 * csr_norm + 0.30 * net_norm

    risk_pref = str(profile.get("risk_preference", "balanced")).strip().lower()
    if risk_pref == "low_premium":
        weights = (0.40, 0.15, 0.15, 0.15, 0.15)
    elif risk_pref == "comprehensive_coverage":
        weights = (0.12, 0.23, 0.27, 0.18, 0.20)
    else:
        weights = (0.25, 0.20, 0.20, 0.15, 0.20)

    score = (
        weights[0] * affordability_score
        + weights[1] * coverage_score
        + weights[2] * benefit_richness
        + weights[3] * cost_waiting_score
        + weights[4] * insurer_network_score
    )
    score = round(max(0.0, min(1.0, score)), 4)

    if score >= 0.78:
        label = 3
    elif score >= 0.62:
        label = 2
    elif score >= 0.45:
        label = 1
    else:
        label = 0
    return score, label


def generate_synthetic_relevance(
    policies_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate deterministic (profile, policy) eligibility and suitability labels."""
    policy_records = policies_df.to_dict(orient="records")
    profile_records = profiles_df.to_dict(orient="records")

    rows: list[dict[str, Any]] = []
    for profile in profile_records:
        for policy in policy_records:
            type_match = (
                str(policy["policy_type"]).lower() == str(profile["preferred_policy_type"]).lower()
            )
            is_eligible, failed_rules = evaluate_eligibility(profile, policy)
            score, label = compute_suitability_score(profile, policy, is_eligible)
            rows.append(
                {
                    "profile_id": profile["profile_id"],
                    "policy_id": policy["policy_id"],
                    "policy_type_match": type_match,
                    "is_eligible": is_eligible,
                    "failed_rules": "|".join(failed_rules),
                    "suitability_score": score,
                    "relevance_label": label,
                }
            )
    return pd.DataFrame(rows, columns=list(RELEVANCE_COLUMNS))


def validate_policies_df(df: pd.DataFrame) -> None:
    """Validate structural, type, domain, and range invariants on policies DataFrame."""
    missing_cols = [c for c in POLICY_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DatasetValidationError(f"Policies missing columns: {missing_cols}")
    if len(df) < 50:
        raise DatasetValidationError(f"Expected >= 50 policies, got {len(df)}")
    if df["policy_id"].isna().any() or not df["policy_id"].is_unique:
        raise DatasetValidationError("policy_id must be non-null and unique")
    if df["policy_code"].isna().any() or not df["policy_code"].is_unique:
        raise DatasetValidationError("policy_code must be non-null and unique")

    for req_text in ("policy_name", "insurer_name", "policy_type", "policy_status"):
        if df[req_text].isna().any() or (df[req_text].astype(str).str.strip() == "").any():
            raise DatasetValidationError(f"Mandatory policy field '{req_text}' contains blanks")

    categories = set(df["policy_type"].astype(str).str.lower())
    if not set(POLICY_CATEGORIES).issubset(categories):
        raise DatasetValidationError(
            f"Missing required categories: {set(POLICY_CATEGORIES) - categories}"
        )
    if not categories.issubset(set(POLICY_CATEGORIES)):
        raise DatasetValidationError(
            f"Unsupported policy categories: {categories - set(POLICY_CATEGORIES)}"
        )

    if (df["min_entry_age"] < 0).any() or (df["max_entry_age"] > 120).any():
        raise DatasetValidationError("Policy entry ages must be within [0, 120]")
    if (df["max_entry_age"] < df["min_entry_age"]).any():
        raise DatasetValidationError("max_entry_age cannot be less than min_entry_age")
    if (df["annual_premium_min"] <= 0).any() or (
        df["annual_premium_max"] < df["annual_premium_min"]
    ).any():
        raise DatasetValidationError("Invalid annual_premium_min/max range")
    if (df["sum_insured_min"] <= 0).any() or (df["sum_insured_max"] < df["sum_insured_min"]).any():
        raise DatasetValidationError("Invalid sum_insured_min/max range")
    if (df["copay_pct"] < 0).any() or (df["copay_pct"] > 100).any():
        raise DatasetValidationError("copay_pct must be within [0, 100]")

    for row in df.to_dict(orient="records"):
        _parse_json_list(row["eligible_applicant_types"], "eligible_applicant_types")
        _parse_json_list(row["available_regions"], "available_regions")
        _parse_json_list(row["covered_benefits"], "covered_benefits")
        _parse_json_list(row["exclusions"], "exclusions")
        _parse_json_dict(row["type_specific_metadata"], "type_specific_metadata")


def validate_profiles_df(df: pd.DataFrame) -> None:
    """Validate user profiles against Pydantic UserProfileRequest and integrity checks."""
    missing_cols = [c for c in PROFILE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DatasetValidationError(f"Profiles missing columns: {missing_cols}")
    if len(df) < 20:
        raise DatasetValidationError(f"Expected >= 20 profiles, got {len(df)}")
    if df["profile_id"].isna().any() or not df["profile_id"].is_unique:
        raise DatasetValidationError("profile_id must be non-null and unique")
    if (df["annual_income_inr"] <= 0).any() or (df["monthly_budget_inr"] <= 0).any():
        raise DatasetValidationError("annual_income_inr and monthly_budget_inr must be > 0")

    categories = set(df["preferred_policy_type"].astype(str).str.lower())
    if not set(POLICY_CATEGORIES).issubset(categories):
        raise DatasetValidationError(
            f"Profiles missing categories: {set(POLICY_CATEGORIES) - categories}"
        )

    for row in df.to_dict(orient="records"):
        try:
            profile_row_to_request(row)
        except Exception as exc:
            raise DatasetValidationError(f"Invalid profile {row.get('profile_id')}: {exc}") from exc


def validate_relevance_df(
    relevance_df: pd.DataFrame,
    policies_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
) -> None:
    """Validate referential integrity, label bounds, and eligibility separation."""
    missing_cols = [c for c in RELEVANCE_COLUMNS if c not in relevance_df.columns]
    if missing_cols:
        raise DatasetValidationError(f"Relevance table missing columns: {missing_cols}")
    if relevance_df.empty:
        raise DatasetValidationError("Relevance table cannot be empty")
    if relevance_df.duplicated(subset=["profile_id", "policy_id"]).any():
        raise DatasetValidationError("Duplicate (profile_id, policy_id) pairs detected")

    valid_profiles = set(profiles_df["profile_id"])
    valid_policies = set(policies_df["policy_id"])
    if not set(relevance_df["profile_id"]).issubset(valid_profiles):
        raise DatasetValidationError("Orphan profile_id found in relevance table")
    if not set(relevance_df["policy_id"]).issubset(valid_policies):
        raise DatasetValidationError("Orphan policy_id found in relevance table")

    if not set(relevance_df["relevance_label"].unique()).issubset({0, 1, 2, 3}):
        raise DatasetValidationError("relevance_label values must be in {0, 1, 2, 3}")
    if (relevance_df["suitability_score"] < 0).any() or (
        relevance_df["suitability_score"] > 1
    ).any():
        raise DatasetValidationError("suitability_score must lie within [0.0, 1.0]")

    ineligible = relevance_df[~relevance_df["is_eligible"].astype(bool)]
    if (ineligible["relevance_label"] != 0).any() or (ineligible["suitability_score"] != 0.0).any():
        raise DatasetValidationError(
            "Ineligible policy pairs must always have label=0 and score=0.0"
        )


def load_synthetic_dataset(
    data_dir: Path | None = None,
    validate: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and optionally validate synthetic policies, profiles, and relevance CSVs."""
    target_dir = data_dir or DEFAULT_DATA_DIR
    policies_path = target_dir / "synthetic_policies.csv"
    profiles_path = target_dir / "synthetic_profiles.csv"
    relevance_path = target_dir / "synthetic_relevance.csv"

    for path in (policies_path, profiles_path, relevance_path):
        if not path.exists():
            raise FileNotFoundError(f"Required dataset file not found: {path}")

    policies_df = pd.read_csv(policies_path)
    profiles_df = pd.read_csv(profiles_path)
    relevance_df = pd.read_csv(relevance_path, keep_default_na=False)
    relevance_df["suitability_score"] = pd.to_numeric(
        relevance_df["suitability_score"], errors="raise"
    )
    relevance_df["relevance_label"] = pd.to_numeric(
        relevance_df["relevance_label"], errors="raise"
    ).astype(int)

    if validate:
        validate_policies_df(policies_df)
        validate_profiles_df(profiles_df)
        validate_relevance_df(relevance_df, policies_df, profiles_df)

    return policies_df, profiles_df, relevance_df


def build_and_save_dataset(
    output_dir: Path | None = None,
    seed: int = DEFAULT_SEED,
    policies_per_category: int = 20,
    profiles_per_category: int = 15,
) -> dict[str, Any]:
    """Generate, validate, and write synthetic CSV artifacts to disk."""
    target_dir = output_dir or DEFAULT_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    policies_df = generate_synthetic_policies(seed=seed, per_category=policies_per_category)
    profiles_df = generate_synthetic_profiles(seed=seed, per_category=profiles_per_category)
    relevance_df = generate_synthetic_relevance(policies_df, profiles_df)

    validate_policies_df(policies_df)
    validate_profiles_df(profiles_df)
    validate_relevance_df(relevance_df, policies_df, profiles_df)

    policies_path = target_dir / "synthetic_policies.csv"
    profiles_path = target_dir / "synthetic_profiles.csv"
    relevance_path = target_dir / "synthetic_relevance.csv"

    policies_df.to_csv(policies_path, index=False)
    profiles_df.to_csv(profiles_path, index=False)
    relevance_df.to_csv(relevance_path, index=False)

    return {
        "seed": seed,
        "output_dir": str(target_dir),
        "policy_count": len(policies_df),
        "profile_count": len(profiles_df),
        "relevance_pair_count": len(relevance_df),
        "eligible_pair_count": int(relevance_df["is_eligible"].sum()),
        "category_counts": policies_df["policy_type"].value_counts().to_dict(),
        "label_distribution": relevance_df["relevance_label"].value_counts().sort_index().to_dict(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CoverageIQ synthetic recommendation dataset."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 42)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Target directory for synthetic CSV files",
    )
    parser.add_argument("--policies-per-category", type=int, default=20)
    parser.add_argument("--profiles-per-category", type=int, default=15)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary = build_and_save_dataset(
        output_dir=args.output_dir,
        seed=args.seed,
        policies_per_category=args.policies_per_category,
        profiles_per_category=args.profiles_per_category,
    )
    print(json.dumps(summary, indent=2))
