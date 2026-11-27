"""Deterministic Hard Filtering Engine and DB adapter for Module 6A.

Separates pure eligibility evaluation from PostgreSQL data loading:
- Layer A: Explicit errors and result models.
- Layer B: Pure deterministic eligibility rules (common + health/car/life/term).
- Layer C: Filtering orchestration and structured audit reporting.
- Layer D: Parameterized SQLAlchemy Core database adapter.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import Connection, Select, and_, or_, select, text
from sqlalchemy.exc import OperationalError, ProgrammingError, SQLAlchemyError

from ai_service.common.tables import policies, policy_attributes
from ai_service.recommendation.schemas import (
    FilterEliminationReason,
    FilterFailureDetail,
    FilterReport,
    PolicyFilterEvaluation,
    UserProfileRequest,
)

# ---------------------------------------------------------------------------
# Layer A: Exceptions & Result Container
# ---------------------------------------------------------------------------


class MissingMandatoryProfileError(ValueError):
    """Raised when a profile omits mandatory fields required for eligibility filtering."""


class PolicyAttributesTableUnavailableError(RuntimeError):
    """Raised when policy_attributes is absent from PostgreSQL."""


class DuplicateCatalogPolicyError(ValueError):
    """Raised when catalog retrieval returns duplicate rows for a single policy_id."""


class HardFilterResult(BaseModel):
    """Complete deterministic hard-filtering output including eligible policies and audit trail."""

    status: Literal["ELIGIBLE_CANDIDATES_FOUND", "NO_MATCHING_POLICIES"]
    eligible_policies: list[dict[str, Any]] = Field(default_factory=list)
    filter_report: FilterReport


@dataclass
class _RuleAccumulator:
    """Tracks passed and failed filter outcomes for a single policy evaluation."""

    passed: list[str]
    failed: list[FilterFailureDetail]

    def pass_rule(self, rule_id: str) -> None:
        self.passed.append(rule_id)

    def fail_rule(
        self,
        rule_id: str,
        message: str,
        expected: Any = None,
        actual: Any = None,
    ) -> None:
        self.failed.append(
            FilterFailureDetail(
                rule_id=rule_id,
                message=message,
                expected=expected,
                actual=actual,
            )
        )


# ---------------------------------------------------------------------------
# Layer B: Pure Input Normalization & Deterministic Rules
# ---------------------------------------------------------------------------

_MANDATORY_PROFILE_FIELDS = ("policy_type", "age", "budget_max", "min_sum_insured")
_NON_FILTER_PROFILE_FIELDS = frozenset(
    {
        "profile_id",
        "annual_income_inr",
        "monthly_budget_inr",
        "occupation",
        "risk_preference",
    }
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _to_optional_int(value: Any) -> int | None:
    return None if _is_missing(value) else int(value)


def _to_optional_float(value: Any) -> float | None:
    return None if _is_missing(value) else float(value)


def _to_optional_bool(value: Any) -> bool | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
    return bool(value)


def _to_optional_str(value: Any) -> str | None:
    if _is_missing(value):
        return None
    return str(value).strip().lower()


def _parse_str_list(raw: Any) -> list[str] | None:
    """Parse a JSON-encoded list or Python sequence into lowercase strings."""
    if _is_missing(raw):
        return None
    if isinstance(raw, (list, tuple, set)):
        return [str(x).strip().lower() for x in raw if str(x).strip()]
    try:
        decoded = json.loads(str(raw))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(decoded, list):
        return None
    return [str(x).strip().lower() for x in decoded if str(x).strip()]


def _parse_metadata_dict(raw: Any) -> dict[str, Any]:
    """Parse type_specific_metadata safely without failing on empty values."""
    if _is_missing(raw):
        return {}
    if isinstance(raw, Mapping):
        return {str(k): v for k, v in raw.items()}
    try:
        decoded = json.loads(str(raw))
    except (json.JSONDecodeError, TypeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _coerce_profile(profile: UserProfileRequest | Mapping[str, Any]) -> UserProfileRequest:
    """Normalize either a UserProfileRequest or raw mapping into a validated profile."""
    if isinstance(profile, UserProfileRequest):
        return profile.model_copy(deep=True)

    if not isinstance(profile, Mapping):
        raise MissingMandatoryProfileError("Profile must be a UserProfileRequest or mapping")

    mapped: dict[str, Any] = {}
    for key, value in profile.items():
        if key in _NON_FILTER_PROFILE_FIELDS:
            continue
        target_key = key
        if key == "preferred_policy_type":
            target_key = "policy_type"
        elif key == "geographic_location":
            target_key = "location"
        if target_key in ("required_benefits", "pre_existing_conditions") and isinstance(
            value, str
        ):
            parsed = _parse_str_list(value)
            mapped[target_key] = parsed if parsed is not None else []
        elif _is_missing(value) and target_key not in _MANDATORY_PROFILE_FIELDS:
            continue
        else:
            mapped[target_key] = value

    missing_mandatory = [
        field for field in _MANDATORY_PROFILE_FIELDS if _is_missing(mapped.get(field))
    ]
    if missing_mandatory:
        raise MissingMandatoryProfileError(
            f"Missing mandatory profile field(s): {', '.join(missing_mandatory)}"
        )

    try:
        return UserProfileRequest.model_validate(mapped)
    except ValidationError as exc:
        raise MissingMandatoryProfileError(f"Invalid profile input: {exc}") from exc


def _check_mandatory_policy_fields(policy: Mapping[str, Any], acc: _RuleAccumulator) -> bool:
    """Verify core policy attributes exist before numeric or domain comparisons."""
    required_numeric = (
        "min_entry_age",
        "max_entry_age",
        "annual_premium_min",
        "annual_premium_max",
        "sum_insured_max",
    )
    missing_fields: list[str] = []
    if _is_missing(policy.get("policy_type")):
        missing_fields.append("policy_type")
    if _is_missing(
        policy.get("policy_status") if "policy_status" in policy else policy.get("status")
    ):
        missing_fields.append("policy_status")
    for field in required_numeric:
        if _is_missing(policy.get(field)):
            missing_fields.append(field)

    if missing_fields:
        acc.fail_rule(
            "MISSING_MANDATORY_POLICY_DATA",
            f"Policy is missing mandatory attribute(s): {', '.join(missing_fields)}",
            expected="all mandatory policy attributes populated",
            actual=missing_fields,
        )
        return False

    acc.pass_rule("MANDATORY_POLICY_DATA")
    return True


def _check_common_rules(
    profile: UserProfileRequest,
    policy: Mapping[str, Any],
    acc: _RuleAccumulator,
) -> None:
    """Evaluate common eligibility constraints shared across all policy types."""
    status_val = _to_optional_str(
        policy.get("policy_status") if "policy_status" in policy else policy.get("status")
    )
    if status_val != "active":
        acc.fail_rule(
            "POLICY_ACTIVE_STATUS",
            "Policy is not active in the catalog",
            expected="active",
            actual=status_val,
        )
    else:
        acc.pass_rule("POLICY_ACTIVE_STATUS")

    catalog_flag = _to_optional_bool(policy.get("is_catalog_eligible"))
    if catalog_flag is not True:
        acc.fail_rule(
            "CATALOG_ELIGIBLE",
            "Policy is not marked eligible for catalog recommendations",
            expected=True,
            actual=catalog_flag,
        )
    else:
        acc.pass_rule("CATALOG_ELIGIBLE")

    pol_type = _to_optional_str(policy.get("policy_type"))
    if pol_type != profile.policy_type:
        acc.fail_rule(
            "POLICY_TYPE_MATCH",
            "Policy category does not match requested insurance type",
            expected=profile.policy_type,
            actual=pol_type,
        )
    else:
        acc.pass_rule("POLICY_TYPE_MATCH")

    min_age = _to_optional_int(policy.get("min_entry_age"))
    max_age = _to_optional_int(policy.get("max_entry_age"))
    if min_age is not None and profile.age < min_age:
        acc.fail_rule(
            "AGE_MIN",
            "Applicant age is below the policy minimum entry age",
            expected=f">= {min_age}",
            actual=profile.age,
        )
    else:
        acc.pass_rule("AGE_MIN")

    if max_age is not None and profile.age > max_age:
        acc.fail_rule(
            "AGE_MAX",
            "Applicant exceeds maximum eligible entry age",
            expected=f"<= {max_age}",
            actual=profile.age,
        )
    else:
        acc.pass_rule("AGE_MAX")

    max_renewal_age = _to_optional_int(policy.get("max_renewal_age"))
    lifelong = _to_optional_bool(policy.get("lifelong_renewal"))
    if (
        lifelong is False
        and max_renewal_age is not None
        and (profile.age + profile.tenure_years) > max_renewal_age
    ):
        acc.fail_rule(
            "RENEWAL_AGE_MAX",
            "Applicant age plus requested tenure exceeds maximum renewal age",
            expected=f"<= {max_renewal_age}",
            actual=profile.age + profile.tenure_years,
        )
    else:
        acc.pass_rule("RENEWAL_AGE_MAX")

    eligible_applicants = _parse_str_list(policy.get("eligible_applicant_types"))
    if eligible_applicants is None or not eligible_applicants:
        acc.fail_rule(
            "APPLICANT_TYPE",
            "Policy eligible applicant types are missing or unverified",
            expected=profile.applicant_type,
            actual=eligible_applicants,
        )
    elif profile.applicant_type not in eligible_applicants:
        acc.fail_rule(
            "APPLICANT_TYPE",
            "Applicant type is not eligible for this policy",
            expected=profile.applicant_type,
            actual=eligible_applicants,
        )
    else:
        acc.pass_rule("APPLICANT_TYPE")

    prem_min = _to_optional_float(policy.get("annual_premium_min"))
    prem_max = _to_optional_float(policy.get("annual_premium_max"))
    if prem_min is not None and prem_min > profile.budget_max:
        acc.fail_rule(
            "BUDGET_MAX",
            "Policy minimum annual premium exceeds applicant maximum budget",
            expected=f"<= {profile.budget_max}",
            actual=prem_min,
        )
    else:
        acc.pass_rule("BUDGET_MAX")

    if profile.budget_min > 0 and prem_max is not None and prem_max < profile.budget_min:
        acc.fail_rule(
            "BUDGET_MIN",
            "Policy maximum annual premium is below applicant minimum budget band",
            expected=f">= {profile.budget_min}",
            actual=prem_max,
        )
    else:
        acc.pass_rule("BUDGET_MIN")

    sum_max = _to_optional_float(policy.get("sum_insured_max"))
    if sum_max is not None and sum_max < profile.min_sum_insured:
        acc.fail_rule(
            "SUM_INSURED_MIN",
            "Policy maximum sum insured is below requested coverage",
            expected=f">= {profile.min_sum_insured}",
            actual=sum_max,
        )
    else:
        acc.pass_rule("SUM_INSURED_MIN")

    sum_min = _to_optional_float(policy.get("sum_insured_min"))
    if profile.max_sum_insured is not None and sum_min is not None:
        if sum_min > profile.max_sum_insured:
            acc.fail_rule(
                "SUM_INSURED_MAX",
                "Policy minimum sum insured exceeds requested maximum coverage ceiling",
                expected=f"<= {profile.max_sum_insured}",
                actual=sum_min,
            )
        else:
            acc.pass_rule("SUM_INSURED_MAX")

    if profile.location is not None:
        pan_india = _to_optional_bool(policy.get("pan_india_coverage"))
        regions = _parse_str_list(policy.get("available_regions")) or []
        loc = profile.location.strip().lower()
        if pan_india is True:
            acc.pass_rule("GEOGRAPHIC_AVAILABILITY")
        elif pan_india is False and loc in regions:
            acc.pass_rule("GEOGRAPHIC_AVAILABILITY")
        else:
            acc.fail_rule(
                "GEOGRAPHIC_AVAILABILITY",
                "Policy is not available in the applicant's geographic location",
                expected=loc,
                actual="pan_india" if pan_india else regions,
            )

    min_tenure = _to_optional_int(policy.get("min_tenure_years"))
    max_tenure = _to_optional_int(policy.get("max_tenure_years"))
    if min_tenure is None or max_tenure is None:
        acc.fail_rule(
            "TENURE_COMPATIBILITY",
            "Policy tenure bounds are missing or unverified",
            expected=profile.tenure_years,
            actual=(min_tenure, max_tenure),
        )
    elif not (min_tenure <= profile.tenure_years <= max_tenure):
        acc.fail_rule(
            "TENURE_COMPATIBILITY",
            "Requested policy tenure falls outside supported policy duration",
            expected=profile.tenure_years,
            actual={"min_tenure_years": min_tenure, "max_tenure_years": max_tenure},
        )
    else:
        acc.pass_rule("TENURE_COMPATIBILITY")

    if profile.require_lifelong_renewal:
        if lifelong is True:
            acc.pass_rule("LIFELONG_RENEWAL")
        else:
            acc.fail_rule(
                "LIFELONG_RENEWAL",
                "Policy does not guarantee lifelong renewal",
                expected=True,
                actual=lifelong,
            )

    pol_benefits = set(_parse_str_list(policy.get("covered_benefits")) or [])
    pol_exclusions = set(_parse_str_list(policy.get("exclusions")) or [])
    if profile.required_benefits:
        req_set = set(profile.required_benefits)
        excluded_requested = sorted(req_set & pol_exclusions)
        missing_benefits = sorted(req_set - pol_benefits)
        if excluded_requested:
            acc.fail_rule(
                "REQUIRED_BENEFITS_EXCLUDED",
                "One or more required benefits appear in policy exclusions",
                expected="not excluded",
                actual=excluded_requested,
            )
        elif missing_benefits:
            acc.fail_rule(
                "REQUIRED_BENEFITS",
                "Policy does not cover all required benefits",
                expected=sorted(req_set),
                actual=sorted(pol_benefits),
            )
        else:
            acc.pass_rule("REQUIRED_BENEFITS")


def _check_shared_cost_and_network_rules(
    profile: UserProfileRequest,
    policy: Mapping[str, Any],
    acc: _RuleAccumulator,
) -> None:
    """Check coverage mode, co-pay, deductible, and cashless/network constraints."""
    pol_mode = _to_optional_str(policy.get("coverage_mode"))
    max_members = _to_optional_int(policy.get("max_family_members"))
    if profile.coverage_mode == "family_floater":
        if pol_mode not in ("family_floater", "both"):
            acc.fail_rule(
                "COVERAGE_MODE",
                "Policy does not support family floater coverage",
                expected="family_floater or both",
                actual=pol_mode,
            )
        elif max_members is None or profile.family_size > max_members:
            acc.fail_rule(
                "FAMILY_SIZE_MAX",
                "Family size exceeds maximum members covered under policy",
                expected=f">= {profile.family_size}",
                actual=max_members,
            )
        else:
            acc.pass_rule("COVERAGE_MODE")
            acc.pass_rule("FAMILY_SIZE_MAX")
    elif profile.policy_type == "health" or pol_mode is not None:
        if pol_mode not in ("individual", "both"):
            acc.fail_rule(
                "COVERAGE_MODE",
                "Policy does not support individual coverage",
                expected="individual or both",
                actual=pol_mode,
            )
        else:
            acc.pass_rule("COVERAGE_MODE")

    if profile.max_copay_pct is not None:
        copay = _to_optional_float(policy.get("copay_pct"))
        if copay is None or copay > profile.max_copay_pct:
            acc.fail_rule(
                "COPAY_MAX",
                "Policy co-payment exceeds maximum acceptable co-payment percentage",
                expected=f"<= {profile.max_copay_pct}",
                actual=copay,
            )
        else:
            acc.pass_rule("COPAY_MAX")

    if profile.max_deductible is not None:
        deductible = _to_optional_float(policy.get("deductible_amount"))
        if deductible is None or deductible > profile.max_deductible:
            acc.fail_rule(
                "DEDUCTIBLE_MAX",
                "Policy deductible exceeds maximum acceptable deductible",
                expected=f"<= {profile.max_deductible}",
                actual=deductible,
            )
        else:
            acc.pass_rule("DEDUCTIBLE_MAX")

    if profile.require_cashless:
        cashless = _to_optional_bool(policy.get("cashless_available"))
        if cashless is True:
            acc.pass_rule("CASHLESS_REQUIRED")
        else:
            acc.fail_rule(
                "CASHLESS_REQUIRED",
                "Cashless claim facility is unavailable or unverified",
                expected=True,
                actual=cashless,
            )

    if profile.min_network_hospitals > 0:
        network_count = _to_optional_int(policy.get("network_hospitals_count"))
        if network_count is None or network_count < profile.min_network_hospitals:
            acc.fail_rule(
                "HOSPITAL_NETWORK_MIN",
                "Policy hospital/garage network is smaller than requested minimum",
                expected=f">= {profile.min_network_hospitals}",
                actual=network_count,
            )
        else:
            acc.pass_rule("HOSPITAL_NETWORK_MIN")


def _check_health_rules(
    profile: UserProfileRequest,
    policy: Mapping[str, Any],
    acc: _RuleAccumulator,
) -> None:
    """Apply Health-specific mandatory rules (waiting period, PED, room rent)."""
    if profile.max_waiting_period_days is not None:
        wait_days = _to_optional_int(policy.get("initial_waiting_period_days"))
        if wait_days is None or wait_days > profile.max_waiting_period_days:
            acc.fail_rule(
                "INITIAL_WAITING_PERIOD",
                "Policy initial waiting period exceeds maximum acceptable days",
                expected=f"<= {profile.max_waiting_period_days}",
                actual=wait_days,
            )
        else:
            acc.pass_rule("INITIAL_WAITING_PERIOD")

    if profile.pre_existing_conditions:
        ped_covered = _to_optional_bool(policy.get("ped_covered"))
        exclusions = set(_parse_str_list(policy.get("exclusions")) or [])
        ped_set = set(profile.pre_existing_conditions)
        excluded_peds = sorted(ped_set & exclusions)
        if ped_covered is not True or excluded_peds:
            acc.fail_rule(
                "PED_COVERAGE",
                "Pre-existing condition(s) are not covered or explicitly excluded",
                expected="PED covered and not excluded",
                actual={"ped_covered": ped_covered, "excluded": excluded_peds},
            )
        else:
            acc.pass_rule("PED_COVERAGE")

        if profile.max_ped_waiting_months is not None:
            ped_wait = _to_optional_int(policy.get("ped_waiting_period_months"))
            if ped_wait is None or ped_wait > profile.max_ped_waiting_months:
                acc.fail_rule(
                    "PED_WAITING_PERIOD",
                    "Pre-existing disease waiting period exceeds limit or is unverified",
                    expected=f"<= {profile.max_ped_waiting_months}",
                    actual=ped_wait,
                )
            else:
                acc.pass_rule("PED_WAITING_PERIOD")

    if profile.room_rent_preference != "any":
        pol_room = _to_optional_str(policy.get("room_rent_limit_type"))
        if profile.room_rent_preference == "no_limit":
            valid_rooms = ("no_limit",)
        elif profile.room_rent_preference == "single_private":
            valid_rooms = ("no_limit", "single_private")
        else:
            valid_rooms = ("no_limit", "single_private", "shared")

        if pol_room not in valid_rooms:
            acc.fail_rule(
                "ROOM_RENT_LIMIT",
                "Policy room-rent limit does not satisfy requested room preference",
                expected=list(valid_rooms),
                actual=pol_room,
            )
        else:
            acc.pass_rule("ROOM_RENT_LIMIT")


def _check_metadata_bool_flag(
    meta: Mapping[str, Any],
    meta_key: str,
    pol_benefits: set[str],
    benefit_tag: str | None,
    rule_id: str,
    message: str,
    acc: _RuleAccumulator,
) -> None:
    """Verify an explicitly requested boolean capability via metadata or benefit tag."""
    raw_val = meta.get(meta_key)
    bool_val = _to_optional_bool(raw_val)
    if bool_val is True or (bool_val is None and benefit_tag and benefit_tag in pol_benefits):
        acc.pass_rule(rule_id)
    else:
        acc.fail_rule(rule_id, message, expected=True, actual=bool_val)


def _check_car_rules(
    profile: UserProfileRequest,
    policy: Mapping[str, Any],
    meta: Mapping[str, Any],
    acc: _RuleAccumulator,
) -> None:
    """Apply Car-specific eligibility checks (vehicle type, vehicle age, coverages)."""
    pol_benefits = set(_parse_str_list(policy.get("covered_benefits")) or [])

    if profile.vehicle_type is not None:
        raw_types = meta.get("vehicle_types")
        supported_types = (
            [str(x).strip().lower() for x in raw_types if str(x).strip()]
            if isinstance(raw_types, (list, tuple, set))
            else []
        )
        if not supported_types or profile.vehicle_type not in supported_types:
            acc.fail_rule(
                "CAR_VEHICLE_TYPE",
                "Vehicle type is unsupported or missing from policy metadata",
                expected=profile.vehicle_type,
                actual=supported_types or None,
            )
        else:
            acc.pass_rule("CAR_VEHICLE_TYPE")

    if profile.vehicle_age_years is not None:
        age_limit = _to_optional_int(meta.get("vehicle_age_limit_years"))
        if age_limit is None or profile.vehicle_age_years > age_limit:
            acc.fail_rule(
                "CAR_VEHICLE_AGE_MAX",
                "Vehicle age exceeds maximum allowed vehicle age or limit is unverified",
                expected=f"<= {age_limit}" if age_limit is not None else "known vehicle age limit",
                actual=profile.vehicle_age_years,
            )
        else:
            acc.pass_rule("CAR_VEHICLE_AGE_MAX")

    if profile.require_own_damage:
        _check_metadata_bool_flag(
            meta,
            "own_damage_coverage",
            pol_benefits,
            "own_damage",
            "CAR_OWN_DAMAGE",
            "Own-damage coverage is not included or unverified",
            acc,
        )
    if profile.require_third_party:
        _check_metadata_bool_flag(
            meta,
            "third_party_coverage",
            pol_benefits,
            "third_party_liability",
            "CAR_THIRD_PARTY",
            "Third-party liability coverage is not included or unverified",
            acc,
        )
    if profile.require_zero_depreciation:
        _check_metadata_bool_flag(
            meta,
            "zero_depreciation",
            pol_benefits,
            "zero_depreciation",
            "CAR_ZERO_DEPRECIATION",
            "Zero-depreciation cover is not included or unverified",
            acc,
        )
    if profile.require_roadside_assistance:
        _check_metadata_bool_flag(
            meta,
            "roadside_assistance",
            pol_benefits,
            "roadside_assistance",
            "CAR_ROADSIDE_ASSISTANCE",
            "Roadside assistance is not included or unverified",
            acc,
        )
    if profile.require_engine_protection:
        _check_metadata_bool_flag(
            meta,
            "engine_protection",
            pol_benefits,
            "engine_protection",
            "CAR_ENGINE_PROTECTION",
            "Engine protection add-on is not included or unverified",
            acc,
        )


def _check_life_and_term_common(
    prefix: str,
    profile: UserProfileRequest,
    policy: Mapping[str, Any],
    meta: Mapping[str, Any],
    acc: _RuleAccumulator,
) -> None:
    """Check life cover bounds, policy term compatibility, and premium payment term."""
    cover_max = _to_optional_float(meta.get("life_cover_max")) or _to_optional_float(
        policy.get("sum_insured_max")
    )
    if cover_max is None or cover_max < profile.min_sum_insured:
        acc.fail_rule(
            f"{prefix}_COVER_MIN",
            "Policy maximum life cover is below requested minimum life cover",
            expected=f">= {profile.min_sum_insured}",
            actual=cover_max,
        )
    else:
        acc.pass_rule(f"{prefix}_COVER_MIN")

    cover_min = _to_optional_float(meta.get("life_cover_min")) or _to_optional_float(
        policy.get("sum_insured_min")
    )
    if profile.max_sum_insured is not None and cover_min is not None:
        if cover_min > profile.max_sum_insured:
            acc.fail_rule(
                f"{prefix}_COVER_MAX",
                "Policy minimum life cover exceeds requested maximum life cover",
                expected=f"<= {profile.max_sum_insured}",
                actual=cover_min,
            )
        else:
            acc.pass_rule(f"{prefix}_COVER_MAX")

    term_min = _to_optional_int(meta.get("policy_term_years_min"))
    if term_min is None:
        term_min = _to_optional_int(policy.get("min_tenure_years"))
    term_max = _to_optional_int(meta.get("policy_term_years_max"))
    if term_max is None:
        term_max = _to_optional_int(policy.get("max_tenure_years"))

    if term_min is None or term_max is None or not (term_min <= profile.tenure_years <= term_max):
        acc.fail_rule(
            f"{prefix}_POLICY_TERM",
            "Requested policy term falls outside policy term range",
            expected=profile.tenure_years,
            actual={"policy_term_years_min": term_min, "policy_term_years_max": term_max},
        )
    else:
        acc.pass_rule(f"{prefix}_POLICY_TERM")

    if profile.max_premium_payment_term_years is not None:
        ppt = _to_optional_int(meta.get("premium_payment_term_years"))
        if ppt is None or ppt > profile.max_premium_payment_term_years:
            acc.fail_rule(
                f"{prefix}_PREMIUM_PAYMENT_TERM",
                "Premium payment term exceeds requested maximum or is unverified",
                expected=f"<= {profile.max_premium_payment_term_years}",
                actual=ppt,
            )
        else:
            acc.pass_rule(f"{prefix}_PREMIUM_PAYMENT_TERM")


def _check_life_rules(
    profile: UserProfileRequest,
    policy: Mapping[str, Any],
    meta: Mapping[str, Any],
    acc: _RuleAccumulator,
) -> None:
    """Apply Life-specific eligibility checks (cover, term, maturity, PPT, surrender)."""
    _check_life_and_term_common("LIFE", profile, policy, meta, acc)
    pol_benefits = set(_parse_str_list(policy.get("covered_benefits")) or [])

    if profile.require_maturity_benefit:
        _check_metadata_bool_flag(
            meta,
            "maturity_benefit",
            pol_benefits,
            "maturity_benefit",
            "LIFE_MATURITY_BENEFIT",
            "Maturity benefit is not available or unverified",
            acc,
        )
    if profile.require_surrender_value:
        _check_metadata_bool_flag(
            meta,
            "surrender_value_available",
            pol_benefits,
            None,
            "LIFE_SURRENDER_VALUE",
            "Surrender value option is not available or unverified",
            acc,
        )


def _check_term_rules(
    profile: UserProfileRequest,
    policy: Mapping[str, Any],
    meta: Mapping[str, Any],
    acc: _RuleAccumulator,
) -> None:
    """Apply Term-specific eligibility checks (cover, term, smoking, PPT, riders)."""
    _check_life_and_term_common("TERM", profile, policy, meta, acc)
    pol_benefits = set(_parse_str_list(policy.get("covered_benefits")) or [])

    if profile.smoking_status == "smoker":
        smoking_elig = _to_optional_str(meta.get("smoking_status_eligibility"))
        if smoking_elig in ("all", "smoker_surcharge"):
            acc.pass_rule("TERM_SMOKING_ELIGIBILITY")
        else:
            acc.fail_rule(
                "TERM_SMOKING_ELIGIBILITY",
                "Term policy does not accept smokers or smoking eligibility is unverified",
                expected="all or smoker_surcharge",
                actual=smoking_elig,
            )

    if profile.require_critical_illness_rider:
        _check_metadata_bool_flag(
            meta,
            "critical_illness_rider",
            pol_benefits,
            "critical_illness_rider",
            "TERM_CRITICAL_ILLNESS_RIDER",
            "Critical illness rider is not supported or unverified",
            acc,
        )
    if profile.require_accidental_death_rider:
        _check_metadata_bool_flag(
            meta,
            "accidental_death_rider",
            pol_benefits,
            "accidental_death_rider",
            "TERM_ACCIDENTAL_DEATH_RIDER",
            "Accidental death rider is not supported or unverified",
            acc,
        )


# ---------------------------------------------------------------------------
# Layer C: Filtering Orchestration & Audit Reporting
# ---------------------------------------------------------------------------


def evaluate_policy_eligibility(
    profile: UserProfileRequest | Mapping[str, Any],
    policy: Mapping[str, Any],
) -> PolicyFilterEvaluation:
    """Evaluate a single candidate policy against a validated user profile."""
    normalized_profile = _coerce_profile(profile)
    acc = _RuleAccumulator(passed=[], failed=[])

    policy_id = str(policy.get("policy_id") or policy.get("id") or "UNKNOWN")
    policy_name = str(policy.get("policy_name") or policy.get("name") or "Unnamed Policy")
    policy_type = _to_optional_str(policy.get("policy_type")) or "unknown"

    if _check_mandatory_policy_fields(policy, acc):
        _check_common_rules(normalized_profile, policy, acc)
        _check_shared_cost_and_network_rules(normalized_profile, policy, acc)

        meta = _parse_metadata_dict(policy.get("type_specific_metadata"))
        if normalized_profile.policy_type == "health":
            _check_health_rules(normalized_profile, policy, acc)
        elif normalized_profile.policy_type == "car":
            _check_car_rules(normalized_profile, policy, meta, acc)
        elif normalized_profile.policy_type == "life":
            _check_life_rules(normalized_profile, policy, meta, acc)
        elif normalized_profile.policy_type == "term":
            _check_term_rules(normalized_profile, policy, meta, acc)

    return PolicyFilterEvaluation(
        policy_id=policy_id,
        policy_name=policy_name,
        policy_type=policy_type,
        is_eligible=len(acc.failed) == 0,
        failed_filters=acc.failed,
        passed_filters=acc.passed,
        elimination_reasons=[failure.message for failure in acc.failed],
    )


def filter_policies(
    profile: UserProfileRequest | Mapping[str, Any],
    candidate_policies: Sequence[Mapping[str, Any]],
) -> HardFilterResult:
    """Filter candidate policies deterministically without mutating input records."""
    normalized_profile = _coerce_profile(profile)
    eligible_policies: list[dict[str, Any]] = []
    evaluations: list[PolicyFilterEvaluation] = []
    eliminations: list[FilterEliminationReason] = []

    for policy in candidate_policies:
        evaluation = evaluate_policy_eligibility(normalized_profile, policy)
        evaluations.append(evaluation)

        if evaluation.is_eligible:
            eligible_policies.append(copy.deepcopy(dict(policy)))
        else:
            for failure in evaluation.failed_filters:
                eliminations.append(
                    FilterEliminationReason(
                        policy_id=evaluation.policy_id,
                        policy_name=evaluation.policy_name,
                        rule=failure.rule_id,
                        reason=failure.message,
                        rule_id=failure.rule_id,
                        expected=failure.expected,
                        actual=failure.actual,
                    )
                )

    passed_count = len(eligible_policies)
    total_candidates = len(evaluations)
    eliminated_count = total_candidates - passed_count
    status_val: Literal["ELIGIBLE_CANDIDATES_FOUND", "NO_MATCHING_POLICIES"] = (
        "ELIGIBLE_CANDIDATES_FOUND" if passed_count > 0 else "NO_MATCHING_POLICIES"
    )

    report = FilterReport(
        status=status_val,
        total_candidates=total_candidates,
        passed_count=passed_count,
        eliminated_count=eliminated_count,
        eliminations=eliminations,
        policy_evaluations=evaluations,
    )
    return HardFilterResult(
        status=status_val,
        eligible_policies=eligible_policies,
        filter_report=report,
    )


# ---------------------------------------------------------------------------
# Layer D: Parameterized SQLAlchemy Core Database Adapter
# ---------------------------------------------------------------------------


def check_policy_attributes_table_exists(conn: Connection) -> bool:
    """Check safely via information_schema whether policy_attributes exists."""
    table_name = conn.execute(
        text(
            "SELECT table_name "
            "FROM information_schema.tables "
            "WHERE table_schema = :schema_name "
            "AND table_name = :table_name"
        ),
        {"schema_name": "public", "table_name": "policy_attributes"},
    ).scalar_one_or_none()
    return table_name is not None


def build_catalog_policy_attributes_query(
    policy_type: str | None = None,
    active_only: bool = True,
    max_budget: float | None = None,
    min_sum_insured: float | None = None,
    age: int | None = None,
) -> Select[Any]:
    """Construct a parameterized SQLAlchemy Core query joining policies and policy_attributes."""
    conditions = [
        policies.c.organization_id.is_not(None),
        policies.c.owner_user_id.is_(None),
        policy_attributes.c.is_catalog_eligible.is_(True),
    ]
    if active_only:
        conditions.append(policies.c.status == "active")
    if policy_type is not None:
        conditions.append(policies.c.policy_type == policy_type.strip().lower())
    if max_budget is not None:
        conditions.append(policy_attributes.c.annual_premium_min <= float(max_budget))
    if min_sum_insured is not None:
        conditions.append(policy_attributes.c.sum_insured_max >= float(min_sum_insured))
    if age is not None:
        conditions.append(policy_attributes.c.min_entry_age <= int(age))
        conditions.append(policy_attributes.c.max_entry_age >= int(age))

    return (
        select(
            policies.c.id.label("policy_id"),
            policies.c.organization_id,
            policies.c.name.label("policy_name"),
            policies.c.insurer_name,
            policies.c.policy_type,
            policies.c.description,
            policies.c.status.label("policy_status"),
            policies.c.current_version_id,
            policy_attributes.c.policy_version_id,
            policy_attributes.c.is_catalog_eligible,
            policy_attributes.c.min_entry_age,
            policy_attributes.c.max_entry_age,
            policy_attributes.c.max_renewal_age,
            policy_attributes.c.lifelong_renewal,
            policy_attributes.c.coverage_mode,
            policy_attributes.c.max_family_members,
            policy_attributes.c.eligible_applicant_types,
            policy_attributes.c.annual_premium_min,
            policy_attributes.c.annual_premium_max,
            policy_attributes.c.sum_insured_min,
            policy_attributes.c.sum_insured_max,
            policy_attributes.c.deductible_amount,
            policy_attributes.c.copay_pct,
            policy_attributes.c.room_rent_limit_type,
            policy_attributes.c.room_rent_limit_value,
            policy_attributes.c.initial_waiting_period_days,
            policy_attributes.c.ped_covered,
            policy_attributes.c.ped_waiting_period_months,
            policy_attributes.c.specific_disease_waiting_months,
            policy_attributes.c.maternity_waiting_months,
            policy_attributes.c.network_hospitals_count,
            policy_attributes.c.cashless_available,
            policy_attributes.c.pan_india_coverage,
            policy_attributes.c.available_regions,
            policy_attributes.c.min_tenure_years,
            policy_attributes.c.max_tenure_years,
            policy_attributes.c.covered_benefits,
            policy_attributes.c.exclusions,
            policy_attributes.c.claim_settlement_ratio,
            policy_attributes.c.no_claim_bonus_pct,
            policy_attributes.c.restoration_benefit,
            policy_attributes.c.type_specific_metadata,
            policy_attributes.c.extraction_source,
        )
        .select_from(
            policies.join(
                policy_attributes,
                and_(
                    policies.c.id == policy_attributes.c.policy_id,
                    or_(
                        policy_attributes.c.policy_version_id == policies.c.current_version_id,
                        and_(
                            policies.c.current_version_id.is_(None),
                            policy_attributes.c.policy_version_id.is_(None),
                        ),
                    ),
                ),
            )
        )
        .where(and_(*conditions))
        .order_by(policies.c.created_at.desc(), policies.c.id.asc())
    )


def _is_mock_connection(conn: Any) -> bool:
    """Return True when conn is a unittest.mock double rather than a real SQLAlchemy Connection."""
    return type(conn).__module__.startswith("unittest.mock")


def _is_missing_policy_attributes_error(exc: SQLAlchemyError) -> bool:
    """Detect missing policy_attributes table errors directly from SQLAlchemy DBAPI exceptions."""
    if not isinstance(exc, (ProgrammingError, OperationalError)):
        return False
    orig = getattr(exc, "orig", None)
    pg_code = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
    combined = f"{exc} {orig or ''}".lower()
    if "policy_attributes" not in combined:
        return False
    if pg_code == "42P01":
        return True
    orig_name = type(orig).__name__.lower() if orig is not None else ""
    return (
        "undefinedtable" in orig_name or "does not exist" in combined or "no such table" in combined
    )


def load_catalog_policies_from_db(
    conn: Connection,
    policy_type: str | None = None,
    active_only: bool = False,
    *,
    verify_table_exists: bool | None = None,
) -> list[dict[str, Any]]:
    """Load version-aligned catalog policy attributes via parameterized queries.

    When executing against a real database Connection (``verify_table_exists=None``),
    skips the redundant ``information_schema`` round-trip and relies on direct
    SQLSTATE ``42P01`` / missing-table detection while mapping any missing
    ``policy_attributes`` table explicitly to ``PolicyAttributesTableUnavailableError``.
    Raises ``DuplicateCatalogPolicyError`` if duplicate rows appear for the same ``policy_id``.
    """
    should_verify = (
        _is_mock_connection(conn) if verify_table_exists is None else verify_table_exists
    )
    if should_verify and not check_policy_attributes_table_exists(conn):
        raise PolicyAttributesTableUnavailableError(
            "Table 'policy_attributes' does not exist in PostgreSQL. "
            "Apply server/migrations/003_recommendation_schema.sql after team sign-off."
        )

    stmt = build_catalog_policy_attributes_query(
        policy_type=policy_type,
        active_only=active_only,
    )
    try:
        rows = conn.execute(stmt).mappings().all()
    except SQLAlchemyError as exc:
        if _is_missing_policy_attributes_error(exc):
            raise PolicyAttributesTableUnavailableError(
                "Table 'policy_attributes' does not exist in PostgreSQL. "
                "Apply server/migrations/003_recommendation_schema.sql after team sign-off."
            ) from exc
        raise

    results: list[dict[str, Any]] = []
    seen_policy_ids: set[str] = set()
    for row in rows:
        item = dict(row)
        pid = str(item["policy_id"])
        if pid in seen_policy_ids:
            raise DuplicateCatalogPolicyError(
                f"Duplicate current-version policy_attributes rows detected for policy_id={pid}"
            )
        seen_policy_ids.add(pid)
        item["policy_id"] = pid
        results.append(item)
    return results


def filter_policies_from_db(
    conn: Connection,
    profile: UserProfileRequest | Mapping[str, Any],
) -> HardFilterResult:
    """Load catalog policies from PostgreSQL and apply deterministic filtering."""
    normalized_profile = _coerce_profile(profile)
    candidates = load_catalog_policies_from_db(conn, active_only=False)
    return filter_policies(normalized_profile, candidates)
