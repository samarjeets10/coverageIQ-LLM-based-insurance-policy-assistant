"""Pydantic v2 request and response schemas for Policy Recommendation (Module 6A)."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PolicyType = Literal["health", "car", "life", "term"]
CoverageMode = Literal["individual", "family_floater"]
RoomRentPreference = Literal["any", "no_limit", "single_private", "shared"]
SmokingStatus = Literal["non_smoker", "smoker"]
RiskPreferenceOption = Literal["low_premium", "balanced", "comprehensive_coverage"]
RankingModelOption = Literal["auto", "weighted", "lightgbm", "xgboost"]


MAX_TAG_LIST_ITEMS = 50
MAX_TAG_ITEM_LENGTH = 100


def _normalize_string_list(values: list[str]) -> list[str]:
    """Normalize string tags by stripping whitespace and deduplicating case-insensitively."""
    seen: set[str] = set()
    normalized: list[str] = []
    for item in values:
        cleaned = item.strip().lower()
        if len(cleaned) > MAX_TAG_ITEM_LENGTH:
            raise ValueError(f"Tag entry length cannot exceed {MAX_TAG_ITEM_LENGTH} characters.")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            normalized.append(cleaned)
    return normalized


class UserProfileRequest(BaseModel):
    """User profile and eligibility requirements (FR-REC-01 & Architecture §3.4)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "policy_type": "health",
                    "age": 32,
                    "budget_min": 8000,
                    "budget_max": 25000,
                    "applicant_type": "salaried",
                    "coverage_mode": "family_floater",
                    "family_size": 3,
                    "min_sum_insured": 500000,
                    "required_benefits": ["cashless_hospitalization", "daycare_procedures"],
                    "pre_existing_conditions": [],
                    "room_rent_preference": "single_private",
                    "require_cashless": True,
                    "min_network_hospitals": 3000,
                    "tenure_years": 1,
                    "annual_income_inr": 1500000,
                    "risk_preference": "balanced",
                }
            ]
        },
    )

    policy_type: PolicyType
    age: int = Field(ge=0, le=120)
    budget_min: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    budget_max: float = Field(gt=0.0, allow_inf_nan=False)
    applicant_type: str = Field(default="salaried", min_length=1, max_length=50)
    coverage_mode: CoverageMode = Field(default="individual")
    family_size: int = Field(default=1, ge=1, le=15)
    min_sum_insured: float = Field(gt=0.0, allow_inf_nan=False)
    max_sum_insured: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    required_benefits: list[str] = Field(default_factory=list, max_length=MAX_TAG_LIST_ITEMS)
    pre_existing_conditions: list[str] = Field(default_factory=list, max_length=MAX_TAG_LIST_ITEMS)
    max_ped_waiting_months: int | None = Field(default=None, ge=0, le=120)
    max_waiting_period_days: int | None = Field(default=None, ge=0, le=365)
    max_copay_pct: float | None = Field(default=None, ge=0.0, le=100.0, allow_inf_nan=False)
    max_deductible: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    room_rent_preference: RoomRentPreference = Field(default="any")
    require_cashless: bool = Field(default=False)
    min_network_hospitals: int = Field(default=0, ge=0, le=1_000_000)
    location: str | None = Field(default=None, max_length=100)
    tenure_years: int = Field(default=1, ge=1, le=50)
    require_lifelong_renewal: bool = Field(default=False)

    # Optional personalization fields used in ML ranking (Stage 7 backward-compatible addition)
    annual_income_inr: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    risk_preference: RiskPreferenceOption | None = Field(default=None)

    # Optional category-specific requirements (Car / Life / Term)
    smoking_status: SmokingStatus | None = Field(default=None)
    vehicle_type: str | None = Field(default=None, max_length=50)
    vehicle_age_years: int | None = Field(default=None, ge=0, le=50)
    require_own_damage: bool = Field(default=False)
    require_third_party: bool = Field(default=False)
    require_zero_depreciation: bool = Field(default=False)
    require_roadside_assistance: bool = Field(default=False)
    require_engine_protection: bool = Field(default=False)
    require_maturity_benefit: bool = Field(default=False)
    require_surrender_value: bool = Field(default=False)
    max_premium_payment_term_years: int | None = Field(default=None, ge=1, le=60)
    require_critical_illness_rider: bool = Field(default=False)
    require_accidental_death_rider: bool = Field(default=False)

    @field_validator(
        "policy_type",
        "applicant_type",
        "smoking_status",
        "vehicle_type",
        "risk_preference",
        "location",
        mode="before",
    )
    @classmethod
    def _normalize_lowercase_str(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip().lower()
            return stripped or None
        return value

    @field_validator("required_benefits", "pre_existing_conditions", mode="before")
    @classmethod
    def _enforce_raw_tag_list_limit(cls, value: Any) -> Any:
        if isinstance(value, list | tuple | set) and len(value) > MAX_TAG_LIST_ITEMS:
            raise ValueError(f"List length exceeds maximum allowed {MAX_TAG_LIST_ITEMS} items.")
        return value

    @field_validator("required_benefits", "pre_existing_conditions")
    @classmethod
    def _clean_tag_lists(cls, value: list[str]) -> list[str]:
        return _normalize_string_list(value)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> "UserProfileRequest":
        if self.budget_min > self.budget_max:
            raise ValueError("budget_min cannot exceed budget_max")
        if self.max_sum_insured is not None and self.min_sum_insured > self.max_sum_insured:
            raise ValueError("min_sum_insured cannot exceed max_sum_insured")
        if self.coverage_mode == "individual" and self.family_size > 1:
            raise ValueError("family_size > 1 requires coverage_mode='family_floater'")
        return self


class TrustedScope(BaseModel):
    """Trusted identity scope forwarded from the Node.js M1 gateway (Architecture §2A).

    Security boundary:
    - The Node.js M1 authentication gateway owns user JWT verification and RBAC role
      attestation; Python NEVER treats a raw browser/client-supplied scope as authentic
      without verifying ``X-Service-Token`` in non-development environments.
    - Private user-uploaded policies (``owner_user_id IS NOT NULL``) remain strictly
      excluded by M5 recommendation queries regardless of scope.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str | None = Field(default=None, min_length=1, max_length=128)
    role: Literal["user", "organization", "admin"] = "user"
    owner_scope: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("user_id", "owner_scope", mode="before")
    @classmethod
    def _strip_optional_scope_id(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("role", mode="before")
    @classmethod
    def _normalize_scope_role(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class RecommendRequest(BaseModel):
    """Unified recommendation payload supporting nested or flat profile payloads."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "profile": {
                        "policy_type": "health",
                        "age": 32,
                        "budget_max": 25000,
                        "min_sum_insured": 500000,
                        "coverage_mode": "individual",
                        "risk_preference": "balanced",
                    },
                    "top_n": 5,
                    "ranking_model": "auto",
                }
            ]
        },
    )

    profile: UserProfileRequest
    scope: TrustedScope | None = None
    top_n: int = Field(default=5, ge=1, le=20)
    ranking_model: RankingModelOption = Field(default="auto")

    @model_validator(mode="before")
    @classmethod
    def _support_flat_profile(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "profile" in data:
            return data
        envelope_keys = {"scope", "top_n", "ranking_model"}
        envelope = {k: data[k] for k in envelope_keys if k in data}
        profile_data = {k: v for k, v in data.items() if k not in envelope_keys}
        return {**envelope, "profile": profile_data}


class FilterFailureDetail(BaseModel):
    """Structured explanation for a single failed hard-filter rule."""

    rule_id: str
    message: str
    expected: Any = None
    actual: Any = None


class PolicyFilterEvaluation(BaseModel):
    """Complete deterministic hard-filter evaluation for one candidate policy."""

    policy_id: str
    policy_name: str
    policy_type: str
    is_eligible: bool
    failed_filters: list[FilterFailureDetail] = Field(default_factory=list)
    passed_filters: list[str] = Field(default_factory=list)
    elimination_reasons: list[str] = Field(default_factory=list)


class FilterEliminationReason(BaseModel):
    """Audit record explaining why a candidate policy failed deterministic filtering."""

    policy_id: str
    policy_name: str
    rule: str
    reason: str
    rule_id: str | None = None
    expected: Any = None
    actual: Any = None


class FilterReport(BaseModel):
    """Summary of deterministic hard-filtering results (FR-REC-05)."""

    status: Literal["ELIGIBLE_CANDIDATES_FOUND", "NO_MATCHING_POLICIES"] = "NO_MATCHING_POLICIES"
    total_candidates: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    eliminated_count: int = Field(ge=0)
    eliminations: list[FilterEliminationReason] = Field(default_factory=list)
    policy_evaluations: list[PolicyFilterEvaluation] = Field(default_factory=list)


class RecommendedPolicy(BaseModel):
    """Single ranked policy entry in Top-N recommendations."""

    policy_id: str
    policy_version_id: str | None = None
    policy_name: str
    insurer_name: str | None = None
    policy_type: str
    rank: int = Field(ge=1)
    score: float
    model_used: str
    annual_premium_min: float
    annual_premium_max: float
    sum_insured_max: float
    key_highlights: list[str] = Field(default_factory=list)
    feature_breakdown: dict[str, float] = Field(default_factory=dict)
    citations: list[dict[str, Any]] = Field(default_factory=list)


class ProfileCompleteness(BaseModel):
    """Explicit completeness & assumption audit for optional personalization fields."""

    is_complete: bool
    missing_fields: list[str] = Field(default_factory=list)
    applied_assumptions: dict[str, Any] = Field(default_factory=dict)
    income_distribution_mismatch_risk: bool = Field(default=False)


class RecommendResponse(BaseModel):
    """Structured recommendation response aligned with Architecture §3.10."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "request_id": "req-0001",
                    "status": "RECOMMENDATIONS_READY",
                    "recommendations": [
                        {
                            "policy_id": "11111111-1111-1111-1111-111111111111",
                            "policy_name": "ArogyaCare Prime Health",
                            "insurer_name": "ArogyaShield General Insurance",
                            "policy_type": "health",
                            "rank": 1,
                            "score": 2.418,
                            "model_used": "coverageiq_lightgbm_ranker",
                            "annual_premium_min": 12000.0,
                            "annual_premium_max": 18000.0,
                            "sum_insured_max": 1500000.0,
                            "key_highlights": ["Cashless network: 8,500 hospitals/garages"],
                            "feature_breakdown": {"affordability_score": 0.52},
                            "citations": [],
                        }
                    ],
                    "filter_report": {
                        "status": "ELIGIBLE_CANDIDATES_FOUND",
                        "total_candidates": 4,
                        "passed_count": 3,
                        "eliminated_count": 1,
                        "eliminations": [],
                        "policy_evaluations": [],
                    },
                    "model_used": "coverageiq_lightgbm_ranker",
                    "model_version": "0.1.0",
                    "feature_schema_version": "1.0.0",
                    "warnings": [],
                    "profile_completeness": {
                        "is_complete": True,
                        "missing_fields": [],
                        "applied_assumptions": {},
                        "income_distribution_mismatch_risk": False,
                    },
                }
            ]
        }
    )

    request_id: str | None = None
    status: Literal["RECOMMENDATIONS_READY", "NO_MATCHING_POLICIES"]
    recommendations: list[RecommendedPolicy] = Field(default_factory=list)
    filter_report: FilterReport
    model_used: str
    model_version: str | None = None
    feature_schema_version: str | None = None
    warnings: list[str] = Field(default_factory=list)
    profile_completeness: ProfileCompleteness | None = None


class ErrorDetailResponse(BaseModel):
    """Standard FastAPI error payload schema for documented HTTP error responses."""

    detail: str


class HealthResponse(BaseModel):
    """Operational readiness & PostgreSQL schema status response for /health endpoints."""

    success: bool
    status: Literal["healthy", "unhealthy"]
    services: dict[str, str]
    database: str | None = None
    readiness: dict[str, Any] | None = None
    model_ready: bool
    model_warmed_up: bool


class AIHealthResponse(HealthResponse):
    """Unified /ai/health response combining M3 Q&A telemetry and M5 readiness."""

    service: str | None = None
    database_mode: str | None = None
    reranker_model: str | None = None
    retrieval: dict[str, Any] | None = None
    evaluator: dict[str, Any] | None = None
