"""Transparent, deterministic weighted scoring backend (Architecture §5 / Phase A).

Provides an explicit, configurable fallback ranking scorer over the validated
Stage 5 59-column feature matrix. This module does NOT depend on LightGBM,
does NOT depend on ``annual_income``, and is only invoked when the caller
explicitly requests ``ranking_model="weighted"``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ai_service.recommendation.features import FEATURE_COLUMNS, RISK_PREFERENCE_ENCODING
from ai_service.recommendation.train_lightgbm import FeatureSchemaValidationError

WEIGHTED_SCORER_NAME = "coverageiq_weighted_scorer"
WEIGHTED_SCORER_VERSION = "1.0.0"

_ENCODED_TO_RISK_PREFERENCE: dict[int, str] = {
    int(round(code)): name for name, code in RISK_PREFERENCE_ENCODING.items()
}


@dataclass(frozen=True)
class WeightedDimensionWeights:
    """Normalized dimension weights for a single customer risk preference profile."""

    affordability: float
    coverage: float
    benefits: float
    cost_waiting: float
    insurer_network: float

    def __post_init__(self) -> None:
        values = (
            self.affordability,
            self.coverage,
            self.benefits,
            self.cost_waiting,
            self.insurer_network,
        )
        if any(not math.isfinite(w) or w < 0.0 for w in values):
            raise ValueError("All weighted scoring dimension weights must be finite and >= 0.")
        total = sum(values)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"Weighted scoring dimension weights must sum to 1.0, got {total:.6f}."
            )


def _default_weights_by_preference() -> dict[str, WeightedDimensionWeights]:
    """Return default risk-preference weights aligned with synthetic suitability rules."""
    return {
        "low_premium": WeightedDimensionWeights(
            affordability=0.40,
            coverage=0.15,
            benefits=0.15,
            cost_waiting=0.15,
            insurer_network=0.15,
        ),
        "balanced": WeightedDimensionWeights(
            affordability=0.25,
            coverage=0.20,
            benefits=0.20,
            cost_waiting=0.15,
            insurer_network=0.20,
        ),
        "comprehensive_coverage": WeightedDimensionWeights(
            affordability=0.12,
            coverage=0.23,
            benefits=0.27,
            cost_waiting=0.18,
            insurer_network=0.20,
        ),
    }


@dataclass(frozen=True)
class WeightedScoringConfig:
    """Configurable scaling and weight parameters for deterministic weighted scoring."""

    weights_by_risk_preference: Mapping[str, WeightedDimensionWeights] = field(
        default_factory=_default_weights_by_preference
    )
    affordability_floor_weight: float = 0.65
    target_budget_midpoint_ratio: float = 0.65
    coverage_log_multiplier: float = 4.0
    benefit_saturation_count: float = 6.0
    restoration_bonus: float = 0.12
    ncb_bonus: float = 0.08
    copay_max_pct: float = 40.0
    deductible_max_inr: float = 25_000.0
    initial_wait_max_days: float = 90.0
    ped_wait_max_days: float = 1_800.0
    csr_floor_pct: float = 85.0
    csr_span_pct: float = 15.0
    default_missing_csr_pct: float = 90.0
    hospital_network_max_count: float = 12_000.0

    def __post_init__(self) -> None:
        required_keys = set(RISK_PREFERENCE_ENCODING.keys())
        missing = required_keys - set(self.weights_by_risk_preference.keys())
        if missing:
            raise ValueError(
                f"WeightedScoringConfig missing weights for risk preferences: {sorted(missing)}"
            )
        if not 0.0 <= self.affordability_floor_weight <= 1.0:
            raise ValueError("affordability_floor_weight must be between 0.0 and 1.0.")
        for name, val in (
            ("coverage_log_multiplier", self.coverage_log_multiplier),
            ("benefit_saturation_count", self.benefit_saturation_count),
            ("copay_max_pct", self.copay_max_pct),
            ("deductible_max_inr", self.deductible_max_inr),
            ("initial_wait_max_days", self.initial_wait_max_days),
            ("ped_wait_max_days", self.ped_wait_max_days),
            ("csr_span_pct", self.csr_span_pct),
            ("hospital_network_max_count", self.hospital_network_max_count),
        ):
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be finite and > 0.0.")

    def weights_for_encoded_preference(self, encoded_value: float) -> WeightedDimensionWeights:
        """Resolve dimension weights cleanly from Stage 5's encoded risk_preference."""
        pref_name = _ENCODED_TO_RISK_PREFERENCE.get(int(round(encoded_value)), "balanced")
        return self.weights_by_risk_preference[pref_name]


@dataclass(frozen=True)
class WeightedScoringResult:
    """Outputs of deterministic weighted scoring including per-policy breakdowns."""

    scores: np.ndarray
    dimension_breakdowns: list[dict[str, float]]


DEFAULT_WEIGHTED_CONFIG = WeightedScoringConfig()


def _validate_weighted_input(
    X: pd.DataFrame,
    *,
    metadata: pd.DataFrame | None = None,
) -> np.ndarray:
    """Validate 59-column feature DataFrame invariants and return dense float64 array."""
    if not isinstance(X, pd.DataFrame):
        raise FeatureSchemaValidationError("Weighted scorer requires a pandas DataFrame input.")
    if tuple(X.columns) != FEATURE_COLUMNS:
        raise FeatureSchemaValidationError(
            "Weighted scorer feature columns do not match active FEATURE_COLUMNS schema."
        )
    if metadata is not None and len(metadata) != len(X):
        raise ValueError(
            f"metadata length ({len(metadata)}) does not match feature matrix rows ({len(X)})."
        )
    if X.empty:
        return np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float64)

    values = X.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise FeatureSchemaValidationError(
            "Weighted scorer input feature matrix contains non-finite (NaN/Inf) values."
        )
    return values


def score_policies_weighted(
    X: pd.DataFrame,
    *,
    config: WeightedScoringConfig | None = None,
    metadata: pd.DataFrame | None = None,
) -> WeightedScoringResult:
    """Compute transparent, deterministic [0, 1] policy scores from Stage 5 features.

    Note: ``annual_income`` and ``has_annual_income`` carry zero weight in the
    weighted formula so missing income never perturbs fallback scores.
    """
    _validate_weighted_input(X, metadata=metadata)
    if X.empty:
        return WeightedScoringResult(
            scores=np.array([], dtype=np.float64),
            dimension_breakdowns=[],
        )

    cfg = config or DEFAULT_WEIGHTED_CONFIG

    def col(name: str) -> np.ndarray:
        return X[name].to_numpy(dtype=np.float64)

    # 1. Affordability (independent of annual_income)
    aff_floor = np.clip(col("affordability_score"), 0.0, 1.0)
    prem_ratio = col("premium_to_budget_ratio")
    midpoint_fit = np.clip(
        1.0 - np.abs(np.minimum(prem_ratio, 1.0) - cfg.target_budget_midpoint_ratio),
        0.0,
        1.0,
    )
    affordability = (
        cfg.affordability_floor_weight * aff_floor
        + (1.0 - cfg.affordability_floor_weight) * midpoint_fit
    )

    # 2. Coverage adequacy & headroom (diminishing marginal return via log1p)
    cov_ratio = np.maximum(0.0, col("coverage_to_requirement_ratio"))
    coverage = np.minimum(
        1.0,
        np.log1p(np.maximum(0.0, cov_ratio - 1.0)) / math.log1p(cfg.coverage_log_multiplier),
    )

    # 3. Benefit richness
    base_benefits = np.minimum(
        1.0, np.maximum(0.0, col("total_benefit_count")) / cfg.benefit_saturation_count
    )
    bonus_benefits = cfg.restoration_bonus * np.clip(
        col("restoration_benefit_indicator"), 0.0, 1.0
    ) + cfg.ncb_bonus * (col("no_claim_bonus_pct") > 0.0).astype(np.float64)
    req_match = np.clip(col("required_benefit_match_ratio"), 0.0, 1.0)
    benefits = np.minimum(1.0, base_benefits + bonus_benefits) * req_match

    # 4. Out-of-pocket cost-sharing & waiting period efficiency
    is_health = col("is_health") > 0.5
    has_ped_cover = col("ped_coverage_indicator") > 0.5
    copay_score = 1.0 - np.minimum(
        1.0, np.maximum(0.0, col("co_payment_percentage")) / cfg.copay_max_pct
    )
    deduct_score = 1.0 - np.minimum(
        1.0, np.maximum(0.0, col("deductible_amount")) / cfg.deductible_max_inr
    )
    init_wait_score = 1.0 - np.minimum(
        1.0, np.maximum(0.0, col("initial_waiting_period_days")) / cfg.initial_wait_max_days
    )
    ped_wait_score = np.maximum(
        0.0, 1.0 - np.maximum(0.0, col("ped_waiting_period_days")) / cfg.ped_wait_max_days
    )
    # Policies without PED coverage cannot earn 0-day PED wait credit in health scoring.
    ped_wait_score = np.where(is_health & ~has_ped_cover, 0.0, ped_wait_score)
    cost_waiting = (
        0.35 * copay_score + 0.25 * deduct_score + 0.20 * init_wait_score + 0.20 * ped_wait_score
    )

    # 5. Insurer reliability & cashless network strength (neutral CSR if unrecorded)
    has_csr = col("has_claim_settlement_ratio") > 0.5
    effective_csr = np.where(has_csr, col("claim_settlement_ratio"), cfg.default_missing_csr_pct)
    csr_norm = np.clip((effective_csr - cfg.csr_floor_pct) / cfg.csr_span_pct, 0.0, 1.0)
    net_norm = np.minimum(
        1.0, np.maximum(0.0, col("hospital_network_count")) / cfg.hospital_network_max_count
    )
    pan_india = np.clip(col("pan_india_coverage_indicator"), 0.0, 1.0)
    insurer_network = np.where(
        is_health,
        0.70 * csr_norm + 0.30 * net_norm,
        0.85 * csr_norm + 0.15 * pan_india,
    )

    risk_encoded = col("risk_preference_encoded")
    raw_scores = np.empty(len(X), dtype=np.float64)
    breakdowns: list[dict[str, float]] = []

    for idx in range(len(X)):
        w = cfg.weights_for_encoded_preference(float(risk_encoded[idx]))
        composite = (
            w.affordability * float(affordability[idx])
            + w.coverage * float(coverage[idx])
            + w.benefits * float(benefits[idx])
            + w.cost_waiting * float(cost_waiting[idx])
            + w.insurer_network * float(insurer_network[idx])
        )
        raw_scores[idx] = composite
        breakdowns.append(
            {
                "weighted_affordability_score": round(float(affordability[idx]), 4),
                "weighted_coverage_score": round(float(coverage[idx]), 4),
                "weighted_benefits_score": round(float(benefits[idx]), 4),
                "weighted_cost_waiting_score": round(float(cost_waiting[idx]), 4),
                "weighted_insurer_network_score": round(float(insurer_network[idx]), 4),
            }
        )

    scores = np.round(np.clip(raw_scores, 0.0, 1.0), 6)
    return WeightedScoringResult(scores=scores, dimension_breakdowns=breakdowns)
