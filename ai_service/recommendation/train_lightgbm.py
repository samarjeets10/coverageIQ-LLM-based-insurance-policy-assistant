"""Stage 6 — LightGBM LambdaRank ranking model for CoverageIQ policy recommendation.

Trains a LightGBM-based learning-to-rank model on the synthetic eligible
(profile, policy) feature matrix produced by Stage 5 (features.py).
Provides group-aware train/val/test splitting, GroupKFold cross-validation,
evaluation, serialization, baseline comparisons, and schema-validated inference.

Developer: Shravan Sambhaji Kapare (Roll No. 203)
Module: M5 — Policy Recommendation
Branch: shravan-feature/recommendation-module
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from ai_service.recommendation.features import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    FORBIDDEN_FEATURE_COLUMNS,
    TrainingFeatureDataset,
    build_training_feature_dataset,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "coverageiq_lightgbm_ranker"
MODEL_VERSION = "0.1.0"

DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "models"

# Validation early-stopping uses NDCG@10 so small validation folds do not
# saturate at 1.0000 on iteration 1 before multi-feature boosting converges.
DEFAULT_EARLY_STOP_EVAL_AT: tuple[int, ...] = (10,)

_DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "lambdarank",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 5,
    "min_child_samples": 5,
    "n_estimators": 300,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "importance_type": "gain",
    "random_state": 42,
    "verbose": -1,
}


# ---------------------------------------------------------------------------
# Exceptions & Data containers
# ---------------------------------------------------------------------------


class IncompatibleModelError(ValueError):
    """Raised when a loaded model's schema does not match current features."""


class FeatureSchemaValidationError(ValueError):
    """Raised when a feature DataFrame violates FEATURE_COLUMNS invariants."""


@dataclass(frozen=True)
class SplitResult:
    """Group-aware train/validation/test split of a TrainingFeatureDataset."""

    X_train: pd.DataFrame
    y_train: pd.Series
    groups_train: pd.Series
    group_sizes_train: np.ndarray

    X_val: pd.DataFrame
    y_val: pd.Series
    groups_val: pd.Series
    group_sizes_val: np.ndarray

    X_test: pd.DataFrame
    y_test: pd.Series
    groups_test: pd.Series
    group_sizes_test: np.ndarray

    metadata_train: pd.DataFrame
    metadata_val: pd.DataFrame
    metadata_test: pd.DataFrame

    train_profiles: frozenset[str] = field(default_factory=frozenset)
    val_profiles: frozenset[str] = field(default_factory=frozenset)
    test_profiles: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class EvalResult:
    """Evaluation metrics for a ranked partition."""

    ndcg_at_3: float
    ndcg_at_5: float
    per_group_ndcg3: dict[str, float]
    per_group_ndcg5: dict[str, float]
    num_groups: int
    num_rows: int
    ndcg_at_1: float = 0.0
    ndcg_at_10: float = 0.0


@dataclass(frozen=True)
class CrossValidationResult:
    """Out-of-fold GroupKFold cross-validation summary across all groups."""

    n_splits: int
    fold_ndcg3: tuple[float, ...]
    fold_ndcg5: tuple[float, ...]
    mean_ndcg3: float
    std_ndcg3: float
    mean_ndcg5: float
    std_ndcg5: float
    category_ndcg3: dict[str, float]
    category_ndcg5: dict[str, float]


@dataclass(frozen=True)
class TrainingResult:
    """Full output of a training run."""

    model: lgb.LGBMRanker
    split: SplitResult
    train_eval: EvalResult
    val_eval: EvalResult
    test_eval: EvalResult
    random_baseline: EvalResult
    cheapest_baseline: EvalResult
    suitability_baseline: EvalResult
    params: dict[str, Any]
    feature_importance: dict[str, float]
    cv_result: CrossValidationResult | None = None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_feature_dataframe(X: pd.DataFrame) -> None:
    """Verify exact FEATURE_COLUMNS schema, no forbidden columns, and finite values."""
    cols = tuple(X.columns)
    forbidden = set(cols) & FORBIDDEN_FEATURE_COLUMNS
    if forbidden:
        raise FeatureSchemaValidationError(
            f"Forbidden target or identifier columns present in X: {sorted(forbidden)}"
        )
    if cols != FEATURE_COLUMNS:
        raise FeatureSchemaValidationError(
            f"Feature matrix columns do not match FEATURE_COLUMNS schema "
            f"(expected {len(FEATURE_COLUMNS)} columns, got {len(cols)})"
        )
    if not X.empty:
        values = X.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise FeatureSchemaValidationError("Feature matrix X contains NaN or infinite values")


# ---------------------------------------------------------------------------
# Splitting & group sizing
# ---------------------------------------------------------------------------


def _compute_group_sizes(groups: pd.Series) -> np.ndarray:
    """Return contiguous group-size counts for LightGBM ranking inputs.

    LightGBM requires rows grouped contiguously by query/profile and
    ``group`` passed as an array of integer sizes (e.g., [5, 3, 7]),
    never string identifiers. Raises ValueError if a group appears in
    non-contiguous runs.
    """
    if groups.empty:
        return np.array([], dtype=np.int32)

    sizes: list[int] = []
    seen: set[Any] = set()
    sentinel = object()
    prev: Any = sentinel
    count = 0

    for val in groups:
        if val == prev:
            count += 1
        else:
            if count > 0:
                sizes.append(count)
                seen.add(prev)
            if val in seen:
                raise ValueError(
                    f"Groups must be sorted contiguously; group {val!r} appeared non-consecutively."
                )
            prev = val
            count = 1

    if count > 0:
        sizes.append(count)

    return np.asarray(sizes, dtype=np.int32)


def _sort_by_group(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Sort all arrays stably by profile_id group."""
    idx = groups.argsort(kind="stable")
    return (
        X.iloc[idx].reset_index(drop=True),
        y.iloc[idx].reset_index(drop=True),
        groups.iloc[idx].reset_index(drop=True),
        metadata.iloc[idx].reset_index(drop=True),
    )


def split_training_data(
    dataset: TrainingFeatureDataset,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> SplitResult:
    """Group-aware train / validation / test split by ``profile_id``.

    Uses ``GroupShuffleSplit`` on ``profile_id`` so no profile appears in more
    than one partition, eliminating query/group data leakage.
    """
    if dataset.X.empty:
        raise ValueError("Cannot split an empty TrainingFeatureDataset")
    _validate_feature_dataframe(dataset.X)

    if train_ratio <= 0.0 or val_ratio <= 0.0 or test_ratio <= 0.0:
        raise ValueError("train_ratio, val_ratio, and test_ratio must all be > 0")
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total:.6f}")

    n_groups = dataset.groups.nunique()
    if n_groups < 3:
        raise ValueError(
            f"Need at least 3 distinct groups for train/val/test split, got {n_groups}"
        )

    X, y, groups, meta = dataset.X, dataset.y, dataset.groups, dataset.metadata

    gss_outer = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(gss_outer.split(X, y, groups))

    X_trainval = X.iloc[trainval_idx].reset_index(drop=True)
    y_trainval = y.iloc[trainval_idx].reset_index(drop=True)
    g_trainval = groups.iloc[trainval_idx].reset_index(drop=True)
    m_trainval = meta.iloc[trainval_idx].reset_index(drop=True)

    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    g_test = groups.iloc[test_idx].reset_index(drop=True)
    m_test = meta.iloc[test_idx].reset_index(drop=True)

    val_frac_inner = val_ratio / (train_ratio + val_ratio)
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=val_frac_inner, random_state=seed)
    train_idx, val_idx = next(gss_inner.split(X_trainval, y_trainval, g_trainval))

    X_train = X_trainval.iloc[train_idx].reset_index(drop=True)
    y_train = y_trainval.iloc[train_idx].reset_index(drop=True)
    g_train = g_trainval.iloc[train_idx].reset_index(drop=True)
    m_train = m_trainval.iloc[train_idx].reset_index(drop=True)

    X_val = X_trainval.iloc[val_idx].reset_index(drop=True)
    y_val = y_trainval.iloc[val_idx].reset_index(drop=True)
    g_val = g_trainval.iloc[val_idx].reset_index(drop=True)
    m_val = m_trainval.iloc[val_idx].reset_index(drop=True)

    X_train, y_train, g_train, m_train = _sort_by_group(X_train, y_train, g_train, m_train)
    X_val, y_val, g_val, m_val = _sort_by_group(X_val, y_val, g_val, m_val)
    X_test, y_test, g_test, m_test = _sort_by_group(X_test, y_test, g_test, m_test)

    train_profiles = frozenset(str(p) for p in g_train.unique())
    val_profiles = frozenset(str(p) for p in g_val.unique())
    test_profiles = frozenset(str(p) for p in g_test.unique())

    if (
        (train_profiles & val_profiles)
        or (train_profiles & test_profiles)
        or (val_profiles & test_profiles)
    ):
        raise RuntimeError("Profile overlap detected across train/val/test partitions")

    if not train_profiles or not val_profiles or not test_profiles:
        raise ValueError("One or more split partitions ended up with zero profile groups")

    return SplitResult(
        X_train=X_train,
        y_train=y_train,
        groups_train=g_train,
        group_sizes_train=_compute_group_sizes(g_train),
        X_val=X_val,
        y_val=y_val,
        groups_val=g_val,
        group_sizes_val=_compute_group_sizes(g_val),
        X_test=X_test,
        y_test=y_test,
        groups_test=g_test,
        group_sizes_test=_compute_group_sizes(g_test),
        metadata_train=m_train,
        metadata_val=m_val,
        metadata_test=m_test,
        train_profiles=train_profiles,
        val_profiles=val_profiles,
        test_profiles=test_profiles,
    )


# ---------------------------------------------------------------------------
# LightGBM dataset construction & parameters
# ---------------------------------------------------------------------------


def create_lightgbm_datasets(
    split: SplitResult,
) -> tuple[lgb.Dataset, lgb.Dataset]:
    """Build native LightGBM ``Dataset`` objects with contiguous group sizes."""
    train_ds = lgb.Dataset(
        split.X_train,
        label=split.y_train,
        group=split.group_sizes_train.tolist(),
        feature_name=list(FEATURE_COLUMNS),
        free_raw_data=False,
    )
    val_ds = lgb.Dataset(
        split.X_val,
        label=split.y_val,
        group=split.group_sizes_val.tolist(),
        reference=train_ds,
        feature_name=list(FEATURE_COLUMNS),
        free_raw_data=False,
    )
    return train_ds, val_ds


def get_default_lgbm_params(seed: int = 42) -> dict[str, Any]:
    """Return default LambdaRank hyperparameters tuned for small-N ranking.

    Notes
    -----
    * ``num_leaves=15`` and ``max_depth=5`` limit tree complexity on ~720 rows.
    * ``subsample=0.8`` + ``subsample_freq=1`` + ``colsample_bytree=0.8``
      provide bagging and feature subsampling regularization.
    * ``eval_metric`` / ``eval_at`` are passed at ``fit()`` time rather than
      constructor kwargs to avoid LightGBM parameter alias warnings.
    """
    params = dict(_DEFAULT_LGBM_PARAMS)
    params["random_state"] = seed
    return params


# ---------------------------------------------------------------------------
# Tie-aware NDCG evaluation & baseline helpers
# ---------------------------------------------------------------------------


def _dcg_at_k_tie_aware(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Compute tie-aware expected DCG@k using exponential gains ``(2**y - 1)``.

    When all predicted scores are distinct, this matches standard descending-sort
    DCG@k identically. When ties exist, items sharing the exact same score receive
    their expected average gain over tied rank positions, preventing input row
    ordering from biasing ties.
    """
    n = len(y_true)
    if n == 0 or k <= 0:
        return 0.0

    order = np.argsort(-y_pred, kind="mergesort")
    sorted_pred = y_pred[order]
    sorted_gains = (2.0 ** y_true[order].astype(np.float64)) - 1.0
    discounts = 1.0 / np.log2(np.arange(min(n, k), dtype=np.float64) + 2.0)

    dcg = 0.0
    idx = 0
    limit = min(n, k)
    while idx < limit:
        end = idx + 1
        while end < n and sorted_pred[end] == sorted_pred[idx]:
            end += 1
        rank_end = min(end, k)
        mean_gain = float(np.mean(sorted_gains[idx:end]))
        dcg += mean_gain * float(np.sum(discounts[idx:rank_end]))
        idx = end
    return dcg


def _ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Tie-aware NDCG@k for a single profile group."""
    if len(y_true) == 0 or k <= 0:
        return 0.0
    ideal_scores = y_true.astype(np.float64)
    idcg = _dcg_at_k_tie_aware(y_true, ideal_scores, k)
    if idcg <= 0.0:
        return 0.0
    dcg = _dcg_at_k_tie_aware(y_true, y_pred, k)
    return dcg / idcg


def _compute_group_ndcg_summary(
    y: pd.Series,
    groups: pd.Series,
    scores: np.ndarray,
    eval_at: tuple[int, ...] = (3, 5),
) -> EvalResult:
    """Shared group-level NDCG evaluator for model predictions and heuristics."""
    if len(y) == 0:
        return EvalResult(
            ndcg_at_3=0.0,
            ndcg_at_5=0.0,
            per_group_ndcg3={},
            per_group_ndcg5={},
            num_groups=0,
            num_rows=0,
            ndcg_at_1=0.0,
            ndcg_at_10=0.0,
        )

    k_first = eval_at[0] if len(eval_at) > 0 else 3
    k_second = eval_at[1] if len(eval_at) > 1 else 5

    y_arr = y.to_numpy()
    unique_groups = groups.unique()

    per1: list[float] = []
    per3: dict[str, float] = {}
    per5: dict[str, float] = {}
    per10: list[float] = []

    for gid in unique_groups:
        mask = (groups == gid).to_numpy()
        y_g = y_arr[mask]
        s_g = scores[mask]
        per1.append(_ndcg_at_k(y_g, s_g, 1))
        per3[str(gid)] = _ndcg_at_k(y_g, s_g, k_first)
        per5[str(gid)] = _ndcg_at_k(y_g, s_g, k_second)
        per10.append(_ndcg_at_k(y_g, s_g, 10))

    return EvalResult(
        ndcg_at_3=float(np.mean(list(per3.values()))),
        ndcg_at_5=float(np.mean(list(per5.values()))),
        per_group_ndcg3=per3,
        per_group_ndcg5=per5,
        num_groups=len(unique_groups),
        num_rows=len(y),
        ndcg_at_1=float(np.mean(per1)),
        ndcg_at_10=float(np.mean(per10)),
    )


def evaluate_ranking(
    model: lgb.LGBMRanker,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    eval_at: tuple[int, ...] = (3, 5),
) -> EvalResult:
    """Compute per-group and mean NDCG@K for a trained ranker."""
    _validate_feature_dataframe(X)
    if X.empty:
        return _compute_group_ndcg_summary(y, groups, np.array([], dtype=np.float64), eval_at)
    preds = predict_ranking_scores(model, X)
    return _compute_group_ndcg_summary(y, groups, preds, eval_at)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def random_baseline_ndcg(
    y: pd.Series,
    groups: pd.Series,
    *,
    eval_at: tuple[int, ...] = (3, 5),
    seed: int = 42,
    n_trials: int = 50,
) -> EvalResult:
    """Mean NDCG@K under random ranking averaged over ``n_trials``."""
    if len(y) == 0:
        return _compute_group_ndcg_summary(y, groups, np.array([], dtype=np.float64), eval_at)

    rng = np.random.RandomState(seed)
    y_arr = y.to_numpy()
    unique_groups = groups.unique()
    k_first = eval_at[0] if len(eval_at) > 0 else 3
    k_second = eval_at[1] if len(eval_at) > 1 else 5

    per_group_1: dict[str, list[float]] = {str(gid): [] for gid in unique_groups}
    per_group_3: dict[str, list[float]] = {str(gid): [] for gid in unique_groups}
    per_group_5: dict[str, list[float]] = {str(gid): [] for gid in unique_groups}
    per_group_10: dict[str, list[float]] = {str(gid): [] for gid in unique_groups}

    for _ in range(n_trials):
        for gid in unique_groups:
            mask = (groups == gid).to_numpy()
            y_g = y_arr[mask]
            rand_scores = rng.rand(len(y_g))
            key = str(gid)
            per_group_1[key].append(_ndcg_at_k(y_g, rand_scores, 1))
            per_group_3[key].append(_ndcg_at_k(y_g, rand_scores, k_first))
            per_group_5[key].append(_ndcg_at_k(y_g, rand_scores, k_second))
            per_group_10[key].append(_ndcg_at_k(y_g, rand_scores, 10))

    avg3 = {k: float(np.mean(v)) for k, v in per_group_3.items()}
    avg5 = {k: float(np.mean(v)) for k, v in per_group_5.items()}
    return EvalResult(
        ndcg_at_3=float(np.mean(list(avg3.values()))),
        ndcg_at_5=float(np.mean(list(avg5.values()))),
        per_group_ndcg3=avg3,
        per_group_ndcg5=avg5,
        num_groups=len(unique_groups),
        num_rows=len(y),
        ndcg_at_1=float(np.mean([np.mean(v) for v in per_group_1.values()])),
        ndcg_at_10=float(np.mean([np.mean(v) for v in per_group_10.values()])),
    )


def cheapest_policy_baseline_ndcg(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    eval_at: tuple[int, ...] = (3, 5),
) -> EvalResult:
    """Simple domain baseline: rank eligible policies strictly by lowest annual premium."""
    if X.empty:
        return _compute_group_ndcg_summary(y, groups, np.array([], dtype=np.float64), eval_at)
    scores = -X["annual_premium"].to_numpy(dtype=np.float64)
    return _compute_group_ndcg_summary(y, groups, scores, eval_at)


def suitability_baseline_ndcg(
    metadata: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    eval_at: tuple[int, ...] = (3, 5),
) -> EvalResult:
    """Evaluation-only ceiling baseline ranking by metadata ``suitability_score``.

    ``suitability_score`` is strictly excluded from training features X and is
    read from metadata solely as an oracle ceiling benchmark.
    """
    scores = metadata["suitability_score"].to_numpy(dtype=np.float64)
    return _compute_group_ndcg_summary(y, groups, scores, eval_at)


# ---------------------------------------------------------------------------
# Model fitting & Cross-Validation
# ---------------------------------------------------------------------------


def _fit_ranker(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    group_sizes_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    group_sizes_val: np.ndarray,
    *,
    params: dict[str, Any] | None,
    seed: int,
    early_stopping_rounds: int,
    early_stop_eval_at: tuple[int, ...] = DEFAULT_EARLY_STOP_EVAL_AT,
) -> tuple[lgb.LGBMRanker, dict[str, Any], int]:
    """Fit an ``LGBMRanker`` cleanly with early stopping and zero warnings."""
    ctor_params = get_default_lgbm_params(seed)
    if params is not None:
        ctor_params.update(params)

    # Route fit-only metric aliases away from LGBMRanker.__init__ to prevent warnings.
    ctor_params.pop("metric", None)
    eval_metric = ctor_params.pop("eval_metric", "ndcg")
    fit_eval_at = list(ctor_params.pop("eval_at", early_stop_eval_at))
    n_estimators = int(ctor_params.pop("n_estimators", 300))
    random_state = int(ctor_params.pop("random_state", seed))

    model = lgb.LGBMRanker(
        n_estimators=n_estimators,
        random_state=random_state,
        **ctor_params,
    )

    callbacks: list[Any] = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model.fit(
        X_train,
        y_train,
        group=group_sizes_train.tolist(),
        eval_X=(X_val,),
        eval_y=(y_val,),
        eval_group=[group_sizes_val.tolist()],
        eval_metric=eval_metric,
        eval_at=fit_eval_at,
        callbacks=callbacks,
    )

    recorded_params = ctor_params | {
        "n_estimators": n_estimators,
        "random_state": random_state,
        "eval_metric": eval_metric,
        "early_stop_eval_at": fit_eval_at,
    }
    return model, recorded_params, random_state


def cross_validate_lightgbm_ranker(
    dataset: TrainingFeatureDataset,
    *,
    n_splits: int = 5,
    params: dict[str, Any] | None = None,
    seed: int = 42,
    n_estimators_cv: int = 30,
) -> CrossValidationResult:
    """Evaluate generalization across all profile groups via GroupKFold CV.

    Using GroupKFold gives out-of-fold predictions for every profile group in
    the dataset (important because single-split test sets have ~9 groups).
    """
    _validate_feature_dataframe(dataset.X)
    n_groups = dataset.groups.nunique()
    if n_groups < n_splits:
        raise ValueError(f"Cannot run {n_splits}-fold GroupKFold with only {n_groups} groups")

    X_sorted, y_sorted, g_sorted, m_sorted = _sort_by_group(
        dataset.X, dataset.y, dataset.groups, dataset.metadata
    )
    gkf = GroupKFold(n_splits=n_splits)

    cv_params = dict(params or {})
    cv_params.setdefault("n_estimators", n_estimators_cv)
    ctor_params = get_default_lgbm_params(seed)
    ctor_params.update(cv_params)
    ctor_params.pop("metric", None)
    ctor_params.pop("eval_metric", None)
    ctor_params.pop("eval_at", None)

    fold_ndcg3: list[float] = []
    fold_ndcg5: list[float] = []
    cat_ndcg3: dict[str, list[float]] = {}
    cat_ndcg5: dict[str, list[float]] = {}

    for train_idx, val_idx in gkf.split(X_sorted, y_sorted, g_sorted):
        X_tr = X_sorted.iloc[train_idx].reset_index(drop=True)
        y_tr = y_sorted.iloc[train_idx].reset_index(drop=True)
        g_tr = g_sorted.iloc[train_idx].reset_index(drop=True)

        X_va = X_sorted.iloc[val_idx].reset_index(drop=True)
        y_va = y_sorted.iloc[val_idx].reset_index(drop=True)
        g_va = g_sorted.iloc[val_idx].reset_index(drop=True)
        m_va = m_sorted.iloc[val_idx].reset_index(drop=True)

        fold_model = lgb.LGBMRanker(**ctor_params)
        fold_model.fit(
            X_tr,
            y_tr,
            group=_compute_group_sizes(g_tr).tolist(),
        )
        ev = evaluate_ranking(fold_model, X_va, y_va, g_va)
        fold_ndcg3.append(ev.ndcg_at_3)
        fold_ndcg5.append(ev.ndcg_at_5)

        prof_categories = m_va.groupby("profile_id")["policy_type"].first().to_dict()
        for pid, score3 in ev.per_group_ndcg3.items():
            cat = str(prof_categories.get(pid, "unknown"))
            cat_ndcg3.setdefault(cat, []).append(score3)
            cat_ndcg5.setdefault(cat, []).append(ev.per_group_ndcg5[pid])

    return CrossValidationResult(
        n_splits=n_splits,
        fold_ndcg3=tuple(fold_ndcg3),
        fold_ndcg5=tuple(fold_ndcg5),
        mean_ndcg3=float(np.mean(fold_ndcg3)),
        std_ndcg3=float(np.std(fold_ndcg3)),
        mean_ndcg5=float(np.mean(fold_ndcg5)),
        std_ndcg5=float(np.std(fold_ndcg5)),
        category_ndcg3={k: float(np.mean(v)) for k, v in sorted(cat_ndcg3.items())},
        category_ndcg5={k: float(np.mean(v)) for k, v in sorted(cat_ndcg5.items())},
    )


def train_lightgbm_ranker(
    dataset: TrainingFeatureDataset | None = None,
    *,
    params: dict[str, Any] | None = None,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    early_stopping_rounds: int = 30,
    run_cv: bool = True,
    cv_splits: int = 5,
) -> TrainingResult:
    """Train a LightGBM LambdaRank model end-to-end with baselines and CV."""
    if dataset is None:
        dataset = build_training_feature_dataset()
    _validate_feature_dataframe(dataset.X)

    split = split_training_data(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    logger.info(
        "Split: train=%d (%d groups), val=%d (%d groups), test=%d (%d groups)",
        len(split.X_train),
        len(split.train_profiles),
        len(split.X_val),
        len(split.val_profiles),
        len(split.X_test),
        len(split.test_profiles),
    )

    model, recorded_params, effective_seed = _fit_ranker(
        split.X_train,
        split.y_train,
        split.group_sizes_train,
        split.X_val,
        split.y_val,
        split.group_sizes_val,
        params=params,
        seed=seed,
        early_stopping_rounds=early_stopping_rounds,
    )

    train_eval = evaluate_ranking(model, split.X_train, split.y_train, split.groups_train)
    val_eval = evaluate_ranking(model, split.X_val, split.y_val, split.groups_val)
    test_eval = evaluate_ranking(model, split.X_test, split.y_test, split.groups_test)

    rand_bl = random_baseline_ndcg(split.y_test, split.groups_test, seed=effective_seed)
    cheap_bl = cheapest_policy_baseline_ndcg(split.X_test, split.y_test, split.groups_test)
    suit_bl = suitability_baseline_ndcg(split.metadata_test, split.y_test, split.groups_test)

    importance_vals = model.feature_importances_
    importance_dict = {
        col: float(imp) for col, imp in zip(FEATURE_COLUMNS, importance_vals, strict=True)
    }

    cv_result: CrossValidationResult | None = None
    if run_cv and dataset.groups.nunique() >= cv_splits:
        cv_n_trees = max(int(model.best_iteration_ or 30), 20)
        cv_result = cross_validate_lightgbm_ranker(
            dataset,
            n_splits=cv_splits,
            params=params,
            seed=effective_seed,
            n_estimators_cv=cv_n_trees,
        )

    return TrainingResult(
        model=model,
        split=split,
        train_eval=train_eval,
        val_eval=val_eval,
        test_eval=test_eval,
        random_baseline=rand_bl,
        cheapest_baseline=cheap_bl,
        suitability_baseline=suit_bl,
        params=recorded_params,
        feature_importance=importance_dict,
        cv_result=cv_result,
    )


# ---------------------------------------------------------------------------
# Serialisation & Loading
# ---------------------------------------------------------------------------


def _serialisable(obj: Any) -> Any:
    """Recursively convert numpy types for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialisable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _eval_to_dict(ev: EvalResult) -> dict[str, Any]:
    return {
        "ndcg@1": ev.ndcg_at_1,
        "ndcg@3": ev.ndcg_at_3,
        "ndcg@5": ev.ndcg_at_5,
        "ndcg@10": ev.ndcg_at_10,
        "num_groups": ev.num_groups,
        "num_rows": ev.num_rows,
    }


def save_model(
    result: TrainingResult,
    output_dir: Path | str | None = None,
) -> Path:
    """Persist trained model, metadata schema manifest, and metrics to disk."""
    out = Path(output_dir) if output_dir is not None else DEFAULT_ARTIFACTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "lightgbm_ranker.joblib"
    joblib.dump(result.model, model_path, compress=3)

    best_iter = int(result.model.best_iteration_ or result.params.get("n_estimators", 0))
    all_y = pd.concat([result.split.y_train, result.split.y_val, result.split.y_test])
    label_dist = {str(k): int(v) for k, v in sorted(all_y.value_counts().items())}

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_type": "LGBMRanker (LambdaRank)",
        "framework": f"lightgbm=={lgb.__version__}",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "num_features": len(FEATURE_COLUMNS),
        "params": _serialisable(result.params),
        "best_iteration": best_iter,
        "n_estimators_actual": best_iter,
        "dataset_stats": {
            "total_rows": int(
                len(result.split.X_train) + len(result.split.X_val) + len(result.split.X_test)
            ),
            "total_groups": int(
                len(result.split.train_profiles)
                + len(result.split.val_profiles)
                + len(result.split.test_profiles)
            ),
            "train_rows": len(result.split.X_train),
            "val_rows": len(result.split.X_val),
            "test_rows": len(result.split.X_test),
            "train_groups": len(result.split.train_profiles),
            "val_groups": len(result.split.val_profiles),
            "test_groups": len(result.split.test_profiles),
            "label_distribution": label_dist,
            "eligible_only_training": 0 not in set(all_y.unique()),
        },
        "seed": result.params.get("random_state", 42),
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "synthetic_data_disclaimer": (
            "This model was trained exclusively on synthetic insurance policy "
            "data generated for development and pipeline validation. It is NOT "
            "suitable for production use without retraining on real-world data."
        ),
    }
    meta_path = out / "lightgbm_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    metrics: dict[str, Any] = {
        "train": _eval_to_dict(result.train_eval),
        "validation": _eval_to_dict(result.val_eval),
        "test": _eval_to_dict(result.test_eval),
        "baseline_random": _eval_to_dict(result.random_baseline),
        "baseline_cheapest_policy": _eval_to_dict(result.cheapest_baseline),
        "baseline_suitability": _eval_to_dict(result.suitability_baseline),
        "feature_importance_top15": dict(
            sorted(result.feature_importance.items(), key=lambda x: -x[1])[:15]
        ),
    }
    if result.cv_result is not None:
        metrics["cross_validation"] = {
            "n_splits": result.cv_result.n_splits,
            "mean_ndcg@3": result.cv_result.mean_ndcg3,
            "std_ndcg@3": result.cv_result.std_ndcg3,
            "mean_ndcg@5": result.cv_result.mean_ndcg5,
            "std_ndcg@5": result.cv_result.std_ndcg5,
            "per_category_ndcg@3": result.cv_result.category_ndcg3,
            "per_category_ndcg@5": result.cv_result.category_ndcg5,
        }

    metrics_path = out / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    logger.info("Model artifacts saved to %s", out)
    return out


def load_model(
    model_dir: Path | str | None = None,
) -> tuple[lgb.LGBMRanker, dict[str, Any]]:
    """Load a saved LightGBM ranker and enforce feature-schema compatibility."""
    model_path = Path(model_dir) if model_dir is not None else DEFAULT_ARTIFACTS_DIR
    joblib_file = model_path / "lightgbm_ranker.joblib"
    meta_file = model_path / "lightgbm_metadata.json"

    if not joblib_file.exists():
        raise FileNotFoundError(f"Model file not found: {joblib_file}")
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")

    metadata = json.loads(meta_file.read_text(encoding="utf-8"))

    saved_version = metadata.get("feature_schema_version", "")
    if saved_version != FEATURE_SCHEMA_VERSION:
        raise IncompatibleModelError(
            f"Feature schema mismatch: saved={saved_version}, current={FEATURE_SCHEMA_VERSION}"
        )

    saved_columns = tuple(metadata.get("feature_columns", []))
    if saved_columns != FEATURE_COLUMNS:
        raise IncompatibleModelError(
            f"Feature column mismatch: saved {len(saved_columns)} columns, "
            f"expected {len(FEATURE_COLUMNS)} columns"
        )

    model = joblib.load(joblib_file)
    if not isinstance(model, lgb.LGBMRanker):
        raise IncompatibleModelError(f"Expected LGBMRanker, got {type(model).__name__}")
    return model, metadata


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_ranking_scores(
    model: lgb.LGBMRanker,
    X: pd.DataFrame,
    *,
    metadata: pd.DataFrame | None = None,
) -> np.ndarray:
    """Return float64 ranking scores (higher = better) with schema validation."""
    _validate_feature_dataframe(X)
    if metadata is not None and len(metadata) != len(X):
        raise ValueError(
            f"metadata length ({len(metadata)}) does not match feature matrix rows ({len(X)})"
        )
    if X.empty:
        return np.array([], dtype=np.float64)
    scores = model.predict(X)
    return np.asarray(scores, dtype=np.float64)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _print_summary(result: TrainingResult, output_dir: Path) -> None:
    """Print a human-readable training summary to stdout."""
    sep = "=" * 76
    print(f"\n{sep}")
    print("  CoverageIQ — LightGBM LambdaRank Training Summary")
    print(sep)
    print(f"  Model:          {MODEL_NAME} v{MODEL_VERSION}")
    print(f"  Framework:      LightGBM {lgb.__version__}")
    print(f"  Feature Schema: {FEATURE_SCHEMA_VERSION} ({len(FEATURE_COLUMNS)} features)")
    print(f"  Seed:           {result.params.get('random_state', 42)}")
    print(f"  Best Iteration: {result.model.best_iteration_}")
    print()

    split = result.split
    print("  Dataset Split:")
    print(f"    Train: {len(split.X_train):>5} rows, {len(split.train_profiles):>3} groups")
    print(f"    Val:   {len(split.X_val):>5} rows, {len(split.val_profiles):>3} groups")
    print(f"    Test:  {len(split.X_test):>5} rows, {len(split.test_profiles):>3} groups")

    all_y = pd.concat([split.y_train, split.y_val, split.y_test])
    dist = dict(sorted(all_y.value_counts().items()))
    print(f"    Labels: {dist}")
    print()

    print("  Evaluation (NDCG):")
    print(
        f"    {'Partition / Baseline':<22} {'NDCG@1':>8}  "
        f"{'NDCG@3':>8}  {'NDCG@5':>8}  {'NDCG@10':>8}"
    )
    print(f"    {'-' * 22} {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
    for name, ev in [
        ("Train", result.train_eval),
        ("Validation", result.val_eval),
        ("Test", result.test_eval),
        ("Random Baseline", result.random_baseline),
        ("Cheapest-Policy BL", result.cheapest_baseline),
        ("Suitability BL (ceil)", result.suitability_baseline),
    ]:
        print(
            f"    {name:<22} {ev.ndcg_at_1:>8.4f}  {ev.ndcg_at_3:>8.4f}  "
            f"{ev.ndcg_at_5:>8.4f}  {ev.ndcg_at_10:>8.4f}"
        )
    print()

    if result.cv_result is not None:
        cv = result.cv_result
        print(
            f"  {cv.n_splits}-Fold GroupKFold OOF CV: "
            f"NDCG@3 = {cv.mean_ndcg3:.4f} ± {cv.std_ndcg3:.4f}, "
            f"NDCG@5 = {cv.mean_ndcg5:.4f} ± {cv.std_ndcg5:.4f}"
        )
        cat_line = ", ".join(f"{cat}={score:.4f}" for cat, score in cv.category_ndcg3.items())
        print(f"    Per-category OOF NDCG@3: {cat_line}")
        print()

    top = sorted(result.feature_importance.items(), key=lambda x: -x[1])[:12]
    print("  Top 12 Features (gain):")
    for rank, (feat, imp) in enumerate(top, 1):
        print(f"    {rank:>2}. {feat:<40} {imp:>10.2f}")
    print()

    print(f"  Artifacts: {output_dir}")
    print(sep)
    print()

    if 0 not in set(all_y.unique()):
        print("  NOTE: Label 0 (ineligible) absent — training uses eligible pairs only.")
        print("        Deterministic hard filtering runs before ranking at inference time.")
    print()


def main() -> None:
    """CLI entry point for Stage 6 LightGBM training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    print("Loading dataset and building features...")
    dataset = build_training_feature_dataset()
    print(
        f"Dataset: {dataset.X.shape[0]} rows x {dataset.X.shape[1]} features, "
        f"{dataset.groups.nunique()} groups"
    )

    print("Training LightGBM LambdaRank model...")
    result = train_lightgbm_ranker(dataset=dataset)

    print("Saving model artifacts...")
    output_dir = save_model(result)

    _print_summary(result, output_dir)


if __name__ == "__main__":
    main()
