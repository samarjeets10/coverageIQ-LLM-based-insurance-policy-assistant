"""Stage 6 — Tests for LightGBM LambdaRank training pipeline.

Developer: Shravan Sambhaji Kapare (Roll No. 203)
Module: M5 — Policy Recommendation
Covers: data checks, group splitting, CV, baselines, training, schema validation,
leakage prevention, serialisation roundtrip, and determinism.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from ai_service.recommendation.features import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    FORBIDDEN_FEATURE_COLUMNS,
    TrainingFeatureDataset,
    build_training_feature_dataset,
)
from ai_service.recommendation.train_lightgbm import (
    MODEL_NAME,
    MODEL_VERSION,
    CrossValidationResult,
    EvalResult,
    FeatureSchemaValidationError,
    IncompatibleModelError,
    SplitResult,
    TrainingResult,
    _compute_group_sizes,
    _ndcg_at_k,
    cheapest_policy_baseline_ndcg,
    create_lightgbm_datasets,
    cross_validate_lightgbm_ranker,
    evaluate_ranking,
    get_default_lgbm_params,
    load_model,
    predict_ranking_scores,
    random_baseline_ndcg,
    save_model,
    split_training_data,
    suitability_baseline_ndcg,
    train_lightgbm_ranker,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_dataset() -> TrainingFeatureDataset:
    """Load the full synthetic eligible dataset once per test module."""
    return build_training_feature_dataset()


@pytest.fixture(scope="module")
def trained_result(full_dataset: TrainingFeatureDataset) -> TrainingResult:
    """Train a model once and reuse across read-only assertions."""
    return train_lightgbm_ranker(dataset=full_dataset, seed=42)


@pytest.fixture()
def artifact_dir(tmp_path: Path) -> Path:
    """Provide an isolated temporary directory for model artifacts."""
    d = tmp_path / "models"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# 1. Dataset integrity & leakage checks
# ---------------------------------------------------------------------------


class TestDatasetIntegrity:
    """Verify the training dataset properties assumed by the ranker."""

    def test_feature_matrix_shape(self, full_dataset: TrainingFeatureDataset) -> None:
        assert full_dataset.X.shape == (720, 59)

    def test_feature_columns_match(self, full_dataset: TrainingFeatureDataset) -> None:
        assert tuple(full_dataset.X.columns) == FEATURE_COLUMNS

    def test_all_features_finite(self, full_dataset: TrainingFeatureDataset) -> None:
        assert np.isfinite(full_dataset.X.to_numpy()).all()

    def test_no_target_leakage_in_x(self, full_dataset: TrainingFeatureDataset) -> None:
        """Verify suitability_score, relevance_label, and IDs are excluded from X."""
        x_cols = set(full_dataset.X.columns)
        leaked = x_cols & FORBIDDEN_FEATURE_COLUMNS
        assert leaked == set(), f"Leaked columns in X: {leaked}"

    def test_label_distribution(self, full_dataset: TrainingFeatureDataset) -> None:
        labels = set(full_dataset.y.unique())
        assert labels <= {1, 2, 3}
        assert len(labels) >= 2, "Need at least 2 distinct relevance grades"

    def test_groups_match_rows(self, full_dataset: TrainingFeatureDataset) -> None:
        assert len(full_dataset.groups) == len(full_dataset.X)
        assert len(full_dataset.metadata) == len(full_dataset.X)


# ---------------------------------------------------------------------------
# 2. Group-aware splitting & contiguous group sizing
# ---------------------------------------------------------------------------


class TestSplitting:
    """Validate group-aware train/val/test split."""

    def test_split_produces_correct_types(self, full_dataset: TrainingFeatureDataset) -> None:
        split = split_training_data(full_dataset)
        assert isinstance(split, SplitResult)
        assert isinstance(split.X_train, pd.DataFrame)
        assert isinstance(split.group_sizes_train, np.ndarray)

    def test_zero_profile_overlap(self, full_dataset: TrainingFeatureDataset) -> None:
        split = split_training_data(full_dataset)
        assert split.train_profiles & split.val_profiles == frozenset()
        assert split.train_profiles & split.test_profiles == frozenset()
        assert split.val_profiles & split.test_profiles == frozenset()

    def test_all_profiles_accounted(self, full_dataset: TrainingFeatureDataset) -> None:
        split = split_training_data(full_dataset)
        combined = split.train_profiles | split.val_profiles | split.test_profiles
        original = frozenset(full_dataset.groups.unique())
        assert combined == original

    def test_all_rows_accounted(self, full_dataset: TrainingFeatureDataset) -> None:
        split = split_training_data(full_dataset)
        total = len(split.X_train) + len(split.X_val) + len(split.X_test)
        assert total == len(full_dataset.X)

    def test_group_sizes_match_sorted_groups(self, full_dataset: TrainingFeatureDataset) -> None:
        split = split_training_data(full_dataset)
        assert split.group_sizes_train.sum() == len(split.X_train)
        assert split.group_sizes_val.sum() == len(split.X_val)
        assert split.group_sizes_test.sum() == len(split.X_test)

    def test_groups_contiguous_within_partition(self, full_dataset: TrainingFeatureDataset) -> None:
        split = split_training_data(full_dataset)
        for grp in [split.groups_train, split.groups_val, split.groups_test]:
            seen: list[str] = []
            current = None
            for g in grp:
                if g != current:
                    assert g not in seen, f"Group {g} appears non-consecutively"
                    seen.append(g)
                    current = g

    def test_invalid_ratio_raises(self, full_dataset: TrainingFeatureDataset) -> None:
        with pytest.raises(ValueError, match="must equal 1.0"):
            split_training_data(full_dataset, train_ratio=0.5, val_ratio=0.2, test_ratio=0.1)

    def test_empty_dataset_raises(self) -> None:
        empty = TrainingFeatureDataset(
            X=pd.DataFrame(columns=list(FEATURE_COLUMNS)).astype(np.float64),
            y=pd.Series([], dtype=np.int64),
            groups=pd.Series([], dtype=object),
            metadata=pd.DataFrame(),
        )
        with pytest.raises(ValueError, match="empty"):
            split_training_data(empty)

    def test_deterministic_split(self, full_dataset: TrainingFeatureDataset) -> None:
        s1 = split_training_data(full_dataset, seed=99)
        s2 = split_training_data(full_dataset, seed=99)
        assert s1.train_profiles == s2.train_profiles
        assert s1.val_profiles == s2.val_profiles
        assert s1.test_profiles == s2.test_profiles

    def test_group_sizes_basic_and_non_contiguous_error(self) -> None:
        g = pd.Series(["a", "a", "b", "b", "b", "c"])
        assert _compute_group_sizes(g).tolist() == [2, 3, 1]
        assert len(_compute_group_sizes(pd.Series([], dtype=object))) == 0

        non_contiguous = pd.Series(["a", "b", "a"])
        with pytest.raises(ValueError, match="contiguously"):
            _compute_group_sizes(non_contiguous)


# ---------------------------------------------------------------------------
# 3. LightGBM dataset construction & parameters
# ---------------------------------------------------------------------------


class TestLightGBMSetup:
    """Validate LightGBM Dataset creation and default parameters."""

    def test_datasets_created(self, full_dataset: TrainingFeatureDataset) -> None:
        split = split_training_data(full_dataset)
        train_ds, val_ds = create_lightgbm_datasets(split)
        assert isinstance(train_ds, lgb.Dataset)
        assert isinstance(val_ds, lgb.Dataset)

    def test_params_dict(self) -> None:
        params = get_default_lgbm_params(seed=123)
        assert params["objective"] == "lambdarank"
        assert params["subsample_freq"] == 1
        assert params["random_state"] == 123

    def test_params_copy_isolation(self) -> None:
        p1 = get_default_lgbm_params()
        p2 = get_default_lgbm_params()
        p1["learning_rate"] = 999
        assert p2["learning_rate"] != 999


# ---------------------------------------------------------------------------
# 4. Tie-aware NDCG calculation
# ---------------------------------------------------------------------------


class TestNDCG:
    """Validate tie-aware internal NDCG calculation."""

    def test_perfect_ranking(self) -> None:
        y = np.array([3, 2, 1, 0])
        p = np.array([10.0, 8.0, 5.0, 1.0])
        assert _ndcg_at_k(y, p, k=4) == pytest.approx(1.0, abs=1e-9)

    def test_empty_group(self) -> None:
        assert _ndcg_at_k(np.array([]), np.array([]), k=3) == 0.0

    def test_all_same_label(self) -> None:
        y = np.array([2, 2, 2])
        p = np.array([0.5, 0.3, 0.1])
        assert _ndcg_at_k(y, p, k=3) == pytest.approx(1.0, abs=1e-9)

    def test_tied_predictions_are_permutation_invariant(self) -> None:
        """Constant zero scores must produce identical expected NDCG regardless of row order."""
        y_desc = np.array([3, 2, 1, 1])
        y_asc = np.array([1, 1, 2, 3])
        zeros = np.zeros(4)
        s1 = _ndcg_at_k(y_desc, zeros, k=3)
        s2 = _ndcg_at_k(y_asc, zeros, k=3)
        assert s1 == pytest.approx(s2, abs=1e-9)
        assert s1 < 1.0


# ---------------------------------------------------------------------------
# 5. Training, evaluation & cross-validation
# ---------------------------------------------------------------------------


class TestTrainingAndEvaluation:
    """Validate end-to-end model training without warnings and metric quality."""

    def test_training_emits_no_lightgbm_warnings(
        self, full_dataset: TrainingFeatureDataset
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = train_lightgbm_ranker(dataset=full_dataset, seed=42, run_cv=False)
        assert isinstance(result.model, lgb.LGBMRanker)

    def test_model_converges_past_single_tree(self, trained_result: TrainingResult) -> None:
        """With NDCG@10 early-stopping guard, ranker learns multi-tree ensemble."""
        assert trained_result.model.best_iteration_ > 1
        used_features = sum(1 for v in trained_result.feature_importance.values() if v > 0.0)
        assert used_features >= 10

    def test_feature_importance_complete(self, trained_result: TrainingResult) -> None:
        assert len(trained_result.feature_importance) == len(FEATURE_COLUMNS)
        for col in FEATURE_COLUMNS:
            assert col in trained_result.feature_importance

    def test_eval_results_and_baselines(self, trained_result: TrainingResult) -> None:
        for ev in [
            trained_result.train_eval,
            trained_result.val_eval,
            trained_result.test_eval,
            trained_result.random_baseline,
            trained_result.cheapest_baseline,
            trained_result.suitability_baseline,
        ]:
            assert isinstance(ev, EvalResult)
            assert 0.0 <= ev.ndcg_at_1 <= 1.0
            assert 0.0 <= ev.ndcg_at_3 <= 1.0
            assert 0.0 <= ev.ndcg_at_5 <= 1.0
            assert 0.0 <= ev.ndcg_at_10 <= 1.0

        # Trained LightGBM ranker must outperform random and naive cheapest baselines
        assert trained_result.test_eval.ndcg_at_3 > trained_result.random_baseline.ndcg_at_3
        assert trained_result.test_eval.ndcg_at_5 > trained_result.random_baseline.ndcg_at_5
        assert trained_result.test_eval.ndcg_at_3 > trained_result.cheapest_baseline.ndcg_at_3
        assert trained_result.test_eval.ndcg_at_5 > trained_result.cheapest_baseline.ndcg_at_5

    def test_baseline_helpers_directly(self, trained_result: TrainingResult) -> None:
        sp = trained_result.split
        rbl = random_baseline_ndcg(sp.y_test, sp.groups_test, seed=42)
        cbl = cheapest_policy_baseline_ndcg(sp.X_test, sp.y_test, sp.groups_test)
        sbl = suitability_baseline_ndcg(sp.metadata_test, sp.y_test, sp.groups_test)
        assert rbl.num_groups == len(sp.test_profiles)
        assert cbl.num_groups == len(sp.test_profiles)
        assert sbl.ndcg_at_3 == pytest.approx(1.0, abs=1e-9)

    def test_cross_validation_result_present(self, trained_result: TrainingResult) -> None:
        cv = trained_result.cv_result
        assert isinstance(cv, CrossValidationResult)
        assert cv.n_splits == 5
        assert cv.mean_ndcg3 > 0.90
        assert cv.mean_ndcg5 > 0.90
        assert set(cv.category_ndcg3.keys()) == {"car", "health", "life", "term"}
        for score in cv.category_ndcg3.values():
            assert score > 0.90

    def test_cross_validation_invalid_splits_raises(
        self, full_dataset: TrainingFeatureDataset
    ) -> None:
        with pytest.raises(ValueError, match="GroupKFold"):
            cross_validate_lightgbm_ranker(full_dataset, n_splits=100)

    def test_empty_evaluation(self, trained_result: TrainingResult) -> None:
        empty_x = pd.DataFrame(columns=list(FEATURE_COLUMNS)).astype(np.float64)
        empty_y = pd.Series([], dtype=np.int64)
        empty_g = pd.Series([], dtype=object)
        ev = evaluate_ranking(trained_result.model, empty_x, empty_y, empty_g)
        assert ev.ndcg_at_3 == 0.0
        assert ev.num_groups == 0


# ---------------------------------------------------------------------------
# 6. Prediction, schema validation & determinism
# ---------------------------------------------------------------------------


class TestPredictionAndDeterminism:
    """Validate inference helpers and strict feature-schema checks."""

    def test_prediction_shape_and_finiteness(self, trained_result: TrainingResult) -> None:
        sp = trained_result.split
        scores = predict_ranking_scores(trained_result.model, sp.X_test, metadata=sp.metadata_test)
        assert scores.shape == (len(sp.X_test),)
        assert np.isfinite(scores).all()

    def test_empty_input_returns_empty(self, trained_result: TrainingResult) -> None:
        empty = pd.DataFrame(columns=list(FEATURE_COLUMNS)).astype(np.float64)
        scores = predict_ranking_scores(trained_result.model, empty)
        assert len(scores) == 0

    def test_predict_rejects_reordered_or_forbidden_columns(
        self, trained_result: TrainingResult
    ) -> None:
        x_sample = trained_result.split.X_test.iloc[:3].copy()

        # Reordered columns
        reordered = x_sample[list(reversed(FEATURE_COLUMNS))]
        with pytest.raises(FeatureSchemaValidationError, match="do not match"):
            predict_ranking_scores(trained_result.model, reordered)

        # Leaked forbidden column
        with_leak = x_sample.copy()
        with_leak["suitability_score"] = 0.9
        with pytest.raises(FeatureSchemaValidationError, match="Forbidden"):
            predict_ranking_scores(trained_result.model, with_leak)

        # Non-finite input
        with_nan = x_sample.copy()
        with_nan.iloc[0, 0] = np.nan
        with pytest.raises(FeatureSchemaValidationError, match="NaN or infinite"):
            predict_ranking_scores(trained_result.model, with_nan)

    def test_metadata_length_mismatch_rejected(self, trained_result: TrainingResult) -> None:
        x_sample = trained_result.split.X_test.iloc[:3]
        bad_meta = trained_result.split.metadata_test.iloc[:1]
        with pytest.raises(ValueError, match="metadata length"):
            predict_ranking_scores(trained_result.model, x_sample, metadata=bad_meta)

    def test_same_seed_same_predictions(self, full_dataset: TrainingFeatureDataset) -> None:
        r1 = train_lightgbm_ranker(dataset=full_dataset, seed=42, run_cv=False)
        r2 = train_lightgbm_ranker(dataset=full_dataset, seed=42, run_cv=False)
        p1 = predict_ranking_scores(r1.model, r1.split.X_test)
        p2 = predict_ranking_scores(r2.model, r2.split.X_test)
        np.testing.assert_array_almost_equal(p1, p2, decimal=10)


# ---------------------------------------------------------------------------
# 7. Serialisation & loading
# ---------------------------------------------------------------------------


class TestSerialisation:
    """Validate save/load roundtrip and schema mismatch rejection."""

    def test_save_creates_files(self, trained_result: TrainingResult, artifact_dir: Path) -> None:
        out = save_model(trained_result, output_dir=artifact_dir)
        assert (out / "lightgbm_ranker.joblib").exists()
        assert (out / "lightgbm_metadata.json").exists()
        assert (out / "training_metrics.json").exists()

    def test_load_roundtrip_and_prediction_parity(
        self, trained_result: TrainingResult, artifact_dir: Path
    ) -> None:
        save_model(trained_result, output_dir=artifact_dir)
        model, metadata = load_model(artifact_dir)
        assert isinstance(model, lgb.LGBMRanker)
        assert metadata["model_name"] == MODEL_NAME
        assert metadata["model_version"] == MODEL_VERSION
        assert metadata["feature_schema_version"] == FEATURE_SCHEMA_VERSION
        assert tuple(metadata["feature_columns"]) == FEATURE_COLUMNS

        x_test = trained_result.split.X_test
        orig = predict_ranking_scores(trained_result.model, x_test)
        loaded = predict_ranking_scores(model, x_test)
        np.testing.assert_array_almost_equal(orig, loaded, decimal=10)

    def test_metadata_and_metrics_contents(
        self, trained_result: TrainingResult, artifact_dir: Path
    ) -> None:
        save_model(trained_result, output_dir=artifact_dir)
        meta = json.loads((artifact_dir / "lightgbm_metadata.json").read_text(encoding="utf-8"))
        required_meta_keys = {
            "model_name",
            "model_version",
            "model_type",
            "framework",
            "feature_schema_version",
            "feature_columns",
            "num_features",
            "params",
            "best_iteration",
            "dataset_stats",
            "seed",
            "timestamp_utc",
            "synthetic_data_disclaimer",
        }
        assert required_meta_keys <= set(meta.keys())
        assert meta["dataset_stats"]["eligible_only_training"] is True

        metrics = json.loads((artifact_dir / "training_metrics.json").read_text(encoding="utf-8"))
        for section in [
            "train",
            "validation",
            "test",
            "baseline_random",
            "baseline_cheapest_policy",
            "baseline_suitability",
            "cross_validation",
            "feature_importance_top15",
        ]:
            assert section in metrics

    def test_schema_version_mismatch_rejected(
        self, trained_result: TrainingResult, artifact_dir: Path
    ) -> None:
        save_model(trained_result, output_dir=artifact_dir)
        meta_path = artifact_dir / "lightgbm_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["feature_schema_version"] = "99.99.99"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(IncompatibleModelError, match="schema mismatch"):
            load_model(artifact_dir)

    def test_column_mismatch_rejected(
        self, trained_result: TrainingResult, artifact_dir: Path
    ) -> None:
        save_model(trained_result, output_dir=artifact_dir)
        meta_path = artifact_dir / "lightgbm_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["feature_columns"] = ["wrong_col_1", "wrong_col_2"]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(IncompatibleModelError, match="column mismatch"):
            load_model(artifact_dir)

    def test_missing_model_file(self, artifact_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_model(artifact_dir)
