"""Automated tests for Stage 3 synthetic dataset generation, validation, and CSV loading."""

import json
from pathlib import Path

import pytest

from ai_service.recommendation.dataset import (
    DEFAULT_SEED,
    POLICY_CATEGORIES,
    DatasetValidationError,
    build_and_save_dataset,
    compute_suitability_score,
    evaluate_eligibility,
    generate_synthetic_policies,
    generate_synthetic_profiles,
    generate_synthetic_relevance,
    load_synthetic_dataset,
    profile_row_to_request,
    validate_policies_df,
    validate_profiles_df,
    validate_relevance_df,
)


def test_checked_in_dataset_loads_and_meets_minimum_counts() -> None:
    """Verify generated CSVs load cleanly and satisfy counts, uniqueness, and category coverage."""
    policies_df, profiles_df, relevance_df = load_synthetic_dataset(validate=True)

    assert len(policies_df) >= 50
    assert len(profiles_df) >= 20
    assert len(relevance_df) == len(policies_df) * len(profiles_df)

    assert set(policies_df["policy_type"].str.lower()) == set(POLICY_CATEGORIES)
    assert set(profiles_df["preferred_policy_type"].str.lower()) == set(POLICY_CATEGORIES)

    assert policies_df["policy_id"].is_unique
    assert policies_df["policy_code"].is_unique
    assert profiles_df["profile_id"].is_unique

    # Every profile must be compatible with UserProfileRequest Pydantic schema
    for row in profiles_df.to_dict(orient="records"):
        req = profile_row_to_request(row)
        assert req.policy_type == row["preferred_policy_type"]


def test_policy_ranges_and_eligibility_rules() -> None:
    """Verify monetary/age ranges and deterministic eligibility rule enforcement."""
    policies_df, profiles_df, relevance_df = load_synthetic_dataset(validate=True)

    assert (policies_df["min_entry_age"] >= 0).all()
    assert (policies_df["max_entry_age"] <= 120).all()
    assert (policies_df["min_entry_age"] <= policies_df["max_entry_age"]).all()
    assert (policies_df["annual_premium_min"] > 0).all()
    assert (policies_df["annual_premium_min"] <= policies_df["annual_premium_max"]).all()
    assert (policies_df["sum_insured_min"] > 0).all()
    assert (policies_df["sum_insured_min"] <= policies_df["sum_insured_max"]).all()

    # Check strict zero-match edge profiles exist across categories
    eligible_per_profile = relevance_df.groupby("profile_id")["is_eligible"].sum()
    assert (eligible_per_profile == 0).sum() >= 4

    # Verify targeted mandatory failure modes directly on evaluate_eligibility
    health_pol = (
        policies_df[
            (policies_df["policy_type"] == "health") & (policies_df["policy_status"] == "active")
        ]
        .iloc[0]
        .to_dict()
    )
    health_prof = profiles_df[profiles_df["preferred_policy_type"] == "health"].iloc[0].to_dict()

    inactive_pol = {**health_pol, "policy_status": "archived"}
    ok, reasons = evaluate_eligibility(health_prof, inactive_pol)
    assert not ok
    assert "inactive_or_non_catalog_policy" in reasons

    over_budget_pol = {
        **health_pol,
        "annual_premium_min": float(health_prof["budget_max"]) + 5000.0,
    }
    ok, reasons = evaluate_eligibility(health_prof, over_budget_pol)
    assert not ok
    assert "exceeds_max_budget" in reasons

    score_inelig, label_inelig = compute_suitability_score(health_prof, over_budget_pol, False)
    assert score_inelig == 0.0
    assert label_inelig == 0

    # Term smoker exclusion rule check
    non_smoker_term = next(
        p
        for p in policies_df[policies_df["policy_type"] == "term"].to_dict(orient="records")
        if json.loads(p["type_specific_metadata"])["smoking_status_eligibility"]
        == "non_smoker_only"
        and p["policy_status"] == "active"
    )
    term_prof = {
        **profiles_df[profiles_df["preferred_policy_type"] == "term"].iloc[0].to_dict(),
        "smoking_status": "smoker",
        "age": 30,
        "budget_max": 100000.0,
        "min_sum_insured": 5000000.0,
        "tenure_years": 20,
    }
    ok, reasons = evaluate_eligibility(term_prof, non_smoker_term)
    assert not ok
    assert "smoker_ineligible" in reasons


def test_relevance_labels_and_referential_integrity() -> None:
    """Verify valid labels, zero orphan references, and strict eligibility separation."""
    policies_df, profiles_df, relevance_df = load_synthetic_dataset(validate=True)

    assert set(relevance_df["profile_id"]).issubset(set(profiles_df["profile_id"]))
    assert set(relevance_df["policy_id"]).issubset(set(policies_df["policy_id"]))
    assert set(relevance_df["relevance_label"].unique()).issubset({0, 1, 2, 3})

    ineligible = relevance_df[~relevance_df["is_eligible"]]
    assert (ineligible["relevance_label"] == 0).all()
    assert (ineligible["suitability_score"] == 0.0).all()

    eligible = relevance_df[relevance_df["is_eligible"]]
    assert (eligible["policy_type_match"]).all()
    assert len(eligible) > 0


def test_dataset_generation_is_byte_reproducible(tmp_path: Path) -> None:
    """Running generation twice with identical seed must produce identical CSV bytes."""
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"

    summary_a = build_and_save_dataset(output_dir=dir_a, seed=DEFAULT_SEED)
    summary_b = build_and_save_dataset(output_dir=dir_b, seed=DEFAULT_SEED)

    assert summary_a["policy_count"] == summary_b["policy_count"]
    assert summary_a["profile_count"] == summary_b["profile_count"]
    assert summary_a["label_distribution"] == summary_b["label_distribution"]

    for filename in (
        "synthetic_policies.csv",
        "synthetic_profiles.csv",
        "synthetic_relevance.csv",
    ):
        assert (dir_a / filename).read_bytes() == (dir_b / filename).read_bytes()


def test_dataset_validation_rejects_corrupt_and_missing_inputs(tmp_path: Path) -> None:
    """Ensure loaders and validators catch missing files, bad ranges, orphans, and label leakage."""
    with pytest.raises(FileNotFoundError):
        load_synthetic_dataset(data_dir=tmp_path / "missing_dir", validate=True)

    policies_df = generate_synthetic_policies(seed=DEFAULT_SEED, per_category=15)
    profiles_df = generate_synthetic_profiles(seed=DEFAULT_SEED, per_category=10)
    relevance_df = generate_synthetic_relevance(policies_df, profiles_df)

    # 1. Missing required column
    with pytest.raises(DatasetValidationError, match="missing columns"):
        validate_policies_df(policies_df.drop(columns=["annual_premium_min"]))

    # 2. Duplicate policy ID
    dup_policies = policies_df.copy()
    dup_policies.loc[1, "policy_id"] = dup_policies.loc[0, "policy_id"]
    with pytest.raises(DatasetValidationError, match="unique"):
        validate_policies_df(dup_policies)

    # 3. Inverted age bounds
    bad_age = policies_df.copy()
    bad_age.loc[0, "min_entry_age"] = 70
    bad_age.loc[0, "max_entry_age"] = 30
    with pytest.raises(DatasetValidationError, match="max_entry_age"):
        validate_policies_df(bad_age)

    # 4. Invalid profile budget
    bad_profile = profiles_df.copy()
    bad_profile.loc[0, "budget_min"] = 999999.0
    with pytest.raises(DatasetValidationError, match="Invalid profile"):
        validate_profiles_df(bad_profile)

    # 5. Orphan policy reference in relevance table
    orphan_rel = relevance_df.copy()
    orphan_rel.loc[0, "policy_id"] = "00000000-0000-0000-0000-000000000000"
    with pytest.raises(DatasetValidationError, match="Orphan policy_id"):
        validate_relevance_df(orphan_rel, policies_df, profiles_df)

    # 6. Leakage / eligibility violation: ineligible policy given positive label
    bad_rel = relevance_df.copy()
    inelig_idx = bad_rel[~bad_rel["is_eligible"]].index[0]
    bad_rel.loc[inelig_idx, "relevance_label"] = 3
    with pytest.raises(DatasetValidationError, match="Ineligible policy pairs"):
        validate_relevance_df(bad_rel, policies_df, profiles_df)

    # 7. Corrupt CSV with empty/missing field loaded from disk
    policies_df.drop(columns=["insurer_name"]).to_csv(
        tmp_path / "synthetic_policies.csv", index=False
    )
    profiles_df.to_csv(tmp_path / "synthetic_profiles.csv", index=False)
    relevance_df.to_csv(tmp_path / "synthetic_relevance.csv", index=False)
    with pytest.raises(DatasetValidationError):
        load_synthetic_dataset(data_dir=tmp_path, validate=True)
