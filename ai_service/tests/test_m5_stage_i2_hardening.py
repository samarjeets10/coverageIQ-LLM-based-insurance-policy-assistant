"""M5.I2 Regression Tests — PED Missingness & Malformed Metadata Hardening.

Developer: Shravan Sambhaji Kapare (Roll No. 203)
Module: M5 — Policy Recommendation
Covers:
- Defect A: Conservative non-zero handling of missing Health PED waiting periods.
- Defect B: Strict rejection of booleans, malformed strings, and non-finite optional numeric
  metadata during Stage 4 filtering and Stage 5 feature extraction without HTTP 500 crashes.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai_service.common.logging import StructuredJsonFormatter
from ai_service.main import create_app
from ai_service.recommendation.coercion import (
    InvalidNumericMetadataError,
    parse_bool,
    parse_optional_bool,
    parse_strict_metadata_mapping,
    parse_strict_optional_float,
    parse_strict_optional_int,
)
from ai_service.recommendation.db import get_db_connection
from ai_service.recommendation.features import (
    CONSERVATIVE_MISSING_HEALTH_PED_WAIT_DAYS,
    CONSERVATIVE_MISSING_HEALTH_PED_WAIT_MONTHS,
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    build_feature_matrix,
    extract_policy_features,
)
from ai_service.recommendation.filters import filter_policies
from ai_service.recommendation.openapi_export import verify_openapi_contract
from ai_service.recommendation.schemas import RecommendRequest, UserProfileRequest
from ai_service.recommendation.service import (
    ModelState,
    _extract_factual_highlights,
    load_model_state,
    recommend_policies_for_request,
)
from ai_service.recommendation.settings import Settings
from ai_service.recommendation.train_lightgbm import predict_ranking_scores
from ai_service.recommendation.weighted import score_policies_weighted


@pytest.fixture(scope="module")
def model_state() -> ModelState:
    state = load_model_state()
    assert state.is_ready is True
    assert state.warmed_up is True
    return state


def _policy(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "policy_id": str(uuid.uuid4()),
        "organization_id": uuid.uuid4(),
        "policy_name": "M5.I2 Hardening Plan",
        "insurer_name": "CoverageIQ Test Insurer",
        "policy_type": "health",
        "description": "Controlled M5.I2 regression fixture.",
        "policy_status": "active",
        "current_version_id": uuid.uuid4(),
        "policy_version_id": uuid.uuid4(),
        "is_catalog_eligible": True,
        "min_entry_age": 18,
        "max_entry_age": 65,
        "max_renewal_age": 85,
        "lifelong_renewal": False,
        "coverage_mode": "both",
        "max_family_members": 4,
        "eligible_applicant_types": ["salaried", "self_employed", "student", "senior_citizen"],
        "annual_premium_min": Decimal("12000.00"),
        "annual_premium_max": Decimal("16000.00"),
        "sum_insured_min": Decimal("500000.00"),
        "sum_insured_max": Decimal("1000000.00"),
        "deductible_amount": Decimal("0.00"),
        "copay_pct": Decimal("0.00"),
        "room_rent_limit_type": "no_limit",
        "room_rent_limit_value": None,
        "initial_waiting_period_days": 30,
        "ped_covered": True,
        "ped_waiting_period_months": 24,
        "specific_disease_waiting_months": 24,
        "maternity_waiting_months": None,
        "network_hospitals_count": 7500,
        "cashless_available": True,
        "pan_india_coverage": True,
        "available_regions": ["mumbai", "pune", "bengaluru"],
        "min_tenure_years": 1,
        "max_tenure_years": 3,
        "covered_benefits": ["cashless_hospitalization", "daycare_procedures", "ambulance_cover"],
        "exclusions": ["cosmetic_surgery"],
        "claim_settlement_ratio": Decimal("96.00"),
        "no_claim_bonus_pct": Decimal("20.00"),
        "restoration_benefit": True,
        "type_specific_metadata": {},
        "extraction_source": "verified",
    }
    row.update(overrides)
    return row


def _profile(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "policy_type": "health",
        "age": 32,
        "budget_min": 5000.0,
        "budget_max": 25000.0,
        "applicant_type": "salaried",
        "coverage_mode": "individual",
        "family_size": 1,
        "min_sum_insured": 500000.0,
        "required_benefits": ["cashless_hospitalization"],
        "pre_existing_conditions": [],
        "tenure_years": 1,
        "annual_income_inr": 1_200_000.0,
        "risk_preference": "balanced",
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# Defect A — Missing Health PED Waiting Period Conservative Semantics
# ---------------------------------------------------------------------------


def test_ped_missing_never_zero_days_and_never_beats_zero_day_wait(
    monkeypatch: pytest.MonkeyPatch,
    model_state: ModelState,
) -> None:
    """Missing Health PED receives conservative 36m (1080d) encoding and never outscores 0m."""
    assert CONSERVATIVE_MISSING_HEALTH_PED_WAIT_MONTHS == 36
    assert CONSERVATIVE_MISSING_HEALTH_PED_WAIT_DAYS == 1080.0
    assert FEATURE_SCHEMA_VERSION == "1.0.0"

    prof = UserProfileRequest.model_validate(_profile())
    pol_zero = _policy(policy_id="ped-00", ped_covered=True, ped_waiting_period_months=0)
    pol_none = _policy(policy_id="ped-na", ped_covered=True, ped_waiting_period_months=None)
    pol_36 = _policy(policy_id="ped-36", ped_covered=True, ped_waiting_period_months=36)
    pol_uncovered = _policy(policy_id="ped-no", ped_covered=False, ped_waiting_period_months=None)

    matrix = build_feature_matrix(prof, [pol_zero, pol_none, pol_36, pol_uncovered])
    assert matrix.shape == (4, 59)
    assert tuple(matrix.columns) == FEATURE_COLUMNS
    assert np.isfinite(matrix.to_numpy()).all()

    assert list(matrix["ped_waiting_period_days"]) == [0.0, 1080.0, 1080.0, 0.0]
    assert list(matrix["ped_coverage_indicator"]) == [1.0, 1.0, 1.0, 0.0]

    w_res = score_policies_weighted(matrix)
    assert np.isfinite(w_res.scores).all()
    w_zero, w_none, w_36, w_uncov = w_res.scores.tolist()
    b_zero, b_none, b_36, b_uncov = [
        b["weighted_cost_waiting_score"] for b in w_res.dimension_breakdowns
    ]

    assert b_zero > b_none
    assert b_none == b_36
    assert w_zero > w_none
    assert w_none == w_36
    assert b_uncov < b_none < b_zero

    # LightGBM remains finite and never boosts missing PED over 0-day or 36-month waits
    assert model_state.model is not None
    lgb_scores = predict_ranking_scores(model_state.model, matrix)
    assert np.isfinite(lgb_scores).all()
    assert lgb_scores[1] == lgb_scores[2]
    assert lgb_scores[1] <= lgb_scores[0]

    # End-to-end service calls across both backends are deterministic & warn cleanly
    catalog = [pol_none, pol_36, pol_zero]
    monkeypatch.setattr(
        "ai_service.recommendation.service.load_catalog_policies_from_db",
        lambda _c, policy_type=None, active_only=False: catalog,
    )
    w_req = RecommendRequest.model_validate({"profile": _profile(), "ranking_model": "weighted"})
    r_w1 = recommend_policies_for_request(w_req, MagicMock(), model_state)
    r_w2 = recommend_policies_for_request(w_req, MagicMock(), model_state)
    assert [(x.policy_id, x.score) for x in r_w1.recommendations] == [
        (x.policy_id, x.score) for x in r_w2.recommendations
    ]
    assert r_w1.recommendations[0].policy_id == "ped-00"
    assert r_w1.recommendations[0].score > r_w1.recommendations[1].score
    assert any("ped_waiting_period_months" in w for w in r_w1.warnings)

    lgb_req = RecommendRequest.model_validate({"profile": _profile(), "ranking_model": "lightgbm"})
    r_l1 = recommend_policies_for_request(lgb_req, MagicMock(), model_state)
    r_l2 = recommend_policies_for_request(lgb_req, MagicMock(), model_state)
    assert [(x.policy_id, x.score) for x in r_l1.recommendations] == [
        (x.policy_id, x.score) for x in r_l2.recommendations
    ]
    assert all(math.isfinite(x.score) for x in r_l1.recommendations)
    assert any("ped_waiting_period_months" in w for w in r_l1.warnings)


def test_non_health_ped_remains_category_isolated() -> None:
    """Non-health categories keep ped_waiting_period_days == 0.0 when PED is None."""
    for ptype, extra in (
        ("car", {"vehicle_type": "sedan"}),
        ("life", {"min_sum_insured": 1_000_000.0, "tenure_years": 20}),
        ("term", {"min_sum_insured": 5_000_000.0, "tenure_years": 20}),
    ):
        prof = UserProfileRequest.model_validate(_profile(policy_type=ptype, **extra))
        pol = _policy(
            policy_type=ptype,
            coverage_mode="individual",
            max_family_members=1,
            max_tenure_years=25,
            ped_covered=True,
            ped_waiting_period_months=None,
            type_specific_metadata={"vehicle_types": ["sedan"]} if ptype == "car" else {},
        )
        feats = extract_policy_features(prof, pol)
        assert feats["ped_waiting_period_days"] == 0.0
        assert feats["ped_coverage_indicator"] == 0.0


# ---------------------------------------------------------------------------
# Defect B — Strict Rejection & Per-Candidate Exclusion of Malformed Metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_val",
    [True, False, "not_a_number", "n/a", "bad", "oops", "NaN", "inf", float("inf"), Decimal("NaN")],
)
def test_strict_numeric_coercion_rejects_booleans_and_malformed_values(bad_val: Any) -> None:
    """Shared strict coercion never treats bools, NaN/Inf strings, or garbage strings as numbers."""
    with pytest.raises(InvalidNumericMetadataError):
        parse_strict_optional_float(bad_val, field_name="test_field")
    with pytest.raises(InvalidNumericMetadataError):
        parse_strict_optional_int(bad_val, field_name="test_int")


def test_strict_coercion_preserves_valid_numbers_and_direct_feature_errors() -> None:
    """Valid numeric strings/Decimals/floats parse cleanly; direct feature calls reject garbage."""
    assert parse_strict_optional_float(None) is None
    assert parse_strict_optional_float("   ") is None
    assert parse_strict_optional_float(float("nan")) is None
    assert parse_strict_optional_float(Decimal("12.50")) == 12.5
    assert parse_strict_optional_float("96.4") == 96.4
    assert parse_strict_optional_int("24") == 24
    assert parse_metadata_empty() == {}

    prof = UserProfileRequest.model_validate(_profile())
    for pol_override in (
        {"deductible_amount": "not_a_number"},
        {"claim_settlement_ratio": "n/a"},
        {"copay_pct": True},
        {"copay_pct": False},
        {"policy_type": "car", "type_specific_metadata": {"vehicle_age_limit_years": "bad"}},
        {"policy_type": "term", "type_specific_metadata": {"life_cover_max": "oops"}},
    ):
        with pytest.raises(InvalidNumericMetadataError):
            extract_policy_features(prof, _policy(**pol_override))


def parse_metadata_empty() -> dict[str, Any]:
    return parse_strict_metadata_mapping(None)


@pytest.mark.parametrize(
    ("case_label", "policy_type", "prof_overrides", "bad_overrides", "expected_field"),
    [
        (
            "deductible_str_no_cap",
            "health",
            {"max_deductible": None},
            {"deductible_amount": "not_a_number"},
            "deductible_amount",
        ),
        (
            "csr_str",
            "health",
            {},
            {"claim_settlement_ratio": "n/a"},
            "claim_settlement_ratio",
        ),
        (
            "copay_bool_no_cap",
            "health",
            {"max_copay_pct": None},
            {"copay_pct": True},
            "copay_pct",
        ),
        (
            "car_meta_limit_no_age",
            "car",
            {"vehicle_type": "sedan", "vehicle_age_years": None},
            {
                "type_specific_metadata": {
                    "vehicle_types": ["sedan"],
                    "vehicle_age_limit_years": "bad",
                }
            },
            "type_specific_metadata.vehicle_age_limit_years",
        ),
        (
            "life_meta_cover",
            "life",
            {
                "min_sum_insured": 1_000_000.0,
                "tenure_years": 20,
                "required_benefits": ["death_benefit"],
            },
            {
                "coverage_mode": "individual",
                "max_family_members": 1,
                "max_tenure_years": 25,
                "covered_benefits": ["death_benefit"],
                "type_specific_metadata": {"life_cover_max": "oops"},
            },
            "type_specific_metadata.life_cover_max",
        ),
        (
            "term_meta_cover",
            "term",
            {
                "min_sum_insured": 1_000_000.0,
                "tenure_years": 20,
                "required_benefits": ["pure_term_life_cover"],
            },
            {
                "coverage_mode": "individual",
                "max_family_members": 1,
                "max_tenure_years": 25,
                "covered_benefits": ["pure_term_life_cover"],
                "type_specific_metadata": {"life_cover_max": "oops"},
            },
            "type_specific_metadata.life_cover_max",
        ),
    ],
)
def test_reproduced_malformed_metadata_excludes_candidate_without_500_or_suppressing_valid(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    model_state: ModelState,
    case_label: str,
    policy_type: str,
    prof_overrides: dict[str, Any],
    bad_overrides: dict[str, Any],
    expected_field: str,
) -> None:
    """All malformed metadata cases fail per-candidate while valid policies rank 200 OK."""
    prof_dict = _profile(policy_type=policy_type, **prof_overrides)
    if policy_type == "car":
        prof_dict["required_benefits"] = []
    prof = UserProfileRequest.model_validate(prof_dict)

    valid_meta: dict[str, Any] = {}
    valid_benefits = ["cashless_hospitalization", "daycare_procedures"]
    if policy_type == "car":
        valid_meta = {"vehicle_types": ["sedan"], "vehicle_age_limit_years": 10}
        valid_benefits = ["own_damage", "third_party_liability"]
    elif policy_type == "life":
        valid_meta = {"life_cover_max": 2_000_000, "policy_term_years_max": 25}
        valid_benefits = ["death_benefit"]
    elif policy_type == "term":
        valid_meta = {"life_cover_max": 5_000_000, "policy_term_years_max": 25}
        valid_benefits = ["pure_term_life_cover"]

    valid_pol = _policy(
        policy_id=f"valid-{case_label}",
        policy_type=policy_type,
        coverage_mode="individual" if policy_type != "health" else "both",
        max_family_members=1 if policy_type != "health" else 4,
        max_tenure_years=25 if policy_type in ("life", "term") else 3,
        covered_benefits=valid_benefits,
        type_specific_metadata=valid_meta,
    )
    bad_pol = _policy(policy_id=f"bad-{case_label}", policy_type=policy_type, **bad_overrides)

    # 1. Pure filtering excludes bad_pol with INVALID_POLICY_METADATA & keeps valid_pol
    f_res = filter_policies(prof, [bad_pol, valid_pol])
    assert f_res.status == "ELIGIBLE_CANDIDATES_FOUND"
    assert [p["policy_id"] for p in f_res.eligible_policies] == [valid_pol["policy_id"]]
    bad_elims = [e for e in f_res.filter_report.eliminations if e.policy_id == bad_pol["policy_id"]]
    assert any(
        e.rule_id == "INVALID_POLICY_METADATA" and expected_field in (e.actual or [])
        for e in bad_elims
    )

    # 2. HTTP & Service paths return 200 RECOMMENDATIONS_READY (NEVER 500) with zero PII in logs
    catalog = [bad_pol, valid_pol]
    app = create_app(Settings(_env_file=None))
    app.dependency_overrides[get_db_connection] = lambda: (yield MagicMock())
    monkeypatch.setattr(
        "ai_service.recommendation.service.load_catalog_policies_from_db",
        lambda _c, policy_type=None, active_only=False: catalog,
    )

    formatter = StructuredJsonFormatter()
    with caplog.at_level(logging.WARNING, logger="ai_service.recommendation.service"):
        with TestClient(app) as client:
            for backend in ("weighted", "lightgbm"):
                caplog.clear()
                res = client.post(
                    "/api/v1/recommend",
                    json={"profile": prof_dict, "ranking_model": backend},
                    headers={"X-Request-ID": f"i2-{case_label}-{backend}"},
                )
                assert res.status_code == 200
                body = res.json()
                assert body["status"] == "RECOMMENDATIONS_READY"
                assert [r["policy_id"] for r in body["recommendations"]] == [valid_pol["policy_id"]]
                assert math.isfinite(body["recommendations"][0]["score"])

                meta_logs = [
                    r
                    for r in caplog.records
                    if getattr(r, "event", None) == "recommendation.policy_metadata_excluded"
                ]
                assert len(meta_logs) == 1
                rendered = formatter.format(meta_logs[0])
                parsed = json.loads(rendered)
                assert parsed["excluded_policy_count"] == 1
                assert expected_field in parsed["invalid_fields"]
                assert "not_a_number" not in rendered
                assert "oops" not in rendered
                assert "1200000" not in rendered

    # 3. Malformed-only catalog returns clean 200 NO_MATCHING_POLICIES instead of 500 crash
    r_only_bad = filter_policies(prof, [bad_pol])
    assert r_only_bad.status == "NO_MATCHING_POLICIES"
    monkeypatch.setattr(
        "ai_service.recommendation.service.load_catalog_policies_from_db",
        lambda _c, policy_type=None, active_only=False: [bad_pol],
    )
    res_only_bad = recommend_policies_for_request(
        RecommendRequest.model_validate({"profile": prof_dict}),
        MagicMock(),
        model_state,
    )
    assert res_only_bad.status == "NO_MATCHING_POLICIES"
    assert res_only_bad.recommendations == []


def test_mixed_catalog_controls_determinism_and_domain_bounds() -> None:
    """Valid numeric controls pass while negative/bool/out-of-range optional fields are excluded."""
    prof = UserProfileRequest.model_validate(_profile())
    pol_valid_zero = _policy(
        policy_id="valid-zero",
        deductible_amount=0,
        copay_pct="0.0",
        claim_settlement_ratio=Decimal("95.5"),
        no_claim_bonus_pct=Decimal("0.00"),
    )
    pol_valid_float = _policy(
        policy_id="valid-float",
        deductible_amount=5000.0,
        copay_pct=10.5,
        claim_settlement_ratio=None,
        no_claim_bonus_pct=50,
    )
    pol_bad_neg_deduct = _policy(policy_id="bad-neg-deduct", deductible_amount=-100)
    pol_bad_csr = _policy(policy_id="bad-csr-gt100", claim_settlement_ratio=105)
    pol_bad_json = _policy(policy_id="bad-json-meta", type_specific_metadata="[not_an_object")

    mixed = [pol_bad_neg_deduct, pol_valid_zero, pol_bad_csr, pol_bad_json, pol_valid_float]
    r1 = filter_policies(prof, mixed)
    r2 = filter_policies(prof, mixed)
    assert r1.model_dump() == r2.model_dump()
    assert [p["policy_id"] for p in r1.eligible_policies] == ["valid-zero", "valid-float"]
    assert r1.filter_report.eliminated_count == 3
    for bad_id in ("bad-neg-deduct", "bad-csr-gt100", "bad-json-meta"):
        assert any(
            e.policy_id == bad_id and e.rule_id == "INVALID_POLICY_METADATA"
            for e in r1.filter_report.eliminations
        )


# ---------------------------------------------------------------------------
# M5.I2 Final Hardening — Boolean Normalization & Cross-Layer Validation Consistency
# ---------------------------------------------------------------------------


def test_boolean_normalization_supported_and_invalid_tokens() -> None:
    """Shared bool normalization maps valid tokens consistently and rejects garbage as None."""
    for truthy in (True, np.bool_(True), 1, "true", "TRUE", " true ", "1", "yes", "Yes"):
        assert parse_optional_bool(truthy) is True
        assert parse_bool(truthy, default=False) is True

    for falsy in (False, np.bool_(False), 0, "false", "FALSE", " 0 ", "no", "No"):
        assert parse_optional_bool(falsy) is False
        assert parse_bool(falsy, default=True) is False

    for missing in (None, "", "   ", float("nan")):
        assert parse_optional_bool(missing) is None
        assert parse_bool(missing, default=False) is False
        assert parse_bool(missing, default=True) is True

    # Invalid non-boolean values must NEVER silently coerce to True via bool(value)
    for invalid in ("maybe", "invalid", "n/a", "unknown", 2, -1, 1.5, ["true"], {"v": 1}):
        assert parse_optional_bool(invalid) is None
        assert parse_bool(invalid, default=False) is False


def test_ped_covered_boolean_equivalence_and_invalid_values_across_layers(
    monkeypatch: pytest.MonkeyPatch,
    model_state: ModelState,
) -> None:
    """Native and string bools produce identical PED scores, warnings, and factual highlights."""
    prof_ped = UserProfileRequest.model_validate(_profile(pre_existing_conditions=["diabetes"]))
    prof_no_ped = UserProfileRequest.model_validate(_profile(pre_existing_conditions=[]))

    equiv_policies = [
        _policy(policy_id="ped-bool", ped_covered=True, ped_waiting_period_months=None),
        _policy(policy_id="ped-str-true", ped_covered="true", ped_waiting_period_months=None),
        _policy(policy_id="ped-str-yes", ped_covered=" Yes ", ped_waiting_period_months=None),
        _policy(policy_id="ped-str-1", ped_covered="1", ped_waiting_period_months=None),
        _policy(policy_id="ped-int-1", ped_covered=1, ped_waiting_period_months=None),
    ]

    # Invalid/falsy PED strings fail PED_COVERAGE when PED is required (never bool("maybe")==True)
    bad_ped_pol = _policy(
        policy_id="ped-invalid", ped_covered="maybe", ped_waiting_period_months=None
    )
    false_ped_pol = _policy(
        policy_id="ped-str-false", ped_covered="false", ped_waiting_period_months=None
    )
    f_ped = filter_policies(prof_ped, [*equiv_policies, bad_ped_pol, false_ped_pol])
    assert [p["policy_id"] for p in f_ped.eligible_policies] == [
        p["policy_id"] for p in equiv_policies
    ]
    assert {e.policy_id for e in f_ped.filter_report.eliminations} == {
        "ped-invalid",
        "ped-str-false",
    }

    # Feature matrix parity across equivalent covered tokens vs false/invalid when PED not required
    feats_equiv = build_feature_matrix(prof_no_ped, equiv_policies)
    assert list(feats_equiv["ped_coverage_indicator"]) == [1.0] * len(equiv_policies)
    assert list(feats_equiv["ped_waiting_period_days"]) == [1080.0] * len(equiv_policies)

    for non_cov_pol in (false_ped_pol, bad_ped_pol):
        feats_nc = extract_policy_features(prof_no_ped, non_cov_pol)
        assert feats_nc["ped_coverage_indicator"] == 0.0
        assert feats_nc["ped_waiting_period_days"] == 0.0

    # Service path: every valid truthy representation triggers identical score, warning & highlight
    expected_warning_substr = "conservative 36-month (1080-day) fallback"
    for pol in equiv_policies:
        monkeypatch.setattr(
            "ai_service.recommendation.service.load_catalog_policies_from_db",
            lambda _c, policy_type=None, active_only=False, row=pol: [row],
        )
        for backend in ("weighted", "lightgbm"):
            resp = recommend_policies_for_request(
                RecommendRequest.model_validate({"profile": _profile(), "ranking_model": backend}),
                MagicMock(),
                model_state,
            )
            assert resp.status == "RECOMMENDATIONS_READY"
            assert any(expected_warning_substr in w for w in resp.warnings), pol["policy_id"]
            highlights = resp.recommendations[0].key_highlights
            assert "Pre-existing conditions covered" in highlights
            assert not any("36-month" in h for h in highlights)

    # Service path with invalid/falsy ped_covered receives no PED credit, warning, or highlight
    for non_cov_pol in (false_ped_pol, bad_ped_pol):
        monkeypatch.setattr(
            "ai_service.recommendation.service.load_catalog_policies_from_db",
            lambda _c, policy_type=None, active_only=False, row=non_cov_pol: [row],
        )
        resp_nc = recommend_policies_for_request(
            RecommendRequest.model_validate({"profile": _profile(), "ranking_model": "weighted"}),
            MagicMock(),
            model_state,
        )
        assert not any(expected_warning_substr in w for w in resp_nc.warnings)
        assert "Pre-existing conditions covered" not in resp_nc.recommendations[0].key_highlights


def test_other_policy_boolean_flags_normalized_consistently(
    monkeypatch: pytest.MonkeyPatch,
    model_state: ModelState,
) -> None:
    """All boolean highlight flags handle native, string, falsy, and invalid tokens consistently."""
    prof = UserProfileRequest.model_validate(_profile())
    pol_native = _policy(
        policy_id="bool-native",
        cashless_available=True,
        lifelong_renewal=True,
        restoration_benefit=True,
        pan_india_coverage=True,
    )
    pol_strings = _policy(
        policy_id="bool-strings",
        cashless_available="yes",
        lifelong_renewal="TRUE",
        restoration_benefit="1",
        pan_india_coverage=1,
    )
    pol_invalid = _policy(
        policy_id="bool-invalid",
        cashless_available="maybe",
        lifelong_renewal="invalid",
        restoration_benefit="no_idea",
        pan_india_coverage=2,
    )

    # Filter parity: valid string booleans pass cashless/lifelong/pan-india; invalid fails closed
    prof_strict = UserProfileRequest.model_validate(
        _profile(require_cashless=True, require_lifelong_renewal=True, location="pune")
    )
    f_strict = filter_policies(prof_strict, [pol_native, pol_strings, pol_invalid])
    assert [p["policy_id"] for p in f_strict.eligible_policies] == ["bool-native", "bool-strings"]

    assert extract_policy_features(prof, pol_native) == extract_policy_features(prof, pol_strings)

    feats_inv = extract_policy_features(prof, pol_invalid)
    assert feats_inv["cashless_availability_indicator"] == 0.0
    assert feats_inv["restoration_benefit_indicator"] == 0.0
    assert feats_inv["pan_india_coverage_indicator"] == 0.0
    assert feats_inv["remaining_eligible_years"] == 53.0  # uses max_renewal_age 85 (not lifelong)

    highlights_native = _extract_factual_highlights(pol_native)
    highlights_strings = _extract_factual_highlights(pol_strings)
    assert highlights_native == highlights_strings
    assert "Cashless network: 7,500 hospitals/garages" in highlights_strings
    assert "Lifelong renewal supported" in highlights_strings
    assert "Sum insured restoration benefit" in highlights_strings
    assert "Pan-India coverage" in highlights_strings

    highlights_inv = _extract_factual_highlights(pol_invalid)
    assert not any("Cashless" in h for h in highlights_inv)
    assert "Lifelong renewal supported" not in highlights_inv
    assert "Sum insured restoration benefit" not in highlights_inv
    assert "Pan-India coverage" not in highlights_inv

    # End-to-end service highlights consistency
    monkeypatch.setattr(
        "ai_service.recommendation.service.load_catalog_policies_from_db",
        lambda _c, policy_type=None, active_only=False: [pol_strings],
    )
    resp = recommend_policies_for_request(
        RecommendRequest.model_validate({"profile": _profile(), "ranking_model": "weighted"}),
        MagicMock(),
        model_state,
    )
    assert resp.recommendations[0].key_highlights == highlights_native


@pytest.mark.parametrize(
    "bad_overrides",
    [
        # Fractional values for Stage 4 integer fields
        {"ped_waiting_period_months": 24.5},
        {"initial_waiting_period_days": 15.5},
        {"network_hospitals_count": 100.25},
        {"max_renewal_age": 80.5},
        {"max_tenure_years": 3.5},
        {"min_entry_age": 18.5},
        # Out-of-range optional numerics
        {"deductible_amount": -100},
        {"annual_premium": -50.0},
        {"room_rent_limit_value": -1.0},
        {"claim_settlement_ratio": -0.1},
        {"claim_settlement_ratio": 100.1},
        {"copay_pct": -0.5},
        {"copay_pct": 100.5},
        {"no_claim_bonus_pct": -1.0},
        {"no_claim_bonus_pct": 200.5},
        # Out-of-range / inverted mandatory bounds
        {"sum_insured_max": 0},
        {"sum_insured_max": -500},
        {"annual_premium_min": 20000, "annual_premium_max": 10000},
        {"min_entry_age": 70, "max_entry_age": 60},
        {"min_entry_age": -1},
        {"max_entry_age": 125},
        # Boolean values in numeric fields
        {"deductible_amount": True},
        {"copay_pct": False},
        {"ped_waiting_period_months": True},
        {"claim_settlement_ratio": False},
        {"min_entry_age": True},
        {"annual_premium_min": True},
        # Out-of-range / malformed type_specific_metadata
        {"policy_type": "car", "type_specific_metadata": {"vehicle_age_limit_years": -1}},
        {"policy_type": "life", "type_specific_metadata": {"life_cover_min": -10}},
        {"policy_type": "term", "type_specific_metadata": {"policy_term_years_max": -5}},
    ],
)
def test_stage4_and_stage5_reject_same_malformed_and_out_of_range_numerics(
    bad_overrides: dict[str, Any],
) -> None:
    """Direct Stage 5 extraction enforces the exact same coercion and bounds as Stage 4 filters."""
    ptype = str(bad_overrides.get("policy_type", "health"))
    prof = UserProfileRequest.model_validate(_profile(policy_type=ptype))
    bad_pol = _policy(**bad_overrides)

    # Stage 4 filters must reject every single one without crashing
    f_res = filter_policies(prof, [bad_pol])
    assert f_res.status == "NO_MATCHING_POLICIES"
    assert f_res.eligible_policies == []

    # Direct Stage 5 calls must reject malformed or out-of-range inputs
    with pytest.raises(InvalidNumericMetadataError):
        extract_policy_features(prof, bad_pol)
    with pytest.raises(InvalidNumericMetadataError):
        build_feature_matrix(prof, [bad_pol])


def test_factual_highlights_handles_invalid_inputs_safely_without_zero_coercion() -> None:
    """_extract_factual_highlights never crashes or treats malformed inputs as zeros."""
    malformed_pol = _policy(
        cashless_available="true",
        network_hospitals_count="bad_count",
        claim_settlement_ratio=True,
        ped_covered="yes",
        ped_waiting_period_months=24.5,
        copay_pct=False,
        lifelong_renewal="true",
        restoration_benefit="1",
        pan_india_coverage="yes",
    )
    highlights = _extract_factual_highlights(malformed_pol)
    assert "Cashless claim facility available" in highlights
    assert not any("hospitals/garages" in h for h in highlights)
    assert not any("Claim settlement ratio" in h for h in highlights)
    assert "Pre-existing conditions covered" in highlights
    assert not any("24" in h or "36-month" in h or "0-month" in h for h in highlights)
    assert "Zero co-payment" not in highlights
    assert "Lifelong renewal supported" in highlights
    assert "Sum insured restoration benefit" in highlights
    assert "Pan-India coverage" in highlights

    for bad_pct in ("not_a_number", -10, 100.5, 150, True, False, float("inf"), float("nan")):
        h_pct = _extract_factual_highlights(
            _policy(claim_settlement_ratio=bad_pct, copay_pct=bad_pct)
        )
        assert "Zero co-payment" not in h_pct
        assert not any("Claim settlement ratio" in h for h in h_pct)

    for bad_int in ("not_a_number", -10, 24.5, True, False, float("inf"), float("nan")):
        h_int = _extract_factual_highlights(
            _policy(
                cashless_available="true",
                network_hospitals_count=bad_int,
                ped_covered="true",
                ped_waiting_period_months=bad_int,
            )
        )
        assert "Cashless claim facility available" in h_int
        assert not any("hospitals/garages" in h for h in h_int)
        assert "Pre-existing conditions covered" in h_int
        assert not any("waiting period" in h for h in h_int)


def test_valid_missing_optional_and_multi_category_numeric_consistency(
    monkeypatch: pytest.MonkeyPatch,
    model_state: ModelState,
) -> None:
    """Valid missing fields and numeric boundaries work identically across all 4 categories."""
    cases = [
        (
            "health",
            {},
            {
                "deductible_amount": Decimal("0.00"),
                "copay_pct": "0",
                "claim_settlement_ratio": "98.5",
                "no_claim_bonus_pct": 200,
                "max_renewal_age": None,
                "room_rent_limit_value": None,
                "ped_waiting_period_months": None,
                "network_hospitals_count": "0",
                "covered_benefits": ["cashless_hospitalization"],
            },
        ),
        (
            "car",
            {"vehicle_type": "sedan", "vehicle_age_years": 3, "required_benefits": []},
            {
                "coverage_mode": "individual",
                "max_family_members": 1,
                "claim_settlement_ratio": None,
                "deductible_amount": None,
                "copay_pct": None,
                "type_specific_metadata": {
                    "vehicle_types": ["sedan"],
                    "vehicle_age_limit_years": 10,
                    "own_damage_coverage": "true",
                },
            },
        ),
        (
            "life",
            {"min_sum_insured": 1_000_000.0, "tenure_years": 20, "required_benefits": []},
            {
                "coverage_mode": "individual",
                "max_family_members": 1,
                "max_tenure_years": 25,
                "deductible_amount": None,
                "copay_pct": None,
                "type_specific_metadata": {
                    "life_cover_min": Decimal("500000"),
                    "life_cover_max": "2000000",
                    "policy_term_years_max": "25",
                    "maturity_benefit": "yes",
                },
            },
        ),
        (
            "term",
            {"min_sum_insured": 2_000_000.0, "tenure_years": 20, "required_benefits": []},
            {
                "coverage_mode": "individual",
                "max_family_members": 1,
                "max_tenure_years": 30,
                "sum_insured_max": Decimal("5000000.00"),
                "type_specific_metadata": {
                    "life_cover_min": 1_000_000,
                    "life_cover_max": 5_000_000,
                    "policy_term_years_max": 30,
                    "critical_illness_rider": "true",
                },
            },
        ),
    ]

    for ptype, prof_over, pol_over in cases:
        prof_dict = _profile(policy_type=ptype, **prof_over)
        prof = UserProfileRequest.model_validate(prof_dict)
        pol = _policy(policy_id=f"valid-{ptype}", policy_type=ptype, **pol_over)

        f_res = filter_policies(prof, [pol])
        assert f_res.status == "ELIGIBLE_CANDIDATES_FOUND", ptype
        matrix = build_feature_matrix(prof, f_res.eligible_policies)
        assert matrix.shape == (1, 59)
        assert np.isfinite(matrix.to_numpy()).all()

        monkeypatch.setattr(
            "ai_service.recommendation.service.load_catalog_policies_from_db",
            lambda _c, policy_type=None, active_only=False, row=pol: [row],
        )
        resp = recommend_policies_for_request(
            RecommendRequest.model_validate({"profile": prof_dict, "ranking_model": "lightgbm"}),
            MagicMock(),
            model_state,
        )
        assert resp.status == "RECOMMENDATIONS_READY"
        assert len(resp.recommendations) == 1
        assert math.isfinite(resp.recommendations[0].score)

    verify_openapi_contract()
