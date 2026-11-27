"""Unit tests for Module 6A Pydantic schemas and endpoint route aliases."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from ai_service.main import create_app
from ai_service.recommendation.db import get_db_connection
from ai_service.recommendation.schemas import RecommendRequest, UserProfileRequest
from ai_service.recommendation.settings import Settings


def test_user_profile_normalizes_and_deduplicates_tags() -> None:
    """Ensure policy_type and benefit/condition lists normalize cleanly."""
    profile = UserProfileRequest(
        policy_type="  Health ",
        age=32,
        budget_min=8000,
        budget_max=20000,
        applicant_type=" Salaried ",
        min_sum_insured=500000,
        required_benefits=["Cashless", "cashless", "  Maternity  "],
        pre_existing_conditions=["Diabetes", "diabetes"],
    )
    assert profile.policy_type == "health"
    assert profile.applicant_type == "salaried"
    assert profile.required_benefits == ["cashless", "maternity"]
    assert profile.pre_existing_conditions == ["diabetes"]


def test_user_profile_rejects_invalid_budget_and_family_size() -> None:
    """Ensure cross-field validation catches budget inversion and individual size > 1."""
    with pytest.raises(ValidationError, match="budget_min cannot exceed budget_max"):
        UserProfileRequest(
            policy_type="health",
            age=30,
            budget_min=25000,
            budget_max=15000,
            min_sum_insured=500000,
        )

    with pytest.raises(ValidationError, match="family_size > 1"):
        UserProfileRequest(
            policy_type="health",
            age=30,
            budget_max=20000,
            coverage_mode="individual",
            family_size=3,
            min_sum_insured=500000,
        )


def test_recommend_request_accepts_both_nested_and_flat_payloads() -> None:
    """Support both {profile, scope} contract and flat profile JSON."""
    nested = RecommendRequest.model_validate(
        {
            "profile": {
                "policy_type": "term",
                "age": 28,
                "budget_max": 15000,
                "min_sum_insured": 10000000,
            },
            "scope": {"user_id": "u-1", "role": "user"},
            "top_n": 3,
        }
    )
    flat = RecommendRequest.model_validate(
        {
            "policy_type": "term",
            "age": 28,
            "budget_max": 15000,
            "min_sum_insured": 10000000,
            "top_n": 3,
        }
    )
    assert nested.profile.policy_type == "term"
    assert flat.profile.policy_type == "term"
    assert nested.top_n == flat.top_n == 3


def test_both_recommend_routes_validate_and_execute_inference(monkeypatch) -> None:
    """Verify /api/v1/recommend and /ai/recommend validate schemas and invoke Stage 7 inference."""
    app = create_app(Settings(_env_file=None))

    def override_conn():
        yield MagicMock()

    app.dependency_overrides[get_db_connection] = override_conn
    monkeypatch.setattr(
        "ai_service.recommendation.service.load_catalog_policies_from_db",
        lambda _conn, active_only=False: [],
    )

    valid_payload = {
        "profile": {
            "policy_type": "health",
            "age": 35,
            "budget_min": 5000,
            "budget_max": 25000,
            "min_sum_insured": 500000,
        },
        "top_n": 5,
    }
    invalid_payload = {"profile": {"policy_type": "invalid_type", "age": 35}}

    with TestClient(app) as client:
        for route in ("/api/v1/recommend", "/ai/recommend"):
            res = client.post(route, json=valid_payload)
            assert res.status_code == 200
            body = res.json()
            assert body["status"] == "NO_MATCHING_POLICIES"

            bad = client.post(route, json=invalid_payload)
            assert bad.status_code == 422


def test_user_profile_accepts_optional_personalization_fields() -> None:
    """Ensure Stage 7 annual_income_inr and risk_preference normalize and remain optional."""
    without_optional = UserProfileRequest(
        policy_type="health",
        age=30,
        budget_max=20000,
        min_sum_insured=500000,
    )
    assert without_optional.annual_income_inr is None
    assert without_optional.risk_preference is None

    with_optional = UserProfileRequest(
        policy_type="health",
        age=30,
        budget_max=20000,
        min_sum_insured=500000,
        annual_income_inr=1_200_000.0,
        risk_preference=" Comprehensive_Coverage ",
    )
    assert with_optional.annual_income_inr == 1_200_000.0
    assert with_optional.risk_preference == "comprehensive_coverage"
