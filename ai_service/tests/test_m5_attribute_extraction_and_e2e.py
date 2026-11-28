"""Checkpoint 5.0 Unit & Endpoint Tests for Grounded Policy Attribute Extraction and M5 E2E."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from ai_service.config import Settings
from ai_service.llm.providers.nvidia_nemotron import NvidiaAuthenticationError, NvidiaTimeoutError
from ai_service.main import create_app
from ai_service.recommendation.attribute_extractor import PolicyAttributeExtractor
from ai_service.recommendation.citations import PolicyCitationRequest, PostgresEvidenceCitationProvider
from ai_service.recommendation.db import get_db_connection
from ai_service.recommendation.settings import Settings as RecSettings
from ai_service.schemas.structured_output import ChunkCategory, PolicyChunk
from ai_service.workers.pipeline import classify_ingestion_error


def _chunk(
    chunk_id: str,
    text: str,
    *,
    page: int = 1,
    clause: str = "Section 1",
    category: ChunkCategory = ChunkCategory.COVERAGE,
) -> PolicyChunk:
    return PolicyChunk(
        chunk_id=chunk_id,
        policy_id="pol-attr-test",
        clause_ref=clause,
        page_number=page,
        page_start=page,
        category=category,
        content_text=text,
    )


def test_offline_attribute_extraction_populates_grounded_fields_and_preserves_missing() -> None:
    chunks = [
        _chunk(
            "chk-sched-1",
            (
                "Entry Age: 18 to 65 years. Lifelong renewal available. "
                "Coverage Mode: Individual and Family Floater up to 4 family members. "
                "Eligible applicants: Salaried and Self-Employed. "
                "Annual Premium: INR 12,000 to 18,000. "
                "Sum Insured: INR 500,000 to 1,500,000. Policy Tenure: 1 to 3 years."
            ),
            page=1,
            clause="Section 1 Schedule",
        ),
        _chunk(
            "chk-excl-2",
            (
                "Initial waiting period: 30 days. Pre-existing conditions covered after 24 months. "
                "Cashless facility available at 7,500 network hospitals Pan-India. "
                "Cosmetic surgery is excluded and not covered."
            ),
            page=2,
            clause="Section 2 Benefits & Exclusions",
            category=ChunkCategory.EXCLUSIONS,
        ),
    ]

    extractor = PolicyAttributeExtractor(
        app_settings=Settings(APP_ENV="development", LLM_PROVIDER="offline", USE_MOCK_STORAGE=True)
    )
    res = extractor.extract_from_chunks(chunks, policy_type="health")

    assert res.extraction_status == "completed"
    assert res.extraction_source == "ingestion_deterministic"
    assert res.min_entry_age == 18
    assert res.max_entry_age == 65
    assert res.lifelong_renewal is True
    assert res.coverage_mode == "both"
    assert res.max_family_members == 4
    assert res.annual_premium_min == 12000.0
    assert res.annual_premium_max == 18000.0
    assert res.sum_insured_max == 1500000.0
    assert res.min_tenure_years == 1
    assert res.max_tenure_years == 3
    assert "salaried" in res.eligible_applicant_types
    assert "cosmetic_surgery" in res.exclusions

    # Attributes absent from PDF chunks MUST stay None (never fabricated defaults)
    assert res.copay_pct is None
    assert res.room_rent_limit_type is None
    assert res.deductible_amount is None

    assert set(res.source_chunk_ids) == {"chk-sched-1", "chk-excl-2"}
    assert "annual_premium_min" in res.field_provenance
    assert res.field_provenance["annual_premium_min"][0]["chunk_id"] == "chk-sched-1"
    assert res.field_provenance["exclusions"][0]["chunk_id"] == "chk-excl-2"
    assert res.field_provenance["exclusions"][0]["page"] == 2


def test_partial_extraction_and_insufficient_data_status() -> None:
    extractor = PolicyAttributeExtractor(
        app_settings=Settings(APP_ENV="development", LLM_PROVIDER="offline", USE_MOCK_STORAGE=True)
    )

    partial_res = extractor.extract_from_chunks(
        [_chunk("chk-part", "Entry Age: 18 to 60 years. Deductible: INR 5,000.")],
        policy_type="health",
    )
    assert partial_res.extraction_status == "partial"
    assert partial_res.min_entry_age == 18
    assert partial_res.deductible_amount == 5000.0
    assert partial_res.annual_premium_min is None

    empty_res = extractor.extract_from_chunks(
        [_chunk("chk-none", "Welcome to our general customer brochure welcome letter.")],
        policy_type="health",
    )
    assert empty_res.extraction_status == "insufficient_data"
    assert empty_res.field_provenance == {}
    assert empty_res.source_chunk_ids == []


def test_llm_extraction_drops_uncited_or_forged_chunk_fields() -> None:
    class MockLlmClient:
        provider_name = "nvidia"

        @staticmethod
        def generate_json(_system_prompt: str, _user_prompt: str):
            return {
                "attributes": {
                    "min_entry_age": 21,
                    "max_entry_age": 60,
                    "annual_premium_min": 14000,
                    "annual_premium_max": 20000,
                    "sum_insured_max": 9999999,
                    "copay_pct": 15,
                    "exclusions": ["self_inflicted_injury"],
                },
                "provenance": {
                    "min_entry_age": ["chk-real-1"],
                    "max_entry_age": ["chk-real-1"],
                    "annual_premium_min": ["chk-real-1"],
                    "annual_premium_max": ["chk-real-1"],
                    # Fabricated chunk ID -> MUST be dropped!
                    "sum_insured_max": ["chk-hallucinated-999"],
                    # Missing citation list -> MUST be dropped!
                    "copay_pct": [],
                    "exclusions": ["chk-real-1"],
                },
                "confidence": 0.91,
            }

    extractor = PolicyAttributeExtractor(
        llm_client=MockLlmClient(),
        app_settings=Settings(
            APP_ENV="development",
            LLM_PROVIDER="nvidia",
            NVIDIA_API_KEY="nvapi-test-only",
            USE_MOCK_STORAGE=True,
        ),
    )
    res = extractor.extract_from_chunks(
        [
            _chunk(
                "chk-real-1",
                "Entry age 21 to 60 years. Annual premium INR 14,000 to 20,000.",
            )
        ],
        policy_type="health",
    )

    assert res.extraction_source == "ingestion_llm"
    assert res.min_entry_age == 21
    assert res.annual_premium_min == 14000.0
    assert res.exclusions == ["self_inflicted_injury"]
    # Hallucinated or uncited attributes rejected
    assert res.sum_insured_max is None
    assert res.copay_pct is None
    assert res.source_chunk_ids == ["chk-real-1"]
    assert res.extraction_status == "partial"


def test_attribute_extraction_provider_errors_classify_without_silent_success() -> None:
    cfg_retryable = classify_ingestion_error(NvidiaTimeoutError("Nvidia timeout"))
    assert cfg_retryable[0] is True

    cfg_perm = classify_ingestion_error(NvidiaAuthenticationError("bad api key"))
    assert cfg_perm[0] is False
    assert cfg_perm[1] == "ATTRIBUTE_EXTRACTION_CONFIG_ERROR"


def test_postgres_evidence_citation_provider_queries_chunks_only() -> None:
    policy_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())
    chunk_id = "chunk-sched-001"

    class FakeResult:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

    class FakeConn:
        def __init__(self):
            self.queries = []

        def execute(self, sql, params):
            self.queries.append((sql, params))
            if "ANY(%(chunk_keys)s::text[])" in sql:
                return FakeResult(
                    [
                        {
                            "policy_id": policy_id,
                            "document_id": doc_id,
                            "version_number": 2,
                            "chunk_key": chunk_id,
                            "page_number": 3,
                            "page_start": 3,
                            "page_end": 3,
                            "clause_ref": "Section 3.2 Premium",
                            "category": "Coverage",
                            "content_text": "Annual premium INR 12,000 for individual coverage.",
                        }
                    ]
                )
            return FakeResult([])

    fake_conn = FakeConn()

    class FakePoolCtx:
        def __enter__(self):
            return fake_conn

        def __exit__(self, *_a):
            return False

    class FakePoolMgr:
        @staticmethod
        def connection():
            return FakePoolCtx()

    provider = PostgresEvidenceCitationProvider(pool_manager=FakePoolMgr(), max_citations_per_policy=3)
    citations_map = provider.get_citations(
        [
            PolicyCitationRequest(
                policy_id=policy_id,
                policy_version_id=doc_id,
                document_id=doc_id,
                document_version=2,
                policy_name="Verified Health Plan",
                policy_type="health",
                rank=1,
                score=1.8,
                model_used="lightgbm",
                field_provenance={"annual_premium_min": [{"chunk_id": chunk_id}]},
            )
        ]
    )
    assert len(citations_map[policy_id]) == 1
    cit = citations_map[policy_id][0]
    assert cit["chunk_id"] == chunk_id
    assert cit["page"] == 3
    assert cit["document_version"] == 2
    assert cit["clause"] == "Section 3.2 Premium"
    assert "policy_chunks" in fake_conn.queries[0][0]


def test_fastapi_recommend_auth_and_grounded_missing_attribute_disclosure(monkeypatch) -> None:
    rec_settings = RecSettings(
        _env_file=None,
        app_env="staging",
        internal_service_token="secret-m5-token",
    )
    app = create_app(rec_settings)

    def override_conn():
        yield object()

    app.dependency_overrides[get_db_connection] = override_conn

    valid_policy_id = str(uuid.uuid4())
    invalid_policy_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())

    fake_candidates = [
        {
            "policy_id": valid_policy_id,
            "policy_version_id": doc_id,
            "document_id": doc_id,
            "document_version": 1,
            "extraction_status": "partial",
            "policy_name": "Grounded Shield Health",
            "insurer_name": "Shield Insurer",
            "policy_type": "health",
            "visibility": "catalog",
            "status": "active",
            "is_catalog_eligible": True,
            "min_entry_age": 18,
            "max_entry_age": 65,
            "lifelong_renewal": True,
            "coverage_mode": "both",
            "max_family_members": 4,
            "eligible_applicant_types": ["salaried"],
            "annual_premium_min": 12000.0,
            "annual_premium_max": 18000.0,
            "sum_insured_min": 500000.0,
            "sum_insured_max": 1500000.0,
            "min_tenure_years": 1,
            "max_tenure_years": 3,
            # Explicitly missing copay_pct, deductible_amount, room_rent_limit_type -> MUST be disclosed!
            "copay_pct": None,
            "deductible_amount": None,
            "room_rent_limit_type": None,
            "exclusions": ["cosmetic_surgery"],
            "covered_benefits": ["cashless_hospitalization"],
            "field_provenance": {"annual_premium_min": [{"chunk_id": "chk-valid-1"}]},
            "source_chunk_ids": ["chk-valid-1"],
            "type_specific_metadata": {},
        },
        {
            "policy_id": invalid_policy_id,
            "policy_version_id": str(uuid.uuid4()),
            "policy_name": "Corrupt Metadata Plan",
            "policy_type": "health",
            "min_entry_age": 70,
            "max_entry_age": 20,  # inverted impossible age range
            "eligible_applicant_types": ["salaried"],
            "annual_premium_min": 10000.0,
            "annual_premium_max": 15000.0,
            "sum_insured_max": 500000.0,
            "min_tenure_years": 1,
            "max_tenure_years": 2,
        },
    ]

    monkeypatch.setattr(
        "ai_service.recommendation.service.load_catalog_policies_from_db",
        lambda _conn, **_kwargs: fake_candidates,
    )

    class StubCitationProvider:
        def get_citations(self, requests, *, request_id=None):
            del request_id
            return {
                req.policy_id: [
                    {
                        "chunk_id": "chk-valid-1",
                        "page": 1,
                        "clause": "Section 1 Schedule",
                        "snippet": "Annual premium INR 12,000 to 18,000.",
                    }
                ]
                for req in requests
            }

    app.state.citation_provider = StubCitationProvider()

    client = TestClient(app)
    body = {
        "profile": {
            "policy_type": "health",
            "age": 32,
            "budget_min": 5000,
            "budget_max": 25000,
            "min_sum_insured": 500000,
            "applicant_type": "salaried",
            "coverage_mode": "individual",
        }
    }

    # 1. Missing service token rejected cleanly (401) on both endpoints
    unauth = client.post("/ai/recommend", json=body)
    assert unauth.status_code == 401
    unauth_backfill = client.post(
        "/internal/recommendations/extract-attributes",
        json={"policy_id": valid_policy_id, "document_id": doc_id},
    )
    assert unauth_backfill.status_code == 401

    # 2. Authorized request returns ONLY valid grounded policy with missing attribute warnings + citations
    auth = client.post(
        "/ai/recommend",
        json=body,
        headers={"X-Service-Token": "secret-m5-token"},
    )
    assert auth.status_code == 200
    payload = auth.json()
    assert payload["status"] == "RECOMMENDATIONS_READY"
    assert payload["filter_report"]["total_candidates"] == 2
    assert payload["filter_report"]["passed_count"] == 1
    assert payload["filter_report"]["eliminated_count"] == 1
    assert len(payload["recommendations"]) == 1

    rec = payload["recommendations"][0]
    assert rec["policy_id"] == valid_policy_id
    assert rec["extraction_status"] == "partial"
    assert "copay_pct" in rec["missing_attributes"]
    assert "cosmetic_surgery" in rec["exclusions"]
    assert len(rec["citations"]) == 1
    assert rec["citations"][0]["chunk_id"] == "chk-valid-1"
