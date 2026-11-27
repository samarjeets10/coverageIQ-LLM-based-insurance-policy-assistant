import json
import logging
from typing import List

import httpx
import pytest
from fastapi.testclient import TestClient

from ai_service.api import routes_qa
from ai_service.config import Settings
from ai_service.database import MockKnowledgeBase
from ai_service.evaluator.decision_gate import DeterministicDecisionGate
from ai_service.evaluator.retry_controller import ClosedRetryLoopOrchestrator
from ai_service.evaluator.verifier import EvaluatorAgent, find_exclusion_contradictions
from ai_service.llm.client import LLMClient, LLMValidationError
from ai_service.llm.generator import GroundedGenerator
from ai_service.llm.providers.nvidia_nemotron import (
    NvidiaConfigurationError,
    NvidiaNemotronProvider,
)
from ai_service.main import app
from ai_service.schemas.structured_output import (
    AtomicClaim,
    ChunkCategory,
    DecisionType,
    EvaluatorResult,
    PolicyChunk,
)


def _verdict_json(
    *,
    verdict: str = "PASS",
    confidence: float = 0.95,
    evidence_supported: bool = True,
    contradiction_found: bool = False,
    completeness_passed: bool = True,
    claim_verdicts=None,
    issues=None,
    summary: str = "Audit completed.",
) -> str:
    return json.dumps({
        "verdict": verdict,
        "confidence_score": confidence,
        "evidence_supported": evidence_supported,
        "contradiction_found": contradiction_found,
        "completeness_passed": completeness_passed,
        "claim_verdicts": claim_verdicts if claim_verdicts is not None else [],
        "issues": issues if issues is not None else [],
        "summary": summary,
    })


def _sample_chunk(chunk_id: str = "chunk_1", category: ChunkCategory = ChunkCategory.COVERAGE) -> PolicyChunk:
    return PolicyChunk(
        chunk_id=chunk_id,
        policy_id="pol_test",
        clause_ref="Section 2.1",
        page_number=5,
        page_start=5,
        page_end=5,
        category=category,
        content_text="Hospitalization for >= 24 hours is covered.",
    )


# =============================================================================
# 1. HTTP Client Lifecycle Tests
# =============================================================================
def test_evaluator_reuses_managed_client_across_repeated_verifications(monkeypatch):
    created_clients: List[LLMClient] = []
    orig_init = LLMClient.__init__

    def tracked_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        created_clients.append(self)

    monkeypatch.setenv("LLM_PROVIDER", "offline")
    monkeypatch.setattr(LLMClient, "__init__", tracked_init)

    evaluator = EvaluatorAgent(confidence_threshold=0.70)
    chunk = _sample_chunk("chunk_h_001")
    draft = {
        "decision": "COVERED",
        "reason": "Covered under Section 2.1.",
        "conditions": [],
        "claims": [AtomicClaim(text="Hospitalization covered.", chunk_ids=["chunk_h_001"], claim_type="coverage")],
    }

    for _ in range(5):
        res = evaluator.verify("Is hospitalization covered?", [chunk], draft)
        assert res.passed is True

    assert len(created_clients) == 1
    evaluator.close()
    assert evaluator._client is None


def test_closed_nvidia_provider_cannot_be_reused():
    provider = NvidiaNemotronProvider(api_key="nvapi-test-key", base_url="https://integrate.api.nvidia.com/v1")
    assert provider.is_closed is False
    provider.close()
    assert provider.is_closed is True
    with pytest.raises(NvidiaConfigurationError, match="closed"):
        provider.complete("sys", "question")


def test_fastapi_lifespan_closes_and_resets_orchestrator(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-lifespan-key")
    routes_qa.close_orchestrator()

    with TestClient(app) as client:
        resp = client.get("/ai/health")
        assert resp.status_code == 200
        orch_1 = routes_qa.get_orchestrator()
        gen_provider_1 = orch_1.generator.client._nvidia_provider
        eval_client_1 = orch_1.evaluator._get_client()
        eval_provider_1 = eval_client_1._nvidia_provider
        assert gen_provider_1.is_closed is False
        assert eval_provider_1.is_closed is False

    # After lifespan exit, clients must be closed and singleton reset
    assert gen_provider_1.is_closed is True
    assert eval_provider_1.is_closed is True
    assert routes_qa._orchestrator is None

    # Subsequent lifespan must get a fresh open orchestrator, never sharing closed clients
    with TestClient(app):
        orch_2 = routes_qa.get_orchestrator()
        assert orch_2 is not orch_1
        assert orch_2.generator.client._nvidia_provider.is_closed is False
        assert orch_2.evaluator._get_client()._nvidia_provider.is_closed is False


# =============================================================================
# 2. Decision Gate Input Validation Tests
# =============================================================================
def test_decision_gate_handles_explicit_none_collections_safely():
    gate = DeterministicDecisionGate(confidence_threshold=0.70)
    chunk = _sample_chunk("chunk_1")
    eval_pass = EvaluatorResult(
        passed=True,
        confidence_score=0.95,
        evidence_supported=True,
        contradiction_found=False,
        citation_valid=True,
        completeness_passed=True,
    )

    # claims=None with COVERED must fail safely (never become valid coverage)
    res_cov_none = gate.evaluate(
        "Is hospitalization covered?",
        [chunk],
        {"decision": "COVERED", "claims": None, "citations": None, "conditions": None},
        eval_pass,
    )
    assert res_cov_none.passed is False
    assert res_cov_none.final_decision == DecisionType.INSUFFICIENT_EVIDENCE
    assert res_cov_none.verified_claims == []
    assert res_cov_none.verified_citations == []

    # claims=None / citations=None / conditions=None with INSUFFICIENT EVIDENCE must not raise TypeError
    res_insuff_none = gate.evaluate(
        "Is dental covered?",
        [chunk],
        {"decision": "INSUFFICIENT EVIDENCE", "claims": None, "citations": None, "conditions": None},
        eval_pass,
    )
    assert res_insuff_none.passed is True
    assert res_insuff_none.final_decision == DecisionType.INSUFFICIENT_EVIDENCE


@pytest.mark.parametrize(
    "bad_draft",
    [
        {"decision": "COVERED", "claims": "not-a-list", "citations": [], "conditions": []},
        {"decision": "COVERED", "claims": [], "citations": {"chunk_id": "chunk_1"}, "conditions": []},
        {"decision": "COVERED", "claims": [], "citations": [], "conditions": 123},
        {"decision": "COVERED", "claims": [123], "citations": [], "conditions": []},
        {"decision": "COVERED", "claims": [{"chunk_ids": ["chunk_1"]}], "citations": [], "conditions": []},
        {"decision": "COVERED", "claims": [], "citations": ["bad-citation"], "conditions": []},
        {"decision": "COVERED", "claims": [], "citations": [], "conditions": [42]},
    ],
)
def test_decision_gate_rejects_non_list_or_malformed_collections(bad_draft):
    gate = DeterministicDecisionGate()
    chunk = _sample_chunk("chunk_1")
    eval_pass = EvaluatorResult(
        passed=True,
        confidence_score=0.95,
        evidence_supported=True,
        contradiction_found=False,
        citation_valid=True,
        completeness_passed=True,
    )
    with pytest.raises(LLMValidationError):
        gate.evaluate("Is hospitalization covered?", [chunk], bad_draft, eval_pass)


# =============================================================================
# 3. Citation Validation in EvaluatorAgent Tests
# =============================================================================
@pytest.mark.parametrize("decision", ["COVERED", "INSUFFICIENT EVIDENCE"])
def test_evaluator_rejects_invalid_top_level_citations_in_positive_and_insufficient_drafts(monkeypatch, decision):
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    evaluator = EvaluatorAgent(confidence_threshold=0.70)
    chunk = _sample_chunk("chunk_real")

    claims = (
        [AtomicClaim(text="Hospitalization covered.", chunk_ids=["chunk_real"], claim_type="coverage")]
        if decision == "COVERED"
        else []
    )
    draft = {
        "decision": decision,
        "reason": "Test reason.",
        "conditions": [],
        "claims": claims,
        "citations": [{"chunk_id": "fabricated_cid", "page": 99}],
    }
    res = evaluator.verify("Is hospitalization covered?", [chunk], draft)
    assert res.passed is False
    assert res.citation_valid is False
    assert any(i.issue_type == "bad_citation" for i in res.issues)


# =============================================================================
# 4. Exception-Aware Exclusion Checks Tests
# =============================================================================
def _cosmetic_exclusion_chunk(text: str) -> PolicyChunk:
    return PolicyChunk(
        chunk_id="chunk_excl_cosmetic",
        policy_id="pol_health",
        clause_ref="Section 4.1 - Cosmetic Exclusion",
        page_number=22,
        category=ChunkCategory.EXCLUSIONS,
        content_text=text,
    )


def test_exclusion_blocks_ordinary_cosmetic_procedure(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    ec = _cosmetic_exclusion_chunk(
        "Section 4.1 Exclusions: Cosmetic, aesthetic or plastic surgery is strictly NOT COVERED under any circumstances, "
        "except when medically necessary arising out of an accident, burn reconstruction, or as part of cancer treatment."
    )
    evaluator = EvaluatorAgent()
    draft = {
        "decision": "COVERED",
        "reason": "Cosmetic rhinoplasty is covered.",
        "conditions": [],
        "claims": [AtomicClaim(text="Cosmetic rhinoplasty is covered.", chunk_ids=["chunk_excl_cosmetic"], claim_type="coverage")],
    }
    res = evaluator.verify("Is cosmetic rhinoplasty covered?", [ec], draft)
    assert res.contradiction_found is True
    assert res.passed is False


def test_exclusion_does_not_infer_exception_from_query_keywords_alone():
    # Case A: Exclusion does NOT include burn exception, even though query mentions burn
    ec_cancer_only = _cosmetic_exclusion_chunk(
        "Section 4.1 Exclusions: Cosmetic or plastic surgery is strictly NOT COVERED, except as part of cancer treatment."
    )
    claim_burn = AtomicClaim(
        text="Cosmetic surgery after burn injury is covered.",
        chunk_ids=["chunk_excl_cosmetic"],
        claim_type="coverage",
    )
    conflicts_a = find_exclusion_contradictions(
        query="Is cosmetic surgery after a burn covered?",
        evidence_chunks=[ec_cancer_only],
        decision="COVERED",
        claims=[claim_burn],
        conditions=[],
    )
    assert len(conflicts_a) == 1

    # Case B: Exclusion includes burn/accident exception, but claim cites unrelated chunk (no evidence supporting exception)
    ec_full = _cosmetic_exclusion_chunk(
        "Section 4.1 Exclusions: Cosmetic, aesthetic or plastic surgery is strictly NOT COVERED, "
        "except when medically necessary arising out of an accident, burn reconstruction, or as part of cancer treatment."
    )
    unrelated = _sample_chunk("chunk_hosp")
    claim_unrelated = AtomicClaim(
        text="Hospitalization for surgery is covered.",
        chunk_ids=["chunk_hosp"],
        claim_type="coverage",
    )
    conflicts_b = find_exclusion_contradictions(
        query="Is cosmetic surgery after a burn covered?",
        evidence_chunks=[ec_full, unrelated],
        decision="COVERED",
        claims=[claim_unrelated],
        conditions=[],
    )
    assert len(conflicts_b) == 1


def test_exclusion_allows_grounded_burn_exception_without_heuristic_granting_coverage():
    ec_full = _cosmetic_exclusion_chunk(
        "Section 4.1 Exclusions: Cosmetic, aesthetic or plastic surgery is strictly NOT COVERED, "
        "except when medically necessary arising out of an accident, burn reconstruction, or as part of cancer treatment."
    )
    claim_grounded = AtomicClaim(
        text="Plastic surgery is covered when medically necessary for burn reconstruction under Section 4.1.",
        chunk_ids=["chunk_excl_cosmetic"],
        claim_type="coverage",
    )
    draft = {
        "decision": "COVERED",
        "reason": "Covered under the Section 4.1 medically necessary burn reconstruction exception.",
        "conditions": ["Must be medically necessary for burn reconstruction."],
        "claims": [claim_grounded],
    }

    # Heuristic lifts contradiction when exception is explicitly grounded in clause, query, and claim
    assert find_exclusion_contradictions(
        query="Is medically necessary plastic surgery for burn reconstruction covered?",
        evidence_chunks=[ec_full],
        decision="COVERED",
        claims=[claim_grounded],
        conditions=draft["conditions"],
    ) == []

    # Heuristic alone does NOT establish coverage if Nemotron verifier rejects the claim
    def ver_fail_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": _verdict_json(
                            verdict="FAIL",
                            confidence=0.30,
                            evidence_supported=False,
                            claim_verdicts=[{
                                "claim_idx": 0,
                                "status": "UNSUPPORTED",
                                "supported": False,
                                "cited_chunk_ids_valid": True,
                                "reason": "Insufficient medical necessity documentation.",
                            }],
                        ),
                    }
                }]
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(ver_fail_handler))
    ver_provider = NvidiaNemotronProvider(api_key="nvapi-test", http_client=http_client)
    evaluator = EvaluatorAgent(client=LLMClient(provider="nvidia", nvidia_provider=ver_provider))
    gate = DeterministicDecisionGate()
    try:
        eval_res = evaluator.verify(
            "Is medically necessary plastic surgery for burn reconstruction covered?",
            [ec_full],
            draft,
        )
        assert eval_res.passed is False
        gate_res = gate.evaluate(
            "Is medically necessary plastic surgery for burn reconstruction covered?",
            [ec_full],
            draft,
            eval_res,
        )
        assert gate_res.passed is False
        assert gate_res.final_decision == DecisionType.INSUFFICIENT_EVIDENCE
    finally:
        http_client.close()

    # And passes when Nemotron verifier independently verifies support
    def ver_pass_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": _verdict_json(
                            verdict="PASS",
                            confidence=0.95,
                            claim_verdicts=[{
                                "claim_idx": 0,
                                "status": "SUPPORTED",
                                "supported": True,
                                "cited_chunk_ids_valid": True,
                                "reason": "Section 4.1 explicitly allows medically necessary burn reconstruction.",
                            }],
                        ),
                    }
                }]
            },
        )

    http_client2 = httpx.Client(transport=httpx.MockTransport(ver_pass_handler))
    ver_provider2 = NvidiaNemotronProvider(api_key="nvapi-test", http_client=http_client2)
    evaluator2 = EvaluatorAgent(client=LLMClient(provider="nvidia", nvidia_provider=ver_provider2))
    try:
        eval_res2 = evaluator2.verify(
            "Is medically necessary plastic surgery for burn reconstruction covered?",
            [ec_full],
            draft,
        )
        gate_res2 = gate.evaluate(
            "Is medically necessary plastic surgery for burn reconstruction covered?",
            [ec_full],
            draft,
            eval_res2,
        )
        assert gate_res2.passed is True
        assert gate_res2.final_decision == DecisionType.COVERED
    finally:
        http_client2.close()


# =============================================================================
# 5. Verifier Edge-Case Coverage Tests
# =============================================================================
def _run_verifier_with_payload(verdict_payload: dict, draft: dict, chunks: List[PolicyChunk]):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": json.dumps(verdict_payload)}}]},
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = NvidiaNemotronProvider(api_key="nvapi-test", http_client=http_client)
    evaluator = EvaluatorAgent(client=LLMClient(provider="nvidia", nvidia_provider=provider))
    try:
        return evaluator.verify("Is hospitalization covered?", chunks, draft)
    finally:
        http_client.close()


def test_verifier_rejects_missing_duplicate_and_out_of_range_verdicts():
    chunks = [_sample_chunk("c1"), _sample_chunk("c2")]
    two_claim_draft = {
        "decision": "COVERED",
        "reason": "Covered.",
        "conditions": [],
        "claims": [
            AtomicClaim(text="Claim 0.", chunk_ids=["c1"], claim_type="coverage"),
            AtomicClaim(text="Claim 1.", chunk_ids=["c2"], claim_type="coverage"),
        ],
    }

    # Missing verdict (only 1 of 2 audited)
    with pytest.raises(LLMValidationError, match="every atomic claim"):
        _run_verifier_with_payload(
            {
                "verdict": "PASS",
                "confidence_score": 0.9,
                "evidence_supported": True,
                "contradiction_found": False,
                "completeness_passed": True,
                "claim_verdicts": [{"claim_idx": 0, "status": "SUPPORTED", "supported": True}],
            },
            two_claim_draft,
            chunks,
        )

    # Duplicate claim_idx=0
    with pytest.raises(LLMValidationError, match="duplicate"):
        _run_verifier_with_payload(
            {
                "verdict": "PASS",
                "confidence_score": 0.9,
                "evidence_supported": True,
                "contradiction_found": False,
                "completeness_passed": True,
                "claim_verdicts": [
                    {"claim_idx": 0, "status": "SUPPORTED", "supported": True},
                    {"claim_idx": 0, "status": "SUPPORTED", "supported": True},
                ],
            },
            two_claim_draft,
            chunks,
        )

    # Out-of-range claim_idx in claim_verdicts
    with pytest.raises(LLMValidationError, match="out-of-range"):
        _run_verifier_with_payload(
            {
                "verdict": "PASS",
                "confidence_score": 0.9,
                "evidence_supported": True,
                "contradiction_found": False,
                "completeness_passed": True,
                "claim_verdicts": [
                    {"claim_idx": 0, "status": "SUPPORTED", "supported": True},
                    {"claim_idx": 5, "status": "SUPPORTED", "supported": True},
                ],
            },
            two_claim_draft,
            chunks,
        )

    # Out-of-range claim_idx in issues
    with pytest.raises(LLMValidationError, match="out-of-range"):
        _run_verifier_with_payload(
            {
                "verdict": "FAIL",
                "confidence_score": 0.3,
                "evidence_supported": False,
                "contradiction_found": False,
                "completeness_passed": True,
                "claim_verdicts": [
                    {"claim_idx": 0, "status": "SUPPORTED", "supported": True},
                    {"claim_idx": 1, "status": "UNSUPPORTED", "supported": False},
                ],
                "issues": [{"claim_idx": 9, "issue_type": "unsupported", "description": "Bad index"}],
            },
            two_claim_draft,
            chunks,
        )


@pytest.mark.parametrize(
    "status,supported,cited_valid,expected_flag",
    [
        ("UNSUPPORTED", False, True, "evidence_supported"),
        ("SUPPORTED", True, False, "citation_valid"),
        ("CONTRADICTED", False, True, "contradiction_found"),
    ],
)
def test_verifier_overrides_overall_pass_when_any_claim_fails(status, supported, cited_valid, expected_flag):
    chunk = _sample_chunk("c1")
    draft = {
        "decision": "COVERED",
        "reason": "Covered.",
        "conditions": [],
        "claims": [AtomicClaim(text="Claim 0.", chunk_ids=["c1"], claim_type="coverage")],
    }
    res = _run_verifier_with_payload(
        {
            "verdict": "PASS",
            "confidence_score": 0.98,
            "evidence_supported": True,
            "contradiction_found": False,
            "completeness_passed": True,
            "claim_verdicts": [
                {
                    "claim_idx": 0,
                    "status": status,
                    "supported": supported,
                    "cited_chunk_ids_valid": cited_valid,
                    "reason": "Audit failure detail.",
                }
            ],
        },
        draft,
        [chunk],
    )
    assert res.passed is False
    if expected_flag == "contradiction_found":
        assert res.contradiction_found is True
    else:
        assert getattr(res, expected_flag) is False


def test_insufficient_evidence_with_non_empty_claims_and_invalid_citations():
    chunk = _sample_chunk("c1")
    gate = DeterministicDecisionGate()

    # 1. Supported claim under INSUFFICIENT EVIDENCE: Gate passes INSUFFICIENT EVIDENCE and clears claims/citations
    draft_supported_insuff = {
        "decision": "INSUFFICIENT EVIDENCE",
        "reason": "Policy mentions hospitalization, not dental.",
        "conditions": ["Some condition"],
        "claims": [AtomicClaim(text="Section 2.1 covers hospitalization only.", chunk_ids=["c1"], claim_type="coverage")],
    }
    res_pass = _run_verifier_with_payload(
        {
            "verdict": "PASS",
            "confidence_score": 0.92,
            "evidence_supported": True,
            "contradiction_found": False,
            "completeness_passed": True,
            "claim_verdicts": [{"claim_idx": 0, "status": "SUPPORTED", "supported": True, "cited_chunk_ids_valid": True}],
        },
        draft_supported_insuff,
        [chunk],
    )
    gate_res = gate.evaluate("Is dental covered?", [chunk], draft_supported_insuff, res_pass)
    assert gate_res.passed is True
    assert gate_res.final_decision == DecisionType.INSUFFICIENT_EVIDENCE
    assert gate_res.verified_claims == []
    assert gate_res.verified_citations == []
    assert gate_res.verified_conditions == []

    # 2. Invalid citation under INSUFFICIENT EVIDENCE: rejected
    draft_bad_cit_insuff = {
        "decision": "INSUFFICIENT EVIDENCE",
        "reason": "Not found.",
        "conditions": [],
        "claims": [],
        "citations": [{"chunk_id": "unknown_chunk_99", "page": 1}],
    }
    res_bad = _run_verifier_with_payload(
        {
            "verdict": "PASS",
            "confidence_score": 0.92,
            "evidence_supported": True,
            "contradiction_found": False,
            "completeness_passed": True,
            "claim_verdicts": [],
        },
        draft_bad_cit_insuff,
        [chunk],
    )
    assert res_bad.passed is False
    assert res_bad.citation_valid is False
    gate_bad = gate.evaluate("Is dental covered?", [chunk], draft_bad_cit_insuff, res_bad)
    assert gate_bad.passed is False
    assert gate_bad.verified_citations == []


# =============================================================================
# 6. Retry Evidence Expansion & Exhaustion Tests
# =============================================================================
def test_retry_expands_evidence_from_5_to_8_chunks_and_exhausts_without_leaking_claims():
    kb = MockKnowledgeBase()
    ten_chunks = [
        PolicyChunk(
            chunk_id=f"chk_exp_{i:02d}",
            policy_id="pol_expand_10",
            clause_ref=f"Section {i}.1",
            page_number=i,
            category=ChunkCategory.COVERAGE,
            content_text=f"Hospitalization clause {i} medical coverage terms.",
        )
        for i in range(1, 11)
    ]
    kb.store_chunks("pol_expand_10", ten_chunks)

    gen_evidence_counts: List[int] = []
    ver_evidence_counts: List[int] = []

    class RecordingGenerator(GroundedGenerator):
        def generate_draft(self, query: str, evidence: List[PolicyChunk]):
            gen_evidence_counts.append(len(evidence))
            return super().generate_draft(query, evidence)

    class RecordingEvaluator(EvaluatorAgent):
        def verify(self, query: str, evidence_chunks: List[PolicyChunk], draft_response):
            ver_evidence_counts.append(len(evidence_chunks))
            return super().verify(query, evidence_chunks, draft_response)

    def gen_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "decision": "COVERED",
                            "reason": "Leaked speculative reason should not survive exhaustion.",
                            "conditions": ["Speculative condition"],
                            "claims": [{
                                "text": "Unsupported speculative claim.",
                                "chunk_ids": ["chk_exp_01"],
                                "claim_type": "coverage",
                            }],
                        }),
                    }
                }]
            },
        )

    def ver_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": _verdict_json(
                            verdict="FAIL",
                            confidence=0.20,
                            evidence_supported=False,
                            claim_verdicts=[{
                                "claim_idx": 0,
                                "status": "UNSUPPORTED",
                                "supported": False,
                                "cited_chunk_ids_valid": True,
                                "reason": "Unsupported speculative claim.",
                            }],
                        ),
                    }
                }]
            },
        )

    gen_http = httpx.Client(transport=httpx.MockTransport(gen_handler))
    ver_http = httpx.Client(transport=httpx.MockTransport(ver_handler))
    gen_prov = NvidiaNemotronProvider(api_key="nvapi-test", http_client=gen_http)
    ver_prov = NvidiaNemotronProvider(api_key="nvapi-test", http_client=ver_http)
    orch = ClosedRetryLoopOrchestrator(
        generator=RecordingGenerator(client=LLMClient(provider="nvidia", nvidia_provider=gen_prov)),
        evaluator=RecordingEvaluator(client=LLMClient(provider="nvidia", nvidia_provider=ver_prov)),
        decision_gate=DeterministicDecisionGate(),
    )
    orch.kb = kb

    try:
        res = orch.execute_qa("Is experimental therapy covered?", "pol_expand_10", max_retries=2)
        assert gen_evidence_counts == [5, 8, 10]
        assert ver_evidence_counts == [5, 8, 10]
        assert res.decision == DecisionType.INSUFFICIENT_EVIDENCE
        assert res.evaluator_status == "VERIFIED_FALLBACK_EXHAUSTED"
        assert res.retry_count == 2
        assert res.citations == []
        assert res.conditions == []
        assert "Leaked speculative reason" not in res.reason
    finally:
        gen_http.close()
        ver_http.close()


# =============================================================================
# 7 & 8. End-to-End FastAPI Error Sanitization & Traceback Logging Tests
# =============================================================================
@pytest.mark.parametrize(
    "scenario,expected_status",
    [
        ("auth", 503),
        ("permission", 503),
        ("config_missing_key", 503),
        ("rate_limit", 429),
        ("timeout", 504),
        ("connection", 502),
        ("malformed_provider_response", 502),
        ("llm_validation_failure", 502),
    ],
)
def test_api_ask_end_to_end_real_orchestrator_sanitizes_errors(monkeypatch, caplog, scenario, expected_status):
    secret_key = "nvapi-end-to-end-super-secret"
    raw_provider_secret = "RAW_PROVIDER_BODY_INTERNAL_TRACE"
    sensitive_question = "Sensitive policyholder query about cancer treatment"

    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    if scenario == "config_missing_key":
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    else:
        monkeypatch.setenv("NVIDIA_API_KEY", secret_key)

    def handler(request: httpx.Request) -> httpx.Response:
        if scenario == "auth":
            return httpx.Response(401, json={"error": {"message": f"{raw_provider_secret} {secret_key}"}})
        if scenario == "permission":
            return httpx.Response(403, json={"error": {"message": f"Forbidden {raw_provider_secret}"}})
        if scenario == "rate_limit":
            return httpx.Response(429, json={"error": {"message": f"Rate limit {raw_provider_secret}"}})
        if scenario == "timeout":
            raise httpx.ReadTimeout(f"Timed out {raw_provider_secret}", request=request)
        if scenario == "connection":
            raise httpx.ConnectError(f"Connect error {raw_provider_secret}", request=request)
        if scenario == "malformed_provider_response":
            return httpx.Response(200, content=b"<<<not json at all>>>")
        if scenario == "llm_validation_failure":
            return httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": json.dumps({"decision": "INVALID_ENUM"})}}]},
            )
        return httpx.Response(200, json={})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    cfg = Settings()
    gen_prov = NvidiaNemotronProvider(
        api_key=None if scenario == "config_missing_key" else secret_key,
        max_retries=0,
        backoff_base_seconds=0.0,
        http_client=http_client,
        app_settings=cfg,
    )
    ver_prov = NvidiaNemotronProvider(
        api_key=None if scenario == "config_missing_key" else secret_key,
        max_retries=0,
        backoff_base_seconds=0.0,
        http_client=http_client,
        app_settings=cfg,
    )
    orch = ClosedRetryLoopOrchestrator(
        generator=GroundedGenerator(client=LLMClient(provider="nvidia", nvidia_provider=gen_prov, app_settings=cfg)),
        evaluator=EvaluatorAgent(
            client=LLMClient(provider="nvidia", nvidia_provider=ver_prov, app_settings=cfg),
            app_settings=cfg,
        ),
        decision_gate=DeterministicDecisionGate(),
    )

    app.dependency_overrides[routes_qa.get_orchestrator] = lambda: orch
    client = TestClient(app)
    try:
        with caplog.at_level(logging.INFO):
            resp = client.post(
                "/ai/ask",
                json={"policy_id": "policy_health_01", "question": sensitive_question},
            )
        assert resp.status_code == expected_status
        assert "INSUFFICIENT EVIDENCE" not in resp.text
        assert secret_key not in resp.text
        assert raw_provider_secret not in resp.text
        assert "NO EVIDENCE" not in resp.text  # internal prompt must never leak
        assert "Bearer" not in resp.text
        assert secret_key not in caplog.text
        assert sensitive_question not in caplog.text
        assert raw_provider_secret not in caplog.text
        assert "Bearer" not in caplog.text
    finally:
        app.dependency_overrides.clear()
        http_client.close()


def test_api_unexpected_500_logs_traceback_location_without_secrets(caplog):
    class CrashingOrchestrator:
        def execute_qa(self, query: str, policy_id: str):
            raise KeyError("SECRET_INTERNAL_DICT_KEY")

    app.dependency_overrides[routes_qa.get_orchestrator] = lambda: CrashingOrchestrator()
    client = TestClient(app)
    try:
        with caplog.at_level(logging.ERROR, logger="coverageiq.api.qa"):
            resp = client.post(
                "/ai/ask",
                json={"policy_id": "policy_health_01", "question": "Sensitive medical condition question"},
            )
        assert resp.status_code == 500
        assert "SECRET_INTERNAL_DICT_KEY" not in resp.text
        assert "SECRET_INTERNAL_DICT_KEY" not in caplog.text
        assert "Sensitive medical condition" not in caplog.text
        assert "trace=" in caplog.text
        assert "test_phase3_hardening.py" in caplog.text
    finally:
        app.dependency_overrides.clear()
