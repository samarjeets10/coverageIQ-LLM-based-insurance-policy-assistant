import json
import logging
from typing import Callable, List

import httpx
import pytest
from fastapi.testclient import TestClient

from ai_service.api import routes_qa
from ai_service.database import get_knowledge_base
from ai_service.evaluator.decision_gate import DeterministicDecisionGate
from ai_service.evaluator.retry_controller import ClosedRetryLoopOrchestrator
from ai_service.evaluator.verifier import EvaluatorAgent
from ai_service.llm.client import LLMClient, LLMJsonParseError, LLMValidationError
from ai_service.llm.generator import GroundedGenerator
from ai_service.llm.providers.nvidia_nemotron import (
    NvidiaAuthenticationError,
    NvidiaNemotronProvider,
    NvidiaRateLimitError,
    NvidiaServerError,
    NvidiaTimeoutError,
)
from ai_service.main import app
from ai_service.schemas.structured_output import (
    AtomicClaim,
    ChunkCategory,
    DecisionType,
    EvaluatorIssue,
    EvaluatorResult,
    PolicyChunk,
)


@pytest.fixture(autouse=True)
def _offline_embeddings_for_verification_tests(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")


def _provider_from_handler(handler: Callable[[httpx.Request], httpx.Response], *, max_retries: int = 0):
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = NvidiaNemotronProvider(
        api_key="nvapi-phase3-secret-key",
        max_retries=max_retries,
        backoff_base_seconds=0.0,
        http_client=http_client,
        sleep_fn=lambda _: None,
    )
    return provider, http_client


def _make_orchestrator(gen_handler, ver_handler, *, max_retries: int = 2):
    gen_provider, gen_http = _provider_from_handler(gen_handler)
    ver_provider, ver_http = _provider_from_handler(ver_handler)
    gen_client = LLMClient(provider="nvidia", nvidia_provider=gen_provider)
    ver_client = LLMClient(provider="nvidia", nvidia_provider=ver_provider)

    orchestrator = ClosedRetryLoopOrchestrator(
        generator=GroundedGenerator(client=gen_client),
        evaluator=EvaluatorAgent(client=ver_client, confidence_threshold=0.70),
        decision_gate=DeterministicDecisionGate(confidence_threshold=0.70),
    )
    return orchestrator, (gen_http, ver_http)


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


def test_supported_claims_pass_nemotron_verification_and_decision_gate():
    gen_calls = 0
    ver_calls = 0
    ver_prompts: List[str] = []

    def gen_handler(request: httpx.Request) -> httpx.Response:
        nonlocal gen_calls
        gen_calls += 1
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "decision": "COVERED",
                            "reason": "In-patient hospitalization is covered for >= 24 hours under Section 2.1.",
                            "conditions": ["Minimum 24 consecutive hours of hospitalization."],
                            "claims": [{
                                "text": "In-patient hospitalization is covered for minimum 24 consecutive hours.",
                                "chunk_ids": ["chunk_h_001"],
                                "claim_type": "coverage",
                            }],
                        }),
                    }
                }]
            },
        )

    def ver_handler(request: httpx.Request) -> httpx.Response:
        nonlocal ver_calls
        ver_calls += 1
        body = json.loads(request.content.decode("utf-8"))
        ver_prompts.append(body["messages"][1]["content"])
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": _verdict_json(
                            verdict="PASS",
                            confidence=0.96,
                            claim_verdicts=[{
                                "claim_idx": 0,
                                "status": "SUPPORTED",
                                "supported": True,
                                "cited_chunk_ids_valid": True,
                                "reason": "Directly supported by chunk_h_001.",
                            }],
                        ),
                    }
                }]
            },
        )

    orch, clients = _make_orchestrator(gen_handler, ver_handler)
    try:
        res = orch.execute_qa("Is in-patient hospitalization covered?", "policy_health_01")
        assert gen_calls == 1
        assert ver_calls == 1
        assert "DRAFT ANSWER TO AUDIT:" in ver_prompts[0]
        assert "chunk_h_001" in ver_prompts[0]
        assert res.decision == DecisionType.COVERED
        assert res.evaluator_status == "VERIFIED_PASS"
        assert res.retry_count == 0
        assert res.confidence_score >= 0.70
        assert any(c.chunk_id == "chunk_h_001" and c.page == 12 for c in res.citations)
    finally:
        for c in clients:
            c.close()


def test_unsupported_claims_rejected_after_bounded_retries():
    gen_calls = 0
    ver_calls = 0

    def gen_handler(request: httpx.Request) -> httpx.Response:
        nonlocal gen_calls
        gen_calls += 1
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "decision": "COVERED",
                            "reason": "Dental checkups are covered up to INR 20,000.",
                            "conditions": [],
                            "claims": [{
                                "text": "Routine dental checkups are covered up to INR 20,000.",
                                "chunk_ids": ["chunk_h_001"],
                                "claim_type": "coverage",
                            }],
                        }),
                    }
                }]
            },
        )

    def ver_handler(request: httpx.Request) -> httpx.Response:
        nonlocal ver_calls
        ver_calls += 1
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
                                "reason": "chunk_h_001 discusses hospitalization, not dental checkups.",
                            }],
                            issues=[{
                                "claim_idx": 0,
                                "issue_type": "unsupported",
                                "description": "Claim is not supported by chunk_h_001.",
                            }],
                        ),
                    }
                }]
            },
        )

    orch, clients = _make_orchestrator(gen_handler, ver_handler)
    try:
        res = orch.execute_qa("Is routine dental checkup covered?", "policy_health_01", max_retries=2)
        assert gen_calls == 3  # 1 initial + 2 retries
        assert ver_calls == 3
        assert res.decision == DecisionType.INSUFFICIENT_EVIDENCE
        assert res.evaluator_status == "VERIFIED_FALLBACK_EXHAUSTED"
        assert res.retry_count == 2
        assert res.citations == []
        assert res.conditions == []
    finally:
        for c in clients:
            c.close()


def test_partially_supported_answer_never_retains_positive_decision_or_unsupported_claim():
    def gen_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "decision": "COVERED",
                            "reason": "Hospitalization and air ambulance are covered.",
                            "conditions": ["Minimum 24 hours stay."],
                            "claims": [
                                {
                                    "text": "In-patient hospitalization is covered for >= 24 hours.",
                                    "chunk_ids": ["chunk_h_001"],
                                    "claim_type": "coverage",
                                },
                                {
                                    "text": "International air ambulance is covered up to 100%.",
                                    "chunk_ids": ["chunk_h_001"],
                                    "claim_type": "coverage",
                                },
                            ],
                        }),
                    }
                }]
            },
        )

    def ver_handler(request: httpx.Request) -> httpx.Response:
        # Even if model marks overall verdict PASS inconsistently, claim 1 is OVERGENERALIZED/unsupported
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": _verdict_json(
                            verdict="PASS",
                            confidence=0.90,
                            evidence_supported=True,
                            claim_verdicts=[
                                {
                                    "claim_idx": 0,
                                    "status": "SUPPORTED",
                                    "supported": True,
                                    "cited_chunk_ids_valid": True,
                                    "reason": "Supported.",
                                },
                                {
                                    "claim_idx": 1,
                                    "status": "OVERGENERALIZED",
                                    "supported": False,
                                    "cited_chunk_ids_valid": True,
                                    "reason": "Air ambulance is not mentioned in chunk_h_001.",
                                },
                            ],
                        ),
                    }
                }]
            },
        )

    orch, clients = _make_orchestrator(gen_handler, ver_handler)
    try:
        res = orch.execute_qa("Is hospitalization and air ambulance covered?", "policy_health_01", max_retries=1)
        assert res.decision == DecisionType.INSUFFICIENT_EVIDENCE
        assert res.evaluator_status == "VERIFIED_FALLBACK_EXHAUSTED"
        assert res.citations == []
        assert res.conditions == []
        assert any(i["issue_type"] == "overgeneralized" for i in res.metadata["evaluator_issues"])
    finally:
        for c in clients:
            c.close()


def test_contradictory_evidence_triggers_retry_and_surfaces_contradiction_if_unresolved():
    gen_calls = 0

    def gen_handler(request: httpx.Request) -> httpx.Response:
        nonlocal gen_calls
        gen_calls += 1
        # First attempt wrongly says COVERED; retry corrects to NOT COVERED using chunk_h_003
        if gen_calls == 1:
            payload = {
                "decision": "COVERED",
                "reason": "Surgical hospitalization is covered.",
                "conditions": [],
                "claims": [{
                    "text": "Hospitalization for surgery is covered.",
                    "chunk_ids": ["chunk_h_001"],
                    "claim_type": "coverage",
                }],
            }
        else:
            payload = {
                "decision": "NOT COVERED",
                "reason": "Cosmetic surgery is explicitly excluded under Section 4.1.",
                "conditions": [],
                "claims": [{
                    "text": "Cosmetic surgery is strictly not covered except post-accident/cancer.",
                    "chunk_ids": ["chunk_h_003"],
                    "claim_type": "exclusion",
                }],
            }
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": json.dumps(payload)}}]})

    def ver_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        if '"decision": "COVERED"' in body["messages"][1]["content"]:
            content = _verdict_json(
                verdict="FAIL",
                confidence=0.25,
                contradiction_found=True,
                claim_verdicts=[{
                    "claim_idx": 0,
                    "status": "CONTRADICTED",
                    "supported": False,
                    "cited_chunk_ids_valid": True,
                    "reason": "Section 4.1 excludes cosmetic surgery.",
                }],
                issues=[{
                    "claim_idx": 0,
                    "issue_type": "contradiction",
                    "description": "Contradicts cosmetic surgery exclusion in chunk_h_003.",
                }],
            )
        else:
            content = _verdict_json(
                verdict="PASS",
                confidence=0.95,
                contradiction_found=False,
                claim_verdicts=[{
                    "claim_idx": 0,
                    "status": "SUPPORTED",
                    "supported": True,
                    "cited_chunk_ids_valid": True,
                    "reason": "Supported by chunk_h_003.",
                }],
            )
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}]})

    orch, clients = _make_orchestrator(gen_handler, ver_handler)
    try:
        res = orch.execute_qa("Is cosmetic surgery covered?", "policy_health_01", max_retries=2)
        assert gen_calls == 2
        assert res.decision == DecisionType.NOT_COVERED
        assert res.evaluator_status == "VERIFIED_PASS"
        assert res.retry_count == 1
        assert any(c.chunk_id == "chunk_h_003" for c in res.citations)
    finally:
        for c in clients:
            c.close()


def test_unresolved_contradiction_surfaced_on_retry_exhaustion():
    def gen_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "decision": "COVERED",
                            "reason": "Covered under Section 2.1.",
                            "conditions": [],
                            "claims": [{
                                "text": "Surgery is covered.",
                                "chunk_ids": ["chunk_h_001"],
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
                            contradiction_found=True,
                            claim_verdicts=[{
                                "claim_idx": 0,
                                "status": "CONTRADICTED",
                                "supported": False,
                                "cited_chunk_ids_valid": True,
                                "reason": "Explicit exclusion applies.",
                            }],
                            issues=[{
                                "claim_idx": 0,
                                "issue_type": "contradiction",
                                "description": "Conflicts with Section 4.1 exclusion.",
                            }],
                        ),
                    }
                }]
            },
        )

    orch, clients = _make_orchestrator(gen_handler, ver_handler)
    try:
        res = orch.execute_qa("Is cosmetic surgery covered?", "policy_health_01", max_retries=1)
        assert res.decision == DecisionType.INSUFFICIENT_EVIDENCE
        assert res.evaluator_status == "VERIFIED_FALLBACK_EXHAUSTED"
        assert "contradictory policy evidence" in res.reason.lower()
        assert res.metadata["contradiction_found"] is True
        assert res.citations == []
    finally:
        for c in clients:
            c.close()


def test_unknown_citation_ids_blocked_by_deterministic_gate_even_if_llm_verifier_passes():
    kb = get_knowledge_base()
    evidence = kb.get_chunks("policy_health_01")
    gate = DeterministicDecisionGate(confidence_threshold=0.70)

    draft = {
        "decision": "COVERED",
        "reason": "Covered.",
        "conditions": [],
        "claims": [
            AtomicClaim(text="Hospitalization is covered.", chunk_ids=["fabricated_chunk_xyz"], claim_type="coverage")
        ],
    }
    # Simulate an overly permissive LLM verifier self-assessment saying PASS
    permissive_eval = EvaluatorResult(
        passed=True,
        confidence_score=0.99,
        evidence_supported=True,
        contradiction_found=False,
        citation_valid=True,
        completeness_passed=True,
        issues=[],
    )

    gate_res = gate.evaluate("Is hospitalization covered?", evidence, draft, permissive_eval)
    assert gate_res.passed is False
    assert gate_res.citation_valid is False
    assert gate_res.final_decision == DecisionType.INSUFFICIENT_EVIDENCE
    assert gate_res.verified_claims == []
    assert gate_res.verified_citations == []
    assert any(i.issue_type == "bad_citation" for i in gate_res.issues)


def test_invalid_verifier_json_propagates_and_never_becomes_policy_decision():
    def gen_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "decision": "COVERED",
                            "reason": "Covered.",
                            "conditions": [],
                            "claims": [{
                                "text": "Hospitalization covered.",
                                "chunk_ids": ["chunk_h_001"],
                                "claim_type": "coverage",
                            }],
                        }),
                    }
                }]
            },
        )

    def ver_bad_json_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "{broken-json"}}]})

    orch, clients = _make_orchestrator(gen_handler, ver_bad_json_handler)
    try:
        with pytest.raises(LLMJsonParseError):
            orch.execute_qa("Is hospitalization covered?", "policy_health_01")
    finally:
        for c in clients:
            c.close()


def test_invalid_verifier_schema_raises_validation_error():
    def gen_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "decision": "COVERED",
                            "reason": "Covered.",
                            "conditions": [],
                            "claims": [{
                                "text": "Hospitalization covered.",
                                "chunk_ids": ["chunk_h_001"],
                                "claim_type": "coverage",
                            }],
                        }),
                    }
                }]
            },
        )

    # Missing claim_verdicts for claim 0 -> invalid verifier payload
    def ver_bad_schema_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": _verdict_json(verdict="PASS", confidence=0.9, claim_verdicts=[]),
                    }
                }]
            },
        )

    orch, clients = _make_orchestrator(gen_handler, ver_bad_schema_handler)
    try:
        with pytest.raises(LLMValidationError):
            orch.execute_qa("Is hospitalization covered?", "policy_health_01")
    finally:
        for c in clients:
            c.close()


@pytest.mark.parametrize(
    "error_mode,expected_exc",
    [
        ("timeout", NvidiaTimeoutError),
        ("auth", NvidiaAuthenticationError),
        ("rate_limit", NvidiaRateLimitError),
        ("server", NvidiaServerError),
    ],
)
def test_verifier_provider_errors_propagate_without_becoming_policy_decision(error_mode, expected_exc):
    def gen_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "decision": "COVERED",
                            "reason": "Covered.",
                            "conditions": [],
                            "claims": [{
                                "text": "Hospitalization covered.",
                                "chunk_ids": ["chunk_h_001"],
                                "claim_type": "coverage",
                            }],
                        }),
                    }
                }]
            },
        )

    def ver_error_handler(request: httpx.Request) -> httpx.Response:
        if error_mode == "timeout":
            raise httpx.ReadTimeout("NVIDIA verifier timeout", request=request)
        if error_mode == "auth":
            return httpx.Response(401, json={"error": {"message": "Invalid bearer token nvapi-secret"}})
        if error_mode == "rate_limit":
            return httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})
        return httpx.Response(503, json={"error": {"message": "Internal upstream failure"}})

    orch, clients = _make_orchestrator(gen_handler, ver_error_handler)
    try:
        with pytest.raises(expected_exc):
            orch.execute_qa("Is hospitalization covered?", "policy_health_01")
    finally:
        for c in clients:
            c.close()


def test_no_evidence_returns_no_data_and_gate_rejects_affirmative_without_evidence():
    def fail_if_called(request: httpx.Request) -> httpx.Response:
        raise AssertionError("LLM should not be called when KB has zero chunks for policy_id.")

    orch, clients = _make_orchestrator(fail_if_called, fail_if_called)
    try:
        res = orch.execute_qa("Is hospitalization covered?", "non_existent_policy_id")
        assert res.decision == DecisionType.INSUFFICIENT_EVIDENCE
        assert res.evaluator_status == "NO_DATA"
        assert res.citations == []
    finally:
        for c in clients:
            c.close()

    gate = DeterministicDecisionGate()
    gate_res = gate.evaluate(
        "Is hospitalization covered?",
        evidence_chunks=[],
        draft_response={"decision": "COVERED", "reason": "Covered.", "conditions": [], "claims": []},
        verifier_result=EvaluatorResult(
            passed=True,
            confidence_score=1.0,
            evidence_supported=True,
            contradiction_found=False,
            citation_valid=True,
            completeness_passed=True,
        ),
    )
    assert gate_res.passed is False
    assert gate_res.final_decision == DecisionType.INSUFFICIENT_EVIDENCE


def test_deterministic_gate_resolves_multipage_citations_from_policy_chunks():
    chunk = PolicyChunk(
        chunk_id="chk_multi_01",
        policy_id="pol_multi",
        clause_ref="Section 5.3 - Benefits Table",
        page_number=6,
        page_start=6,
        page_end=8,
        category=ChunkCategory.CONDITIONS,
        content_text="Waiting period of 24 months applies.",
    )
    gate = DeterministicDecisionGate()
    draft = {
        "decision": "CONDITIONAL",
        "reason": "Subject to 24-month waiting period.",
        "conditions": ["24 months waiting period."],
        "claims": [AtomicClaim(text="24-month waiting period applies.", chunk_ids=["chk_multi_01"], claim_type="condition")],
        "citations": [{"chunk_id": "chk_multi_01", "page": 999}],
    }
    eval_res = EvaluatorResult(
        passed=True,
        confidence_score=0.92,
        evidence_supported=True,
        contradiction_found=False,
        citation_valid=True,
        completeness_passed=True,
    )
    gate_res = gate.evaluate("What is the waiting period?", [chunk], draft, eval_res)
    assert gate_res.passed is True
    assert gate_res.verified_citations[0].page == 6
    assert gate_res.verified_citations[0].page_end == 8
    assert gate_res.verified_citations[0].page_range == "6-8"


@pytest.mark.parametrize(
    "exc,expected_status",
    [
        (NvidiaTimeoutError("SECRET_TIMEOUT_DETAIL https://internal/url nvapi-leak"), 504),
        (NvidiaRateLimitError("SECRET_RATE_LIMIT_DETAIL", status_code=429), 429),
        (NvidiaAuthenticationError("SECRET_AUTH_KEY nvapi-123456", status_code=401), 503),
        (NvidiaServerError("RAW_UPSTREAM_BODY stacktrace /etc/passwd", status_code=502), 502),
        (LLMJsonParseError("RAW_JSON_PARSE_FAILURE {model secret leak}"), 502),
        (RuntimeError("INTERNAL_TRACEBACK_SECRET_DB_DSN"), 500),
    ],
)
def test_api_error_responses_do_not_leak_internal_details(exc, expected_status, caplog):
    class FailingOrchestrator:
        def execute_qa(self, query: str, policy_id: str):
            raise exc

    app.dependency_overrides[routes_qa.get_orchestrator] = lambda: FailingOrchestrator()
    client = TestClient(app)
    try:
        with caplog.at_level(logging.INFO, logger="coverageiq.api.qa"):
            resp = client.post(
                "/ai/ask",
                json={"policy_id": "policy_health_01", "question": "Sensitive question about cancer surgery"},
            )
        assert resp.status_code == expected_status
        body_text = resp.text
        assert "SECRET_" not in body_text
        assert "RAW_" not in body_text
        assert "INTERNAL_" not in body_text
        assert "nvapi-" not in body_text
        assert "traceback" not in body_text.lower()
        assert "Sensitive question" not in caplog.text
        assert "nvapi-" not in caplog.text
        assert "RAW_UPSTREAM_BODY" not in caplog.text
    finally:
        app.dependency_overrides.clear()
