import json
import logging
from typing import Any, Dict, List

import httpx
import pytest
from fastapi.testclient import TestClient

from ai_service.api import routes_qa
from ai_service.config import Settings
from ai_service.database import get_knowledge_base
from ai_service.evaluator.decision_gate import DeterministicDecisionGate
from ai_service.evaluator.retry_controller import ClosedRetryLoopOrchestrator
from ai_service.evaluator.verifier import EvaluatorAgent
from ai_service.llm.client import LLMClient, LLMJsonParseError, LLMValidationError, parse_llm_json
from ai_service.llm.generator import GroundedGenerator
from ai_service.llm.prompts import GROUNDED_SYSTEM_PROMPT, VERIFIER_SYSTEM_PROMPT
from ai_service.llm.providers.nvidia_nemotron import (
    NvidiaNemotronProvider,
    NvidiaResponseError,
    NvidiaTokenLimitError,
)
from ai_service.main import app
from ai_service.schemas.structured_output import AtomicClaim, DecisionType, EvaluatorResult


def _verdict_json(
    *,
    verdict: str = "PASS",
    confidence: float = 0.95,
    evidence_supported: bool = True,
    contradiction_found: bool = False,
    completeness_passed: bool = True,
    claim_verdicts: list | None = None,
    issues: list | None = None,
) -> str:
    return json.dumps(
        {
            "verdict": verdict,
            "confidence_score": confidence,
            "evidence_supported": evidence_supported,
            "contradiction_found": contradiction_found,
            "completeness_passed": completeness_passed,
            "claim_verdicts": claim_verdicts or [],
            "issues": issues or [],
            "summary": "Audit summary.",
        }
    )


# =============================================================================
# 1. Token Budget Configuration & Payload Tests
# =============================================================================
def test_default_nvidia_max_tokens_is_8192(monkeypatch):
    monkeypatch.delenv("NVIDIA_MAX_TOKENS", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-budget-test")

    cfg = Settings()
    assert cfg.NVIDIA_MAX_TOKENS == 8192

    provider = NvidiaNemotronProvider(app_settings=cfg)
    try:
        assert provider.max_tokens == 8192
    finally:
        provider.close()


def test_nvidia_max_tokens_env_override(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-budget-override")
    monkeypatch.setenv("NVIDIA_MAX_TOKENS", "12288")

    captured_payload: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": '{"ok": true}'}, "finish_reason": "stop"}]},
        )

    cfg = Settings()
    assert cfg.NVIDIA_MAX_TOKENS == 12288

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = NvidiaNemotronProvider(app_settings=cfg, http_client=http_client)
    try:
        assert provider.max_tokens == 12288
        provider.complete("sys", "hi")
        assert captured_payload["max_tokens"] == 12288
    finally:
        http_client.close()


def test_provider_payload_uses_configured_and_override_max_tokens(monkeypatch):
    monkeypatch.delenv("NVIDIA_MAX_TOKENS", raising=False)
    captured: List[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.append(body["max_tokens"])
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}]},
        )

    cfg = Settings(NVIDIA_API_KEY="nvapi-cfg-key", NVIDIA_MAX_TOKENS=6144)
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = NvidiaNemotronProvider(app_settings=cfg, http_client=http_client)
    try:
        provider.generate_text("sys", "hello")
        provider.generate_text("sys", "hello", max_tokens=2048)
        assert captured == [6144, 2048]
    finally:
        http_client.close()


# =============================================================================
# 2. Token Exhaustion, Truncation & Error Isolation Tests
# =============================================================================
@pytest.mark.parametrize("content_value", [None, "", '{"decision": "COVERED", "rea'])
def test_finish_reason_length_raises_token_limit_error_with_redacted_metadata(monkeypatch, caplog, content_value):
    calls = 0
    secret_key = "nvapi-secret-length-test"
    sensitive_prompt = "CONFIDENTIAL_USER_POLICY_PROMPT"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"x-request-id": "req-len-123"},
            json={
                "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
                "usage": {"prompt_tokens": 900, "completion_tokens": 8192, "total_tokens": 9092},
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": content_value,
                            "reasoning_content": "Internal reasoning exhausting token limit...",
                        },
                    }
                ],
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = NvidiaNemotronProvider(api_key=secret_key, max_retries=2, http_client=http_client)
    client = LLMClient(provider="nvidia", nvidia_provider=provider)

    def fail_offline(*args, **kwargs):
        raise AssertionError("Offline heuristic fallback must never run after NVIDIA token exhaustion.")

    monkeypatch.setattr(client, "_offline_grounded_synthesis", fail_offline)

    try:
        with caplog.at_level(logging.ERROR):
            with pytest.raises(NvidiaTokenLimitError) as exc_info:
                client.generate_json("sys", sensitive_prompt)

        err = exc_info.value
        assert isinstance(err, NvidiaResponseError)
        assert err.retryable is False
        assert calls == 1  # Not retried
        assert err.finish_reason == "length"
        assert err.request_id == "req-len-123"
        assert err.usage is not None and err.usage.total_tokens == 9092
        assert err.safe_metadata()["finish_reason"] == "length"
        assert client.last_metadata == {
            "provider": "nvidia",
            "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
            "finish_reason": "length",
            "request_id": "req-len-123",
            "usage": {"prompt_tokens": 900, "completion_tokens": 8192, "total_tokens": 9092},
        }
        assert secret_key not in caplog.text
        assert sensitive_prompt not in caplog.text
        assert "Bearer" not in caplog.text
    finally:
        http_client.close()


@pytest.mark.parametrize("content_value", [None, "", "   "])
def test_empty_or_missing_content_with_stop_raises_nvidia_response_error_and_preserves_metadata(monkeypatch, content_value):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"x-request-id": "req-empty-456"},
            json={
                "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content_value}}],
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = NvidiaNemotronProvider(api_key="nvapi-empty-test", max_retries=0, http_client=http_client)
    client = LLMClient(provider="nvidia", nvidia_provider=provider)
    monkeypatch.setattr(client, "_offline_grounded_synthesis", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("No offline fallback")))

    try:
        with pytest.raises(NvidiaResponseError) as exc_info:
            client.generate_json("sys", "question")
        assert exc_info.value.finish_reason == "stop"
        assert exc_info.value.request_id == "req-empty-456"
        assert client.last_metadata is not None
        assert client.last_metadata["request_id"] == "req-empty-456"
    finally:
        http_client.close()


@pytest.mark.parametrize(
    "truncated_payload",
    [
        '{"decision": "COVERED", "reason": "Incomplete',
        '```json\n{"decision": "COVERED", "claims": []',
    ],
)
def test_truncated_json_raises_llm_json_parse_error_without_offline_fallback(monkeypatch, caplog, truncated_payload):
    sensitive_prompt = "SENSITIVE_TRUNCATION_QUERY"

    with pytest.raises(LLMJsonParseError):
        parse_llm_json(truncated_payload)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"x-request-id": "req-trunc-789"},
            json={
                "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
                "usage": {"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250},
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": truncated_payload}}],
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = NvidiaNemotronProvider(api_key="nvapi-trunc-secret", max_retries=0, http_client=http_client)
    client = LLMClient(provider="nvidia", nvidia_provider=provider)
    monkeypatch.setattr(client, "_offline_grounded_synthesis", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("No offline fallback")))

    try:
        with caplog.at_level(logging.ERROR):
            with pytest.raises(LLMJsonParseError):
                client.generate_json("sys", sensitive_prompt)
        assert client.last_metadata is not None
        assert client.last_metadata["request_id"] == "req-trunc-789"
        assert sensitive_prompt not in caplog.text
        assert "nvapi-trunc-secret" not in caplog.text
    finally:
        http_client.close()


def test_token_exhaustion_never_converts_into_policy_decision_end_to_end(monkeypatch):
    def length_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"x-request-id": "req-e2e-len"},
            json={
                "usage": {"prompt_tokens": 1200, "completion_tokens": 8192, "total_tokens": 9392},
                "choices": [{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": None}}],
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(length_handler))
    gen_prov = NvidiaNemotronProvider(api_key="nvapi-e2e-len", max_retries=0, http_client=http_client)
    ver_prov = NvidiaNemotronProvider(api_key="nvapi-e2e-len", max_retries=0, http_client=http_client)
    orch = ClosedRetryLoopOrchestrator(
        generator=GroundedGenerator(client=LLMClient(provider="nvidia", nvidia_provider=gen_prov)),
        evaluator=EvaluatorAgent(client=LLMClient(provider="nvidia", nvidia_provider=ver_prov)),
        decision_gate=DeterministicDecisionGate(),
    )

    with pytest.raises(NvidiaTokenLimitError):
        orch.execute_qa("Is hospitalization covered?", "policy_health_01")

    app.dependency_overrides[routes_qa.get_orchestrator] = lambda: orch
    client = TestClient(app)
    try:
        resp = client.post("/ai/ask", json={"policy_id": "policy_health_01", "question": "Is hospitalization covered?"})
        assert resp.status_code == 502
        assert "INSUFFICIENT EVIDENCE" not in resp.text
        assert "req-e2e-len" not in resp.text
    finally:
        app.dependency_overrides.clear()
        http_client.close()


# =============================================================================
# 3. Exclusion Proviso Grounding Regression Tests (Real PolicyChunk Data)
# =============================================================================
def test_prompts_mandate_inline_exception_preservation_and_proviso_auditing():
    assert "inline" in GROUNDED_SYSTEM_PROMPT.lower()
    assert "under any circumstances" in GROUNDED_SYSTEM_PROMPT.lower()
    assert "never invent" in GROUNDED_SYSTEM_PROMPT.lower()
    assert "inline exceptions" in VERIFIER_SYSTEM_PROMPT.lower()
    assert "overgeneralized" in VERIFIER_SYSTEM_PROMPT.lower()


def test_exclusion_clause_with_explicit_exception_regression_on_real_policy_chunks():
    kb = get_knowledge_base()
    all_chunks = kb.get_chunks("policy_health_01")

    # 1. Qualified exclusion claim preserving Section 4.1 carve-outs inline passes verification & Gate
    def ver_qualified_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": _verdict_json(
                                verdict="PASS",
                                confidence=0.96,
                                claim_verdicts=[
                                    {
                                        "claim_idx": 0,
                                        "status": "SUPPORTED",
                                        "supported": True,
                                        "cited_chunk_ids_valid": True,
                                        "reason": "Exclusion and documented exceptions preserved inline.",
                                    }
                                ],
                            ),
                        }
                    }
                ]
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(ver_qualified_handler))
    evaluator = EvaluatorAgent(
        client=LLMClient(provider="nvidia", nvidia_provider=NvidiaNemotronProvider(api_key="nvapi-test", http_client=http_client))
    )
    gate = DeterministicDecisionGate()

    qualified_draft = {
        "decision": "NOT COVERED",
        "reason": "Elective aesthetic cosmetic rhinoplasty is excluded under Section 4.1; exceptions for medically necessary accident, burn reconstruction, or cancer treatment do not apply.",
        "conditions": [
            "Except when medically necessary arising out of an accident, burn reconstruction, or cancer treatment (Section 4.1)."
        ],
        "claims": [
            AtomicClaim(
                text=(
                    "Cosmetic, aesthetic or plastic surgery is excluded under Section 4.1 except when medically "
                    "necessary arising out of an accident, burn reconstruction, or as part of cancer treatment."
                ),
                chunk_ids=["chunk_h_003"],
                claim_type="exclusion",
            )
        ],
    }

    try:
        eval_res = evaluator.verify(
            "Are expenses for an elective rhinoplasty purely for aesthetic cosmetic improvement covered?",
            all_chunks,
            qualified_draft,
        )
        gate_res = gate.evaluate(
            "Are expenses for an elective rhinoplasty purely for aesthetic cosmetic improvement covered?",
            all_chunks,
            qualified_draft,
            eval_res,
        )
        assert eval_res.passed is True
        assert gate_res.passed is True
        assert gate_res.final_decision == DecisionType.NOT_COVERED
        assert len(gate_res.verified_citations) == 1
        assert gate_res.verified_citations[0].chunk_id == "chunk_h_003"
    finally:
        http_client.close()

    # 2. Grounded valid burn-reconstruction exception claim on real policy_health_01 chunks passes
    def ver_burn_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": _verdict_json(
                                verdict="PASS",
                                confidence=0.95,
                                claim_verdicts=[
                                    {
                                        "claim_idx": 0,
                                        "status": "SUPPORTED",
                                        "supported": True,
                                        "cited_chunk_ids_valid": True,
                                        "reason": "Section 4.1 explicitly covers medically necessary burn reconstruction.",
                                    }
                                ],
                            ),
                        }
                    }
                ]
            },
        )

    http_client_burn = httpx.Client(transport=httpx.MockTransport(ver_burn_handler))
    eval_burn = EvaluatorAgent(
        client=LLMClient(
            provider="nvidia",
            nvidia_provider=NvidiaNemotronProvider(api_key="nvapi-test", http_client=http_client_burn),
        )
    )
    burn_draft = {
        "decision": "COVERED",
        "reason": "Covered under the Section 4.1 exception for medically necessary burn reconstruction.",
        "conditions": ["Must be medically necessary arising out of burn reconstruction (Section 4.1)."],
        "claims": [
            AtomicClaim(
                text=(
                    "While cosmetic or plastic surgery is generally excluded, reconstructive plastic surgery is "
                    "covered when medically necessary arising out of burn reconstruction under Section 4.1."
                ),
                chunk_ids=["chunk_h_003"],
                claim_type="coverage",
            )
        ],
    }
    try:
        burn_eval_res = eval_burn.verify(
            "Is medically necessary plastic surgery for burn reconstruction covered?",
            all_chunks,
            burn_draft,
        )
        burn_gate_res = gate.evaluate(
            "Is medically necessary plastic surgery for burn reconstruction covered?",
            all_chunks,
            burn_draft,
            burn_eval_res,
        )
        assert burn_eval_res.passed is True
        assert burn_gate_res.passed is True
        assert burn_gate_res.final_decision == DecisionType.COVERED
        assert [c.chunk_id for c in burn_gate_res.verified_citations] == ["chunk_h_003"]
    finally:
        http_client_burn.close()


def test_ordinary_excluded_procedure_without_exception_blocked_deterministically():
    kb = get_knowledge_base()
    all_chunks = kb.get_chunks("policy_health_01")
    gate = DeterministicDecisionGate()

    for query, cid, text in [
        (
            "Can I claim for elective aesthetic cosmetic rhinoplasty?",
            "chunk_h_003",
            "Elective cosmetic rhinoplasty is covered up to Sum Insured.",
        ),
        (
            "Are hospitalization bills from intentional self-inflicted injury covered?",
            "chunk_h_006",
            "Self-inflicted injury treatment is covered under Section 4.5.",
        ),
    ]:
        draft = {
            "decision": "COVERED",
            "reason": text,
            "conditions": [],
            "claims": [AtomicClaim(text=text, chunk_ids=[cid], claim_type="coverage")],
        }
        # Even if an LLM verifier mistakenly returned PASS, deterministic gate must catch contradiction and block
        optimistic_eval = EvaluatorResult(
            passed=True,
            confidence_score=0.95,
            evidence_supported=True,
            contradiction_found=False,
            citation_valid=True,
            completeness_passed=True,
        )
        gate_res = gate.evaluate(query, all_chunks, draft, optimistic_eval)
        assert gate_res.passed is False
        assert gate_res.contradiction_found is True
        assert gate_res.final_decision == DecisionType.INSUFFICIENT_EVIDENCE
        assert gate_res.verified_claims == []
        assert gate_res.verified_citations == []


# =============================================================================
# 4. Empty INSUFFICIENT EVIDENCE Optimization & Validation Tests
# =============================================================================
def test_empty_insufficient_evidence_skips_verifier_llm_call():
    kb = get_knowledge_base()
    all_chunks = kb.get_chunks("policy_health_01")

    def fail_if_verifier_called(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Verifier LLM must not be called for valid empty INSUFFICIENT EVIDENCE drafts.")

    http_client = httpx.Client(transport=httpx.MockTransport(fail_if_verifier_called))
    ver_prov = NvidiaNemotronProvider(api_key="nvapi-skip-test", http_client=http_client)
    evaluator = EvaluatorAgent(client=LLMClient(provider="nvidia", nvidia_provider=ver_prov))
    gate = DeterministicDecisionGate()

    empty_insuff_draft = {
        "decision": "INSUFFICIENT EVIDENCE",
        "reason": "The supplied policy clauses contain no information on dental cleanings or waiting periods for dental care.",
        "conditions": [],
        "claims": [],
        "citations": [],
    }

    try:
        # Even when query mentions 'waiting period' and evidence has maternity/PED condition chunks,
        # an empty INSUFFICIENT EVIDENCE draft must not fail deterministic waiting-period completeness.
        eval_res = evaluator.verify("What is the waiting period for routine dental cleaning?", all_chunks, empty_insuff_draft)
        assert eval_res.passed is True
        assert eval_res.confidence_score == 0.95
        assert "skipped verifier LLM call" in eval_res.raw_evaluator_comment

        gate_res = gate.evaluate("What is the waiting period for routine dental cleaning?", all_chunks, empty_insuff_draft, eval_res)
        assert gate_res.passed is True
        assert gate_res.final_decision == DecisionType.INSUFFICIENT_EVIDENCE
        assert gate_res.verified_claims == []
        assert gate_res.verified_citations == []
        assert gate_res.verified_conditions == []
    finally:
        http_client.close()


def test_orchestrator_skips_verifier_call_on_empty_insufficient_evidence():
    gen_calls = 0
    ver_calls = 0

    def gen_handler(request: httpx.Request) -> httpx.Response:
        nonlocal gen_calls
        gen_calls += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "decision": "INSUFFICIENT EVIDENCE",
                                    "reason": "No dental clauses found in policy evidence.",
                                    "conditions": [],
                                    "claims": [],
                                }
                            ),
                        }
                    }
                ]
            },
        )

    def ver_handler(request: httpx.Request) -> httpx.Response:
        nonlocal ver_calls
        ver_calls += 1
        return httpx.Response(500, json={"error": {"message": "Should not run"}})

    gen_http = httpx.Client(transport=httpx.MockTransport(gen_handler))
    ver_http = httpx.Client(transport=httpx.MockTransport(ver_handler))
    orch = ClosedRetryLoopOrchestrator(
        generator=GroundedGenerator(
            client=LLMClient(provider="nvidia", nvidia_provider=NvidiaNemotronProvider(api_key="nvapi-test", http_client=gen_http))
        ),
        evaluator=EvaluatorAgent(
            client=LLMClient(provider="nvidia", nvidia_provider=NvidiaNemotronProvider(api_key="nvapi-test", http_client=ver_http))
        ),
        decision_gate=DeterministicDecisionGate(),
    )

    try:
        res = orch.execute_qa("Are routine dental cleanings covered?", "policy_health_01")
        assert gen_calls == 1
        assert ver_calls == 0
        assert res.decision == DecisionType.INSUFFICIENT_EVIDENCE
        assert res.evaluator_status == "VERIFIED_PASS"
        assert res.retry_count == 0
        assert res.citations == []
        assert res.conditions == []
        assert res.metadata is not None and res.metadata.get("verifier_llm_skipped") is True
    finally:
        gen_http.close()
        ver_http.close()


def test_insufficient_evidence_with_non_empty_claims_conditions_or_citations_still_calls_verifier():
    kb = get_knowledge_base()
    chunks = [kb.get_chunks("policy_health_01")[0]]
    cid = chunks[0].chunk_id

    cases = [
        {
            "decision": "INSUFFICIENT EVIDENCE",
            "reason": "Only hospitalization exists.",
            "conditions": [],
            "claims": [AtomicClaim(text="Hospitalization exists in Section 2.1.", chunk_ids=[cid], claim_type="coverage")],
            "claim_verdicts": [{"claim_idx": 0, "status": "SUPPORTED", "supported": True, "cited_chunk_ids_valid": True}],
        },
        {
            "decision": "INSUFFICIENT EVIDENCE",
            "reason": "Conditions listed.",
            "conditions": ["Prior insurer history required"],
            "claims": [],
            "claim_verdicts": [],
        },
        {
            "decision": "INSUFFICIENT EVIDENCE",
            "reason": "Citation present.",
            "conditions": [],
            "claims": [],
            "citations": [{"chunk_id": cid, "page": 12}],
            "claim_verdicts": [],
        },
    ]

    for case in cases:
        ver_calls = 0

        def handler(request: httpx.Request, payload_cv=case["claim_verdicts"]) -> httpx.Response:
            nonlocal ver_calls
            ver_calls += 1
            return httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": _verdict_json(claim_verdicts=payload_cv)}}]},
            )

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        evaluator = EvaluatorAgent(
            client=LLMClient(provider="nvidia", nvidia_provider=NvidiaNemotronProvider(api_key="nvapi-test", http_client=http_client))
        )
        draft = {k: v for k, v in case.items() if k != "claim_verdicts"}
        try:
            res = evaluator.verify("Is dental covered?", chunks, draft)
            assert ver_calls == 1
            assert res.passed is True
        finally:
            http_client.close()


@pytest.mark.parametrize(
    "bad_draft",
    [
        "not-a-dict",
        {"reason": "Missing decision.", "claims": [], "citations": [], "conditions": []},
        {"decision": "MAYBE", "reason": "Bad decision.", "claims": [], "citations": [], "conditions": []},
        {"decision": "INSUFFICIENT EVIDENCE", "reason": "", "claims": [], "citations": [], "conditions": []},
        {"decision": "INSUFFICIENT EVIDENCE", "reason": "Ok", "claims": "bad", "citations": [], "conditions": []},
        {"decision": "INSUFFICIENT EVIDENCE", "reason": "Ok", "claims": [], "citations": {"chunk_id": "x"}, "conditions": []},
        {"decision": "INSUFFICIENT EVIDENCE", "reason": "Ok", "claims": [], "citations": [], "conditions": [123]},
    ],
)
def test_empty_insufficient_evidence_optimization_never_bypasses_malformed_validation(bad_draft):
    kb = get_knowledge_base()
    chunks = [kb.get_chunks("policy_health_01")[0]]
    evaluator = EvaluatorAgent(provider="offline")

    with pytest.raises(LLMValidationError):
        evaluator.verify("Is dental covered?", chunks, bad_draft)  # type: ignore[arg-type]
