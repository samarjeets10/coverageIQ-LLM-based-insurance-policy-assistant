import json
import logging
import os
import re
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ai_service.config import Settings, settings
from ai_service.llm.client import LLMClient, LLMValidationError
from ai_service.llm.prompts import VERIFIER_SYSTEM_PROMPT, format_verification_prompt
from ai_service.llm.providers.nvidia_nemotron import NvidiaNemotronProvider
from ai_service.schemas.structured_output import (
    AtomicClaim,
    ChunkCategory,
    CitationItem,
    DecisionType,
    EvaluatorIssue,
    EvaluatorResult,
    PolicyChunk,
)

logger = logging.getLogger("coverageiq.evaluator")

_EXCLUSION_TOPIC_GROUPS = (
    ("cosmetic",),
    ("plastic surgery",),
    ("aesthetic",),
    ("self-inflicted", "self-injury"),
    ("substance abuse", "abuse of alcohol", "intoxicating drugs"),
)
_EXCEPTION_SPLIT_RE = re.compile(
    r"\b(?:except(?:\s+(?:when|for|in|where|as))?|unless|provided\s+that|other\s+than|save\s+for)\b",
    re.IGNORECASE,
)
_GENERIC_EXCEPTION_STOPWORDS = {
    "when",
    "with",
    "from",
    "that",
    "this",
    "part",
    "only",
    "such",
    "other",
    "than",
    "save",
    "under",
    "into",
    "upon",
    "arising",
    "medically",
    "necessary",
    "treatment",
    "treatments",
    "surgery",
    "procedure",
    "procedures",
    "coverage",
    "covered",
    "clause",
    "policy",
    "unless",
    "except",
    "provided",
    "reconstruction",
    "reconstructive",
}


def find_exclusion_contradictions(
    query: str,
    evidence_chunks: Sequence[PolicyChunk],
    decision: str,
    claims: Sequence[AtomicClaim],
    conditions: Sequence[str],
) -> List[Tuple[PolicyChunk, str]]:
    """
    Conservative, clause-aware exclusion contradiction detector.
    Only lifts an exclusion contradiction on a COVERED decision when:
    1. The exclusion sentence itself contains explicit exception language,
    2. A concrete exception trigger from that exception segment appears in BOTH
       the query and an AtomicClaim that explicitly cites that exclusion chunk, and
    3. Any 'medically necessary' qualifier in the exception is acknowledged in
       the claim or conditions.
    Never independently establishes coverage on its own.
    """
    if decision != "COVERED":
        return []

    q_lower = query.lower()
    q_tokens = set(re.findall(r"\b[a-z]{4,}\b", q_lower))
    conflicts: List[Tuple[PolicyChunk, str]] = []

    for ec in evidence_chunks:
        if ec.category != ChunkCategory.EXCLUSIONS:
            continue
        ec_lower = ec.content_text.lower()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ec_lower) if s.strip()]
        if not sentences:
            sentences = [ec_lower]

        for topic_group in _EXCLUSION_TOPIC_GROUPS:
            if not any(alias in q_lower for alias in topic_group):
                continue
            matched_ec_terms = [alias for alias in topic_group if alias in ec_lower]
            if not matched_ec_terms:
                continue

            # Inspect the specific sentence(s) containing the exclusion term
            matching_sentences = [s for s in sentences if any(alias in s for alias in matched_ec_terms)]
            exception_grounded = False

            for sent in matching_sentences:
                term_indices = [sent.find(alias) for alias in matched_ec_terms if alias in sent]
                term_idx = min(term_indices) if term_indices else 0
                exc_match = _EXCEPTION_SPLIT_RE.search(sent, pos=term_idx)
                if not exc_match:
                    continue

                exception_segment = sent[exc_match.end():]
                exc_tokens = set(re.findall(r"\b[a-z]{4,}\b", exception_segment)) - _GENERIC_EXCEPTION_STOPWORDS
                shared_query_exc = q_tokens.intersection(exc_tokens)
                if not shared_query_exc:
                    continue

                requires_med_necessity = "medically necessary" in exception_segment
                citing_claims = [cl for cl in claims if ec.chunk_id in cl.chunk_ids]

                for cl in citing_claims:
                    cl_lower = cl.text.lower()
                    cl_tokens = set(re.findall(r"\b[a-z]{4,}\b", cl_lower))
                    if not shared_query_exc.intersection(cl_tokens):
                        continue
                    if requires_med_necessity:
                        has_med_ack = (
                            "medically necessary" in cl_lower
                            or "medical necessity" in cl_lower
                            or any(
                                "medically necessary" in c.lower() or "medical necessity" in c.lower()
                                for c in conditions
                            )
                        )
                        if not has_med_ack:
                            continue
                    exception_grounded = True
                    break

                if exception_grounded:
                    break

            if not exception_grounded:
                conflicts.append((ec, matched_ec_terms[0]))

    return conflicts


class _ClaimVerdictPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claim_idx: int = Field(ge=0)
    status: Literal["SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "OVERGENERALIZED"]
    supported: bool
    cited_chunk_ids_valid: bool = True
    reason: str = ""


class _VerifierIssuePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claim_idx: Optional[int] = None
    issue_type: Literal["unsupported", "contradiction", "bad_citation", "missing_exclusion", "overgeneralized"]
    description: str

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Issue description must be a non-empty string.")
        return v.strip()


class _RawVerifierPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    verdict: Literal["PASS", "FAIL"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    evidence_supported: bool
    contradiction_found: bool
    completeness_passed: bool
    claim_verdicts: List[_ClaimVerdictPayload] = Field(default_factory=list)
    issues: List[_VerifierIssuePayload] = Field(default_factory=list)
    summary: str = ""


class EvaluatorAgent:
    """
    Module 5 (Stage 8): Independent Evaluator Agent (Verifier).
    Reuses a managed LLMClient across verification calls (unless closed),
    auditing every atomic claim, evidence support, contradictions, overgeneralization,
    completeness, and citation validity against ONLY supplied evidence.
    """

    def __init__(
        self,
        confidence_threshold: Optional[float] = None,
        client: Optional[LLMClient] = None,
        provider: Optional[str] = None,
        nvidia_provider: Optional[NvidiaNemotronProvider] = None,
        app_settings: Optional[Settings] = None,
    ):
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None else settings.CONFIDENCE_THRESHOLD
        )
        self._app_settings = app_settings or settings
        self._owns_client = client is None
        self._client = client
        self._provider_override = provider
        self._nvidia_provider = nvidia_provider

    def _effective_provider_name(self) -> str:
        raw = (
            self._provider_override
            if self._provider_override is not None
            else (os.getenv("LLM_PROVIDER") or self._app_settings.LLM_PROVIDER)
        )
        return raw.strip().lower()

    def _get_client(self) -> LLMClient:
        if self._client is not None:
            if not self._owns_client:
                return self._client
            # Refresh owned client only if environment provider changed (e.g., across tests)
            # or if owned provider client was previously closed.
            expected_provider = self._effective_provider_name()
            provider_closed = (
                self._client._nvidia_provider is not None and self._client._nvidia_provider.is_closed
            )
            if self._client.provider_name == expected_provider and not provider_closed:
                return self._client
            self.close()

        self._client = LLMClient(
            provider=self._provider_override,
            nvidia_provider=self._nvidia_provider,
            app_settings=self._app_settings,
        )
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    @staticmethod
    def _resolve_page_bounds(chunk: PolicyChunk) -> tuple[int, Optional[int]]:
        meta_start = chunk.metadata.get("page_start") if chunk.metadata else None
        meta_end = chunk.metadata.get("page_end") if chunk.metadata else None
        p_start = chunk.page_start if chunk.page_start is not None else (
            int(meta_start) if meta_start is not None else chunk.page_number
        )
        p_end = chunk.page_end if chunk.page_end is not None else (
            int(meta_end) if meta_end is not None else None
        )
        return p_start, p_end

    def _format_evidence_chunks(self, chunks: List[PolicyChunk]) -> str:
        lines = []
        for i, c in enumerate(chunks, start=1):
            p_start, p_end = self._resolve_page_bounds(c)
            p_display = f"{p_start}-{p_end}" if p_end is not None and p_end != p_start else str(p_start)
            lines.append(
                f"[{i}] CHUNK_ID: {c.chunk_id} | PAGE: {p_display} | "
                f"CLAUSE: {c.clause_ref} | CATEGORY: {c.category.value}\n"
                f"CONTENT: {c.content_text}\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _parse_decision(raw_decision: Any) -> str:
        if isinstance(raw_decision, DecisionType):
            return raw_decision.value
        if isinstance(raw_decision, str):
            try:
                return DecisionType(raw_decision.strip()).value
            except ValueError as exc:
                raise LLMValidationError(f"Invalid draft decision value '{raw_decision}'.") from exc
        raise LLMValidationError("Draft decision must be a valid DecisionType.")

    @staticmethod
    def _validate_reason(raw_reason: Any) -> str:
        if not isinstance(raw_reason, str) or not raw_reason.strip():
            raise LLMValidationError("Draft 'reason' must be a non-empty string.")
        return raw_reason.strip()

    @staticmethod
    def _normalize_claims(raw_claims: Any) -> List[AtomicClaim]:
        if raw_claims is None:
            return []
        if not isinstance(raw_claims, list):
            raise LLMValidationError("Draft 'claims' must be a list.")
        normalized: List[AtomicClaim] = []
        for item in raw_claims:
            if isinstance(item, AtomicClaim):
                normalized.append(item)
            elif isinstance(item, dict):
                try:
                    normalized.append(AtomicClaim.model_validate(item))
                except ValidationError as exc:
                    raise LLMValidationError(f"Malformed draft claim: {exc}") from exc
            else:
                raise LLMValidationError("Each draft claim must be an AtomicClaim or dict.")
        return normalized

    @staticmethod
    def _normalize_conditions(raw_conditions: Any) -> List[str]:
        if raw_conditions is None:
            return []
        if not isinstance(raw_conditions, list):
            raise LLMValidationError("Draft 'conditions' must be a list.")
        normalized: List[str] = []
        for item in raw_conditions:
            if not isinstance(item, str):
                raise LLMValidationError("Each condition entry must be a string.")
            if item.strip():
                normalized.append(item.strip())
        return normalized

    @staticmethod
    def _normalize_citations(raw_citations: Any) -> List[Any]:
        if raw_citations is None:
            return []
        if not isinstance(raw_citations, list):
            raise LLMValidationError("Draft 'citations' must be a list.")
        for item in raw_citations:
            if not isinstance(item, (CitationItem, dict)):
                raise LLMValidationError("Each draft citation must be a CitationItem or dict.")
        return raw_citations

    @staticmethod
    def _format_draft_for_audit(
        decision: str,
        reason: str,
        conditions: List[str],
        claims: List[AtomicClaim],
    ) -> str:
        payload = {
            "decision": decision,
            "reason": reason,
            "conditions": conditions,
            "claims": [
                {
                    "claim_idx": idx,
                    "text": cl.text,
                    "chunk_ids": cl.chunk_ids,
                    "claim_type": cl.claim_type,
                }
                for idx, cl in enumerate(claims)
            ],
        }
        return json.dumps(payload, indent=2)

    def _run_deterministic_checks(
        self,
        query: str,
        evidence_chunks: List[PolicyChunk],
        decision: str,
        conditions: List[str],
        claims: List[AtomicClaim],
        citations: Optional[List[Any]] = None,
    ) -> tuple[bool, bool, bool, bool, List[EvaluatorIssue]]:
        issues: List[EvaluatorIssue] = []
        chunk_map = {c.chunk_id: c for c in evidence_chunks}

        # 1. Citation Accuracy Check (both claims and top-level citations)
        citation_valid = True
        for idx, cl in enumerate(claims):
            for cid in cl.chunk_ids:
                if not isinstance(cid, str) or cid not in chunk_map:
                    citation_valid = False
                    issues.append(
                        EvaluatorIssue(
                            claim_idx=idx,
                            issue_type="bad_citation",
                            description=f"Claim cites unknown or unretrieved chunk_id '{cid}'.",
                        )
                    )

        for cit in citations or []:
            cid = cit.chunk_id if isinstance(cit, CitationItem) else cit.get("chunk_id")
            if not isinstance(cid, str) or not cid.strip() or cid not in chunk_map:
                citation_valid = False
                issues.append(
                    EvaluatorIssue(
                        claim_idx=None,
                        issue_type="bad_citation",
                        description=f"Draft citations contain unknown or invalid chunk_id '{cid}'.",
                    )
                )

        # 2. Evidence Support Check
        evidence_supported = True
        if decision != "INSUFFICIENT EVIDENCE" and not claims:
            evidence_supported = False
            issues.append(
                EvaluatorIssue(
                    claim_idx=None,
                    issue_type="unsupported",
                    description="Affirmative decision made without supporting atomic claims.",
                )
            )

        for idx, cl in enumerate(claims):
            if not cl.text or not cl.text.strip():
                evidence_supported = False
                issues.append(
                    EvaluatorIssue(
                        claim_idx=idx,
                        issue_type="unsupported",
                        description=f"Claim {idx} has empty text.",
                    )
                )
            if not cl.chunk_ids:
                evidence_supported = False
                issues.append(
                    EvaluatorIssue(
                        claim_idx=idx,
                        issue_type="unsupported",
                        description=f"Claim text '{cl.text[:60]}' has no chunk citations.",
                    )
                )

        # 3. Clause-Aware Exclusion Contradiction Check
        contradiction_found = False
        conflicts = find_exclusion_contradictions(
            query=query,
            evidence_chunks=evidence_chunks,
            decision=decision,
            claims=claims,
            conditions=conditions,
        )
        for ec, term in conflicts:
            contradiction_found = True
            issues.append(
                EvaluatorIssue(
                    claim_idx=None,
                    issue_type="contradiction",
                    description=(
                        f"Contradiction: Decision is COVERED, but Clause '{ec.clause_ref}' "
                        f"excludes '{term}' without established exception support."
                    ),
                )
            )

        # 4. Deterministic Completeness Check
        completeness_passed = True
        if decision != DecisionType.INSUFFICIENT_EVIDENCE.value:
            q_lower = query.lower()
            condition_chunks = [c for c in evidence_chunks if c.category == ChunkCategory.CONDITIONS]
            for cc in condition_chunks:
                cc_text = cc.content_text.lower()
                if "waiting period" in cc_text and "waiting period" in q_lower:
                    if not any("waiting" in str(cond).lower() for cond in conditions):
                        completeness_passed = False
                        issues.append(
                            EvaluatorIssue(
                                claim_idx=None,
                                issue_type="missing_exclusion",
                                description=f"Relevant waiting period in '{cc.clause_ref}' was omitted from conditions.",
                            )
                        )

        return evidence_supported, contradiction_found, citation_valid, completeness_passed, issues

    def _run_llm_verification(
        self,
        client: LLMClient,
        query: str,
        evidence_chunks: List[PolicyChunk],
        decision: str,
        reason: str,
        conditions: List[str],
        claims: List[AtomicClaim],
    ) -> _RawVerifierPayload:
        evidence_text = self._format_evidence_chunks(evidence_chunks)
        draft_json = self._format_draft_for_audit(decision, reason, conditions, claims)
        prompt = format_verification_prompt(query, draft_json, evidence_text)

        raw_verdict = client.generate_json(VERIFIER_SYSTEM_PROMPT, prompt)
        if not isinstance(raw_verdict, dict):
            raise LLMValidationError("Verifier output must be a JSON object.")

        try:
            parsed = _RawVerifierPayload.model_validate(raw_verdict)
        except ValidationError as exc:
            raise LLMValidationError(f"Verifier output failed schema validation: {exc}") from exc

        seen_indices = set()
        for cv in parsed.claim_verdicts:
            if cv.claim_idx >= len(claims):
                raise LLMValidationError(
                    f"Verifier returned out-of-range claim_idx={cv.claim_idx} for {len(claims)} claims."
                )
            if cv.claim_idx in seen_indices:
                raise LLMValidationError(f"Verifier returned duplicate claim_idx={cv.claim_idx}.")
            seen_indices.add(cv.claim_idx)

        if claims and len(seen_indices) != len(claims):
            raise LLMValidationError(
                f"Verifier audited {len(seen_indices)} of {len(claims)} claims; every atomic claim must be audited."
            )

        for iss in parsed.issues:
            if iss.claim_idx is not None and (iss.claim_idx < 0 or iss.claim_idx >= len(claims)):
                raise LLMValidationError(
                    f"Verifier issue references out-of-range claim_idx={iss.claim_idx}."
                )

        return parsed

    def verify(
        self,
        query: str,
        evidence_chunks: List[PolicyChunk],
        draft_response: Dict[str, Any],
    ) -> EvaluatorResult:
        if not isinstance(draft_response, dict):
            raise LLMValidationError("Draft response must be a dictionary.")

        decision = self._parse_decision(draft_response.get("decision"))
        reason = self._validate_reason(draft_response.get("reason"))
        claims = self._normalize_claims(draft_response.get("claims"))
        conditions = self._normalize_conditions(draft_response.get("conditions"))
        citations = self._normalize_citations(draft_response.get("citations"))

        (
            det_evidence_supported,
            det_contradiction_found,
            det_citation_valid,
            det_completeness_passed,
            issues,
        ) = self._run_deterministic_checks(
            query=query,
            evidence_chunks=evidence_chunks,
            decision=decision,
            conditions=conditions,
            claims=claims,
            citations=citations,
        )

        evidence_supported = det_evidence_supported
        contradiction_found = det_contradiction_found
        citation_valid = det_citation_valid
        completeness_passed = det_completeness_passed

        if (
            decision == DecisionType.INSUFFICIENT_EVIDENCE.value
            and not claims
            and not citations
            and not conditions
            and evidence_supported
            and not contradiction_found
            and citation_valid
            and completeness_passed
            and not issues
        ):
            logger.info("Skipping verifier LLM call for deterministically valid empty INSUFFICIENT EVIDENCE draft.")
            return EvaluatorResult(
                passed=True,
                confidence_score=0.95,
                evidence_supported=True,
                contradiction_found=False,
                citation_valid=True,
                completeness_passed=True,
                issues=[],
                raw_evaluator_comment=(
                    "Deterministic validation passed for empty INSUFFICIENT EVIDENCE draft; "
                    "skipped verifier LLM call."
                ),
            )

        client = self._get_client()
        llm_confidence: Optional[float] = None
        llm_verdict_pass = True
        audit_comment = f"Evaluated {len(claims)} claims against {len(evidence_chunks)} evidence chunks."

        if client.provider_name == "nvidia":
            if not evidence_chunks:
                llm_confidence = 0.95 if decision == "INSUFFICIENT EVIDENCE" and not claims else 0.10
                llm_verdict_pass = decision == "INSUFFICIENT EVIDENCE" and not claims
            else:
                llm_audit = self._run_llm_verification(
                    client=client,
                    query=query,
                    evidence_chunks=evidence_chunks,
                    decision=decision,
                    reason=reason,
                    conditions=conditions,
                    claims=claims,
                )
                llm_confidence = llm_audit.confidence_score
                llm_verdict_pass = llm_audit.verdict == "PASS"

                if not llm_audit.evidence_supported:
                    evidence_supported = False
                if llm_audit.contradiction_found:
                    contradiction_found = True
                if not llm_audit.completeness_passed:
                    completeness_passed = False

                for cv in llm_audit.claim_verdicts:
                    if not cv.cited_chunk_ids_valid:
                        citation_valid = False
                        issues.append(
                            EvaluatorIssue(
                                claim_idx=cv.claim_idx,
                                issue_type="bad_citation",
                                description=cv.reason or f"Claim {cv.claim_idx} citation invalid per verifier.",
                            )
                        )
                    if not cv.supported or cv.status != "SUPPORTED":
                        if cv.status == "CONTRADICTED":
                            contradiction_found = True
                            issue_type = "contradiction"
                        elif cv.status == "OVERGENERALIZED":
                            evidence_supported = False
                            issue_type = "overgeneralized"
                        else:
                            evidence_supported = False
                            issue_type = "unsupported"
                        issues.append(
                            EvaluatorIssue(
                                claim_idx=cv.claim_idx,
                                issue_type=issue_type,
                                description=cv.reason or f"Claim {cv.claim_idx} marked {cv.status} by verifier.",
                            )
                        )

                for iss in llm_audit.issues:
                    candidate = EvaluatorIssue(
                        claim_idx=iss.claim_idx,
                        issue_type=iss.issue_type,
                        description=iss.description,
                    )
                    if candidate not in issues:
                        issues.append(candidate)
                    if iss.issue_type == "bad_citation":
                        citation_valid = False
                    elif iss.issue_type == "contradiction":
                        contradiction_found = True
                    elif iss.issue_type == "missing_exclusion":
                        completeness_passed = False
                    elif iss.issue_type in ("unsupported", "overgeneralized"):
                        evidence_supported = False

                if llm_audit.summary:
                    audit_comment = llm_audit.summary

        det_confidence = 1.0
        if not evidence_supported:
            det_confidence -= 0.40
        if contradiction_found:
            det_confidence -= 0.50
        if not citation_valid:
            det_confidence -= 0.30
        if not completeness_passed:
            det_confidence -= 0.15
        det_confidence -= len(issues) * 0.05
        det_confidence = max(0.10, min(round(det_confidence, 2), 1.0))

        final_confidence = (
            min(round(llm_confidence, 2), det_confidence) if llm_confidence is not None else det_confidence
        )

        passed = (
            llm_verdict_pass
            and evidence_supported
            and not contradiction_found
            and citation_valid
            and completeness_passed
            and (final_confidence >= self.confidence_threshold)
        )

        return EvaluatorResult(
            passed=passed,
            confidence_score=final_confidence,
            evidence_supported=evidence_supported,
            contradiction_found=contradiction_found,
            citation_valid=citation_valid,
            completeness_passed=completeness_passed,
            issues=issues,
            raw_evaluator_comment=audit_comment,
        )
