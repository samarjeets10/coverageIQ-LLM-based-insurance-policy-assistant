from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, ValidationError

from ai_service.config import settings
from ai_service.evaluator.verifier import find_exclusion_contradictions
from ai_service.llm.client import LLMValidationError
from ai_service.schemas.structured_output import (
    AtomicClaim,
    ChunkCategory,
    CitationItem,
    DecisionType,
    EvaluatorIssue,
    EvaluatorResult,
    PolicyChunk,
)


class DecisionGateResult(BaseModel):
    """
    Outcome of the deterministic decision gate.
    Kept strictly separate from the Nemotron verification LLM call so that
    model self-assessment alone never authorizes an ungrounded response.
    """

    passed: bool
    final_decision: DecisionType
    confidence_score: float = Field(ge=0.0, le=1.0)
    evidence_supported: bool
    contradiction_found: bool
    citation_valid: bool
    completeness_passed: bool
    verified_claims: List[AtomicClaim] = Field(default_factory=list)
    verified_citations: List[CitationItem] = Field(default_factory=list)
    verified_conditions: List[str] = Field(default_factory=list)
    issues: List[EvaluatorIssue] = Field(default_factory=list)


class DeterministicDecisionGate:
    """
    Independent deterministic safety gate enforcing NO EVIDENCE = NO CLAIM.
    Validates collection types, citations, claim grounding, clause-aware exclusion
    contradictions, and confidence thresholds after verification.
    """

    def __init__(self, confidence_threshold: Optional[float] = None):
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None else settings.CONFIDENCE_THRESHOLD
        )

    @staticmethod
    def _parse_decision(raw_decision: Any) -> DecisionType:
        if isinstance(raw_decision, DecisionType):
            return raw_decision
        if isinstance(raw_decision, str):
            try:
                return DecisionType(raw_decision.strip())
            except ValueError as exc:
                raise LLMValidationError(f"Invalid draft decision value '{raw_decision}'.") from exc
        raise LLMValidationError("Draft decision must be a valid DecisionType.")

    @staticmethod
    def _validate_conditions(raw_conditions: Any) -> List[str]:
        if raw_conditions is None:
            return []
        if not isinstance(raw_conditions, list):
            raise LLMValidationError("Draft 'conditions' must be a list of strings.")
        result: List[str] = []
        for item in raw_conditions:
            if not isinstance(item, str):
                raise LLMValidationError("Each condition entry must be a string.")
            if item.strip():
                result.append(item.strip())
        return result

    @staticmethod
    def _validate_claims(raw_claims: Any) -> List[AtomicClaim]:
        if raw_claims is None:
            return []
        if not isinstance(raw_claims, list):
            raise LLMValidationError("Draft 'claims' must be a list.")
        claims: List[AtomicClaim] = []
        for cl in raw_claims:
            if isinstance(cl, AtomicClaim):
                claims.append(cl)
            elif isinstance(cl, dict):
                try:
                    claims.append(AtomicClaim.model_validate(cl))
                except ValidationError as exc:
                    raise LLMValidationError(f"Malformed draft claim: {exc}") from exc
            else:
                raise LLMValidationError("Each draft claim must be an AtomicClaim or dict.")
        return claims

    @staticmethod
    def _validate_citations(raw_citations: Any) -> List[Any]:
        if raw_citations is None:
            return []
        if not isinstance(raw_citations, list):
            raise LLMValidationError("Draft 'citations' must be a list.")
        for cit in raw_citations:
            if not isinstance(cit, (CitationItem, dict)):
                raise LLMValidationError("Each draft citation must be a CitationItem or dict.")
        return raw_citations

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

    def _build_verified_citations(
        self,
        claims: List[AtomicClaim],
        chunk_lookup: Dict[str, PolicyChunk],
    ) -> List[CitationItem]:
        seen: Set[str] = set()
        citations: List[CitationItem] = []
        for cl in claims:
            for cid in cl.chunk_ids:
                if cid in chunk_lookup and cid not in seen:
                    seen.add(cid)
                    chunk = chunk_lookup[cid]
                    p_start, p_end = self._resolve_page_bounds(chunk)
                    citations.append(
                        CitationItem(
                            chunk_id=chunk.chunk_id,
                            page=p_start,
                            page_end=p_end if p_end is not None and p_end != p_start else None,
                            clause=chunk.clause_ref,
                            category=chunk.category.value,
                            snippet=(
                                chunk.content_text[:160] + "..."
                                if len(chunk.content_text) > 160
                                else chunk.content_text
                            ),
                        )
                    )
        return citations

    def evaluate(
        self,
        query: str,
        evidence_chunks: List[PolicyChunk],
        draft_response: Dict[str, Any],
        verifier_result: EvaluatorResult,
    ) -> DecisionGateResult:
        if not isinstance(draft_response, dict):
            raise LLMValidationError("Draft response must be a dictionary.")

        draft_decision = self._parse_decision(draft_response.get("decision"))
        conditions = self._validate_conditions(draft_response.get("conditions"))
        claims = self._validate_claims(draft_response.get("claims"))
        raw_citations = self._validate_citations(draft_response.get("citations"))

        chunk_lookup = {c.chunk_id: c for c in evidence_chunks}
        issues: List[EvaluatorIssue] = list(verifier_result.issues)

        evidence_supported = verifier_result.evidence_supported
        contradiction_found = verifier_result.contradiction_found
        citation_valid = verifier_result.citation_valid
        completeness_passed = verifier_result.completeness_passed

        # Track claim indices rejected by either the verifier or deterministic checks
        rejected_claim_indices: Set[int] = set()
        for iss in issues:
            if iss.claim_idx is not None and iss.issue_type in (
                "unsupported",
                "contradiction",
                "bad_citation",
                "overgeneralized",
            ):
                rejected_claim_indices.add(iss.claim_idx)

        # Independent check 1: Validate every draft claim's chunk_ids against retrieved evidence
        for idx, cl in enumerate(claims):
            if not cl.text or not cl.text.strip():
                evidence_supported = False
                rejected_claim_indices.add(idx)
                issues.append(
                    EvaluatorIssue(
                        claim_idx=idx,
                        issue_type="unsupported",
                        description=f"Claim {idx} has empty text.",
                    )
                )
            if not cl.chunk_ids:
                evidence_supported = False
                rejected_claim_indices.add(idx)
                issues.append(
                    EvaluatorIssue(
                        claim_idx=idx,
                        issue_type="unsupported",
                        description=f"Claim {idx} has no chunk_ids.",
                    )
                )
            for cid in cl.chunk_ids:
                if not isinstance(cid, str) or cid not in chunk_lookup:
                    citation_valid = False
                    rejected_claim_indices.add(idx)
                    issues.append(
                        EvaluatorIssue(
                            claim_idx=idx,
                            issue_type="bad_citation",
                            description=f"Claim {idx} cites unknown chunk_id '{cid}'.",
                        )
                    )

        # Independent check 2: Reject invalid citations passed in draft["citations"]
        for cit in raw_citations:
            cid = cit.chunk_id if isinstance(cit, CitationItem) else cit.get("chunk_id")
            if not isinstance(cid, str) or not cid.strip() or cid not in chunk_lookup:
                citation_valid = False
                issues.append(
                    EvaluatorIssue(
                        claim_idx=None,
                        issue_type="bad_citation",
                        description=f"Draft citations contain unknown or invalid chunk_id '{cid}'.",
                    )
                )

        # Independent check 3: Positive / affirmative decisions require evidence and supported claims
        if draft_decision != DecisionType.INSUFFICIENT_EVIDENCE:
            if not evidence_chunks or not claims:
                evidence_supported = False
                issues.append(
                    EvaluatorIssue(
                        claim_idx=None,
                        issue_type="unsupported",
                        description=(
                            f"Decision '{draft_decision.value}' cannot be issued without "
                            "retrieved evidence and supporting claims."
                        ),
                    )
                )

        # Independent check 4: Clause-aware deterministic exclusion contradiction guard
        conflicts = find_exclusion_contradictions(
            query=query,
            evidence_chunks=evidence_chunks,
            decision=draft_decision.value,
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
                        f"Contradiction: Decision is COVERED while Clause '{ec.clause_ref}' "
                        f"excludes '{term}' without established exception support."
                    ),
                )
            )

        # Independent check 5: Deterministic waiting-period completeness guard
        if draft_decision != DecisionType.INSUFFICIENT_EVIDENCE:
            q_lower = query.lower()
            for cc in evidence_chunks:
                if cc.category == ChunkCategory.CONDITIONS:
                    if "waiting period" in cc.content_text.lower() and "waiting period" in q_lower:
                        if not any("waiting" in cond.lower() for cond in conditions):
                            completeness_passed = False
                            issues.append(
                                EvaluatorIssue(
                                    claim_idx=None,
                                    issue_type="missing_exclusion",
                                    description=f"Waiting period condition from '{cc.clause_ref}' omitted.",
                                )
                            )

        deduped_issues: List[EvaluatorIssue] = []
        for iss in issues:
            if iss not in deduped_issues:
                deduped_issues.append(iss)

        det_conf = 1.0
        if not evidence_supported:
            det_conf -= 0.40
        if contradiction_found:
            det_conf -= 0.50
        if not citation_valid:
            det_conf -= 0.30
        if not completeness_passed:
            det_conf -= 0.15
        det_conf -= len(deduped_issues) * 0.05
        det_conf = max(0.10, min(round(det_conf, 2), 1.0))

        final_confidence = min(round(verifier_result.confidence_score, 2), det_conf)
        all_claims_supported = len(rejected_claim_indices) == 0

        gate_passed = (
            verifier_result.passed
            and evidence_supported
            and not contradiction_found
            and citation_valid
            and completeness_passed
            and all_claims_supported
            and (final_confidence >= self.confidence_threshold)
        )

        if not gate_passed:
            return DecisionGateResult(
                passed=False,
                final_decision=DecisionType.INSUFFICIENT_EVIDENCE,
                confidence_score=final_confidence,
                evidence_supported=evidence_supported,
                contradiction_found=contradiction_found,
                citation_valid=citation_valid,
                completeness_passed=completeness_passed,
                verified_claims=[],
                verified_citations=[],
                verified_conditions=[],
                issues=deduped_issues,
            )

        if draft_decision == DecisionType.INSUFFICIENT_EVIDENCE:
            return DecisionGateResult(
                passed=True,
                final_decision=DecisionType.INSUFFICIENT_EVIDENCE,
                confidence_score=final_confidence,
                evidence_supported=evidence_supported,
                contradiction_found=contradiction_found,
                citation_valid=citation_valid,
                completeness_passed=completeness_passed,
                verified_claims=[],
                verified_citations=[],
                verified_conditions=[],
                issues=deduped_issues,
            )

        verified_citations = self._build_verified_citations(claims, chunk_lookup)
        if not verified_citations:
            return DecisionGateResult(
                passed=False,
                final_decision=DecisionType.INSUFFICIENT_EVIDENCE,
                confidence_score=min(final_confidence, 0.50),
                evidence_supported=False,
                contradiction_found=contradiction_found,
                citation_valid=False,
                completeness_passed=completeness_passed,
                verified_claims=[],
                verified_citations=[],
                verified_conditions=[],
                issues=deduped_issues
                + [
                    EvaluatorIssue(
                        claim_idx=None,
                        issue_type="bad_citation",
                        description="No valid retrieved citations resolved for affirmative decision.",
                    )
                ],
            )

        return DecisionGateResult(
            passed=True,
            final_decision=draft_decision,
            confidence_score=final_confidence,
            evidence_supported=evidence_supported,
            contradiction_found=contradiction_found,
            citation_valid=citation_valid,
            completeness_passed=completeness_passed,
            verified_claims=claims,
            verified_citations=verified_citations,
            verified_conditions=conditions,
            issues=deduped_issues,
        )
