import logging
from typing import List, Dict, Any, Set
from ai_service.schemas.structured_output import (
    PolicyChunk,
    EvaluatorResult,
    EvaluatorIssue,
    ChunkCategory
)
from ai_service.config import settings

logger = logging.getLogger("coverageiq.evaluator")

class EvaluatorAgent:
    """
    Module 5 (Stage 8): Independent Evaluator Agent (Verifier)
    Audits draft responses across five critical dimensions:
    1. Evidence Support Check
    2. Contradiction Detection
    3. Citation Accuracy Check
    4. Completeness & Hallucination Check
    5. Calibrated Confidence Scoring
    """
    def __init__(self, confidence_threshold: float = None):
        self.confidence_threshold = confidence_threshold or settings.CONFIDENCE_THRESHOLD

    def verify(
        self,
        query: str,
        evidence_chunks: List[PolicyChunk],
        draft_response: Dict[str, Any]
    ) -> EvaluatorResult:
        issues: List[EvaluatorIssue] = []
        chunk_map = {c.chunk_id: c for c in evidence_chunks}
        
        claims = draft_response.get("claims", [])
        decision = draft_response.get("decision", "INSUFFICIENT EVIDENCE")
        conditions = draft_response.get("conditions", [])
        
        # 1. Check Citation Accuracy
        citation_valid = True
        cited_cids: Set[str] = set()
        for idx, cl in enumerate(claims):
            for cid in cl.chunk_ids:
                cited_cids.add(cid)
                if cid not in chunk_map:
                    citation_valid = False
                    issues.append(
                        EvaluatorIssue(
                            claim_idx=idx,
                            issue_type="bad_citation",
                            description=f"Claim cites unknown or unretrieved chunk_id '{cid}'."
                        )
                    )

        # 2. Check Evidence Support
        evidence_supported = True
        if decision != "INSUFFICIENT EVIDENCE" and not claims:
            evidence_supported = False
            issues.append(
                EvaluatorIssue(
                    claim_idx=None,
                    issue_type="unsupported",
                    description="Affirmative decision made without supporting atomic claims."
                )
            )

        for idx, cl in enumerate(claims):
            if not cl.chunk_ids:
                evidence_supported = False
                issues.append(
                    EvaluatorIssue(
                        claim_idx=idx,
                        issue_type="unsupported",
                        description=f"Claim text '{cl.text[:60]}' has no chunk citations."
                    )
                )

        # 3. Check Contradictions (e.g. Asserting COVERED when an active exclusion exists in evidence)
        contradiction_found = False
        exclusion_chunks = [c for c in evidence_chunks if c.category == ChunkCategory.EXCLUSIONS]
        q_lower = query.lower()
        
        # Check if an exclusion chunk directly matches the query but decision was COVERED
        for ec in exclusion_chunks:
            ec_text = ec.content_text.lower()
            # If query asks about cosmetic surgery and exclusion says cosmetic surgery is NOT COVERED
            for term in ["cosmetic", "plastic surgery", "aesthetic", "self-inflicted", "substance abuse"]:
                if term in q_lower and term in ec_text:
                    if decision == "COVERED":
                        contradiction_found = True
                        issues.append(
                            EvaluatorIssue(
                                claim_idx=None,
                                issue_type="contradiction",
                                description=(
                                    f"Contradiction: Decision is COVERED, but Clause '{ec.clause_ref}' "
                                    f"explicitly excludes '{term}'."
                                )
                            )
                        )

        # 4. Completeness Check (Missing conditions / waiting periods)
        completeness_passed = True
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
                            description=f"Relevant waiting period in '{cc.clause_ref}' was omitted from conditions."
                        )
                    )

        # 5. Calibrated Confidence Scoring
        confidence = 1.0
        if not evidence_supported:
            confidence -= 0.40
        if contradiction_found:
            confidence -= 0.50
        if not citation_valid:
            confidence -= 0.30
        if not completeness_passed:
            confidence -= 0.15

        # Penalty for issue counts
        confidence -= len(issues) * 0.05
        confidence = max(0.10, min(round(confidence, 2), 1.0))

        # Check overall pass criteria
        passed = (
            evidence_supported
            and not contradiction_found
            and citation_valid
            and completeness_passed
            and (confidence >= self.confidence_threshold)
        )

        return EvaluatorResult(
            passed=passed,
            confidence_score=confidence,
            evidence_supported=evidence_supported,
            contradiction_found=contradiction_found,
            citation_valid=citation_valid,
            completeness_passed=completeness_passed,
            issues=issues,
            raw_evaluator_comment=f"Evaluated {len(claims)} claims against {len(evidence_chunks)} evidence chunks."
        )
