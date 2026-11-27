from ai_service.evaluator.verifier import EvaluatorAgent
from ai_service.database import get_knowledge_base
from ai_service.schemas.structured_output import AtomicClaim

def test_evaluator_passes_valid_evidence():
    kb = get_knowledge_base()
    evidence = kb.get_chunks("policy_health_01")
    evaluator = EvaluatorAgent(confidence_threshold=0.70)

    query = "Is in-patient hospitalization covered?"
    valid_draft = {
        "decision": "COVERED",
        "reason": "Hospitalization is covered under Section 2.1 for stays >= 24 hours.",
        "conditions": ["Minimum 24 hours stay required"],
        "claims": [
            AtomicClaim(
                text="In-patient hospitalization is covered for minimum 24 hours.",
                chunk_ids=["chunk_h_001"],
                claim_type="coverage"
            )
        ]
    }

    result = evaluator.verify(query, evidence, valid_draft)
    assert result.passed is True
    assert result.evidence_supported is True
    assert result.contradiction_found is False
    assert result.citation_valid is True
    assert result.confidence_score >= 0.70

def test_evaluator_catches_unsupported_claim():
    kb = get_knowledge_base()
    evidence = kb.get_chunks("policy_health_01")
    evaluator = EvaluatorAgent(confidence_threshold=0.70)

    query = "Is dental checkup covered?"
    unsupported_draft = {
        "decision": "COVERED",
        "reason": "Dental is covered up to INR 20,000.",
        "conditions": [],
        "claims": [
            AtomicClaim(
                text="Dental surgery is covered up to INR 20,000.",
                chunk_ids=[],  # No citation!
                claim_type="coverage"
            )
        ]
    }

    result = evaluator.verify(query, evidence, unsupported_draft)
    assert result.passed is False
    assert result.evidence_supported is False
    assert any(i.issue_type == "unsupported" for i in result.issues)

def test_evaluator_catches_contradiction():
    kb = get_knowledge_base()
    evidence = kb.get_chunks("policy_health_01")
    evaluator = EvaluatorAgent(confidence_threshold=0.70)

    query = "Is cosmetic surgery covered?"
    contradictory_draft = {
        "decision": "COVERED",  # Contradicts Section 4.1 which excludes cosmetic surgery!
        "reason": "Cosmetic surgery is fully covered under the plan.",
        "conditions": [],
        "claims": [
            AtomicClaim(
                text="Cosmetic surgery is covered.",
                chunk_ids=["chunk_h_003"],
                claim_type="coverage"
            )
        ]
    }

    result = evaluator.verify(query, evidence, contradictory_draft)
    assert result.passed is False
    assert result.contradiction_found is True
    assert any(i.issue_type == "contradiction" for i in result.issues)

def test_evaluator_catches_bad_citation():
    kb = get_knowledge_base()
    evidence = kb.get_chunks("policy_health_01")
    evaluator = EvaluatorAgent(confidence_threshold=0.70)

    query = "Is hospitalization covered?"
    fake_citation_draft = {
        "decision": "COVERED",
        "reason": "Hospitalization is covered.",
        "conditions": [],
        "claims": [
            AtomicClaim(
                text="Hospitalization is covered.",
                chunk_ids=["non_existent_chunk_999"],
                claim_type="coverage"
            )
        ]
    }

    result = evaluator.verify(query, evidence, fake_citation_draft)
    assert result.passed is False
    assert result.citation_valid is False
    assert any(i.issue_type == "bad_citation" for i in result.issues)
