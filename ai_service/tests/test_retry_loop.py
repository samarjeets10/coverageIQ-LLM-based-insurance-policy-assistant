from ai_service.evaluator.retry_controller import ClosedRetryLoopOrchestrator
from ai_service.schemas.structured_output import DecisionType

def test_closed_retry_loop_hospitalization_covered():
    orchestrator = ClosedRetryLoopOrchestrator()
    response = orchestrator.execute_qa(
        query="Is in-patient hospitalization covered?",
        policy_id="policy_health_01"
    )
    
    assert response.decision == DecisionType.COVERED
    assert response.evaluator_status == "VERIFIED_PASS"
    assert response.confidence_score >= 0.70
    assert len(response.citations) > 0
    assert any(c.page == 12 for c in response.citations)
    assert any("Section 2.1" in c.clause for c in response.citations)

def test_closed_retry_loop_cosmetic_surgery_excluded():
    orchestrator = ClosedRetryLoopOrchestrator()
    response = orchestrator.execute_qa(
        query="Can I claim for cosmetic surgery?",
        policy_id="policy_health_01"
    )
    
    assert response.decision == DecisionType.NOT_COVERED
    assert response.evaluator_status == "VERIFIED_PASS"
    assert len(response.citations) > 0
    assert any("Section 4.1" in c.clause for c in response.citations)

def test_closed_retry_loop_maternity_conditional():
    orchestrator = ClosedRetryLoopOrchestrator()
    response = orchestrator.execute_qa(
        query="Is maternity delivery covered?",
        policy_id="policy_health_01"
    )
    
    assert response.decision == DecisionType.CONDITIONAL
    assert response.evaluator_status == "VERIFIED_PASS"
    assert len(response.conditions) > 0
    assert any("24" in cond for cond in response.conditions)

def test_closed_retry_loop_unanswerable_fallback_insufficient_evidence():
    orchestrator = ClosedRetryLoopOrchestrator()
    response = orchestrator.execute_qa(
        query="Does this policy pay for space travel accidents or dental implants?",
        policy_id="policy_health_01"
    )
    
    # Enforces NO EVIDENCE = NO CLAIM: Never hallucinate!
    assert response.decision == DecisionType.INSUFFICIENT_EVIDENCE
    assert "VERIFIED" in response.evaluator_status
