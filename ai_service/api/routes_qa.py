import logging
from fastapi import APIRouter, HTTPException, Depends
from ai_service.schemas.query import AskQueryRequest
from ai_service.schemas.structured_output import StructuredAnswerResponse
from ai_service.evaluator.retry_controller import ClosedRetryLoopOrchestrator

logger = logging.getLogger("coverageiq.api.qa")
router = APIRouter()

# Dependency injector for orchestrator
_orchestrator = None

def get_orchestrator() -> ClosedRetryLoopOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ClosedRetryLoopOrchestrator()
    return _orchestrator

@router.post("/ask", response_model=StructuredAnswerResponse, summary="Query insurance policy with evidence grounding & evaluator retry loop")
def ask_coverage_query(
    payload: AskQueryRequest,
    orchestrator: ClosedRetryLoopOrchestrator = Depends(get_orchestrator)
) -> StructuredAnswerResponse:
    """
    Module 4 & 5 (6C Pipeline):
    - Hybrid Retrieval (Vector + BM25)
    - Cross-Encoder Reranker
    - Grounded LLM Prompting ('NO EVIDENCE = NO CLAIM')
    - Evaluator Agent Verification (5 dimensions)
    - Closed Retry Loop (up to 2 retries)
    """
    logger.info(f"Received query on policy '{payload.policy_id}': {payload.question}")
    try:
        response = orchestrator.execute_qa(
            query=payload.question,
            policy_id=payload.policy_id
        )
        return response
    except Exception as e:
        logger.exception(f"Error processing policy QA query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
