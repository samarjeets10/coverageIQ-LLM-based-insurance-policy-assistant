import logging
import os
import traceback
from fastapi import APIRouter, Depends, HTTPException

from ai_service.evaluator.retry_controller import ClosedRetryLoopOrchestrator
from ai_service.llm.client import LLMGenerationError
from ai_service.llm.providers.nvidia_nemotron import (
    NvidiaAuthenticationError,
    NvidiaConfigurationError,
    NvidiaPermissionError,
    NvidiaProviderError,
    NvidiaRateLimitError,
    NvidiaTimeoutError,
)
from ai_service.schemas.query import AskQueryRequest
from ai_service.schemas.structured_output import StructuredAnswerResponse

logger = logging.getLogger("coverageiq.api.qa")
router = APIRouter()

# Dependency injector for orchestrator
_orchestrator = None


def get_orchestrator() -> ClosedRetryLoopOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ClosedRetryLoopOrchestrator()
    return _orchestrator


def close_orchestrator() -> None:
    global _orchestrator
    if _orchestrator is not None:
        _orchestrator.close()
        _orchestrator = None


@router.post(
    "/ask",
    response_model=StructuredAnswerResponse,
    summary="Query insurance policy with evidence grounding & evaluator retry loop",
)
def ask_coverage_query(
    payload: AskQueryRequest,
    orchestrator: ClosedRetryLoopOrchestrator = Depends(get_orchestrator),
) -> StructuredAnswerResponse:
    """
    Module 4 & 5 (6C Pipeline):
    - Hybrid Retrieval (Vector + BM25)
    - Cross-Encoder Reranker
    - Grounded LLM Prompting ('NO EVIDENCE = NO CLAIM')
    - Independent Verifier + Deterministic Decision Gate
    - Closed Retry Loop (up to 2 retries)
    """
    logger.info(
        "Received QA query request: policy_id=%s question_length=%d",
        payload.policy_id,
        len(payload.question),
    )
    try:
        return orchestrator.execute_qa(
            query=payload.question,
            policy_id=payload.policy_id,
        )
    except NvidiaTimeoutError as exc:
        logger.error(
            "QA request failed due to upstream provider timeout: policy_id=%s error_type=%s",
            payload.policy_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=504,
            detail="AI provider request timed out while processing the policy query.",
        )
    except NvidiaRateLimitError as exc:
        logger.warning(
            "QA request throttled by upstream provider: policy_id=%s error_type=%s status_code=%s",
            payload.policy_id,
            type(exc).__name__,
            exc.status_code,
        )
        raise HTTPException(
            status_code=429,
            detail="AI provider rate limit exceeded. Please retry shortly.",
        )
    except (NvidiaConfigurationError, NvidiaAuthenticationError, NvidiaPermissionError) as exc:
        logger.error(
            "QA request blocked by provider configuration or authorization error: policy_id=%s error_type=%s status_code=%s",
            payload.policy_id,
            type(exc).__name__,
            exc.status_code,
        )
        raise HTTPException(
            status_code=503,
            detail="AI provider service is currently unavailable.",
        )
    except NvidiaProviderError as exc:
        logger.error(
            "QA request failed due to upstream provider error: policy_id=%s error_type=%s status_code=%s finish_reason=%s request_id=%s",
            payload.policy_id,
            type(exc).__name__,
            exc.status_code,
            getattr(exc, "finish_reason", None),
            getattr(exc, "request_id", None),
        )
        raise HTTPException(
            status_code=502,
            detail="Upstream AI provider failed to complete the request.",
        )
    except LLMGenerationError as exc:
        logger.error(
            "QA request failed due to invalid AI structured output: policy_id=%s error_type=%s",
            payload.policy_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=502,
            detail="AI service failed to produce a valid verifiable response.",
        )
    except Exception as exc:
        frames = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ else []
        trace_locs = [f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in frames[-5:]]
        logger.error(
            "Unexpected error processing policy QA query: policy_id=%s error_type=%s trace=%s",
            payload.policy_id,
            type(exc).__name__,
            " -> ".join(trace_locs) if trace_locs else "unavailable",
        )
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the policy query.",
        )
