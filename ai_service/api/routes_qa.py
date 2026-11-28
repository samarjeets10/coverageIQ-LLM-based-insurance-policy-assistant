import logging
import os
import traceback
from typing import Optional
from fastapi import APIRouter, Depends, Header, HTTPException

from ai_service.config import Settings, settings
from ai_service.evaluator.retry_controller import ClosedRetryLoopOrchestrator
from ai_service.llm.client import (
    ExternalProviderConfigurationError,
    ExternalProviderError,
    LLMGenerationError,
)
from ai_service.llm.providers.nvidia_nemotron import (
    NvidiaAuthenticationError,
    NvidiaConfigurationError,
    NvidiaPermissionError,
    NvidiaProviderError,
    NvidiaRateLimitError,
    NvidiaTimeoutError,
)
from ai_service.persistence import (
    AuthorizedScope,
    DatabaseAuthorizationError,
    DatabaseError,
)
from ai_service.schemas.query import AskQueryRequest
from ai_service.schemas.structured_output import StructuredAnswerResponse

logger = logging.getLogger("coverageiq.api.qa")
router = APIRouter()

# Dependency injector for orchestrator
_orchestrator = None


def _runtime_settings() -> Settings:
    watch_vars = (
        "APP_ENV",
        "ENVIRONMENT",
        "NODE_ENV",
        "DEBUG",
        "USE_MOCK_STORAGE",
        "DATABASE_URL",
        "INTERNAL_SERVICE_TOKEN",
        "LLM_PROVIDER",
        "NVIDIA_API_KEY",
        "GEMINI_API_KEY",
        "EMBEDDING_PROVIDER",
    )
    if any(os.getenv(k) is not None for k in watch_vars):
        return Settings()
    return settings


def get_orchestrator() -> ClosedRetryLoopOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        try:
            _orchestrator = ClosedRetryLoopOrchestrator()
        except (ValueError, ExternalProviderConfigurationError) as exc:
            logger.error("Failed to initialize QA orchestrator due to config error: %s", type(exc).__name__)
            raise HTTPException(
                status_code=503,
                detail="AI service storage/provider configuration is invalid.",
            ) from exc
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
    x_service_token: Optional[str] = Header(default=None, alias="X-Service-Token"),
) -> StructuredAnswerResponse:
    """
    Module 4 & 5 (6C Pipeline):
    - Hybrid Retrieval (Vector + BM25) over MockKnowledgeBase or PostgreSQL 16 + pgvector
    - Cross-Encoder Reranker
    - Grounded LLM Prompting ('NO EVIDENCE = NO CLAIM')
    - Independent Verifier + Deterministic Decision Gate
    - Closed Retry Loop (up to 2 retries)
    """
    try:
        cfg = _runtime_settings()
    except ValueError as exc:
        logger.error("QA configuration validation failed cleanly: %s", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="AI service storage/provider configuration is invalid.",
        )

    if cfg.is_non_test_env and cfg.USE_MOCK_STORAGE:
        raise HTTPException(
            status_code=503,
            detail="Mock storage mode is disabled in non-test environments.",
        )

    logger.info(
        "Received QA query request: policy_id=%s question_length=%d mode=%s",
        payload.policy_id,
        len(payload.question),
        "mock" if cfg.USE_MOCK_STORAGE else "postgresql",
    )
    try:
        if not cfg.USE_MOCK_STORAGE:
            if payload.scope is None:
                raise DatabaseAuthorizationError("Missing caller scope context.")
            auth_scope = AuthorizedScope.from_gateway_context(
                payload.scope,
                provided_service_token=x_service_token,
                app_settings=cfg,
                allow_catalog_read=bool(payload.scope.allow_catalog_read),
            )
            return orchestrator.execute_qa(
                query=payload.question,
                policy_id=payload.policy_id,
                scope=auth_scope,
            )

        return orchestrator.execute_qa(
            query=payload.question,
            policy_id=payload.policy_id,
        )
    except DatabaseAuthorizationError as exc:
        logger.warning(
            "QA request authorization denied: policy_id=%s error_type=%s",
            payload.policy_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=403,
            detail="Access denied for requested policy scope.",
        )
    except DatabaseError as exc:
        logger.error(
            "QA request failed due to database service error: policy_id=%s error_type=%s",
            payload.policy_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="Policy storage service is currently unavailable.",
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
    except ExternalProviderConfigurationError:
        logger.error(
            "QA request blocked by external provider configuration error: policy_id=%s",
            payload.policy_id,
        )
        raise HTTPException(
            status_code=503,
            detail="AI provider service is currently unavailable.",
        )
    except ExternalProviderError as exc:
        logger.error(
            "QA request failed due to external provider error: policy_id=%s status=%s timeout=%s",
            payload.policy_id,
            exc.status_code,
            exc.is_timeout,
        )
        if exc.is_timeout:
            raise HTTPException(
                status_code=504,
                detail="AI provider request timed out while processing the policy query.",
            )
        if exc.status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="AI provider rate limit exceeded. Please retry shortly.",
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
