"""FastAPI router for Policy Recommendation endpoints."""

import hmac
import logging
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from fastapi.security import APIKeyHeader
from sqlalchemy import Connection

from ai_service.common.logging import emit_structured_event
from ai_service.recommendation.citations import (
    NullCitationProvider,
    RecommendationCitationProvider,
)
from ai_service.recommendation.db import get_db_connection
from ai_service.recommendation.schemas import (
    ErrorDetailResponse,
    RecommendRequest,
    RecommendResponse,
)
from ai_service.recommendation.service import (
    InferenceProcessingError,
    ModelState,
    ServiceDependencyUnavailableError,
    UnsupportedRankingModelError,
    load_model_state,
    recommend_policies_for_request,
)
from ai_service.recommendation.settings import Settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["recommendation"])

_service_token_header = APIKeyHeader(
    name="X-Service-Token",
    scheme_name="ServiceTokenAuth",
    description=(
        "Shared internal service token supplied by the Node.js API gateway. "
        "Mandatory in all non-development environments."
    ),
    auto_error=False,
)


def verify_service_token(
    request: Request,
    x_service_token: Annotated[str | None, Depends(_service_token_header)] = None,
) -> None:
    """Enforce constant-time X-Service-Token authentication (fail-closed outside dev)."""
    raw_req_id = request.headers.get("X-Request-ID")
    req_id = raw_req_id.strip()[:128] if raw_req_id and raw_req_id.strip() else None
    err_headers = {"X-Request-ID": req_id} if req_id else None

    settings: Settings = request.app.state.settings
    token_secret = settings.ai_service_token
    expected = token_secret.get_secret_value().strip() if token_secret is not None else ""

    if not settings.is_development_env or settings.ai_service_require_auth is True:
        if not expected:
            emit_structured_event(
                logger,
                level=logging.ERROR,
                event="auth.misconfigured",
                message="Service token enforcement active without configured token.",
                request_id=req_id,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing service token",
                headers=err_headers,
            )
    elif token_secret is None:
        return
    elif not expected:
        emit_structured_event(
            logger,
            level=logging.WARNING,
            event="auth.rejected",
            message="Service token authentication failed: configured token is empty.",
            request_id=req_id,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing service token",
            headers=err_headers,
        )

    if not x_service_token or not hmac.compare_digest(x_service_token, expected):
        emit_structured_event(
            logger,
            level=logging.WARNING,
            event="auth.rejected",
            message="Service token authentication failed.",
            request_id=req_id,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing service token",
            headers=err_headers,
        )


def get_model_state(request: Request) -> ModelState:
    """Return the cached ModelState loaded at startup (loading once if uninitialized)."""
    cached: ModelState | None = getattr(request.app.state, "model_state", None)
    if cached is None:
        settings: Settings = request.app.state.settings
        cached = load_model_state(settings.model_artifacts_dir)
        request.app.state.model_state = cached
    return cached


def get_citation_provider(request: Request) -> RecommendationCitationProvider:
    """Return configured RecommendationCitationProvider (defaults to NullCitationProvider)."""
    provider: RecommendationCitationProvider | None = getattr(
        request.app.state, "citation_provider", None
    )
    if provider is None:
        provider = NullCitationProvider()
        request.app.state.citation_provider = provider
    return provider


_RECOMMEND_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_401_UNAUTHORIZED: {
        "model": ErrorDetailResponse,
        "description": "Invalid or missing X-Service-Token header.",
    },
    status.HTTP_413_CONTENT_TOO_LARGE: {
        "model": ErrorDetailResponse,
        "description": "Request JSON body exceeds configured byte size limit.",
    },
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "description": "Validation error on user profile, nested scope, or resource limits."
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": ErrorDetailResponse,
        "description": "Unexpected feature construction, catalog integrity, or scoring error.",
    },
    status.HTTP_501_NOT_IMPLEMENTED: {
        "model": ErrorDetailResponse,
        "description": "Requested ranking_model (e.g., 'xgboost') is not yet implemented.",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorDetailResponse,
        "description": "LightGBM artifacts unavailable/corrupt or PostgreSQL catalog unreachable.",
    },
}


@router.post(
    "/api/v1/recommend",
    response_model=RecommendResponse,
    summary="Recommend policies (Final Design Report endpoint)",
    description=(
        "Evaluates candidate policies from PostgreSQL using deterministic Stage 4 hard "
        "eligibility filters, computes the 59-column Stage 5 feature matrix on eligible "
        "candidates only, and ranks Top-N policies via LightGBM ('auto'/'lightgbm') or "
        "transparent weighted scoring ('weighted'). Requires Node.js M1 gateway "
        "authentication (`X-Service-Token`) in non-development environments; private user "
        "uploads (`owner_user_id IS NOT NULL`) are never returned regardless of `scope`."
    ),
    responses=_RECOMMEND_RESPONSES,
    dependencies=[Depends(verify_service_token)],
)
@router.post(
    "/ai/recommend",
    response_model=RecommendResponse,
    summary="Recommend policies (Internal AI service contract alias)",
    description=(
        "Internal Node-to-Python recommendation alias sharing exact authentication, "
        "filtering, scoring, and warning contracts with /api/v1/recommend."
    ),
    responses=_RECOMMEND_RESPONSES,
    dependencies=[Depends(verify_service_token)],
)
def recommend_policies(
    payload: RecommendRequest,
    response: Response,
    conn: Annotated[Connection, Depends(get_db_connection)],
    model_state: Annotated[ModelState, Depends(get_model_state)],
    citation_provider: Annotated[RecommendationCitationProvider, Depends(get_citation_provider)],
    x_request_id: Annotated[str | None, Header(alias="X-Request-ID")] = None,
) -> RecommendResponse:
    """Filter eligible catalog policies deterministically and rank via LightGBM or weighted."""
    cleaned_id = x_request_id.strip()[:128] if x_request_id and x_request_id.strip() else None
    request_id = cleaned_id or str(uuid.uuid4())

    try:
        result = recommend_policies_for_request(
            payload,
            conn=conn,
            model_state=model_state,
            request_id=request_id,
            citation_provider=citation_provider,
        )
    except UnsupportedRankingModelError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
            headers={"X-Request-ID": request_id},
        ) from exc
    except ServiceDependencyUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
            headers={"X-Request-ID": request_id},
        ) from exc
    except InferenceProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
            headers={"X-Request-ID": request_id},
        ) from exc
    except ValueError as exc:
        logger.exception("Unhandled ValueError guarded at router boundary [%s]", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Recommendation request failed internal data validation.",
            headers={"X-Request-ID": request_id},
        ) from exc

    if result.request_id:
        response.headers["X-Request-ID"] = result.request_id
    return result
