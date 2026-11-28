"""FastAPI router for Policy Recommendation and Attribute Backfill endpoints."""

import hmac
import logging
import os
import re
from typing import Annotated, Any
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Connection

from ai_service.common.logging import emit_structured_event
from ai_service.config import Settings as CoreSettings, settings as core_settings
from ai_service.llm.client import ExternalProviderConfigurationError, ExternalProviderError
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
    DatabaseError,
    PolicyKnowledgeRepository,
    get_db_pool_manager,
    init_db_pool,
)
from ai_service.recommendation.attribute_extractor import PolicyAttributeExtractor
from ai_service.recommendation.citations import (
    NullCitationProvider,
    PostgresEvidenceCitationProvider,
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

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")

_service_token_header = APIKeyHeader(
    name="X-Service-Token",
    scheme_name="ServiceTokenAuth",
    description=(
        "Shared internal service token supplied by the Node.js API gateway. "
        "Mandatory in all non-development environments."
    ),
    auto_error=False,
)


def _clean_request_id(raw_request_id: str | None) -> str | None:
    if not raw_request_id:
        return None
    stripped = _CONTROL_CHAR_RE.sub("", raw_request_id).strip()[:128]
    return stripped or None


def verify_service_token(
    request: Request,
    x_service_token: Annotated[str | None, Depends(_service_token_header)] = None,
) -> None:
    """Enforce constant-time X-Service-Token authentication (fail-closed outside dev)."""
    req_id = _clean_request_id(request.headers.get("X-Request-ID"))
    err_headers = {"X-Request-ID": req_id} if req_id else None

    rec_settings: Settings = getattr(request.app.state, "settings", None) or Settings()
    expected = rec_settings.effective_service_token

    if not rec_settings.auth_enforced:
        return

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
        settings: Settings = getattr(request.app.state, "settings", None) or Settings()
        cached = load_model_state(settings.model_artifacts_dir)
        request.app.state.model_state = cached
    return cached


def get_citation_provider(request: Request) -> RecommendationCitationProvider:
    """Return configured RecommendationCitationProvider (defaults to PostgresEvidenceCitationProvider)."""
    provider: RecommendationCitationProvider | None = getattr(
        request.app.state, "citation_provider", None
    )
    if provider is None:
        use_mock = bool(getattr(request.app.state, "use_mock_storage", False))
        provider = (
            NullCitationProvider() if use_mock else PostgresEvidenceCitationProvider()
        )
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
    responses=_RECOMMEND_RESPONSES,
    dependencies=[Depends(verify_service_token)],
)
@router.post(
    "/ai/recommend",
    response_model=RecommendResponse,
    summary="Recommend policies (Internal AI service endpoint)",
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
    request_id = _clean_request_id(x_request_id) or str(uuid.uuid4())

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


class ExtractAttributesInternalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: uuid.UUID
    document_id: uuid.UUID
    is_catalog_eligible: bool | None = None


def _resolve_core_settings() -> CoreSettings:
    watch = (
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
    )
    if any(os.getenv(k) is not None for k in watch):
        return CoreSettings()
    return core_settings


@router.post(
    "/internal/recommendations/extract-attributes",
    summary="Extract and persist structured policy_attributes for an already ingested document",
    dependencies=[Depends(verify_service_token)],
)
def extract_policy_attributes_internal(
    payload: ExtractAttributesInternalRequest,
) -> dict[str, Any]:
    """Backfill or refresh grounded policy_attributes from persisted policy_chunks."""
    cfg = _resolve_core_settings()
    if cfg.USE_MOCK_STORAGE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Attribute extraction requires PostgreSQL storage mode.",
        )

    try:
        pool_mgr = get_db_pool_manager()
    except Exception:
        pool_mgr = init_db_pool(verify_schema=False, app_settings=cfg)

    repo = PolicyKnowledgeRepository(pool_mgr)
    try:
        with pool_mgr.connection() as conn:
            pol_row = conn.execute(
                """
                SELECT id, owner_scope, policy_type, visibility, status
                FROM policies
                WHERE id = %(policy_id)s
                LIMIT 1
                """,
                {"policy_id": payload.policy_id},
            ).fetchone()
            doc_row = conn.execute(
                """
                SELECT id, ingestion_status, is_archived
                FROM policy_documents
                WHERE id = %(document_id)s AND policy_id = %(policy_id)s
                LIMIT 1
                """,
                {"policy_id": payload.policy_id, "document_id": payload.document_id},
            ).fetchone()

        if pol_row is None or doc_row is None:
            raise HTTPException(status_code=404, detail="Policy or document not found.")
        if doc_row.get("is_archived") or pol_row.get("status") == "archived":
            raise HTTPException(status_code=409, detail="Archived documents cannot be extracted.")
        if doc_row.get("ingestion_status") != "completed":
            raise HTTPException(status_code=409, detail="Document ingestion is not completed yet.")

        auth_scope = AuthorizedScope.from_gateway_context(
            {
                "user_id": "internal_worker",
                "role": "admin",
                "owner_scope": str(pol_row["owner_scope"]),
            },
            trusted_internal=True,
            allow_catalog_read=True,
        )
        chunks = repo.get_policy_chunks(
            policy_id=payload.policy_id,
            scope=auth_scope,
            document_id=payload.document_id,
        )
        if not chunks:
            raise HTTPException(status_code=409, detail="No indexed chunks found for document.")

        extractor = PolicyAttributeExtractor(app_settings=cfg)
        try:
            extracted = extractor.extract_from_chunks(
                chunks,
                policy_type=str(pol_row["policy_type"]),
            )
        finally:
            extractor.close()

        saved = repo.upsert_policy_attributes(
            policy_id=payload.policy_id,
            document_id=payload.document_id,
            attributes=extracted,
            is_catalog_eligible=payload.is_catalog_eligible,
        )
        return {
            "policy_id": str(payload.policy_id),
            "document_id": str(payload.document_id),
            "extraction_status": saved["extraction_status"],
            "is_catalog_eligible": bool(saved["is_catalog_eligible"]),
            "source_chunk_ids": saved["source_chunk_ids"],
        }
    except HTTPException:
        raise
    except (ExternalProviderConfigurationError, NvidiaConfigurationError, NvidiaAuthenticationError, NvidiaPermissionError):
        raise HTTPException(status_code=503, detail="AI provider service is currently unavailable.")
    except NvidiaTimeoutError:
        raise HTTPException(status_code=504, detail="AI provider timed out during attribute extraction.")
    except NvidiaRateLimitError:
        raise HTTPException(status_code=429, detail="AI provider rate limit exceeded during extraction.")
    except (NvidiaProviderError, ExternalProviderError):
        raise HTTPException(status_code=502, detail="AI provider failed during attribute extraction.")
    except DatabaseError:
        raise HTTPException(status_code=503, detail="Database error while extracting policy attributes.")
