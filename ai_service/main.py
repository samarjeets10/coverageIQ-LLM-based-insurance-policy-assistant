"""Unified FastAPI entry point for CoverageIQ AI Service (M3/M7/M8 QA + M5 Recommendation)."""

from __future__ import annotations

import math
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated, Any

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fastapi import Depends, FastAPI, Request, status  # noqa: E402
from fastapi.encoders import jsonable_encoder  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from sqlalchemy import Connection  # noqa: E402
from sqlalchemy.exc import SQLAlchemyError  # noqa: E402

from ai_service.api.routes_qa import router as qa_router  # noqa: E402
from ai_service.common.logging import configure_logging  # noqa: E402
from ai_service.config import settings as qa_settings  # noqa: E402
from ai_service.recommendation.citations import NullCitationProvider  # noqa: E402
from ai_service.recommendation.db import (  # noqa: E402
    check_database_connection,
    create_db_engine,
    get_db_connection,
    inspect_database_readiness,
)
from ai_service.recommendation.router import router as recommendation_router  # noqa: E402
from ai_service.recommendation.schemas import AIHealthResponse, HealthResponse  # noqa: E402
from ai_service.recommendation.service import ModelState, load_model_state  # noqa: E402
from ai_service.recommendation.settings import Settings, get_settings  # noqa: E402


class RequestBodyLimitMiddleware:
    """ASGI middleware enforcing a bounded HTTP request payload size before JSON parsing."""

    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    @staticmethod
    async def _empty_receive() -> dict[str, Any]:
        return {"type": "http.disconnect"}

    async def _send_too_large(self, scope: dict[str, Any], send: Any) -> None:
        response = JSONResponse(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            content={
                "detail": (
                    f"Request payload exceeds maximum allowed size of {self.max_bytes} bytes."
                )
            },
        )
        await response(scope, self._empty_receive, send)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for header_name, header_val in scope.get("headers", []):
            if header_name.lower() == b"content-length":
                try:
                    if int(header_val.decode("ascii")) > self.max_bytes:
                        await self._send_too_large(scope, send)
                        return
                except ValueError:
                    pass

        chunks: list[bytes] = []
        total_bytes = 0
        more_body = True
        while more_body:
            message = await receive()
            chunk = message.get("body", b"")
            total_bytes += len(chunk)
            if total_bytes > self.max_bytes:
                await self._send_too_large(scope, send)
                return
            chunks.append(chunk)
            more_body = bool(message.get("more_body", False))

        buffered = b"".join(chunks)
        consumed = False

        async def replay_receive() -> dict[str, Any]:
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": buffered, "more_body": False}
            return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send)


def _sanitize_validation_errors(obj: Any) -> Any:
    """Replace non-finite floats cleanly so Starlette JSONResponse never fails on 422 errors."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_validation_errors(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_sanitize_validation_errors(item) for item in obj]
    return obj


def _upstream_health_telemetry() -> dict[str, Any]:
    """Build upstream M3/M7/M8 configuration telemetry preserved on /ai/health."""
    return {
        "service": qa_settings.SERVICE_NAME,
        "database_mode": "mock_memory" if qa_settings.USE_MOCK_STORAGE else "postgresql_pgvector",
        "reranker_model": qa_settings.RERANKER_MODEL_NAME,
        "retrieval": {
            "rrf_k": qa_settings.RRF_K,
            "weights": {
                "vector": qa_settings.WEIGHT_VECTOR,
                "bm25": qa_settings.WEIGHT_BM25,
            },
        },
        "evaluator": {
            "max_retries": qa_settings.MAX_RETRIES,
            "confidence_threshold": qa_settings.CONFIDENCE_THRESHOLD,
        },
    }


@contextmanager
def _optional_db_connection(request: Request):
    """Yield an overridden or live DB connection if available for /ai/health probing."""
    override = request.app.dependency_overrides.get(get_db_connection)
    if override is not None:
        dep_result = override()
        if hasattr(dep_result, "__next__"):
            conn = next(dep_result)
            try:
                yield conn
            finally:
                try:
                    dep_result.close()
                except Exception:
                    pass
        else:
            yield dep_result
        return

    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        cfg: Settings = request.app.state.settings
        if cfg.db_password is None:
            yield None
            return
        engine = create_db_engine(cfg)
        request.app.state.db_engine = engine

    with engine.connect() as conn:
        yield conn


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the unified FastAPI application instance."""
    resolved_settings = settings or get_settings()
    resolved_settings._validate_security_configuration()
    configure_logging(level=resolved_settings.log_level, json_format=resolved_settings.log_json)

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
        app_instance.state.settings = resolved_settings
        app_instance.state.db_engine = create_db_engine(resolved_settings)
        app_instance.state.model_state = load_model_state(resolved_settings.model_artifacts_dir)
        app_instance.state.citation_provider = NullCitationProvider()
        try:
            yield
        finally:
            app_instance.state.db_engine.dispose()

    app = FastAPI(
        title="CoverageIQ AI Service",
        description=(
            "Unified CoverageIQ Python AI Service housing Evidence-Grounded Policy Q&A "
            "(Hybrid Search, Cross-Encoder Reranker, Evaluator Retry Loop) and Policy "
            "Recommendation (Module M5 / 6A: PostgreSQL catalog filtering, 59-feature "
            "engineering, and LightGBM / Weighted ranking)."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=resolved_settings.max_request_body_bytes,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = resolved_settings
    app.state.model_state = None
    app.state.citation_provider = NullCitationProvider()

    def _model_ready_flag(request: Request) -> bool:
        model_state: ModelState | None = getattr(request.app.state, "model_state", None)
        return bool(model_state is not None and model_state.is_ready)

    def _model_warmed_up_flag(request: Request) -> bool:
        model_state: ModelState | None = getattr(request.app.state, "model_state", None)
        return bool(model_state is not None and model_state.warmed_up)

    def _health_db_down_payload(
        model_ready: bool,
        model_warmed_up: bool,
        *,
        include_upstream: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "success": False,
            "status": "unhealthy",
            "services": {
                "api": "up",
                "database": "down",
                "model": "ready" if model_ready else "unavailable",
            },
            "model_ready": model_ready,
            "model_warmed_up": model_warmed_up,
        }
        if include_upstream:
            payload.update(_upstream_health_telemetry())
        return payload

    @app.exception_handler(RequestValidationError)
    def handle_request_validation_error(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Convert Pydantic validation errors (including NaN/Inf inputs) to HTTP 422 safely."""
        safe_errors = jsonable_encoder(_sanitize_validation_errors(exc.errors()))
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": safe_errors},
        )

    @app.exception_handler(SQLAlchemyError)
    def handle_sqlalchemy_error(request: Request, _exc: SQLAlchemyError) -> JSONResponse:
        """Convert unhandled DB connection/dependency errors into controlled HTTP 503."""
        if request.url.path in ("/health", "/ai/health"):
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=_health_db_down_payload(
                    _model_ready_flag(request),
                    _model_warmed_up_flag(request),
                    include_upstream=(request.url.path == "/ai/health"),
                ),
            )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "PostgreSQL database connection is unavailable."},
        )

    @app.get(
        "/health",
        tags=["health"],
        response_model=HealthResponse,
        responses={
            status.HTTP_200_OK: {
                "model": HealthResponse,
                "description": "Service healthy and PostgreSQL reachable.",
            },
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "model": HealthResponse,
                "description": "PostgreSQL database connection is unavailable.",
            },
        },
    )
    def health_check(
        request: Request,
        conn: Annotated[Connection, Depends(get_db_connection)],
    ) -> JSONResponse:
        """Report M5 recommendation health, PostgreSQL readiness, and model state."""
        model_ready = _model_ready_flag(request)
        model_warmed_up = _model_warmed_up_flag(request)

        try:
            db_info = check_database_connection(conn)
            readiness = inspect_database_readiness(conn)
            payload: dict[str, Any] = {
                "success": True,
                "status": "healthy",
                "services": {
                    "api": "up",
                    "database": "up",
                    "model": "ready" if model_ready else "unavailable",
                },
                "database": db_info["database"],
                "readiness": readiness,
                "model_ready": model_ready,
                "model_warmed_up": model_warmed_up,
            }
            return JSONResponse(status_code=status.HTTP_200_OK, content=payload)
        except SQLAlchemyError:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=_health_db_down_payload(model_ready, model_warmed_up),
            )

    @app.get(
        "/ai/health",
        tags=["Health"],
        response_model=AIHealthResponse,
        responses={
            status.HTTP_200_OK: {
                "model": AIHealthResponse,
                "description": "Unified AI service health across M3 Q&A and M5 recommendation.",
            },
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "model": AIHealthResponse,
                "description": "PostgreSQL database connection is unavailable.",
            },
        },
    )
    def ai_health_check(request: Request) -> JSONResponse:
        """Unified AI health endpoint combining upstream M3 telemetry and M5 DB readiness."""
        model_ready = _model_ready_flag(request)
        model_warmed_up = _model_warmed_up_flag(request)
        telemetry = _upstream_health_telemetry()

        try:
            with _optional_db_connection(request) as conn:
                if conn is not None:
                    db_info = check_database_connection(conn)
                    readiness = inspect_database_readiness(conn)
                    telemetry["database_mode"] = "postgresql_pgvector"
                    payload: dict[str, Any] = {
                        "success": True,
                        "status": "healthy",
                        "services": {
                            "api": "up",
                            "database": "up",
                            "model": "ready" if model_ready else "unavailable",
                        },
                        "database": db_info["database"],
                        "readiness": readiness,
                        "model_ready": model_ready,
                        "model_warmed_up": model_warmed_up,
                        **telemetry,
                    }
                    return JSONResponse(status_code=status.HTTP_200_OK, content=payload)
        except SQLAlchemyError:
            if not qa_settings.USE_MOCK_STORAGE or (
                get_db_connection in request.app.dependency_overrides
            ):
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content=_health_db_down_payload(
                        model_ready, model_warmed_up, include_upstream=True
                    ),
                )

        if qa_settings.USE_MOCK_STORAGE:
            payload = {
                "success": True,
                "status": "healthy",
                "services": {
                    "api": "up",
                    "database": "mock_memory",
                    "model": "ready" if model_ready else "unavailable",
                },
                "database": None,
                "readiness": None,
                "model_ready": model_ready,
                "model_warmed_up": model_warmed_up,
                **telemetry,
            }
            return JSONResponse(status_code=status.HTTP_200_OK, content=payload)

        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_health_db_down_payload(model_ready, model_warmed_up, include_upstream=True),
        )

    app.include_router(qa_router, prefix="/ai", tags=["Policy Q&A & Verification"])
    app.include_router(recommendation_router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ai_service.main:app", host=qa_settings.HOST, port=qa_settings.PORT, reload=True)
