from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_service.api.routes_ingestion import router as ingestion_router
from ai_service.api.routes_qa import close_orchestrator, router as qa_router
from ai_service.common.logging import configure_logging
from ai_service.config import Settings, settings
from ai_service.persistence import close_db_pool, init_db_pool
from ai_service.recommendation.citations import (
    NullCitationProvider,
    PostgresEvidenceCitationProvider,
)
from ai_service.recommendation.router import router as recommendation_router
from ai_service.recommendation.service import ModelState, load_model_state
from ai_service.recommendation.settings import Settings as RecommendationSettings

logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("coverageiq.main")


def _effective_runtime_settings() -> Settings:
    """Resolve runtime settings honoring post-import test env overrides safely."""
    watch_vars = (
        "APP_ENV",
        "ENVIRONMENT",
        "NODE_ENV",
        "DEBUG",
        "USE_MOCK_STORAGE",
        "DATABASE_URL",
        "INTERNAL_SERVICE_TOKEN",
        "REDIS_URL",
        "CELERY_BROKER_URL",
        "CELERY_RESULT_BACKEND",
    )
    if any(os.getenv(k) is not None for k in watch_vars):
        return Settings()
    return settings


def create_app(rec_settings: RecommendationSettings | None = None) -> FastAPI:
    """Create unified CoverageIQ FastAPI app serving Q&A (M7/M8), Ingestion (M2), and Recommendation (M5)."""
    resolved_rec_settings = rec_settings or RecommendationSettings()
    configure_logging(
        level=resolved_rec_settings.log_level,
        json_format=resolved_rec_settings.log_json,
    )

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI):
        cfg = _effective_runtime_settings()
        fastapi_app.state.use_mock_storage = cfg.USE_MOCK_STORAGE
        if not cfg.USE_MOCK_STORAGE:
            init_db_pool(verify_schema=True, app_settings=cfg)
            fastapi_app.state.citation_provider = PostgresEvidenceCitationProvider()
        else:
            fastapi_app.state.citation_provider = NullCitationProvider()

        if getattr(fastapi_app.state, "model_state", None) is None:
            fastapi_app.state.model_state = load_model_state(
                resolved_rec_settings.model_artifacts_dir
            )

        try:
            yield
        finally:
            engine = getattr(fastapi_app.state, "db_engine", None)
            if engine is not None:
                engine.dispose()
                fastapi_app.state.db_engine = None
            close_orchestrator()
            close_db_pool()

    fastapi_app = FastAPI(
        title="CoverageIQ AI Service",
        description=(
            "Evidence-Grounded Insurance Policy Intelligence System "
            "(Hybrid Search, Grounded LLM, Evaluator Agent & M5 Policy Recommendations)"
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    fastapi_app.state.settings = resolved_rec_settings
    fastapi_app.state.use_mock_storage = settings.USE_MOCK_STORAGE
    fastapi_app.state.model_state = load_model_state(resolved_rec_settings.model_artifacts_dir)
    fastapi_app.state.citation_provider = (
        NullCitationProvider()
        if settings.USE_MOCK_STORAGE
        else PostgresEvidenceCitationProvider()
    )

    _cors_origins_raw = os.getenv("INTERNAL_CORS_ORIGINS", "")
    _allowed_cors_origins = [
        origin.strip().rstrip("/")
        for origin in _cors_origins_raw.split(",")
        if origin.strip() and origin.strip() != "*"
    ]

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_cors_origins,
        allow_credentials=bool(_allowed_cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-Service-Token", "X-Request-Id"],
    )

    # Mount unified AI routes
    fastapi_app.include_router(qa_router, prefix="/ai", tags=["Policy Q&A & Verification"])
    fastapi_app.include_router(
        ingestion_router, prefix="/internal/ingestion", tags=["Internal Ingestion Enqueue"]
    )
    fastapi_app.include_router(recommendation_router)

    @fastapi_app.get("/ai/health", tags=["Health"])
    @fastapi_app.get("/health", tags=["Health"])
    def health_check():
        cfg = _effective_runtime_settings()
        mstate: ModelState | None = getattr(fastapi_app.state, "model_state", None)
        return {
            "success": True,
            "status": "healthy",
            "service": cfg.SERVICE_NAME,
            "database_mode": "mock_memory" if cfg.USE_MOCK_STORAGE else "postgresql_pgvector",
            "reranker_model": cfg.RERANKER_MODEL_NAME,
            "retrieval": {
                "rrf_k": cfg.RRF_K,
                "weights": {
                    "vector": cfg.WEIGHT_VECTOR,
                    "bm25": cfg.WEIGHT_BM25,
                },
            },
            "evaluator": {
                "max_retries": cfg.MAX_RETRIES,
                "confidence_threshold": cfg.CONFIDENCE_THRESHOLD,
            },
            "model_ready": bool(mstate and mstate.is_ready),
            "model_warmed_up": bool(mstate and mstate.warmed_up),
        }

    return fastapi_app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ai_service.main:app", host=settings.HOST, port=settings.PORT, reload=True)
