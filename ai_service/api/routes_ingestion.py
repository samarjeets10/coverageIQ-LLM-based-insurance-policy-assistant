import hmac
import logging
import os
import threading
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException

from ai_service.config import Settings, settings
from ai_service.persistence import (
    DatabaseAuthorizationError,
    DatabaseError,
    DatabaseNotFoundError,
    DatabaseValidationError,
    PolicyKnowledgeRepository,
    get_db_pool_manager,
    init_db_pool,
)
from ai_service.schemas.ingestion_api import InternalEnqueueRequest, InternalEnqueueResponse
from ai_service.storage import DocumentPathSecurityError, LocalFilesystemDocumentStorage
from ai_service.workers import (
    IngestionTaskPayload,
    celery_app,
    create_celery_app,
    sanitize_worker_error,
)

logger = logging.getLogger("coverageiq.api.ingestion")
router = APIRouter()

_producer_lock = threading.Lock()
_cached_producers: Dict[tuple[str, str], Any] = {}


def _runtime_settings() -> Settings:
    watch_vars = (
        "USE_MOCK_STORAGE",
        "DATABASE_URL",
        "INTERNAL_SERVICE_TOKEN",
        "REDIS_URL",
        "CELERY_BROKER_URL",
        "CELERY_RESULT_BACKEND",
        "CELERY_INGESTION_QUEUE",
        "POLICY_UPLOAD_ROOT",
    )
    if any(os.getenv(k) is not None for k in watch_vars):
        return Settings()
    return settings


def _verify_internal_service_token(provided_token: Optional[str], cfg: Settings) -> None:
    expected = cfg.INTERNAL_SERVICE_TOKEN
    if not expected or not provided_token or not hmac.compare_digest(provided_token, expected):
        raise HTTPException(
            status_code=403,
            detail="Forbidden: invalid or missing internal service token.",
        )


def _get_repository(cfg: Settings) -> PolicyKnowledgeRepository:
    mgr = get_db_pool_manager()
    if not mgr.is_open:
        mgr = init_db_pool(verify_schema=True, app_settings=cfg)
    return PolicyKnowledgeRepository(mgr)


def get_celery_producer(cfg: Settings):
    if (
        cfg.CELERY_BROKER_URL == settings.CELERY_BROKER_URL
        and cfg.CELERY_RESULT_BACKEND == settings.CELERY_RESULT_BACKEND
    ):
        return celery_app

    key = (cfg.CELERY_BROKER_URL, cfg.CELERY_RESULT_BACKEND)
    with _producer_lock:
        app_inst = _cached_producers.get(key)
        if app_inst is None:
            app_inst = create_celery_app(cfg)
            _cached_producers[key] = app_inst
        return app_inst


def dispatch_ingestion_to_celery(
    *,
    task_payload: IngestionTaskPayload,
    task_id: str,
    cfg: Settings,
) -> None:
    producer = get_celery_producer(cfg)
    producer.send_task(
        "coverageiq.ingestion.ingest_policy_document",
        args=[task_payload.model_dump(mode="json")],
        task_id=task_id,
        queue=cfg.CELERY_INGESTION_QUEUE,
    )


@router.post(
    "/enqueue",
    response_model=InternalEnqueueResponse,
    summary="Authenticated internal endpoint to validate and enqueue a persisted PDF ingestion job",
)
def enqueue_ingestion_job(
    payload: InternalEnqueueRequest,
    x_service_token: Optional[str] = Header(default=None, alias="X-Service-Token"),
) -> InternalEnqueueResponse:
    """
    Internal Node.js -> FastAPI ingestion dispatch endpoint.

    - Authenticates via X-Service-Token using constant-time comparison.
    - Loads persisted job, policy, and document rows from PostgreSQL.
    - Resolves trusted owner_scope strictly from DB ownership.
    - Enqueues idempotently to the configured Celery ingestion queue.
    """
    cfg = _runtime_settings()
    _verify_internal_service_token(x_service_token, cfg)

    if cfg.USE_MOCK_STORAGE:
        raise HTTPException(
            status_code=503,
            detail="Ingestion enqueue is disabled while USE_MOCK_STORAGE=true.",
        )

    try:
        repo = _get_repository(cfg)
        state, job, policy, doc, trusted_scope = repo.prepare_job_for_enqueue(
            job_id=payload.job_id,
            policy_id=payload.policy_id,
            document_id=payload.document_id,
        )

        if state == "already_completed":
            return InternalEnqueueResponse(
                job_id=job.id,
                policy_id=policy.id,
                document_id=doc.id,
                request_id=payload.request_id,
                enqueued=False,
                already_enqueued=False,
                already_completed=True,
                status=job.status,
                stage=job.current_stage,
                task_id=job.celery_task_id,
                queue=cfg.CELERY_INGESTION_QUEUE,
            )

        if state == "already_enqueued":
            return InternalEnqueueResponse(
                job_id=job.id,
                policy_id=policy.id,
                document_id=doc.id,
                request_id=payload.request_id,
                enqueued=False,
                already_enqueued=True,
                already_completed=False,
                status=job.status,
                stage=job.current_stage,
                task_id=job.celery_task_id,
                queue=cfg.CELERY_INGESTION_QUEUE,
            )

        storage = LocalFilesystemDocumentStorage(app_settings=cfg)
        storage.resolve_safe_path(doc.storage_path)

        task_payload = IngestionTaskPayload(
            job_id=job.id,
            policy_id=policy.id,
            document_id=doc.id,
            owner_scope=trusted_scope.owner_scope,
            storage_ref=doc.storage_path,
            expected_sha256=doc.file_hash_sha256,
            request_id=payload.request_id,
        )
        planned_task_id = f"ingest-{job.id}"

        try:
            dispatch_ingestion_to_celery(
                task_payload=task_payload,
                task_id=planned_task_id,
                cfg=cfg,
            )
        except Exception as broker_exc:
            sanitized = sanitize_worker_error(broker_exc)
            logger.error(
                "Celery enqueue failed for job_id=%s request_id=%s error=%s",
                job.id,
                payload.request_id,
                sanitized,
            )
            try:
                repo.record_enqueue_failure(
                    job_id=job.id,
                    scope=trusted_scope,
                    error_code="ENQUEUE_BROKER_UNAVAILABLE",
                    error_message="Async ingestion broker temporarily unavailable; reconciliation scheduled.",
                    request_id=payload.request_id,
                )
            except Exception:
                pass
            raise HTTPException(
                status_code=503,
                detail="Async ingestion broker temporarily unavailable; retry/reconciliation scheduled.",
            )

        updated_job = repo.record_enqueue_dispatch(
            job_id=job.id,
            celery_task_id=planned_task_id,
            scope=trusted_scope,
            request_id=payload.request_id,
        )
        logger.info(
            "Enqueued Celery ingestion job: job_id=%s policy_id=%s document_id=%s task_id=%s request_id=%s",
            job.id,
            policy.id,
            doc.id,
            planned_task_id,
            payload.request_id,
        )
        return InternalEnqueueResponse(
            job_id=updated_job.id,
            policy_id=policy.id,
            document_id=doc.id,
            request_id=payload.request_id,
            enqueued=True,
            already_enqueued=False,
            already_completed=False,
            status=updated_job.status,
            stage=updated_job.current_stage,
            task_id=updated_job.celery_task_id or planned_task_id,
            queue=cfg.CELERY_INGESTION_QUEUE,
        )

    except HTTPException:
        raise
    except DocumentPathSecurityError as exc:
        logger.warning("Rejected invalid document storage path on enqueue: job_id=%s", payload.job_id)
        raise HTTPException(status_code=400, detail=sanitize_worker_error(exc))
    except DatabaseAuthorizationError as exc:
        logger.warning("Authorization rejected for enqueue: job_id=%s error=%s", payload.job_id, sanitize_worker_error(exc))
        raise HTTPException(status_code=403, detail="Access denied for requested ingestion job.")
    except DatabaseNotFoundError:
        raise HTTPException(status_code=404, detail="Ingestion job, policy, or document not found.")
    except DatabaseValidationError as exc:
        raise HTTPException(status_code=409, detail=sanitize_worker_error(exc))
    except DatabaseError as exc:
        logger.error("Database failure on internal enqueue: job_id=%s error_type=%s", payload.job_id, type(exc).__name__)
        raise HTTPException(status_code=503, detail="Policy persistence database is temporarily unavailable.")
