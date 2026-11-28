import logging
import random
from typing import Any, Dict, Optional

from pydantic import ValidationError

from ai_service.persistence import AuthorizedScope, PolicyKnowledgeRepository
from ai_service.workers.celery_app import (
    WorkerResourceManager,
    celery_app,
    get_worker_resources,
)
from ai_service.workers.contracts import IngestionStage, IngestionTaskPayload
from ai_service.workers.pipeline import (
    IngestionPermanentError,
    IngestionPipelineExecutor,
    classify_ingestion_error,
)

logger = logging.getLogger("coverageiq.workers.tasks")


def _record_retry_or_failure(
    *,
    resources: WorkerResourceManager,
    repository: Optional[PolicyKnowledgeRepository],
    payload: Optional[IngestionTaskPayload],
    scope: Optional[AuthorizedScope],
    is_retry: bool,
    error_code: str,
    error_message: str,
) -> None:
    if payload is None or scope is None:
        return
    try:
        repo = repository or PolicyKnowledgeRepository(resources.get_pool())
        if is_retry:
            repo.transition_ingestion_job_stage(
                job_id=payload.job_id,
                document_id=payload.document_id,
                scope=scope,
                status="queued",
                stage=IngestionStage.RETRY_SCHEDULED.value,
                doc_status="queued",
                error_code=error_code,
                error_message=error_message,
            )
        else:
            repo.transition_ingestion_job_stage(
                job_id=payload.job_id,
                document_id=payload.document_id,
                scope=scope,
                status="failed",
                stage=IngestionStage.FAILED.value,
                doc_status="failed",
                error_code=error_code,
                error_message=error_message,
                mark_completed_at=True,
            )
    except Exception as db_exc:
        logger.error(
            "Failed updating ingestion job status after failure: job_id=%s error_type=%s",
            payload.job_id,
            type(db_exc).__name__,
        )


def execute_ingestion_task(
    *,
    task: Any,
    raw_payload: Dict[str, Any],
    executor: Optional[IngestionPipelineExecutor] = None,
    resources: Optional[WorkerResourceManager] = None,
    repository: Optional[PolicyKnowledgeRepository] = None,
) -> Dict[str, Any]:
    """
    Core Celery execution wrapper handling atomic DB status updates and bounded
    exponential-backoff retries strictly for transient failures.
    """
    res_mgr = resources or get_worker_resources()
    cfg = res_mgr.get_settings()
    task_id = getattr(getattr(task, "request", None), "id", None)
    retries_so_far = int(getattr(getattr(task, "request", None), "retries", 0) or 0)
    is_celery_retry = retries_so_far > 0

    validated_payload: Optional[IngestionTaskPayload] = None
    verified_scope: Optional[AuthorizedScope] = None
    pipe = executor or IngestionPipelineExecutor(resources=res_mgr, repository=repository)
    effective_repo = repository or getattr(pipe, "_repository", None)
    if effective_repo is None and executor is None:
        effective_repo = PolicyKnowledgeRepository(res_mgr.get_pool())

    try:
        validated_payload = IngestionTaskPayload.model_validate(raw_payload)
        if effective_repo is not None:
            # Obtain DB-verified scope upfront so mid-pipeline failures can update job status safely.
            _, _, _, verified_scope = effective_repo.fetch_worker_job_context(
                job_id=validated_payload.job_id,
                policy_id=validated_payload.policy_id,
                document_id=validated_payload.document_id,
                claimed_owner_scope=validated_payload.owner_scope,
            )

        result, verified_scope, validated_payload = pipe.execute(
            raw_payload,
            task_id=task_id,
            is_celery_retry=is_celery_retry,
        )
        return result.to_json_dict()

    except ValidationError as exc:
        _, code, msg = classify_ingestion_error(exc)
        logger.error("Rejected malformed ingestion task payload: error_code=%s", code)
        raise IngestionPermanentError(f"[{code}] {msg}") from exc

    except Exception as exc:
        is_transient, error_code, sanitized_msg = classify_ingestion_error(exc)
        max_retries = cfg.CELERY_TASK_MAX_RETRIES

        if is_transient and retries_so_far < max_retries:
            countdown = min(
                cfg.CELERY_TASK_RETRY_BACKOFF_MAX,
                int(cfg.CELERY_TASK_RETRY_BACKOFF_BASE * (2**retries_so_far) + random.uniform(0.0, 1.0)),
            )
            logger.warning(
                "Transient ingestion failure; scheduling retry %d/%d in %ds: job_id=%s error_code=%s",
                retries_so_far + 1,
                max_retries,
                countdown,
                getattr(validated_payload, "job_id", "unknown"),
                error_code,
            )
            _record_retry_or_failure(
                resources=res_mgr,
                repository=effective_repo,
                payload=validated_payload,
                scope=verified_scope,
                is_retry=True,
                error_code=error_code,
                error_message=sanitized_msg,
            )
            if task is not None and hasattr(task, "retry"):
                raise task.retry(exc=exc, countdown=countdown)
            raise

        logger.error(
            "Permanent or retry-exhausted ingestion failure: job_id=%s transient=%s retries=%d error_code=%s msg=%s",
            getattr(validated_payload, "job_id", "unknown"),
            is_transient,
            retries_so_far,
            error_code,
            sanitized_msg,
        )
        _record_retry_or_failure(
            resources=res_mgr,
            repository=effective_repo,
            payload=validated_payload,
            scope=verified_scope,
            is_retry=False,
            error_code=error_code,
            error_message=sanitized_msg,
        )
        raise IngestionPermanentError(f"[{error_code}] {sanitized_msg}") from exc


@celery_app.task(
    bind=True,
    name="coverageiq.ingestion.ingest_policy_document",
    acks_late=True,
    reject_on_worker_lost=True,
    track_started=True,
)
def ingest_policy_document_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Celery asynchronous task executing PDF validation, extraction/OCR, structuring, chunking, BGE-M3 embedding, and PG persistence."""
    return execute_ingestion_task(task=self, raw_payload=payload)


@celery_app.task(
    bind=True,
    name="coverageiq.ingestion.ocr_policy_page",
    acks_late=True,
    reject_on_worker_lost=True,
    track_started=True,
)
def ocr_policy_page_task(self, storage_ref: str, page_number: int) -> Dict[str, Any]:
    """Dedicated OCR queue task descriptor for offloading page-level CPU OCR when queued separately."""
    res_mgr = get_worker_resources()
    storage = res_mgr.get_storage()
    ocr_engine = res_mgr.get_ocr_engine()
    import fitz

    with storage.resolve_for_reading(storage_ref) as handle:
        with fitz.open(handle.local_path) as doc:
            if page_number < 1 or page_number > len(doc):
                raise IngestionPermanentError(f"[INVALID_PAGE] Page {page_number} out of range (1..{len(doc)}).")
            res = ocr_engine.process_page(doc[page_number - 1], page_number)
            return res.model_dump()
