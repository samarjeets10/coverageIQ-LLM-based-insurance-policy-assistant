from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import fitz
import pytest

from ai_service.config import Settings
from ai_service.embeddings import (
    EmbeddingConfigurationError,
    EmbeddingCudaOutOfMemoryError,
    EmbeddingInferenceError,
    EmbeddingModelLoadError,
    EmbeddingOutputValidationError,
    OfflineHashEmbedder,
)
from ai_service.ingestion import (
    CorruptedPDFError,
    DocumentTooLargeError,
    EmptyDocumentError,
    InvalidPDFSignatureError,
    MockOCREngine,
    PasswordProtectedPDFError,
    PathTraversalError,
)
from ai_service.persistence import (
    AuthorizedScope,
    ChunkEmbeddingRecord,
    DatabaseAuthorizationError,
    DatabaseConnectionError,
    DatabaseConstraintError,
    DatabaseNotFoundError,
    IngestionJobRecord,
    PersistedChunkRecord,
    PolicyDocumentRecord,
    PolicyRecord,
)
from ai_service.schemas.structured_output import PolicyChunk
from ai_service.storage import (
    DocumentNotFoundError,
    DocumentPathSecurityError,
    LocalFilesystemDocumentStorage,
)
from ai_service.workers import (
    IngestionExecutionResult,
    IngestionPermanentError,
    IngestionPipelineExecutor,
    IngestionStage,
    IngestionTaskPayload,
    SHA256MismatchError,
    TransientIngestionError,
    WorkerResourceManager,
    celery_app,
    classify_ingestion_error,
    create_celery_app,
    execute_ingestion_task,
    ingest_policy_document_task,
    ocr_policy_page_task,
    sanitize_worker_error,
)


def _make_sample_pdf_bytes(*pages_text: str) -> bytes:
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    return doc.tobytes()


class _FakeTaskRetrySignal(Exception):
    def __init__(self, exc: Exception, countdown: Optional[int]):
        super().__init__(str(exc))
        self.exc = exc
        self.countdown = countdown


class _DummyCeleryTask:
    def __init__(self, task_id: str = "task-unit-001", retries: int = 0):
        self.request = type("Req", (), {"id": task_id, "retries": retries})()
        self.retry_calls: List[Dict[str, Any]] = []

    def retry(self, *, exc: Exception, countdown: Optional[int] = None):
        self.retry_calls.append({"exc": exc, "countdown": countdown})
        raise _FakeTaskRetrySignal(exc=exc, countdown=countdown)


class _InMemoryWorkerRepo:
    def __init__(
        self,
        *,
        job: IngestionJobRecord,
        policy: PolicyRecord,
        document: PolicyDocumentRecord,
    ):
        self.job = job
        self.policy = policy
        self.document = document
        self.stage_history: List[Dict[str, Any]] = []
        self.persisted_chunks: List[PersistedChunkRecord] = []
        self.persisted_embeddings: List[ChunkEmbeddingRecord] = []
        self.complete_calls = 0

    def fetch_worker_job_context(
        self,
        *,
        job_id: UUID,
        policy_id: UUID,
        document_id: UUID,
        claimed_owner_scope: str,
    ):
        if job_id != self.job.id:
            raise DatabaseNotFoundError("Job not found.")
        if policy_id != self.policy.id or document_id != self.document.id:
            raise DatabaseNotFoundError("Policy/document not found.")
        if (
            self.job.owner_scope != claimed_owner_scope
            or self.policy.owner_scope != claimed_owner_scope
            or self.job.policy_id != policy_id
            or self.job.document_id != document_id
        ):
            raise DatabaseAuthorizationError("Scope or identifier mismatch.")
        scope = AuthorizedScope(
            user_id=self.policy.owner_id,
            role="user",
            owner_scope=self.policy.owner_scope,
            allow_catalog_read=False,
            trusted_gateway_context=True,
        )
        return self.job, self.policy, self.document, scope

    def claim_ingestion_job_for_execution(
        self,
        *,
        job_id: UUID,
        document_id: UUID,
        task_id: Optional[str],
        scope: AuthorizedScope,
        initial_stage: str = "VALIDATING",
        is_celery_retry: bool = False,
        stale_lock_seconds: int = 600,
    ):
        if self.job.status == "completed" and self.document.ingestion_status == "completed":
            return "already_completed", self.job, self.document
        if self.job.status == "processing" and not is_celery_retry and self.job.celery_task_id != task_id:
            return "duplicate_in_progress", self.job, self.document

        now = datetime.now(timezone.utc)
        self.job = self.job.model_copy(
            update={
                "status": "processing",
                "current_stage": initial_stage,
                "celery_task_id": task_id or self.job.celery_task_id,
                "attempt_count": self.job.attempt_count + 1,
                "updated_at": now,
            }
        )
        self.document = self.document.model_copy(update={"ingestion_status": "processing", "updated_at": now})
        self.stage_history.append({"status": "processing", "stage": initial_stage, "doc_status": "processing"})
        return "claimed", self.job, self.document

    def transition_ingestion_job_stage(
        self,
        *,
        job_id: UUID,
        document_id: UUID,
        scope: AuthorizedScope,
        status: str,
        stage: str,
        doc_status: Optional[str] = None,
        total_chunks: Optional[int] = None,
        processed_chunks: Optional[int] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        metadata_patch: Optional[Dict[str, Any]] = None,
        mark_completed_at: bool = False,
    ):
        now = datetime.now(timezone.utc)
        self.job = self.job.model_copy(
            update={
                "status": status,
                "current_stage": stage,
                "total_chunks": total_chunks if total_chunks is not None else self.job.total_chunks,
                "processed_chunks": (
                    processed_chunks if processed_chunks is not None else self.job.processed_chunks
                ),
                "error_code": error_code,
                "error_message": error_message,
                "completed_at": now if mark_completed_at else self.job.completed_at,
                "updated_at": now,
            }
        )
        if doc_status is not None:
            self.document = self.document.model_copy(update={"ingestion_status": doc_status, "updated_at": now})
        self.stage_history.append(
            {
                "status": status,
                "stage": stage,
                "doc_status": doc_status,
                "error_code": error_code,
                "error_message": error_message,
            }
        )
        return self.job

    def complete_document_ingestion(
        self,
        *,
        job_id: UUID,
        policy_id: UUID,
        document_id: UUID,
        embedded_chunks: List[PolicyChunk],
        model_name: str,
        model_revision: str,
        page_count: int,
        document_metadata: Optional[Dict[str, Any]],
        job_metadata: Optional[Dict[str, Any]],
        scope: AuthorizedScope,
        is_normalized: bool = True,
        activate_document: bool = True,
    ):
        self.complete_calls += 1
        now = datetime.now(timezone.utc)
        pcs: List[PersistedChunkRecord] = []
        embs: List[ChunkEmbeddingRecord] = []
        for idx, ch in enumerate(embedded_chunks):
            assert ch.embedding is not None and len(ch.embedding) == 1024
            pc = PersistedChunkRecord(
                id=uuid4(),
                policy_id=policy_id,
                document_id=document_id,
                chunk_key=ch.chunk_id,
                chunk_index=idx,
                section_id=None,
                clause_ref=ch.clause_ref,
                category=ch.category.value,
                page_number=ch.page_number,
                page_start=ch.page_start or ch.page_number,
                page_end=ch.page_end or ch.page_number,
                content_text=ch.content_text,
                source_block_ids=list(ch.source_block_ids),
                table_id=ch.table_id,
                extraction_method="DIGITAL",
                token_count=None,
                metadata=dict(ch.metadata),
                created_at=now,
                updated_at=now,
            )
            ce = ChunkEmbeddingRecord(
                id=uuid4(),
                chunk_id=pc.id,
                policy_id=policy_id,
                document_id=document_id,
                model_name=model_name,
                model_revision=model_revision,
                embedding_dim=1024,
                is_normalized=is_normalized,
                created_at=now,
                updated_at=now,
            )
            pcs.append(pc)
            embs.append(ce)
        self.persisted_chunks = pcs
        self.persisted_embeddings = embs
        self.document = self.document.model_copy(
            update={"ingestion_status": "completed", "page_count": page_count, "updated_at": now}
        )
        if activate_document:
            self.policy = self.policy.model_copy(
                update={"active_version_id": document_id, "status": "active", "updated_at": now}
            )
        self.job = self.job.model_copy(
            update={
                "status": "completed",
                "current_stage": IngestionStage.COMPLETED.value,
                "total_chunks": len(pcs),
                "processed_chunks": len(pcs),
                "completed_at": now,
                "updated_at": now,
            }
        )
        return pcs, embs, self.job


def _make_fake_records(
    *,
    storage_path: str,
    file_hash_sha256: str,
    file_size_bytes: int,
    owner_id: str = "alice",
    owner_scope: str = "user:alice",
) -> tuple[IngestionJobRecord, PolicyRecord, PolicyDocumentRecord]:
    now = datetime.now(timezone.utc)
    policy_id = uuid4()
    document_id = uuid4()
    job_id = uuid4()
    policy = PolicyRecord(
        id=policy_id,
        external_policy_key="pol_unit_async",
        title="Unit Health Plan",
        insurer_name="CoverageIQ Assurance",
        policy_type="health",
        visibility="private",
        owner_id=owner_id,
        owner_scope=owner_scope,
        status="draft",
        active_version_id=None,
        metadata={},
        created_at=now,
        updated_at=now,
    )
    document = PolicyDocumentRecord(
        id=document_id,
        policy_id=policy_id,
        version_number=1,
        version_label="v1",
        file_name="unit_health.pdf",
        storage_path=storage_path,
        mime_type="application/pdf",
        file_size_bytes=file_size_bytes,
        file_hash_sha256=file_hash_sha256,
        page_count=0,
        is_encrypted=False,
        is_archived=False,
        ingestion_status="queued",
        document_metadata={},
        created_at=now,
        updated_at=now,
    )
    job = IngestionJobRecord(
        id=job_id,
        policy_id=policy_id,
        document_id=document_id,
        external_policy_key="pol_unit_async",
        document_version_snapshot=1,
        file_hash_sha256=file_hash_sha256,
        owner_scope=owner_scope,
        celery_task_id=None,
        status="queued",
        current_stage=IngestionStage.QUEUED.value,
        total_chunks=0,
        processed_chunks=0,
        attempt_count=0,
        error_code=None,
        error_message=None,
        metadata={},
        started_at=None,
        completed_at=None,
        created_at=now,
        updated_at=now,
    )
    return job, policy, document


# =============================================================================
# 1. Celery App & Routing Configuration Tests
# =============================================================================
def test_celery_app_configuration_and_task_registration():
    cfg = Settings(
        EMBEDDING_PROVIDER="offline",
        CELERY_BROKER_URL="redis://127.0.0.1:6399/1",
        CELERY_RESULT_BACKEND="redis://127.0.0.1:6399/1",
        CELERY_WORKER_CONCURRENCY=1,
    )
    app = create_celery_app(cfg)

    assert app.conf.task_serializer == "json"
    assert app.conf.result_serializer == "json"
    assert list(app.conf.accept_content) == ["json"]
    assert app.conf.task_acks_late is True
    assert app.conf.task_reject_on_worker_lost is True
    assert app.conf.task_track_started is True
    assert app.conf.worker_prefetch_multiplier == 1
    assert app.conf.worker_concurrency == 1
    assert app.conf.task_always_eager is False

    queue_names = {q.name for q in app.conf.task_queues}
    assert queue_names == {"ingestion", "ocr"}
    assert app.conf.task_routes["coverageiq.ingestion.ingest_policy_document"] == {"queue": "ingestion"}
    assert app.conf.task_routes["coverageiq.ingestion.ocr_policy_page"] == {"queue": "ocr"}

    # Module-level tasks registered on shared celery_app
    assert ingest_policy_document_task.name == "coverageiq.ingestion.ingest_policy_document"
    assert ocr_policy_page_task.name == "coverageiq.ingestion.ocr_policy_page"
    assert "coverageiq.ingestion.ingest_policy_document" in celery_app.tasks
    assert "coverageiq.ingestion.ocr_policy_page" in celery_app.tasks


# =============================================================================
# 2. Task Payload Validation (No Raw Bytes / Embeddings / Extra Keys)
# =============================================================================
def test_task_payload_contract_rejects_raw_bytes_embeddings_and_invalid_fields():
    valid = {
        "job_id": str(uuid4()),
        "policy_id": str(uuid4()),
        "document_id": str(uuid4()),
        "owner_scope": "user:alice",
        "storage_ref": "alice/pol/doc.pdf",
        "expected_sha256": "a" * 64,
        "request_id": "req-123",
    }
    parsed = IngestionTaskPayload.model_validate(valid)
    assert parsed.owner_scope == "user:alice"

    for extra_key in ("raw_pdf_bytes", "pdf_base64", "embeddings", "chunks", "text"):
        bad = {**valid, extra_key: "forbidden"}
        with pytest.raises(Exception):
            IngestionTaskPayload.model_validate(bad)

    with pytest.raises(Exception):
        IngestionTaskPayload.model_validate({**valid, "owner_scope": "user:*"})

    with pytest.raises(Exception):
        IngestionTaskPayload.model_validate({**valid, "expected_sha256": "not-sha256"})


# =============================================================================
# 3. Document Storage Abstraction Security Tests
# =============================================================================
def test_local_filesystem_document_storage_jails_paths_and_rejects_escapes(tmp_path: Path):
    upload_root = tmp_path / "uploads" / "policies"
    outside_secret = tmp_path / "secret.pdf"
    outside_secret.write_bytes(b"%PDF-1.4\nsecret")

    cfg = Settings(POLICY_UPLOAD_ROOT=str(upload_root), EMBEDDING_PROVIDER="offline")
    store = LocalFilesystemDocumentStorage(app_settings=cfg)

    saved_path = store.put_bytes("user_alice/policy_1/v1.pdf", b"%PDF-1.4\nhello")
    assert saved_path.exists()
    assert store.exists("user_alice/policy_1/v1.pdf") is True

    with store.resolve_for_reading("user_alice/policy_1/v1.pdf") as handle:
        assert handle.local_path == saved_path.resolve()
        assert handle.file_size_bytes > 0

    for bad_ref in ("../secret.pdf", "sub/../../secret.pdf", str(outside_secret), "bad\x00.pdf", "   "):
        with pytest.raises(DocumentPathSecurityError):
            store.resolve_safe_path(bad_ref)

    escape_link = upload_root / "escape.pdf"
    escape_link.symlink_to(outside_secret)
    with pytest.raises(DocumentPathSecurityError):
        store.resolve_safe_path("escape.pdf")

    with pytest.raises(DocumentNotFoundError):
        with store.resolve_for_reading("user_alice/policy_1/missing.pdf"):
            pass


# =============================================================================
# 4. Error Sanitization & Transient vs Permanent Classification
# =============================================================================
def test_error_classification_and_dsn_sanitization():
    msg = sanitize_worker_error(
        RuntimeError("Failure connecting postgresql://admin:supersecret@db.internal:5432/prod_db   now")
    )
    assert "supersecret" not in msg
    assert "postgresql://<redacted>" in msg

    permanent_errors = [
        InvalidPDFSignatureError("bad header"),
        PasswordProtectedPDFError("encrypted"),
        CorruptedPDFError("corrupt xref"),
        EmptyDocumentError("empty"),
        DocumentTooLargeError("too large"),
        DocumentPathSecurityError("escape"),
        PathTraversalError("traversal"),
        DocumentNotFoundError("missing file"),
        FileNotFoundError("not found"),
        PermissionError("eacces"),
        SHA256MismatchError("digest mismatch"),
        DatabaseAuthorizationError("forbidden"),
        DatabaseNotFoundError("missing job"),
        DatabaseConstraintError("fk violation"),
        EmbeddingCudaOutOfMemoryError("CUDA out of memory"),
        EmbeddingModelLoadError("load failed"),
        EmbeddingOutputValidationError("bad dim"),
        EmbeddingConfigurationError("bad dim config"),
        EmbeddingInferenceError("inference err"),
    ]
    for err in permanent_errors:
        is_transient, code, _ = classify_ingestion_error(err)
        assert is_transient is False, f"{type(err).__name__} must be permanent: got {code}"

    transient_errors = [
        DatabaseConnectionError("pool exhausted"),
        TransientIngestionError("redis blip"),
        TimeoutError("socket timeout"),
        ConnectionError("connection reset"),
        OSError("EIO"),
    ]
    for err in transient_errors:
        is_transient, code, _ = classify_ingestion_error(err)
        assert is_transient is True, f"{type(err).__name__} must be transient: got {code}"


# =============================================================================
# 5. Permanent & Transient Task Retry Behavior (No Fake Success)
# =============================================================================
@pytest.mark.parametrize(
    ("raised_exc", "expected_code"),
    [
        (InvalidPDFSignatureError("not a pdf"), "INVALID_PDF_SIGNATURE"),
        (SHA256MismatchError("hash mismatch"), "SHA256_MISMATCH"),
        (DocumentNotFoundError("gone"), "DOCUMENT_FILE_MISSING"),
        (EmbeddingCudaOutOfMemoryError("CUDA OOM during batch"), "EMBEDDING_CUDA_OOM"),
    ],
)
def test_permanent_failures_never_retry_and_mark_job_failed(tmp_path: Path, raised_exc, expected_code):
    pdf_bytes = _make_sample_pdf_bytes("Section 1 Coverage: Hospitalization is covered.")
    sha = hashlib.sha256(pdf_bytes).hexdigest()
    job, policy, doc = _make_fake_records(
        storage_path="alice/unit.pdf",
        file_hash_sha256=sha,
        file_size_bytes=len(pdf_bytes),
    )
    repo = _InMemoryWorkerRepo(job=job, policy=policy, document=doc)
    cfg = Settings(
        POLICY_UPLOAD_ROOT=str(tmp_path / "uploads"),
        EMBEDDING_PROVIDER="offline",
        CELERY_TASK_MAX_RETRIES=3,
    )
    res_mgr = WorkerResourceManager(app_settings=cfg)

    class _FailingPipe:
        _repository = repo

        def execute(self, raw_payload, *, task_id=None, is_celery_retry=False):
            raise raised_exc

    task = _DummyCeleryTask(retries=0)
    payload = {
        "job_id": str(job.id),
        "policy_id": str(policy.id),
        "document_id": str(doc.id),
        "owner_scope": "user:alice",
    }

    with pytest.raises(IngestionPermanentError) as exc_info:
        execute_ingestion_task(task=task, raw_payload=payload, executor=_FailingPipe(), resources=res_mgr, repository=repo)

    assert expected_code in str(exc_info.value)
    assert task.retry_calls == []
    assert repo.job.status == "failed"
    assert repo.job.current_stage == IngestionStage.FAILED.value
    assert repo.job.error_code == expected_code
    assert repo.document.ingestion_status == "failed"
    assert repo.complete_calls == 0


def test_transient_failure_retries_with_bounded_backoff_then_fails_cleanly_when_exhausted(tmp_path: Path):
    pdf_bytes = _make_sample_pdf_bytes("Section 1 Coverage: Hospitalization is covered.")
    sha = hashlib.sha256(pdf_bytes).hexdigest()
    job, policy, doc = _make_fake_records(
        storage_path="alice/unit.pdf",
        file_hash_sha256=sha,
        file_size_bytes=len(pdf_bytes),
    )
    repo = _InMemoryWorkerRepo(job=job, policy=policy, document=doc)
    cfg = Settings(
        POLICY_UPLOAD_ROOT=str(tmp_path / "uploads"),
        EMBEDDING_PROVIDER="offline",
        CELERY_TASK_MAX_RETRIES=2,
        CELERY_TASK_RETRY_BACKOFF_BASE=5,
        CELERY_TASK_RETRY_BACKOFF_MAX=12,
    )
    res_mgr = WorkerResourceManager(app_settings=cfg)

    class _TransientPipe:
        _repository = repo

        def execute(self, raw_payload, *, task_id=None, is_celery_retry=False):
            raise DatabaseConnectionError("transient postgres blip postgresql://u:pw@db:5432/x")

    payload = {
        "job_id": str(job.id),
        "policy_id": str(policy.id),
        "document_id": str(doc.id),
        "owner_scope": "user:alice",
    }

    # Attempt 0: schedules retry with backoff >= 5 and <= 12
    task0 = _DummyCeleryTask(retries=0)
    with pytest.raises(_FakeTaskRetrySignal) as r0:
        execute_ingestion_task(task=task0, raw_payload=payload, executor=_TransientPipe(), resources=res_mgr, repository=repo)
    assert 5 <= r0.value.countdown <= 6
    assert repo.job.status == "queued"
    assert repo.job.current_stage == IngestionStage.RETRY_SCHEDULED.value
    assert "pw@" not in (repo.job.error_message or "")

    # Attempt 1: backoff capped at CELERY_TASK_RETRY_BACKOFF_MAX (12) even if base*2=10 (+jitter)
    task1 = _DummyCeleryTask(retries=1)
    with pytest.raises(_FakeTaskRetrySignal) as r1:
        execute_ingestion_task(task=task1, raw_payload=payload, executor=_TransientPipe(), resources=res_mgr, repository=repo)
    assert 10 <= r1.value.countdown <= 11

    # Attempt 2 == max_retries -> exhausted, fails cleanly without retrying
    task2 = _DummyCeleryTask(retries=2)
    with pytest.raises(IngestionPermanentError) as final_err:
        execute_ingestion_task(task=task2, raw_payload=payload, executor=_TransientPipe(), resources=res_mgr, repository=repo)
    assert "DATABASE_CONNECTION_TRANSIENT" in str(final_err.value)
    assert task2.retry_calls == []
    assert repo.job.status == "failed"
    assert repo.job.current_stage == IngestionStage.FAILED.value
    assert repo.document.ingestion_status == "failed"


# =============================================================================
# 6. WorkerResourceManager Lazy GPU / Embedder Reuse & Shutdown Cleanup
# =============================================================================
def test_worker_resource_manager_reuses_single_batch1_embedder_and_cleans_up(tmp_path: Path):
    cfg = Settings(
        POLICY_UPLOAD_ROOT=str(tmp_path / "uploads"),
        EMBEDDING_PROVIDER="offline",
        EMBEDDING_BATCH_SIZE=8,
        CELERY_INGESTION_EMBEDDING_BATCH_SIZE=1,
    )
    mgr = WorkerResourceManager(app_settings=cfg)
    assert mgr._embedder is None

    emb1 = mgr.get_embedder()
    emb2 = mgr.get_embedder()
    assert emb1 is emb2
    assert emb1.dimension == 1024

    mgr.shutdown()
    assert mgr._embedder is None
    assert mgr._pool is None

    # Also verify lazy BgeM3Embedder gets CELERY_INGESTION_EMBEDDING_BATCH_SIZE=1 without loading weights
    cfg_bge = Settings(
        POLICY_UPLOAD_ROOT=str(tmp_path / "uploads"),
        EMBEDDING_PROVIDER="bge-m3",
        EMBEDDING_BATCH_SIZE=8,
        CELERY_INGESTION_EMBEDDING_BATCH_SIZE=1,
        EMBEDDING_LOCAL_FILES_ONLY=True,
    )
    mgr_bge = WorkerResourceManager(app_settings=cfg_bge)
    bge_emb = mgr_bge.get_embedder()
    assert bge_emb.batch_size == 1
    assert bge_emb.is_loaded is False
    assert mgr_bge.get_embedder() is bge_emb
    mgr_bge.shutdown()


# =============================================================================
# 7. End-to-End Pipeline Unit Flow (Real PyMuPDF + Chunker + OfflineHashEmbedder)
# =============================================================================
def test_pipeline_executor_runs_all_stages_and_prevents_chunker_from_embedding(tmp_path: Path):
    policy_text = (
        "Section 2.1 In-Patient Hospitalization Coverage\n"
        "The company indemnifies reasonable and customary medical expenses incurred during "
        "in-patient hospitalization for a minimum of twenty-four consecutive hours, including "
        "room rent up to 1% of sum insured per day, ICU charges up to 2% per day, surgeon fees, "
        "diagnostic imaging, and medicines.\n\n"
        "Section 4.1 Exclusions\n"
        "Cosmetic or aesthetic plastic surgery is excluded unless medically required for "
        "reconstructive surgery following an accidental injury or medically diagnosed burns."
    )
    pdf_bytes = _make_sample_pdf_bytes(policy_text)
    sha = hashlib.sha256(pdf_bytes).hexdigest()

    cfg = Settings(
        POLICY_UPLOAD_ROOT=str(tmp_path / "uploads"),
        EMBEDDING_PROVIDER="offline",
        CELERY_INGESTION_EMBEDDING_BATCH_SIZE=1,
    )
    res_mgr = WorkerResourceManager(app_settings=cfg)
    storage = res_mgr.get_storage()
    storage.put_bytes("alice/pol_unit_async/v1.pdf", pdf_bytes)

    job, policy, doc = _make_fake_records(
        storage_path="alice/pol_unit_async/v1.pdf",
        file_hash_sha256=sha,
        file_size_bytes=len(pdf_bytes),
    )
    repo = _InMemoryWorkerRepo(job=job, policy=policy, document=doc)
    executor = IngestionPipelineExecutor(
        resources=res_mgr,
        repository=repo,
        ocr_engine=MockOCREngine(),
        app_settings=cfg,
    )

    result = execute_ingestion_task(
        task=_DummyCeleryTask(task_id="task-unit-success"),
        raw_payload={
            "job_id": str(job.id),
            "policy_id": str(policy.id),
            "document_id": str(doc.id),
            "owner_scope": "user:alice",
            "storage_ref": "alice/pol_unit_async/v1.pdf",
            "expected_sha256": sha,
        },
        executor=executor,
        resources=res_mgr,
        repository=repo,
    )

    assert result["status"] == "completed"
    assert result["stage"] == IngestionStage.COMPLETED.value
    assert result["embedding_dim"] == 1024
    assert result["chunk_count"] >= 1
    assert repo.complete_calls == 1
    assert len(repo.persisted_chunks) == result["chunk_count"]
    assert len(repo.persisted_embeddings) == result["chunk_count"]

    recorded_stages = [entry["stage"] for entry in repo.stage_history]
    assert recorded_stages == [
        "VALIDATING",
        "EXTRACTING",
        "STRUCTURING",
        "CHUNKING",
        "EMBEDDING",
        "EXTRACTING_ATTRIBUTES",
        "PERSISTING",
    ]

    # Idempotent re-execution skips cleanly without duplicating chunks
    repeat = execute_ingestion_task(
        task=_DummyCeleryTask(task_id="task-unit-repeat"),
        raw_payload={
            "job_id": str(job.id),
            "policy_id": str(policy.id),
            "document_id": str(doc.id),
            "owner_scope": "user:alice",
        },
        executor=executor,
        resources=res_mgr,
        repository=repo,
    )
    assert repeat["already_completed"] is True
    assert repo.complete_calls == 1


def test_pipeline_executor_blocks_tampered_sha256_and_mismatched_storage_ref(tmp_path: Path):
    pdf_bytes = _make_sample_pdf_bytes("Section 1 Benefits: Covered.")
    good_sha = hashlib.sha256(pdf_bytes).hexdigest()

    cfg = Settings(POLICY_UPLOAD_ROOT=str(tmp_path / "uploads"), EMBEDDING_PROVIDER="offline")
    res_mgr = WorkerResourceManager(app_settings=cfg)
    storage = res_mgr.get_storage()
    storage.put_bytes("alice/tamper.pdf", b"%PDF-1.4\ntampered content")

    job, policy, doc = _make_fake_records(
        storage_path="alice/tamper.pdf",
        file_hash_sha256=good_sha,
        file_size_bytes=len(pdf_bytes),
    )
    repo = _InMemoryWorkerRepo(job=job, policy=policy, document=doc)
    executor = IngestionPipelineExecutor(resources=res_mgr, repository=repo, ocr_engine=MockOCREngine(), app_settings=cfg)

    with pytest.raises(IngestionPermanentError) as ref_err:
        execute_ingestion_task(
            task=_DummyCeleryTask(),
            raw_payload={
                "job_id": str(job.id),
                "policy_id": str(policy.id),
                "document_id": str(doc.id),
                "owner_scope": "user:alice",
                "storage_ref": "alice/other.pdf",
            },
            executor=executor,
            resources=res_mgr,
            repository=repo,
        )
    assert "INVALID_STORAGE_PATH" in str(ref_err.value)
    assert repo.complete_calls == 0

    # Reset job back to queued and verify tampered on-disk PDF fails SHA/signature verification
    repo.job = repo.job.model_copy(update={"status": "queued"})
    repo.document = repo.document.model_copy(update={"ingestion_status": "queued"})
    storage.put_bytes("alice/tamper.pdf", pdf_bytes + b"\n%extra")
    with pytest.raises(IngestionPermanentError) as sha_err:
        execute_ingestion_task(
            task=_DummyCeleryTask(),
            raw_payload={
                "job_id": str(job.id),
                "policy_id": str(policy.id),
                "document_id": str(doc.id),
                "owner_scope": "user:alice",
            },
            executor=executor,
            resources=res_mgr,
            repository=repo,
        )
    assert "SHA256_MISMATCH" in str(sha_err.value)
    assert repo.job.status == "failed"
    assert repo.complete_calls == 0


def test_pipeline_executor_materializes_scanned_ocr_boxes_into_chunks(tmp_path: Path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 400, 600), 0)
    pix.set_rect(pix.irect, (238, 238, 238))
    page.insert_image(fitz.Rect(50, 50, 545, 792), stream=pix.tobytes())
    scanned_pdf = doc.tobytes()
    doc.close()

    sha = hashlib.sha256(scanned_pdf).hexdigest()
    cfg = Settings(
        POLICY_UPLOAD_ROOT=str(tmp_path / "uploads"),
        EMBEDDING_PROVIDER="offline",
        CELERY_INGESTION_EMBEDDING_BATCH_SIZE=1,
    )
    res_mgr = WorkerResourceManager(app_settings=cfg)
    storage = res_mgr.get_storage()
    storage.put_bytes("alice/scanned_unit.pdf", scanned_pdf)

    job, policy, doc_rec = _make_fake_records(
        storage_path="alice/scanned_unit.pdf",
        file_hash_sha256=sha,
        file_size_bytes=len(scanned_pdf),
    )
    repo = _InMemoryWorkerRepo(job=job, policy=policy, document=doc_rec)
    ocr_engine = MockOCREngine(
        mock_boxes=[
            ("Section 3.1 Scanned Benefit Clause", 0.94, (60.0, 140.0, 520.0, 170.0)),
            (
                "Emergency hospitalization expenses incurred for accidental injuries are covered up to Sum Insured.",
                0.92,
                (60.0, 190.0, 520.0, 260.0),
            ),
        ]
    )
    executor = IngestionPipelineExecutor(
        resources=res_mgr,
        repository=repo,
        ocr_engine=ocr_engine,
        app_settings=cfg,
    )

    result = execute_ingestion_task(
        task=_DummyCeleryTask(task_id="task-unit-scanned"),
        raw_payload={
            "job_id": str(job.id),
            "policy_id": str(policy.id),
            "document_id": str(doc_rec.id),
            "owner_scope": "user:alice",
        },
        executor=executor,
        resources=res_mgr,
        repository=repo,
    )

    assert result["status"] == "completed"
    assert result["ocr_pages_count"] == 1
    assert result["chunk_count"] >= 1
    assert any(entry["stage"] == "OCR" for entry in repo.stage_history)
    assert any("accidental injuries" in ch.content_text.lower() for ch in repo.persisted_chunks)
