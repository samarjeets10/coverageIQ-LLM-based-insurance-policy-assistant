import hashlib
import inspect
import logging
from pathlib import Path
import time
from typing import Callable, Dict, Optional, Tuple
from urllib.parse import urlsplit
import re

from pydantic import ValidationError

from ai_service.config import Settings
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
from ai_service.recommendation.attribute_extractor import PolicyAttributeExtractor
from ai_service.embeddings import (
    BaseEmbedder,
    EmbeddingConfigurationError,
    EmbeddingCudaOutOfMemoryError,
    EmbeddingError,
    EmbeddingInferenceError,
    EmbeddingModelLoadError,
    EmbeddingOutputValidationError,
)
from ai_service.ingestion import (
    BaseOCREngine,
    CorruptedPDFError,
    DocumentStructurer,
    DocumentTooLargeError,
    EmptyDocumentError,
    IngestionError,
    InvalidPDFSignatureError,
    PDFValidator,
    PasswordProtectedPDFError,
    PathTraversalError,
    PyMuPDFLoader,
    SemanticChunker,
)
from ai_service.persistence import (
    AuthorizedScope,
    DatabaseAuthorizationError,
    DatabaseConnectionError,
    DatabaseConstraintError,
    DatabaseMigrationError,
    DatabaseNotFoundError,
    DatabaseValidationError,
    PolicyKnowledgeRepository,
)
from ai_service.schemas.ingestion import (
    ExtractedBlock,
    ExtractedDocument,
    ExtractedLine,
    ExtractedSpan,
)
from ai_service.storage import (
    DocumentNotFoundError,
    DocumentPathSecurityError,
    DocumentStorageBackend,
)
from ai_service.workers.celery_app import WorkerResourceManager, get_worker_resources
from ai_service.workers.contracts import (
    IngestionExecutionResult,
    IngestionStage,
    IngestionTaskPayload,
)

logger = logging.getLogger("coverageiq.workers.ingestion")

_DSN_RE = re.compile(r"postgres(?:ql)?://[^\s'\"]+", re.IGNORECASE)


class SHA256MismatchError(IngestionError):
    """Raised when on-disk PDF checksum differs from registered document SHA-256."""


class IngestionPermanentError(IngestionError):
    """Raised when an ingestion failure cannot succeed cleanly via retry."""


class TransientIngestionError(Exception):
    """Raised on temporary connectivity/infrastructure failures eligible for bounded Celery retry."""


def sanitize_worker_error(exc: Exception) -> str:
    """Produce a safe, secret-free error description capped in length with DSNs masked."""
    raw = str(exc) if str(exc) else type(exc).__name__
    redacted = _DSN_RE.sub("postgresql://<redacted>", raw)
    # Collapse whitespace and limit length so DB logs/columns never store huge traces or file paths
    cleaned = " ".join(redacted.split())
    return cleaned[:400]


def classify_ingestion_error(exc: Exception) -> Tuple[bool, str, str]:
    """
    Classify exceptions as transient (retryable) vs permanent (no retry).
    Returns:
        (is_transient, error_code, sanitized_error_message)
    """
    msg = sanitize_worker_error(exc)

    # 1. Explicit permanent security, file, PDF, schema, or embedding failures
    if isinstance(exc, DatabaseAuthorizationError):
        return False, "UNAUTHORIZED_SCOPE", msg
    if isinstance(exc, DatabaseNotFoundError):
        return False, "RECORD_NOT_FOUND", msg
    if isinstance(exc, DatabaseMigrationError):
        return False, "DATABASE_MIGRATION_REQUIRED", msg
    if isinstance(exc, (DatabaseConstraintError, DatabaseValidationError)):
        return False, "DATABASE_VALIDATION_ERROR", msg
    if isinstance(exc, ValidationError):
        return False, "INVALID_TASK_PAYLOAD", msg

    if isinstance(exc, (DocumentPathSecurityError, PathTraversalError)):
        return False, "INVALID_STORAGE_PATH", msg
    if isinstance(exc, (DocumentNotFoundError, FileNotFoundError)):
        return False, "DOCUMENT_FILE_MISSING", msg
    if isinstance(exc, PermissionError):
        return False, "DOCUMENT_ACCESS_DENIED", msg
    if isinstance(exc, SHA256MismatchError):
        return False, "SHA256_MISMATCH", msg
    if isinstance(exc, InvalidPDFSignatureError):
        return False, "INVALID_PDF_SIGNATURE", msg
    if isinstance(exc, PasswordProtectedPDFError):
        return False, "PASSWORD_PROTECTED_PDF", msg
    if isinstance(exc, DocumentTooLargeError):
        return False, "DOCUMENT_TOO_LARGE", msg
    if isinstance(exc, EmptyDocumentError):
        return False, "EMPTY_OR_UNEXTRACTABLE_PDF", msg
    if isinstance(exc, CorruptedPDFError):
        return False, "CORRUPTED_PDF", msg
    if isinstance(exc, (IngestionPermanentError, IngestionError)):
        return False, "PERMANENT_INGESTION_ERROR", msg

    if isinstance(exc, EmbeddingCudaOutOfMemoryError):
        return False, "EMBEDDING_CUDA_OOM", msg
    if isinstance(exc, EmbeddingModelLoadError):
        return False, "EMBEDDING_MODEL_LOAD_FAILED", msg
    if isinstance(exc, EmbeddingOutputValidationError):
        return False, "EMBEDDING_OUTPUT_INVALID", msg
    if isinstance(exc, EmbeddingConfigurationError):
        return False, "EMBEDDING_CONFIGURATION_ERROR", msg
    if isinstance(exc, (EmbeddingInferenceError, EmbeddingError)):
        return False, "EMBEDDING_INFERENCE_FAILED", msg

    if isinstance(
        exc,
        (
            ExternalProviderConfigurationError,
            NvidiaConfigurationError,
            NvidiaAuthenticationError,
            NvidiaPermissionError,
        ),
    ):
        return False, "ATTRIBUTE_EXTRACTION_CONFIG_ERROR", msg
    if isinstance(exc, LLMGenerationError):
        return True, "ATTRIBUTE_EXTRACTION_LLM_INVALID", msg
    if isinstance(exc, (NvidiaTimeoutError, NvidiaRateLimitError, NvidiaProviderError, ExternalProviderError)):
        return True, "ATTRIBUTE_EXTRACTION_TRANSIENT", msg

    # 2. Explicit transient database / network / broker I/O failures
    if isinstance(exc, DatabaseConnectionError):
        return True, "DATABASE_CONNECTION_TRANSIENT", msg
    if isinstance(exc, TransientIngestionError):
        return True, "TRANSIENT_INGESTION_ERROR", msg
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True, "TRANSIENT_IO_ERROR", msg

    # Default fail-safe: non-retryable unexpected error
    return False, f"UNEXPECTED_{type(exc).__name__.upper()}", f"{type(exc).__name__}: {msg}"


class _TrackingOCRProxy(BaseOCREngine):
    """Lightweight delegating wrapper around BaseOCREngine to emit stage=OCR only when OCR runs."""

    def __init__(self, inner: BaseOCREngine, on_first_ocr_page: Callable[[], None]):
        self._inner = inner
        self._on_first_ocr_page = on_first_ocr_page
        self._notified = False
        self.ocr_pages_count = 0

    def is_available(self) -> bool:
        return self._inner.is_available()

    def initialize(self) -> bool:
        return self._inner.initialize()

    def process_page(self, page, page_number, dpi=None, clip_rect=None):
        if not self._notified:
            self._notified = True
            self._on_first_ocr_page()
        self.ocr_pages_count += 1
        return self._inner.process_page(page, page_number, dpi=dpi, clip_rect=clip_rect)


def _resolve_model_revision(embedder: BaseEmbedder) -> str:
    if embedder.model_name == "offline-sha256-hash":
        return "offline-sha256-v1"
    try:
        from huggingface_hub import snapshot_download

        snap_path = snapshot_download(embedder.model_name, local_files_only=True)
        return Path(snap_path).name
    except Exception:
        return "local-default"


def _materialize_ocr_text_blocks(extracted_doc: ExtractedDocument) -> None:
    """Populate structured text blocks from OCR results for scanned/hybrid pages."""
    for page in extracted_doc.pages:
        ocr_res = page.ocr_result
        if ocr_res is None or not ocr_res.success or not ocr_res.boxes:
            continue

        existing_text_blocks = [
            b for b in page.blocks if b.block_type == "text" and b.text.strip()
        ]
        ocr_blocks: list[ExtractedBlock] = []
        base_idx = len(page.blocks)

        for idx, obox in enumerate(ocr_res.boxes):
            txt = obox.text.strip()
            if not txt:
                continue
            overlaps_digital = any(
                PyMuPDFLoader._boxes_overlap_significantly(obox.bbox, b.bbox, threshold=0.25)
                for b in existing_text_blocks
            )
            if overlaps_digital:
                continue

            span = ExtractedSpan(
                text=txt,
                font_name="ocr",
                font_size=10.0,
                bbox=obox.bbox,
            )
            line = ExtractedLine(text=txt, bbox=obox.bbox, spans=[span])
            ocr_blocks.append(
                ExtractedBlock(
                    block_id=f"{extracted_doc.doc_id}_p{page.page_number}_ocr_{idx}",
                    page_number=page.page_number,
                    raw_block_idx=base_idx + idx,
                    reading_order_idx=base_idx + idx,
                    block_type="text",
                    bbox=obox.bbox,
                    lines=[line],
                    text=txt,
                )
            )

        if ocr_blocks:
            combined = [b for b in page.blocks if b.block_type != "image"] + ocr_blocks
            ordered = PyMuPDFLoader.reconstruct_reading_order(
                combined, page_width=page.width, page_height=page.height
            )
            for order_idx, blk in enumerate(ordered):
                blk.reading_order_idx = order_idx
            page.blocks = ordered


class IngestionPipelineExecutor:
    """
    Executes the full asynchronous ingestion workflow using existing CoverageIQ components:
    1. Validate payload & authorize against persisted job/policy/document rows.
    2. Resolve document handle cleanly from storage root.
    3. Verify PDF header, structural readability, size, and SHA-256 hash.
    4. Extract text blocks & run OCR when needed via PyMuPDFLoader + BaseOCREngine.
    5. Structure hierarchy & tables via DocumentStructurer.
    6. Generate semantic chunks (embedding=None) via SemanticChunker.
    7. Generate 1024-dim embeddings via existing BGE-M3 embedder outside DB transactions.
    8. Atomically persist chunks, embeddings, active document version, and COMPLETED status.
    """

    def __init__(
        self,
        *,
        resources: Optional[WorkerResourceManager] = None,
        repository: Optional[PolicyKnowledgeRepository] = None,
        storage: Optional[DocumentStorageBackend] = None,
        embedder: Optional[BaseEmbedder] = None,
        ocr_engine: Optional[BaseOCREngine] = None,
        structurer: Optional[DocumentStructurer] = None,
        chunker: Optional[SemanticChunker] = None,
        attribute_extractor: Optional[PolicyAttributeExtractor] = None,
        app_settings: Optional[Settings] = None,
    ):
        self._resources = resources or get_worker_resources()
        self._repository = repository
        self._storage = storage
        self._embedder = embedder
        self._ocr_engine = ocr_engine
        self._structurer = structurer
        self._chunker = chunker
        self._attribute_extractor = attribute_extractor
        self._app_settings = app_settings

    def _repo(self) -> PolicyKnowledgeRepository:
        if self._repository is not None:
            return self._repository
        return PolicyKnowledgeRepository(self._resources.get_pool())

    def _store(self) -> DocumentStorageBackend:
        return self._storage or self._resources.get_storage()

    def _get_embedder(self) -> BaseEmbedder:
        return self._embedder or self._resources.get_embedder()

    def execute(
        self,
        raw_payload: Dict,
        *,
        task_id: Optional[str] = None,
        is_celery_retry: bool = False,
    ) -> Tuple[IngestionExecutionResult, Optional[AuthorizedScope], Optional[IngestionTaskPayload]]:
        t0 = time.perf_counter()
        stage_timings: Dict[str, float] = {}
        payload = IngestionTaskPayload.model_validate(raw_payload)
        repo = self._repo()
        cfg = self._app_settings or self._resources.get_settings()

        # 1. Verify Celery message against persisted DB ownership BEFORE running anything
        job, policy, doc, derived_scope = repo.fetch_worker_job_context(
            job_id=payload.job_id,
            policy_id=payload.policy_id,
            document_id=payload.document_id,
            claimed_owner_scope=payload.owner_scope,
        )

        # 2. Claim job atomically (handles idempotent completion & concurrent duplicate tasks)
        claim_state, claimed_job, claimed_doc = repo.claim_ingestion_job_for_execution(
            job_id=payload.job_id,
            document_id=payload.document_id,
            task_id=task_id,
            scope=derived_scope,
            initial_stage=IngestionStage.VALIDATING.value,
            is_celery_retry=is_celery_retry,
            stale_lock_seconds=cfg.CELERY_TASK_TIME_LIMIT,
        )

        if claim_state == "already_completed":
            logger.info(
                "Skipping ingestion task cleanly; job already completed: job_id=%s document_id=%s",
                payload.job_id,
                payload.document_id,
            )
            return (
                IngestionExecutionResult(
                    job_id=str(payload.job_id),
                    policy_id=str(payload.policy_id),
                    document_id=str(payload.document_id),
                    status="completed",
                    stage=IngestionStage.COMPLETED.value,
                    already_completed=True,
                    page_count=claimed_doc.page_count,
                    chunk_count=claimed_job.processed_chunks,
                    processing_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                ),
                derived_scope,
                payload,
            )

        if claim_state == "duplicate_in_progress":
            logger.info(
                "Skipping duplicate active ingestion task: job_id=%s task_id=%s",
                payload.job_id,
                task_id,
            )
            return (
                IngestionExecutionResult(
                    job_id=str(payload.job_id),
                    policy_id=str(payload.policy_id),
                    document_id=str(payload.document_id),
                    status="processing",
                    stage=claimed_job.current_stage or IngestionStage.VALIDATING.value,
                    duplicate_skipped=True,
                    processing_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                ),
                derived_scope,
                payload,
            )

        storage = self._store()
        if payload.storage_ref is not None:
            ref_path = storage.resolve_safe_path(payload.storage_ref)
            doc_path = storage.resolve_safe_path(claimed_doc.storage_path)
            if ref_path != doc_path:
                raise DocumentPathSecurityError(
                    "Payload storage_ref does not match registered document storage path."
                )

        if payload.expected_sha256 is not None and payload.expected_sha256 != claimed_doc.file_hash_sha256.lower():
            raise SHA256MismatchError("Payload expected_sha256 does not match registered document hash.")

        # 3. Resolve file handle from storage backend & verify PDF integrity + SHA-256
        t_val0 = time.perf_counter()
        with storage.resolve_for_reading(claimed_doc.storage_path) as doc_handle:
            root_bound = getattr(storage, "root_dir", None)
            safe_path, actual_sha256, file_size = PDFValidator.validate_file(
                doc_handle.local_path,
                allowed_root=root_bound,
                max_size_bytes=cfg.MAX_PDF_SIZE_BYTES,
            )
            if actual_sha256.lower() != claimed_doc.file_hash_sha256.lower():
                raise SHA256MismatchError(
                    f"PDF file SHA-256 checksum mismatch for document '{claimed_doc.id}'."
                )
            stage_timings["validating_ms"] = round((time.perf_counter() - t_val0) * 1000.0, 3)

            # 4. Extract PDF blocks (and OCR when required) WITHOUT open DB transactions
            repo.transition_ingestion_job_stage(
                job_id=payload.job_id,
                document_id=payload.document_id,
                scope=derived_scope,
                status="processing",
                stage=IngestionStage.EXTRACTING.value,
            )
            ocr_base = self._ocr_engine or self._resources.get_ocr_engine()

            def _mark_ocr_stage() -> None:
                repo.transition_ingestion_job_stage(
                    job_id=payload.job_id,
                    document_id=payload.document_id,
                    scope=derived_scope,
                    status="processing",
                    stage=IngestionStage.OCR.value,
                )

            ocr_proxy = _TrackingOCRProxy(ocr_base, _mark_ocr_stage)
            loader = PyMuPDFLoader(
                allowed_root=root_bound,
                max_size_bytes=cfg.MAX_PDF_SIZE_BYTES,
                ocr_engine=ocr_proxy,
            )
            t_ext0 = time.perf_counter()
            extracted_doc = loader.load_document(safe_path, doc_id=f"doc_{claimed_doc.id.hex[:12]}")
            _materialize_ocr_text_blocks(extracted_doc)
            stage_timings["extract_and_ocr_ms"] = round((time.perf_counter() - t_ext0) * 1000.0, 3)

            # 5. Structure document hierarchy & tables
            repo.transition_ingestion_job_stage(
                job_id=payload.job_id,
                document_id=payload.document_id,
                scope=derived_scope,
                status="processing",
                stage=IngestionStage.STRUCTURING.value,
            )
            t_str0 = time.perf_counter()
            structurer = self._structurer or DocumentStructurer()
            structured_doc = structurer.structure_document(extracted_doc, pdf_path=safe_path)
            stage_timings["structuring_ms"] = round((time.perf_counter() - t_str0) * 1000.0, 3)

        # 6. Generate Semantic Chunks (SemanticChunker keeps embedding=None)
        repo.transition_ingestion_job_stage(
            job_id=payload.job_id,
            document_id=payload.document_id,
            scope=derived_scope,
            status="processing",
            stage=IngestionStage.CHUNKING.value,
        )
        t_chk0 = time.perf_counter()
        ocr_conf_map: Dict[int, float] = {
            page.page_number: page.ocr_result.mean_confidence
            for page in extracted_doc.pages
            if page.ocr_result and page.ocr_result.success and page.ocr_result.mean_confidence is not None
        }
        chunker = self._chunker or SemanticChunker(tokenizer_name=cfg.EMBEDDING_MODEL_NAME)
        chunks = chunker.chunk_document(
            structured_doc,
            policy_id=policy.external_policy_key,
            policy_version=f"v{claimed_doc.version_number}",
            ocr_confidence_by_page=ocr_conf_map or None,
        )
        stage_timings["chunking_ms"] = round((time.perf_counter() - t_chk0) * 1000.0, 3)

        if not chunks:
            raise EmptyDocumentError("No extractable clauses or chunks found in document.")

        # 7. Generate 1024-dim embeddings separately (DB transaction CLOSED during GPU/CPU inference)
        repo.transition_ingestion_job_stage(
            job_id=payload.job_id,
            document_id=payload.document_id,
            scope=derived_scope,
            status="processing",
            stage=IngestionStage.EMBEDDING.value,
            total_chunks=len(chunks),
        )
        t_emb0 = time.perf_counter()
        embedder = self._get_embedder()
        embedded_chunks = embedder.embed_chunks(chunks)
        model_revision = _resolve_model_revision(embedder)
        stage_timings["embedding_ms"] = round((time.perf_counter() - t_emb0) * 1000.0, 3)

        # 7b. Extract structured policy attributes cleanly (idempotent if already extracted on retry)
        repo.transition_ingestion_job_stage(
            job_id=payload.job_id,
            document_id=payload.document_id,
            scope=derived_scope,
            status="processing",
            stage=IngestionStage.EXTRACTING_ATTRIBUTES.value,
            total_chunks=len(embedded_chunks),
        )
        t_attr0 = time.perf_counter()
        existing_attr_row = (
            repo.get_policy_attributes(policy_id=payload.policy_id, document_id=payload.document_id)
            if hasattr(repo, "get_policy_attributes")
            else None
        )
        if existing_attr_row and existing_attr_row.get("extraction_status") in (
            "completed",
            "partial",
            "insufficient_data",
        ):
            extracted_attributes = existing_attr_row
        else:
            extractor = self._attribute_extractor or PolicyAttributeExtractor(app_settings=cfg)
            owns_extractor = self._attribute_extractor is None
            try:
                extracted_attributes = extractor.extract_from_chunks(
                    embedded_chunks,
                    policy_type=policy.policy_type,
                )
            finally:
                if owns_extractor:
                    extractor.close()
        stage_timings["attribute_extraction_ms"] = round((time.perf_counter() - t_attr0) * 1000.0, 3)
        attr_status = (
            extracted_attributes.get("extraction_status")
            if isinstance(extracted_attributes, dict)
            else getattr(extracted_attributes, "extraction_status", None)
        )

        # 8. Persist chunks + embeddings + attributes + document completion atomically
        repo.transition_ingestion_job_stage(
            job_id=payload.job_id,
            document_id=payload.document_id,
            scope=derived_scope,
            status="processing",
            stage=IngestionStage.PERSISTING.value,
            total_chunks=len(embedded_chunks),
        )
        t_per0 = time.perf_counter()
        total_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        doc_meta_update = {
            "title": structured_doc.document_title or extracted_doc.metadata.title,
            "file_size_bytes": file_size,
            "extraction_warnings": structured_doc.extraction_warnings,
            "ocr_pages_count": ocr_proxy.ocr_pages_count,
            "attribute_extraction_status": attr_status,
        }
        job_meta_update = {
            "request_id": payload.request_id,
            "stage_timings_ms": stage_timings,
            "processing_ms": total_ms,
            "embedding_model_name": embedder.model_name,
            "embedding_model_revision": model_revision,
            "ocr_pages_count": ocr_proxy.ocr_pages_count,
            "chunk_count": len(embedded_chunks),
            "attribute_extraction_status": attr_status,
        }

        complete_kwargs: Dict[str, object] = {
            "job_id": payload.job_id,
            "policy_id": payload.policy_id,
            "document_id": payload.document_id,
            "embedded_chunks": embedded_chunks,
            "model_name": embedder.model_name,
            "model_revision": model_revision,
            "page_count": extracted_doc.metadata.page_count,
            "document_metadata": doc_meta_update,
            "job_metadata": job_meta_update,
            "scope": derived_scope,
            "is_normalized": embedder.normalize,
            "activate_document": True,
        }
        if "extracted_attributes" in inspect.signature(repo.complete_document_ingestion).parameters:
            complete_kwargs["extracted_attributes"] = extracted_attributes

        persisted_chunks, persisted_embs, completed_job = repo.complete_document_ingestion(
            **complete_kwargs
        )
        stage_timings["persisting_ms"] = round((time.perf_counter() - t_per0) * 1000.0, 3)

        final_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        logger.info(
            "Document ingestion completed cleanly: job_id=%s policy_id=%s document_id=%s chunks=%d pages=%d attr_status=%s duration_ms=%.2f",
            payload.job_id,
            payload.policy_id,
            payload.document_id,
            len(persisted_chunks),
            extracted_doc.metadata.page_count,
            attr_status,
            final_ms,
        )
        return (
            IngestionExecutionResult(
                job_id=str(completed_job.id),
                policy_id=str(payload.policy_id),
                document_id=str(payload.document_id),
                status="completed",
                stage=IngestionStage.COMPLETED.value,
                page_count=extracted_doc.metadata.page_count,
                ocr_pages_count=ocr_proxy.ocr_pages_count,
                chunk_count=len(persisted_chunks),
                embedding_dim=persisted_embs[0].embedding_dim,
                model_name=embedder.model_name,
                model_revision=model_revision,
                attribute_extraction_status=attr_status,
                processing_ms=final_ms,
                stage_timings_ms=stage_timings,
            ),
            derived_scope,
            payload,
        )
