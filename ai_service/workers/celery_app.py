import logging
import os
import threading
from typing import Optional

from celery import Celery
from celery.signals import worker_process_shutdown, worker_shutdown
from kombu import Queue

from ai_service.config import Settings, settings
from ai_service.embeddings import BaseEmbedder, create_embedder
from ai_service.ingestion.ocr_engine import BaseOCREngine, PaddleOCREngine
from ai_service.persistence import PostgresPoolManager, close_db_pool, init_db_pool, redact_dsn
from ai_service.storage import DocumentStorageBackend, LocalFilesystemDocumentStorage

logger = logging.getLogger("coverageiq.workers.celery")


def resolve_worker_settings(explicit: Optional[Settings] = None) -> Settings:
    if explicit is not None:
        return explicit
    return Settings()


def create_celery_app(app_settings: Optional[Settings] = None) -> Celery:
    """Create and configure the centralized CoverageIQ Celery application."""
    cfg = resolve_worker_settings(app_settings)

    app = Celery(
        "coverageiq_worker",
        broker=cfg.CELERY_BROKER_URL,
        backend=cfg.CELERY_RESULT_BACKEND,
        include=["ai_service.workers.tasks"],
    )

    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        worker_concurrency=cfg.CELERY_WORKER_CONCURRENCY,
        task_soft_time_limit=cfg.CELERY_TASK_SOFT_TIME_LIMIT,
        task_time_limit=cfg.CELERY_TASK_TIME_LIMIT,
        broker_connection_retry_on_startup=True,
        result_expires=3600,
        task_default_queue=cfg.CELERY_INGESTION_QUEUE,
        task_queues=(
            Queue(cfg.CELERY_INGESTION_QUEUE),
            Queue(cfg.CELERY_OCR_QUEUE),
        ),
        task_routes={
            "coverageiq.ingestion.ingest_policy_document": {"queue": cfg.CELERY_INGESTION_QUEUE},
            "coverageiq.ingestion.ocr_policy_page": {"queue": cfg.CELERY_OCR_QUEUE},
        },
        task_always_eager=False,
    )
    return app


class WorkerResourceManager:
    """
    Process-wide lazy resource holder for Celery workers.

    - Reuses one PostgresPoolManager across tasks in the worker process.
    - Reuses one lazy BgeM3Embedder instance across tasks so a 4 GB VRAM GPU
      only loads BAAI/bge-m3 once when worker concurrency is 1.
    - Releases DB pool and CUDA cache cleanly on worker shutdown.
    """

    def __init__(self, app_settings: Optional[Settings] = None):
        self._explicit_settings = app_settings
        self._lock = threading.Lock()
        self._pool: Optional[PostgresPoolManager] = None
        self._embedder: Optional[BaseEmbedder] = None
        self._embedder_signature: Optional[tuple] = None
        self._storage: Optional[DocumentStorageBackend] = None
        self._storage_root: Optional[str] = None
        self._ocr_engine: Optional[BaseOCREngine] = None

    def get_settings(self) -> Settings:
        return resolve_worker_settings(self._explicit_settings)

    def get_pool(self) -> PostgresPoolManager:
        cfg = self.get_settings()
        with self._lock:
            if self._pool is not None and self._pool.is_open and self._pool.redacted_dsn == redact_dsn(cfg.DATABASE_URL):
                return self._pool
            if self._pool is not None:
                self._pool.close()
                self._pool = None
            mgr = init_db_pool(verify_schema=True, app_settings=cfg)
            self._pool = mgr
            return mgr

    def get_embedder(self) -> BaseEmbedder:
        cfg = self.get_settings()
        eff_batch = cfg.CELERY_INGESTION_EMBEDDING_BATCH_SIZE
        sig = (
            cfg.EMBEDDING_PROVIDER,
            cfg.EMBEDDING_MODEL_NAME,
            cfg.EMBEDDING_DIM,
            eff_batch,
            cfg.EMBEDDING_DEVICE,
            cfg.EMBEDDING_NORMALIZE,
            cfg.EMBEDDING_LOCAL_FILES_ONLY,
        )
        with self._lock:
            if self._embedder is not None and self._embedder_signature == sig:
                return self._embedder

            worker_cfg = cfg.model_copy(update={"EMBEDDING_BATCH_SIZE": eff_batch})
            self._embedder = create_embedder(provider=cfg.EMBEDDING_PROVIDER, app_settings=worker_cfg)
            self._embedder_signature = sig
            return self._embedder

    def get_storage(self) -> DocumentStorageBackend:
        cfg = self.get_settings()
        with self._lock:
            if self._storage is not None and self._storage_root == cfg.POLICY_UPLOAD_ROOT:
                return self._storage
            self._storage = LocalFilesystemDocumentStorage(app_settings=cfg)
            self._storage_root = cfg.POLICY_UPLOAD_ROOT
            return self._storage

    def get_ocr_engine(self) -> BaseOCREngine:
        with self._lock:
            if self._ocr_engine is None:
                self._ocr_engine = PaddleOCREngine()
            return self._ocr_engine

    def shutdown(self) -> None:
        with self._lock:
            logger.info("Shutting down CoverageIQ worker resources cleanly.")
            if self._pool is not None:
                try:
                    self._pool.close()
                finally:
                    self._pool = None
            else:
                close_db_pool()
            if self._embedder is not None and hasattr(self._embedder, "close"):
                try:
                    self._embedder.close()
                except Exception:
                    pass
            self._embedder = None
            self._embedder_signature = None
            if self._ocr_engine is not None and hasattr(self._ocr_engine, "close"):
                try:
                    self._ocr_engine.close()
                except Exception:
                    pass
            self._ocr_engine = None
            self._storage = None
            self._storage_root = None

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


celery_app = create_celery_app(settings)

_resource_lock = threading.Lock()
_worker_resources: Optional[WorkerResourceManager] = None


def get_worker_resources() -> WorkerResourceManager:
    global _worker_resources
    with _resource_lock:
        if _worker_resources is None:
            _worker_resources = WorkerResourceManager()
        return _worker_resources


def shutdown_worker_resources() -> None:
    global _worker_resources
    with _resource_lock:
        mgr = _worker_resources
        _worker_resources = None
    if mgr is not None:
        mgr.shutdown()
    else:
        close_db_pool()


@worker_process_shutdown.connect
def _on_worker_process_shutdown(**_kwargs) -> None:
    shutdown_worker_resources()


@worker_shutdown.connect
def _on_worker_shutdown(**_kwargs) -> None:
    shutdown_worker_resources()
