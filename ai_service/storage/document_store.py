from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union

from pydantic import BaseModel

from ai_service.config import Settings, settings


class DocumentStorageError(Exception):
    """Base error for CoverageIQ document storage access failures."""


class DocumentNotFoundError(DocumentStorageError):
    """Raised when a referenced document object does not exist in storage."""


class DocumentPathSecurityError(DocumentStorageError):
    """Raised when a storage reference attempts path traversal or escapes storage root."""


class StoredDocumentHandle(BaseModel):
    """Handle to a verified local readable copy of a stored policy document."""

    storage_ref: str
    local_path: Path
    file_size_bytes: int


class DocumentStorageBackend(ABC):
    """
    Storage abstraction keeping raw PDF bytes out of Redis and PostgreSQL.

    Local filesystem storage is used for single-node dev/test environments under
    POLICY_UPLOAD_ROOT. In multi-node deployments, implement this ABC for shared
    object storage (e.g., S3/MinIO/GCS) so Celery workers fetch PDFs to a scoped
    temporary file during execution without requiring a shared host filesystem.
    """

    @abstractmethod
    @contextmanager
    def resolve_for_reading(self, storage_ref: str) -> Iterator[StoredDocumentHandle]:
        """Yield a verified readable handle to the PDF artifact."""

    @abstractmethod
    def exists(self, storage_ref: str) -> bool:
        """Return True if the storage reference resolves to an existing object inside root."""


class LocalFilesystemDocumentStorage(DocumentStorageBackend):
    """Root-jailed local document store validating all keys against POLICY_UPLOAD_ROOT."""

    def __init__(self, root_dir: Optional[Union[str, Path]] = None, *, app_settings: Optional[Settings] = None):
        cfg = app_settings or settings
        raw_root = root_dir if root_dir is not None else cfg.POLICY_UPLOAD_ROOT
        if not raw_root or not str(raw_root).strip():
            raise DocumentPathSecurityError("POLICY_UPLOAD_ROOT cannot be empty.")
        self.root_dir = Path(raw_root).resolve()

    def resolve_safe_path(self, storage_ref: str) -> Path:
        if not isinstance(storage_ref, str) or not storage_ref.strip():
            raise DocumentPathSecurityError("Storage reference must be a non-empty string.")
        cleaned = storage_ref.strip()
        if "\x00" in cleaned:
            raise DocumentPathSecurityError("Null byte detected in storage reference.")

        candidate = Path(cleaned)
        # Disallow traversal segments outright even before resolve()
        if ".." in candidate.parts:
            raise DocumentPathSecurityError("Directory traversal segments ('..') are forbidden in storage references.")

        target = candidate if candidate.is_absolute() else (self.root_dir / candidate)
        resolved = target.resolve()
        try:
            resolved.relative_to(self.root_dir)
        except ValueError as exc:
            raise DocumentPathSecurityError(
                "Resolved storage path escapes the configured POLICY_UPLOAD_ROOT."
            ) from exc

        return resolved

    @contextmanager
    def resolve_for_reading(self, storage_ref: str) -> Iterator[StoredDocumentHandle]:
        resolved = self.resolve_safe_path(storage_ref)
        if not resolved.exists() or not resolved.is_file():
            raise DocumentNotFoundError("Stored document artifact does not exist or is not a regular file.")

        try:
            rel_key = str(resolved.relative_to(self.root_dir))
        except ValueError as exc:
            raise DocumentPathSecurityError("Resolved path escaped storage root.") from exc

        yield StoredDocumentHandle(
            storage_ref=rel_key,
            local_path=resolved,
            file_size_bytes=resolved.stat().st_size,
        )

    def put_bytes(self, relative_key: str, data: bytes) -> Path:
        """Helper to store bytes inside the jailed root (for tests and local staging)."""
        if Path(relative_key).is_absolute():
            raise DocumentPathSecurityError("put_bytes requires a relative storage key inside POLICY_UPLOAD_ROOT.")
        resolved = self.resolve_safe_path(relative_key)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_bytes(data)
        return resolved

    def exists(self, storage_ref: str) -> bool:
        try:
            resolved = self.resolve_safe_path(storage_ref)
        except DocumentPathSecurityError:
            return False
        return resolved.is_file()


def get_document_storage(*, app_settings: Optional[Settings] = None) -> LocalFilesystemDocumentStorage:
    return LocalFilesystemDocumentStorage(app_settings=app_settings)
