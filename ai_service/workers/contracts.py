from enum import Enum
import re
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_FORBIDDEN_SCOPE_RE = re.compile(r"[*%;\s\x00]")


class IngestionStage(str, Enum):
    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"
    EXTRACTING = "EXTRACTING"
    OCR = "OCR"
    STRUCTURING = "STRUCTURING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    EXTRACTING_ATTRIBUTES = "EXTRACTING_ATTRIBUTES"
    PERSISTING = "PERSISTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RETRY_SCHEDULED = "RETRY_SCHEDULED"


class IngestionTaskPayload(BaseModel):
    """
    Internal Node.js -> Celery JSON enqueue contract.

    Carries database identifiers and controlled storage references ONLY.
    Forbids raw PDF bytes, embeddings, or extracted text blobs inside Redis messages.
    """

    model_config = ConfigDict(extra="forbid")

    job_id: UUID = Field(..., description="Pre-created PostgreSQL ingestion_jobs.id")
    policy_id: UUID = Field(..., description="Target policies.id UUID")
    document_id: UUID = Field(..., description="Target policy_documents.id UUID")
    owner_scope: str = Field(..., min_length=1, description="Claimed tenant owner_scope verified against DB records")
    storage_ref: Optional[str] = Field(
        default=None,
        description="Optional relative storage key validated against persisted document.storage_path and POLICY_UPLOAD_ROOT",
    )
    expected_sha256: Optional[str] = Field(
        default=None,
        description="Optional 64-hex SHA-256 checksum cross-checked against document record and PDF bytes",
    )
    request_id: Optional[str] = Field(default=None, max_length=128)

    @field_validator("owner_scope")
    @classmethod
    def validate_owner_scope(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned or _FORBIDDEN_SCOPE_RE.search(cleaned):
            raise ValueError("owner_scope must be non-empty without wildcards or control characters.")
        return cleaned

    @field_validator("expected_sha256")
    @classmethod
    def validate_expected_sha256(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        cleaned = v.strip().lower()
        if not _SHA256_RE.fullmatch(cleaned):
            raise ValueError("expected_sha256 must be a valid 64-character hexadecimal SHA-256 digest.")
        return cleaned


class IngestionExecutionResult(BaseModel):
    """Sanitized JSON-serializable task result written to the Celery Redis result backend."""

    job_id: str
    policy_id: str
    document_id: str
    status: str
    stage: str
    duplicate_skipped: bool = False
    already_completed: bool = False
    page_count: int = 0
    ocr_pages_count: int = 0
    chunk_count: int = 0
    embedding_dim: int = 1024
    model_name: Optional[str] = None
    model_revision: Optional[str] = None
    attribute_extraction_status: Optional[str] = None
    processing_ms: float = 0.0
    stage_timings_ms: Dict[str, float] = Field(default_factory=dict)
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    def to_json_dict(self) -> Dict[str, Any]:
        return self.model_dump()
