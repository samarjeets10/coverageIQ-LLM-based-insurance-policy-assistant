from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class InternalEnqueueRequest(BaseModel):
    """
    Authenticated internal Node.js -> FastAPI enqueue payload.

    Accepts persisted database identifiers ONLY. Arbitrary owner_scope overrides,
    storage paths, raw PDF bytes, extracted text, or embeddings are forbidden.
    """

    model_config = ConfigDict(extra="forbid")

    job_id: UUID = Field(..., description="Persisted PostgreSQL ingestion_jobs.id")
    policy_id: UUID = Field(..., description="Persisted PostgreSQL policies.id")
    document_id: UUID = Field(..., description="Persisted PostgreSQL policy_documents.id")
    request_id: Optional[str] = Field(default=None, max_length=128)


class InternalEnqueueResponse(BaseModel):
    """Sanitized response returned to the Node.js gateway after idempotent Celery dispatch."""

    job_id: UUID
    policy_id: UUID
    document_id: UUID
    request_id: Optional[str] = None
    enqueued: bool
    already_enqueued: bool = False
    already_completed: bool = False
    status: str
    stage: Optional[str] = None
    task_id: Optional[str] = None
    queue: Optional[str] = None
