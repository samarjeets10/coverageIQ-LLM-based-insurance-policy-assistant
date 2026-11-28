from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from ai_service.schemas.structured_output import PolicyChunk

PolicyTypeLiteral = Literal["health", "car", "life", "term", "travel", "motor", "pension", "general"]
VisibilityLiteral = Literal["private", "catalog"]
PolicyStatusLiteral = Literal["draft", "active", "inactive", "archived"]
DocumentIngestionStatusLiteral = Literal["pending", "queued", "processing", "completed", "failed", "archived"]
JobStatusLiteral = Literal["queued", "processing", "completed", "failed", "cancelled"]


class PolicyRecord(BaseModel):
    id: UUID
    external_policy_key: str
    title: str
    insurer_name: Optional[str] = None
    policy_type: PolicyTypeLiteral = "general"
    visibility: VisibilityLiteral = "private"
    owner_id: str
    owner_scope: str
    status: PolicyStatusLiteral = "draft"
    active_version_id: Optional[UUID] = None
    archived_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class PolicyDocumentRecord(BaseModel):
    id: UUID
    policy_id: UUID
    version_number: int
    version_label: Optional[str] = None
    file_name: str
    storage_path: str
    mime_type: str = "application/pdf"
    file_size_bytes: int
    file_hash_sha256: str
    page_count: int = 0
    is_encrypted: bool = False
    ingestion_status: DocumentIngestionStatusLiteral = "pending"
    is_archived: bool = False
    archived_at: Optional[datetime] = None
    document_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class PersistedChunkRecord(BaseModel):
    id: UUID
    policy_id: UUID
    document_id: UUID
    chunk_key: str
    chunk_index: int
    section_id: Optional[str] = None
    clause_ref: str
    category: str
    page_number: int
    page_start: int
    page_end: int
    content_text: str
    source_block_ids: List[str] = Field(default_factory=list)
    table_id: Optional[str] = None
    extraction_method: str = "DIGITAL"
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class ChunkEmbeddingRecord(BaseModel):
    id: UUID
    chunk_id: UUID
    policy_id: UUID
    document_id: UUID
    model_name: str
    model_revision: str
    embedding_dim: int = 1024
    is_normalized: bool = True
    created_at: datetime
    updated_at: datetime


class IngestionJobRecord(BaseModel):
    id: UUID
    policy_id: Optional[UUID] = None
    document_id: Optional[UUID] = None
    external_policy_key: str
    document_version_snapshot: Optional[int] = None
    file_hash_sha256: Optional[str] = None
    owner_scope: str
    celery_task_id: Optional[str] = None
    status: JobStatusLiteral = "queued"
    current_stage: Optional[str] = None
    total_chunks: int = 0
    processed_chunks: int = 0
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    attempt_count: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class VectorSearchHit(BaseModel):
    chunk_uuid: UUID
    document_id: UUID
    chunk: PolicyChunk
    cosine_similarity: float
    model_name: str
    model_revision: str


class SparseSearchHit(BaseModel):
    """
    PostgreSQL full-text search hit (`tsvector` + GIN + `ts_rank_cd`).
    Note: PostgreSQL FTS is NOT true Okapi BM25.
    """

    chunk_uuid: UUID
    document_id: UUID
    chunk: PolicyChunk
    lexical_rank_score: float


class AuthorizedPolicyTarget(BaseModel):
    """Authorized target policy and document version resolved prior to chunk/vector retrieval."""

    policy_id: UUID
    external_policy_key: str
    document_id: Optional[UUID] = None
    owner_scope: str
    visibility: VisibilityLiteral
    status: PolicyStatusLiteral
