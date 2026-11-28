from datetime import datetime, timezone
import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import UUID

from pgvector import Vector
from psycopg.types.json import Jsonb

from ai_service.persistence.exceptions import (
    DatabaseAuthorizationError,
    DatabaseConstraintError,
    DatabaseNotFoundError,
    DatabaseValidationError,
)
from ai_service.persistence.models import (
    AuthorizedPolicyTarget,
    ChunkEmbeddingRecord,
    DocumentIngestionStatusLiteral,
    IngestionJobRecord,
    JobStatusLiteral,
    PersistedChunkRecord,
    PolicyDocumentRecord,
    PolicyRecord,
    PolicyStatusLiteral,
    PolicyTypeLiteral,
    SparseSearchHit,
    VectorSearchHit,
    VisibilityLiteral,
)
from ai_service.persistence.pool import PostgresPoolManager
from ai_service.persistence.security import (
    AuthorizedScope,
    build_policy_scope_predicate,
    validate_authorized_scope,
)
from ai_service.schemas.structured_output import ChunkCategory, PolicyChunk

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_EMBEDDING_DIM = 1024
_VALID_EXTRACTION_METHODS = ("DIGITAL", "OCR", "HYBRID")


def _validate_sha256(value: Optional[str], *, allow_none: bool = False) -> Optional[str]:
    if value is None:
        if allow_none:
            return None
        raise DatabaseValidationError("file_hash_sha256 must not be None.")
    cleaned = value.strip().lower()
    if not _SHA256_RE.fullmatch(cleaned):
        raise DatabaseValidationError("file_hash_sha256 must be a valid 64-character lowercase hex SHA-256 digest.")
    return cleaned


def _validate_embedding_vector(vector: Sequence[float], *, label: str = "embedding") -> List[float]:
    if isinstance(vector, (str, bytes)):
        raise DatabaseValidationError(f"{label} must be a numeric sequence of length {_EXPECTED_EMBEDDING_DIM}.")
    try:
        values = [float(x) for x in vector]
    except (TypeError, ValueError) as exc:
        raise DatabaseValidationError(f"{label} contains non-numeric values.") from exc

    if len(values) != _EXPECTED_EMBEDDING_DIM:
        raise DatabaseValidationError(
            f"{label} must have exactly {_EXPECTED_EMBEDDING_DIM} dimensions, got {len(values)}."
        )
    if not all(math.isfinite(x) for x in values):
        raise DatabaseValidationError(f"{label} contains non-finite (NaN or Inf) values.")
    return values


def _row_to_policy_chunk(row: Dict[str, Any]) -> PolicyChunk:
    raw_cat = row["category"]
    try:
        category = ChunkCategory(raw_cat)
    except ValueError:
        category = ChunkCategory.GENERAL

    meta = dict(row.get("metadata") or {})
    if row.get("id") is not None:
        meta.setdefault("db_chunk_uuid", str(row["id"]))
    if row.get("document_id") is not None:
        meta.setdefault("document_id", str(row["document_id"]))
    if row.get("policy_id") is not None:
        meta.setdefault("policy_uuid", str(row["policy_id"]))

    return PolicyChunk(
        chunk_id=row["chunk_key"],
        policy_id=row.get("external_policy_key") or str(row["policy_id"]),
        clause_ref=row["clause_ref"],
        page_number=int(row["page_number"]),
        category=category,
        content_text=row["content_text"],
        embedding=None,
        page_start=row.get("page_start"),
        page_end=row.get("page_end"),
        source_block_ids=list(row.get("source_block_ids") or []),
        table_id=row.get("table_id"),
        metadata=meta,
    )


class PolicyKnowledgeRepository:
    """
    Parameterized PostgreSQL 16 + pgvector repository for CoverageIQ policies,
    versioned documents, semantic chunks, 1024-dim BGE-M3 embeddings, and ingestion jobs.

    Security & retrieval guarantees:
    - All SQL statements use parameterized query bindings (zero string interpolation of user input).
    - Every query enforces a gateway-verified AuthorizedScope (`owner_scope` tenant boundary).
    - Chunk IDs are scoped per document version `(document_id, chunk_key)` to prevent collisions.
    - Sparse SQL search (`tsvector` + GIN + `ts_rank_cd`) is PostgreSQL full-text search, NOT true BM25.
    """

    def __init__(self, pool_manager: PostgresPoolManager):
        self.db = pool_manager

    # -------------------------------------------------------------------------
    # Policies
    # -------------------------------------------------------------------------
    def upsert_policy(
        self,
        *,
        external_policy_key: str,
        title: str,
        owner_id: str,
        scope: AuthorizedScope,
        insurer_name: Optional[str] = None,
        policy_type: PolicyTypeLiteral = "general",
        visibility: VisibilityLiteral = "private",
        status: PolicyStatusLiteral = "draft",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PolicyRecord:
        auth = validate_authorized_scope(scope)
        if not external_policy_key.strip() or not title.strip() or not owner_id.strip():
            raise DatabaseValidationError("external_policy_key, title, and owner_id must be non-empty.")

        with self.db.transaction() as conn:
            existing = conn.execute(
                "SELECT id, owner_scope FROM policies WHERE external_policy_key = %(key)s FOR UPDATE",
                {"key": external_policy_key.strip()},
            ).fetchone()
            if existing is not None and existing["owner_scope"] != auth.owner_scope:
                raise DatabaseAuthorizationError(
                    "Policy external_policy_key belongs to a different owner_scope."
                )

            row = conn.execute(
                """
                INSERT INTO policies (
                    external_policy_key, title, insurer_name, policy_type,
                    visibility, owner_id, owner_scope, status, metadata
                )
                VALUES (
                    %(external_policy_key)s, %(title)s, %(insurer_name)s, %(policy_type)s,
                    %(visibility)s, %(owner_id)s, %(owner_scope)s, %(status)s, %(metadata)s
                )
                ON CONFLICT (external_policy_key) DO UPDATE SET
                    title = EXCLUDED.title,
                    insurer_name = COALESCE(EXCLUDED.insurer_name, policies.insurer_name),
                    policy_type = EXCLUDED.policy_type,
                    visibility = EXCLUDED.visibility,
                    status = EXCLUDED.status,
                    metadata = policies.metadata || EXCLUDED.metadata,
                    updated_at = NOW()
                RETURNING *
                """,
                {
                    "external_policy_key": external_policy_key.strip(),
                    "title": title.strip(),
                    "insurer_name": insurer_name.strip() if insurer_name else None,
                    "policy_type": policy_type,
                    "visibility": visibility,
                    "owner_id": owner_id.strip(),
                    "owner_scope": auth.owner_scope,
                    "status": status,
                    "metadata": Jsonb(dict(metadata or {})),
                },
            ).fetchone()
            assert row is not None
            return PolicyRecord.model_validate(row)

    def get_policy(self, policy_id: UUID, *, scope: AuthorizedScope) -> Optional[PolicyRecord]:
        auth = validate_authorized_scope(scope)
        scope_sql, scope_params = build_policy_scope_predicate(auth, alias="p", for_write=False)
        with self.db.connection() as conn:
            row = conn.execute(
                f"SELECT p.* FROM policies p WHERE p.id = %(policy_id)s AND {scope_sql}",
                {"policy_id": policy_id, **scope_params},
            ).fetchone()
            return PolicyRecord.model_validate(row) if row is not None else None

    def get_policy_by_external_key(self, external_policy_key: str, *, scope: AuthorizedScope) -> Optional[PolicyRecord]:
        auth = validate_authorized_scope(scope)
        scope_sql, scope_params = build_policy_scope_predicate(auth, alias="p", for_write=False)
        with self.db.connection() as conn:
            row = conn.execute(
                f"SELECT p.* FROM policies p WHERE p.external_policy_key = %(key)s AND {scope_sql}",
                {"key": external_policy_key.strip(), **scope_params},
            ).fetchone()
            return PolicyRecord.model_validate(row) if row is not None else None

    def set_active_document_version(
        self,
        *,
        policy_id: UUID,
        document_id: UUID,
        scope: AuthorizedScope,
    ) -> PolicyRecord:
        auth = validate_authorized_scope(scope)
        with self.db.transaction() as conn:
            self._assert_policy_writable(conn, policy_id, auth)
            row = conn.execute(
                """
                UPDATE policies p
                SET active_version_id = %(document_id)s,
                    updated_at = NOW()
                WHERE p.id = %(policy_id)s
                RETURNING p.*
                """,
                {"policy_id": policy_id, "document_id": document_id},
            ).fetchone()
            if row is None:
                raise DatabaseNotFoundError("Policy not found within authorized scope.")
            return PolicyRecord.model_validate(row)

    # -------------------------------------------------------------------------
    # Policy Documents & Versioning
    # -------------------------------------------------------------------------
    def _assert_policy_writable(self, conn: Any, policy_id: UUID, scope: AuthorizedScope) -> None:
        scope_sql, scope_params = build_policy_scope_predicate(scope, alias="p", for_write=True)
        row = conn.execute(
            f"SELECT p.id FROM policies p WHERE p.id = %(policy_id)s AND {scope_sql}",
            {"policy_id": policy_id, **scope_params},
        ).fetchone()
        if row is None:
            raise DatabaseAuthorizationError(
                "Target policy does not exist or is outside caller write authorization scope."
            )

    def create_document(
        self,
        *,
        policy_id: UUID,
        version_number: int,
        file_name: str,
        storage_path: str,
        file_size_bytes: int,
        file_hash_sha256: str,
        scope: AuthorizedScope,
        version_label: Optional[str] = None,
        mime_type: str = "application/pdf",
        page_count: int = 0,
        is_encrypted: bool = False,
        ingestion_status: DocumentIngestionStatusLiteral = "pending",
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> PolicyDocumentRecord:
        auth = validate_authorized_scope(scope)
        clean_hash = _validate_sha256(file_hash_sha256)
        if version_number < 1 or file_size_bytes < 0 or page_count < 0:
            raise DatabaseValidationError("Invalid version_number, file_size_bytes, or page_count.")

        with self.db.transaction() as conn:
            self._assert_policy_writable(conn, policy_id, auth)
            row = conn.execute(
                """
                INSERT INTO policy_documents (
                    policy_id, version_number, version_label, file_name, storage_path,
                    mime_type, file_size_bytes, file_hash_sha256, page_count,
                    is_encrypted, ingestion_status, document_metadata
                )
                VALUES (
                    %(policy_id)s, %(version_number)s, %(version_label)s, %(file_name)s, %(storage_path)s,
                    %(mime_type)s, %(file_size_bytes)s, %(file_hash_sha256)s, %(page_count)s,
                    %(is_encrypted)s, %(ingestion_status)s, %(document_metadata)s
                )
                RETURNING *
                """,
                {
                    "policy_id": policy_id,
                    "version_number": version_number,
                    "version_label": version_label,
                    "file_name": file_name.strip(),
                    "storage_path": storage_path.strip(),
                    "mime_type": mime_type.strip(),
                    "file_size_bytes": file_size_bytes,
                    "file_hash_sha256": clean_hash,
                    "page_count": page_count,
                    "is_encrypted": is_encrypted,
                    "ingestion_status": ingestion_status,
                    "document_metadata": Jsonb(dict(document_metadata or {})),
                },
            ).fetchone()
            assert row is not None
            return PolicyDocumentRecord.model_validate(row)

    def get_document(self, document_id: UUID, *, scope: AuthorizedScope) -> Optional[PolicyDocumentRecord]:
        auth = validate_authorized_scope(scope)
        scope_sql, scope_params = build_policy_scope_predicate(auth, alias="p", for_write=False)
        with self.db.connection() as conn:
            row = conn.execute(
                f"""
                SELECT d.*
                FROM policy_documents d
                JOIN policies p ON p.id = d.policy_id
                WHERE d.id = %(document_id)s
                  AND {scope_sql}
                """,
                {"document_id": document_id, **scope_params},
            ).fetchone()
            return PolicyDocumentRecord.model_validate(row) if row is not None else None

    def _lock_owned_document(self, conn: Any, document_id: UUID, scope: AuthorizedScope) -> Optional[str]:
        row = conn.execute(
            """
            SELECT d.id, p.owner_scope
            FROM policy_documents d
            JOIN policies p ON p.id = d.policy_id
            WHERE d.id = %(document_id)s
            FOR UPDATE OF d
            """,
            {"document_id": document_id},
        ).fetchone()
        if row is None:
            return None
        if row["owner_scope"] != scope.owner_scope:
            raise DatabaseAuthorizationError(
                "Target policy document belongs to a different owner_scope."
            )
        return row["owner_scope"]

    def archive_document(self, document_id: UUID, *, scope: AuthorizedScope) -> PolicyDocumentRecord:
        auth = validate_authorized_scope(scope)
        with self.db.transaction() as conn:
            owner = self._lock_owned_document(conn, document_id, auth)
            if owner is None:
                raise DatabaseNotFoundError("Document not found.")
            row = conn.execute(
                """
                UPDATE policy_documents d
                SET is_archived = TRUE,
                    archived_at = COALESCE(d.archived_at, NOW()),
                    ingestion_status = 'archived',
                    updated_at = NOW()
                WHERE d.id = %(document_id)s
                RETURNING d.*
                """,
                {"document_id": document_id},
            ).fetchone()
            assert row is not None
            return PolicyDocumentRecord.model_validate(row)

    def delete_document(self, document_id: UUID, *, scope: AuthorizedScope) -> bool:
        auth = validate_authorized_scope(scope)
        with self.db.transaction() as conn:
            owner = self._lock_owned_document(conn, document_id, auth)
            if owner is None:
                return False
            row = conn.execute(
                "DELETE FROM policy_documents d WHERE d.id = %(document_id)s RETURNING d.id",
                {"document_id": document_id},
            ).fetchone()
            return row is not None

    # -------------------------------------------------------------------------
    # Policy Chunks & 1024-dim Embeddings (Idempotent, Transactional)
    # -------------------------------------------------------------------------
    def _assert_document_writable(self, conn: Any, policy_id: UUID, document_id: UUID, scope: AuthorizedScope) -> None:
        scope_sql, scope_params = build_policy_scope_predicate(scope, alias="p", for_write=True)
        row = conn.execute(
            f"""
            SELECT d.id
            FROM policy_documents d
            JOIN policies p ON p.id = d.policy_id
            WHERE d.id = %(document_id)s
              AND d.policy_id = %(policy_id)s
              AND {scope_sql}
            """,
            {"document_id": document_id, "policy_id": policy_id, **scope_params},
        ).fetchone()
        if row is None:
            raise DatabaseAuthorizationError(
                "Target policy document does not exist or is outside caller write authorization scope."
            )

    @staticmethod
    def _upsert_chunk_in_tx(
        conn: Any,
        *,
        policy_id: UUID,
        document_id: UUID,
        chunk: PolicyChunk,
        chunk_index: int,
    ) -> PersistedChunkRecord:
        meta = dict(chunk.metadata or {})
        raw_method = str(meta.get("extraction_method") or "DIGITAL").upper()
        extraction_method = raw_method if raw_method in _VALID_EXTRACTION_METHODS else "DIGITAL"
        token_count = meta.get("token_count")
        if not isinstance(token_count, int) or token_count < 0:
            token_count = None
        section_id = meta.get("section_id") if isinstance(meta.get("section_id"), str) else None
        p_start = int(chunk.page_start if chunk.page_start is not None else chunk.page_number)
        p_end = int(chunk.page_end if chunk.page_end is not None else p_start)
        if p_end < p_start:
            p_end = p_start

        row = conn.execute(
            """
            INSERT INTO policy_chunks (
                policy_id, document_id, chunk_key, chunk_index, section_id,
                clause_ref, category, page_number, page_start, page_end,
                content_text, source_block_ids, table_id, extraction_method,
                token_count, metadata
            )
            VALUES (
                %(policy_id)s, %(document_id)s, %(chunk_key)s, %(chunk_index)s, %(section_id)s,
                %(clause_ref)s, %(category)s, %(page_number)s, %(page_start)s, %(page_end)s,
                %(content_text)s, %(source_block_ids)s, %(table_id)s, %(extraction_method)s,
                %(token_count)s, %(metadata)s
            )
            ON CONFLICT (document_id, chunk_key) DO UPDATE SET
                chunk_index = EXCLUDED.chunk_index,
                section_id = EXCLUDED.section_id,
                clause_ref = EXCLUDED.clause_ref,
                category = EXCLUDED.category,
                page_number = EXCLUDED.page_number,
                page_start = EXCLUDED.page_start,
                page_end = EXCLUDED.page_end,
                content_text = EXCLUDED.content_text,
                source_block_ids = EXCLUDED.source_block_ids,
                table_id = EXCLUDED.table_id,
                extraction_method = EXCLUDED.extraction_method,
                token_count = EXCLUDED.token_count,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            RETURNING *
            """,
            {
                "policy_id": policy_id,
                "document_id": document_id,
                "chunk_key": chunk.chunk_id,
                "chunk_index": chunk_index,
                "section_id": section_id,
                "clause_ref": chunk.clause_ref,
                "category": chunk.category.value,
                "page_number": chunk.page_number,
                "page_start": p_start,
                "page_end": p_end,
                "content_text": chunk.content_text,
                "source_block_ids": list(chunk.source_block_ids),
                "table_id": chunk.table_id,
                "extraction_method": extraction_method,
                "token_count": token_count,
                "metadata": Jsonb(meta),
            },
        ).fetchone()
        assert row is not None
        return PersistedChunkRecord.model_validate(row)

    @staticmethod
    def _upsert_embedding_in_tx(
        conn: Any,
        *,
        chunk_uuid: UUID,
        policy_id: UUID,
        document_id: UUID,
        vector_values: Sequence[float],
        model_name: str,
        model_revision: str,
        is_normalized: bool,
    ) -> ChunkEmbeddingRecord:
        validated = _validate_embedding_vector(vector_values, label=f"chunk {chunk_uuid}")
        row = conn.execute(
            """
            INSERT INTO chunk_embeddings (
                chunk_id, policy_id, document_id, model_name, model_revision,
                embedding_dim, is_normalized, embedding
            )
            VALUES (
                %(chunk_id)s, %(policy_id)s, %(document_id)s, %(model_name)s, %(model_revision)s,
                1024, %(is_normalized)s, %(embedding)s
            )
            ON CONFLICT (chunk_id, model_name, model_revision) DO UPDATE SET
                is_normalized = EXCLUDED.is_normalized,
                embedding = EXCLUDED.embedding,
                updated_at = NOW()
            RETURNING id, chunk_id, policy_id, document_id, model_name, model_revision,
                      embedding_dim, is_normalized, created_at, updated_at
            """,
            {
                "chunk_id": chunk_uuid,
                "policy_id": policy_id,
                "document_id": document_id,
                "model_name": model_name.strip(),
                "model_revision": model_revision.strip(),
                "is_normalized": bool(is_normalized),
                "embedding": Vector(validated),
            },
        ).fetchone()
        assert row is not None
        return ChunkEmbeddingRecord.model_validate(row)

    def persist_chunks(
        self,
        *,
        policy_id: UUID,
        document_id: UUID,
        chunks: Sequence[PolicyChunk],
        scope: AuthorizedScope,
    ) -> List[PersistedChunkRecord]:
        auth = validate_authorized_scope(scope)
        if not chunks:
            return []
        with self.db.transaction() as conn:
            self._assert_document_writable(conn, policy_id, document_id, auth)
            return [
                self._upsert_chunk_in_tx(
                    conn,
                    policy_id=policy_id,
                    document_id=document_id,
                    chunk=chunk,
                    chunk_index=idx,
                )
                for idx, chunk in enumerate(chunks)
            ]

    def persist_chunk_embeddings(
        self,
        *,
        policy_id: UUID,
        document_id: UUID,
        embeddings_by_chunk_uuid: Sequence[Tuple[UUID, Sequence[float]]],
        model_name: str,
        model_revision: str,
        scope: AuthorizedScope,
        is_normalized: bool = True,
    ) -> List[ChunkEmbeddingRecord]:
        auth = validate_authorized_scope(scope)
        if not model_name.strip() or not model_revision.strip():
            raise DatabaseValidationError("model_name and model_revision must be non-empty.")
        if not embeddings_by_chunk_uuid:
            return []

        with self.db.transaction() as conn:
            self._assert_document_writable(conn, policy_id, document_id, auth)
            return [
                self._upsert_embedding_in_tx(
                    conn,
                    chunk_uuid=chunk_uuid,
                    policy_id=policy_id,
                    document_id=document_id,
                    vector_values=vec,
                    model_name=model_name,
                    model_revision=model_revision,
                    is_normalized=is_normalized,
                )
                for chunk_uuid, vec in embeddings_by_chunk_uuid
            ]

    def persist_embedded_chunks(
        self,
        *,
        policy_id: UUID,
        document_id: UUID,
        embedded_chunks: Sequence[PolicyChunk],
        model_name: str,
        model_revision: str,
        scope: AuthorizedScope,
        is_normalized: bool = True,
    ) -> Tuple[List[PersistedChunkRecord], List[ChunkEmbeddingRecord]]:
        """Persist chunks and their 1024-dim BGE-M3 embeddings inside a single ACID transaction."""
        auth = validate_authorized_scope(scope)
        if not model_name.strip() or not model_revision.strip():
            raise DatabaseValidationError("model_name and model_revision must be non-empty.")
        if not embedded_chunks:
            return [], []

        with self.db.transaction() as conn:
            self._assert_document_writable(conn, policy_id, document_id, auth)
            persisted_chunks: List[PersistedChunkRecord] = []
            persisted_embeddings: List[ChunkEmbeddingRecord] = []
            for idx, chunk in enumerate(embedded_chunks):
                if chunk.embedding is None:
                    raise DatabaseValidationError(
                        f"Chunk '{chunk.chunk_id}' is missing its 1024-dim embedding."
                    )
                pc = self._upsert_chunk_in_tx(
                    conn,
                    policy_id=policy_id,
                    document_id=document_id,
                    chunk=chunk,
                    chunk_index=idx,
                )
                ce = self._upsert_embedding_in_tx(
                    conn,
                    chunk_uuid=pc.id,
                    policy_id=policy_id,
                    document_id=document_id,
                    vector_values=chunk.embedding,
                    model_name=model_name,
                    model_revision=model_revision,
                    is_normalized=is_normalized,
                )
                persisted_chunks.append(pc)
                persisted_embeddings.append(ce)
            return persisted_chunks, persisted_embeddings

    def _resolve_target_document_sql(
        self,
        document_id: Optional[UUID],
        *,
        policy_alias: str = "p",
        doc_alias: str = "d",
    ) -> Tuple[str, Dict[str, Any]]:
        if document_id is not None:
            return f"{doc_alias}.id = %(document_id)s AND {doc_alias}.is_archived = FALSE", {
                "document_id": document_id
            }
        return (
            f"{doc_alias}.id = {policy_alias}.active_version_id AND {doc_alias}.is_archived = FALSE",
            {},
        )

    def get_policy_chunks(
        self,
        *,
        policy_id: UUID,
        scope: AuthorizedScope,
        document_id: Optional[UUID] = None,
    ) -> List[PolicyChunk]:
        auth = validate_authorized_scope(scope)
        scope_sql, scope_params = build_policy_scope_predicate(auth, alias="p", for_write=False)
        doc_sql, doc_params = self._resolve_target_document_sql(document_id)

        with self.db.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT pc.*, p.external_policy_key
                FROM policy_chunks pc
                JOIN policy_documents d ON d.id = pc.document_id AND d.policy_id = pc.policy_id
                JOIN policies p ON p.id = d.policy_id
                WHERE p.id = %(policy_id)s
                  AND {doc_sql}
                  AND {scope_sql}
                ORDER BY pc.chunk_index ASC, pc.created_at ASC
                """,
                {"policy_id": policy_id, **doc_params, **scope_params},
            ).fetchall()
            return [_row_to_policy_chunk(r) for r in rows]

    # -------------------------------------------------------------------------
    # Scoped Search Infrastructure (pgvector HNSW Cosine + PostgreSQL FTS GIN)
    # -------------------------------------------------------------------------
    def authorize_policy_access(
        self,
        policy_identifier: str | UUID,
        *,
        scope: AuthorizedScope,
        document_id: Optional[UUID] = None,
    ) -> Optional[AuthorizedPolicyTarget]:
        """
        Validate tenant/catalog authorization and target document availability BEFORE retrieval.

        Returns None only when no matching policy record exists at all.
        Raises DatabaseAuthorizationError immediately if the policy exists outside
        the caller's scope, is archived, is an inactive catalog entry, or if an
        explicit document_id belongs to another policy or is archived.
        """
        auth = validate_authorized_scope(scope)
        raw_id = str(policy_identifier).strip() if isinstance(policy_identifier, UUID) else (
            policy_identifier.strip() if isinstance(policy_identifier, str) else ""
        )
        if not raw_id:
            raise DatabaseValidationError("policy_id must be a non-empty string or UUID.")

        parsed_uuid: Optional[UUID] = None
        try:
            parsed_uuid = policy_identifier if isinstance(policy_identifier, UUID) else UUID(raw_id)
        except ValueError:
            parsed_uuid = None

        with self.db.connection() as conn:
            if parsed_uuid is not None:
                row = conn.execute(
                    """
                    SELECT p.id, p.external_policy_key, p.owner_scope, p.visibility,
                           p.status, p.active_version_id
                    FROM policies p
                    WHERE p.id = %(policy_uuid)s OR p.external_policy_key = %(policy_key)s
                    LIMIT 1
                    """,
                    {"policy_uuid": parsed_uuid, "policy_key": raw_id},
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT p.id, p.external_policy_key, p.owner_scope, p.visibility,
                           p.status, p.active_version_id
                    FROM policies p
                    WHERE p.external_policy_key = %(policy_key)s
                    LIMIT 1
                    """,
                    {"policy_key": raw_id},
                ).fetchone()

            if row is None:
                return None

            is_owner = row["owner_scope"] == auth.owner_scope
            is_active_catalog = (
                auth.allow_catalog_read
                and row["visibility"] == "catalog"
                and row["status"] == "active"
            )
            if row["status"] == "archived" or not (is_owner or is_active_catalog):
                raise DatabaseAuthorizationError(
                    "Access denied: target policy is outside caller authorization scope or inactive."
                )

            target_doc_id = document_id if document_id is not None else row["active_version_id"]
            if target_doc_id is not None:
                doc_row = conn.execute(
                    """
                    SELECT d.id
                    FROM policy_documents d
                    WHERE d.id = %(document_id)s
                      AND d.policy_id = %(policy_id)s
                      AND d.is_archived = FALSE
                    """,
                    {"document_id": target_doc_id, "policy_id": row["id"]},
                ).fetchone()
                if doc_row is None:
                    raise DatabaseAuthorizationError(
                        "Target policy document version is archived or does not belong to authorized policy."
                    )

            return AuthorizedPolicyTarget(
                policy_id=row["id"],
                external_policy_key=row["external_policy_key"],
                document_id=target_doc_id,
                owner_scope=row["owner_scope"],
                visibility=row["visibility"],
                status=row["status"],
            )

    def vector_search(
        self,
        *,
        policy_id: UUID,
        query_embedding: Sequence[float],
        scope: AuthorizedScope,
        document_id: Optional[UUID] = None,
        top_n: int = 10,
        model_name: str = "BAAI/bge-m3",
        ef_search: Optional[int] = None,
        iterative_scan: Optional[str] = None,
    ) -> List[VectorSearchHit]:
        """Dense cosine-distance retrieval backed by pgvector HNSW (`vector_cosine_ops`) with mandatory tenant scope."""
        auth = validate_authorized_scope(scope)
        if top_n <= 0:
            return []
        if ef_search is not None and ef_search < 1:
            raise DatabaseValidationError("ef_search must be a positive integer.")
        if iterative_scan is not None and iterative_scan not in ("off", "relaxed_order", "strict_order"):
            raise DatabaseValidationError("iterative_scan must be 'off', 'relaxed_order', or 'strict_order'.")

        q_vec = Vector(_validate_embedding_vector(query_embedding, label="query_embedding"))
        scope_sql, scope_params = build_policy_scope_predicate(auth, alias="p", for_write=False)
        doc_sql, doc_params = self._resolve_target_document_sql(document_id)

        with self.db.transaction() as conn:
            if ef_search is not None:
                conn.execute(
                    "SELECT set_config('hnsw.ef_search', %(ef_search)s, true)",
                    {"ef_search": str(int(ef_search))},
                )
            if iterative_scan is not None and iterative_scan != "off":
                conn.execute(
                    "SELECT set_config('hnsw.iterative_scan', %(iter_scan)s, true)",
                    {"iter_scan": iterative_scan},
                )

            rows = conn.execute(
                f"""
                SELECT
                    pc.*,
                    p.external_policy_key,
                    ce.model_name,
                    ce.model_revision,
                    1.0 - (ce.embedding <=> %(q_vec)s::vector) AS cosine_similarity
                FROM chunk_embeddings ce
                JOIN policy_chunks pc
                    ON pc.id = ce.chunk_id AND pc.document_id = ce.document_id AND pc.policy_id = ce.policy_id
                JOIN policy_documents d
                    ON d.id = pc.document_id AND d.policy_id = pc.policy_id
                JOIN policies p
                    ON p.id = d.policy_id
                WHERE ce.policy_id = %(policy_id)s
                  AND ce.model_name = %(model_name)s
                  AND {doc_sql}
                  AND {scope_sql}
                ORDER BY ce.embedding <=> %(q_vec)s::vector ASC
                LIMIT %(top_n)s
                """,
                {
                    "q_vec": q_vec,
                    "policy_id": policy_id,
                    "model_name": model_name.strip(),
                    "top_n": int(top_n),
                    **doc_params,
                    **scope_params,
                },
            ).fetchall()

        hits: List[VectorSearchHit] = []
        for row in rows:
            sim = float(row["cosine_similarity"])
            chunk = _row_to_policy_chunk(row).model_copy(update={"similarity_score": sim})
            hits.append(
                VectorSearchHit(
                    chunk_uuid=row["id"],
                    document_id=row["document_id"],
                    chunk=chunk,
                    cosine_similarity=sim,
                    model_name=row["model_name"],
                    model_revision=row["model_revision"],
                )
            )
        return hits

    def fts_search(
        self,
        *,
        policy_id: UUID,
        query_text: str,
        scope: AuthorizedScope,
        document_id: Optional[UUID] = None,
        top_n: int = 10,
    ) -> List[SparseSearchHit]:
        """
        Sparse lexical retrieval using PostgreSQL `tsvector` + GIN + `ts_rank_cd`.

        Note: PostgreSQL full-text search is NOT true Okapi BM25; it does not apply
        collection-wide IDF saturation (k1) or average-document-length normalization (b).
        """
        auth = validate_authorized_scope(scope)
        if top_n <= 0 or not isinstance(query_text, str) or not query_text.strip():
            return []
        scope_sql, scope_params = build_policy_scope_predicate(auth, alias="p", for_write=False)
        doc_sql, doc_params = self._resolve_target_document_sql(document_id)

        with self.db.connection() as conn:
            rows = conn.execute(
                f"""
                WITH q AS (
                    SELECT websearch_to_tsquery('english', %(query_text)s) AS tsq
                )
                SELECT
                    pc.*,
                    p.external_policy_key,
                    ts_rank_cd(pc.search_tsv, q.tsq) AS lexical_rank_score
                FROM policy_chunks pc
                CROSS JOIN q
                JOIN policy_documents d
                    ON d.id = pc.document_id AND d.policy_id = pc.policy_id
                JOIN policies p
                    ON p.id = d.policy_id
                WHERE pc.policy_id = %(policy_id)s
                  AND {doc_sql}
                  AND {scope_sql}
                  AND pc.search_tsv @@ q.tsq
                ORDER BY lexical_rank_score DESC, pc.chunk_index ASC
                LIMIT %(top_n)s
                """,
                {
                    "query_text": query_text.strip(),
                    "policy_id": policy_id,
                    "top_n": int(top_n),
                    **doc_params,
                    **scope_params,
                },
            ).fetchall()

        hits: List[SparseSearchHit] = []
        for row in rows:
            score = float(row["lexical_rank_score"])
            hits.append(
                SparseSearchHit(
                    chunk_uuid=row["id"],
                    document_id=row["document_id"],
                    chunk=_row_to_policy_chunk(row),
                    lexical_rank_score=score,
                )
            )
        return hits

    # -------------------------------------------------------------------------
    # Ingestion Jobs (Persistent History Across Document Archive/Delete)
    # -------------------------------------------------------------------------
    def create_ingestion_job(
        self,
        *,
        external_policy_key: str,
        scope: AuthorizedScope,
        policy_id: Optional[UUID] = None,
        document_id: Optional[UUID] = None,
        document_version_snapshot: Optional[int] = None,
        file_hash_sha256: Optional[str] = None,
        celery_task_id: Optional[str] = None,
        status: JobStatusLiteral = "queued",
        current_stage: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestionJobRecord:
        auth = validate_authorized_scope(scope)
        clean_hash = _validate_sha256(file_hash_sha256, allow_none=True)
        with self.db.transaction() as conn:
            if policy_id is not None:
                self._assert_policy_writable(conn, policy_id, auth)
            row = conn.execute(
                """
                INSERT INTO ingestion_jobs (
                    policy_id, document_id, external_policy_key, document_version_snapshot,
                    file_hash_sha256, owner_scope, celery_task_id, status,
                    current_stage, metadata
                )
                VALUES (
                    %(policy_id)s, %(document_id)s, %(external_policy_key)s, %(document_version_snapshot)s,
                    %(file_hash_sha256)s, %(owner_scope)s, %(celery_task_id)s, %(status)s,
                    %(current_stage)s, %(metadata)s
                )
                RETURNING *
                """,
                {
                    "policy_id": policy_id,
                    "document_id": document_id,
                    "external_policy_key": external_policy_key.strip(),
                    "document_version_snapshot": document_version_snapshot,
                    "file_hash_sha256": clean_hash,
                    "owner_scope": auth.owner_scope,
                    "celery_task_id": celery_task_id,
                    "status": status,
                    "current_stage": current_stage,
                    "metadata": Jsonb(dict(metadata or {})),
                },
            ).fetchone()
            assert row is not None
            return IngestionJobRecord.model_validate(row)

    def update_ingestion_job_status(
        self,
        *,
        job_id: UUID,
        status: JobStatusLiteral,
        scope: AuthorizedScope,
        current_stage: Optional[str] = None,
        total_chunks: Optional[int] = None,
        processed_chunks: Optional[int] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        increment_attempt: bool = False,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
    ) -> IngestionJobRecord:
        auth = validate_authorized_scope(scope)
        with self.db.transaction() as conn:
            existing = conn.execute(
                "SELECT id, owner_scope FROM ingestion_jobs WHERE id = %(job_id)s FOR UPDATE",
                {"job_id": job_id},
            ).fetchone()
            if existing is None:
                raise DatabaseNotFoundError("Ingestion job not found.")
            if existing["owner_scope"] != auth.owner_scope:
                raise DatabaseAuthorizationError("Ingestion job belongs to a different owner_scope.")

            row = conn.execute(
                """
                UPDATE ingestion_jobs j
                SET status = %(status)s,
                    current_stage = COALESCE(%(current_stage)s, j.current_stage),
                    total_chunks = COALESCE(%(total_chunks)s, j.total_chunks),
                    processed_chunks = COALESCE(%(processed_chunks)s, j.processed_chunks),
                    error_code = %(error_code)s,
                    error_message = %(error_message)s,
                    attempt_count = j.attempt_count + CASE WHEN %(increment_attempt)s THEN 1 ELSE 0 END,
                    started_at = COALESCE(%(started_at)s, j.started_at),
                    completed_at = COALESCE(%(completed_at)s, j.completed_at),
                    updated_at = NOW()
                WHERE j.id = %(job_id)s
                RETURNING j.*
                """,
                {
                    "job_id": job_id,
                    "status": status,
                    "current_stage": current_stage,
                    "total_chunks": total_chunks,
                    "processed_chunks": processed_chunks,
                    "error_code": error_code,
                    "error_message": error_message,
                    "increment_attempt": increment_attempt,
                    "started_at": started_at,
                    "completed_at": completed_at,
                },
            ).fetchone()
            assert row is not None
            return IngestionJobRecord.model_validate(row)

    def get_ingestion_job(self, job_id: UUID, *, scope: AuthorizedScope) -> Optional[IngestionJobRecord]:
        auth = validate_authorized_scope(scope)
        with self.db.connection() as conn:
            row = conn.execute(
                """
                SELECT j.*
                FROM ingestion_jobs j
                WHERE j.id = %(job_id)s
                  AND j.owner_scope = %(owner_scope)s
                """,
                {"job_id": job_id, "owner_scope": auth.owner_scope},
            ).fetchone()
            return IngestionJobRecord.model_validate(row) if row is not None else None

    # -------------------------------------------------------------------------
    # Asynchronous Worker Lifecycle Operations (Runtime Role, Atomic Transitions)
    # -------------------------------------------------------------------------
    def fetch_worker_job_context(
        self,
        *,
        job_id: UUID,
        policy_id: UUID,
        document_id: UUID,
        claimed_owner_scope: str,
    ) -> Tuple[IngestionJobRecord, PolicyRecord, PolicyDocumentRecord, AuthorizedScope]:
        """
        Validate Celery payload identifiers against persisted DB records before ingestion.
        Returns a derived AuthorizedScope backed by DB ownership rather than trusting message input.
        """
        if not isinstance(claimed_owner_scope, str) or not claimed_owner_scope.strip():
            raise DatabaseAuthorizationError("Celery ingestion payload missing owner_scope.")

        with self.db.connection() as conn:
            job_row = conn.execute(
                "SELECT * FROM ingestion_jobs WHERE id = %(job_id)s",
                {"job_id": job_id},
            ).fetchone()
            if job_row is None:
                raise DatabaseNotFoundError(f"Ingestion job '{job_id}' does not exist.")

            pol_row = conn.execute(
                "SELECT * FROM policies WHERE id = %(policy_id)s",
                {"policy_id": policy_id},
            ).fetchone()
            if pol_row is None:
                raise DatabaseNotFoundError(f"Policy '{policy_id}' does not exist.")

            doc_row = conn.execute(
                "SELECT * FROM policy_documents WHERE id = %(document_id)s AND policy_id = %(policy_id)s",
                {"document_id": document_id, "policy_id": policy_id},
            ).fetchone()
            if doc_row is None:
                raise DatabaseNotFoundError(
                    f"Document '{document_id}' does not exist for policy '{policy_id}'."
                )

        if (
            job_row["policy_id"] != policy_id
            or job_row["document_id"] != document_id
            or job_row["owner_scope"] != claimed_owner_scope.strip()
            or pol_row["owner_scope"] != claimed_owner_scope.strip()
        ):
            raise DatabaseAuthorizationError(
                "Ingestion task identifiers or owner_scope mismatch persisted job/policy ownership."
            )

        if pol_row["status"] == "archived" or doc_row["is_archived"]:
            raise DatabaseAuthorizationError("Cannot ingest archived policy or archived document version.")

        derived_scope = AuthorizedScope(
            user_id=pol_row["owner_id"],
            role="user",
            owner_scope=pol_row["owner_scope"],
            allow_catalog_read=False,
            trusted_gateway_context=True,
        )
        return (
            IngestionJobRecord.model_validate(job_row),
            PolicyRecord.model_validate(pol_row),
            PolicyDocumentRecord.model_validate(doc_row),
            derived_scope,
        )

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
    ) -> Tuple[str, IngestionJobRecord, PolicyDocumentRecord]:
        """
        Atomically claim an ingestion job under row-level locks (`FOR UPDATE`).

        Returns:
            ('claimed' | 'already_completed' | 'duplicate_in_progress', job, document)
        """
        auth = validate_authorized_scope(scope)
        with self.db.transaction() as conn:
            job_row = conn.execute(
                "SELECT * FROM ingestion_jobs WHERE id = %(job_id)s FOR UPDATE",
                {"job_id": job_id},
            ).fetchone()
            if job_row is None:
                raise DatabaseNotFoundError(f"Ingestion job '{job_id}' not found.")
            if job_row["owner_scope"] != auth.owner_scope:
                raise DatabaseAuthorizationError("Job belongs to a different owner_scope.")

            doc_row = conn.execute(
                """
                SELECT d.*
                FROM policy_documents d
                JOIN policies p ON p.id = d.policy_id
                WHERE d.id = %(document_id)s
                  AND d.policy_id = %(policy_id)s
                  AND p.owner_scope = %(owner_scope)s
                FOR UPDATE OF d
                """,
                {
                    "document_id": document_id,
                    "policy_id": job_row["policy_id"],
                    "owner_scope": auth.owner_scope,
                },
            ).fetchone()
            if doc_row is None:
                raise DatabaseNotFoundError("Document missing or unauthorized for claimed ingestion job.")

            job = IngestionJobRecord.model_validate(job_row)
            doc = PolicyDocumentRecord.model_validate(doc_row)

            # Safe idempotent handling: already completed job + completed document
            if job.status == "completed" and doc.ingestion_status == "completed":
                return "already_completed", job, doc

            if job.status == "cancelled":
                raise DatabaseAuthorizationError("Ingestion job was cancelled and cannot execute.")

            # Duplicate active task detection when another task_id currently holds processing state
            if job.status == "processing" and not is_celery_retry:
                same_task = task_id is not None and job.celery_task_id == task_id
                age_s = (datetime.now(timezone.utc) - job.updated_at).total_seconds()
                if not same_task and job.celery_task_id is not None and age_s < stale_lock_seconds:
                    return "duplicate_in_progress", job, doc

            updated_job_row = conn.execute(
                """
                UPDATE ingestion_jobs
                SET status = 'processing',
                    current_stage = %(stage)s,
                    celery_task_id = COALESCE(%(task_id)s, celery_task_id),
                    attempt_count = attempt_count + 1,
                    started_at = COALESCE(started_at, NOW()),
                    error_code = NULL,
                    error_message = NULL,
                    updated_at = NOW()
                WHERE id = %(job_id)s
                RETURNING *
                """,
                {"job_id": job_id, "stage": initial_stage, "task_id": task_id},
            ).fetchone()

            updated_doc_row = conn.execute(
                """
                UPDATE policy_documents
                SET ingestion_status = 'processing',
                    updated_at = NOW()
                WHERE id = %(document_id)s
                RETURNING *
                """,
                {"document_id": document_id},
            ).fetchone()
            assert updated_job_row is not None and updated_doc_row is not None
            return (
                "claimed",
                IngestionJobRecord.model_validate(updated_job_row),
                PolicyDocumentRecord.model_validate(updated_doc_row),
            )

    def transition_ingestion_job_stage(
        self,
        *,
        job_id: UUID,
        document_id: UUID,
        scope: AuthorizedScope,
        status: JobStatusLiteral,
        stage: str,
        doc_status: Optional[DocumentIngestionStatusLiteral] = None,
        total_chunks: Optional[int] = None,
        processed_chunks: Optional[int] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        metadata_patch: Optional[Dict[str, Any]] = None,
        mark_completed_at: bool = False,
    ) -> IngestionJobRecord:
        """Persist an atomic stage transition outside long-running OCR/embedding compute."""
        auth = validate_authorized_scope(scope)
        with self.db.transaction() as conn:
            row = conn.execute(
                """
                UPDATE ingestion_jobs
                SET status = %(status)s,
                    current_stage = %(stage)s,
                    total_chunks = COALESCE(%(total_chunks)s, total_chunks),
                    processed_chunks = COALESCE(%(processed_chunks)s, processed_chunks),
                    error_code = %(error_code)s,
                    error_message = %(error_message)s,
                    completed_at = CASE WHEN %(mark_completed_at)s THEN NOW() ELSE completed_at END,
                    metadata = metadata || %(metadata_patch)s,
                    updated_at = NOW()
                WHERE id = %(job_id)s
                  AND owner_scope = %(owner_scope)s
                RETURNING *
                """,
                {
                    "job_id": job_id,
                    "owner_scope": auth.owner_scope,
                    "status": status,
                    "stage": stage,
                    "total_chunks": total_chunks,
                    "processed_chunks": processed_chunks,
                    "error_code": error_code,
                    "error_message": error_message,
                    "mark_completed_at": mark_completed_at,
                    "metadata_patch": Jsonb(dict(metadata_patch or {})),
                },
            ).fetchone()
            if row is None:
                raise DatabaseNotFoundError("Ingestion job not found within authorized scope.")

            if doc_status is not None:
                conn.execute(
                    """
                    UPDATE policy_documents d
                    SET ingestion_status = %(doc_status)s,
                        updated_at = NOW()
                    FROM policies p
                    WHERE d.id = %(document_id)s
                      AND p.id = d.policy_id
                      AND p.owner_scope = %(owner_scope)s
                    """,
                    {
                        "document_id": document_id,
                        "doc_status": doc_status,
                        "owner_scope": auth.owner_scope,
                    },
                )
            return IngestionJobRecord.model_validate(row)

    def get_policy_attributes(
        self,
        *,
        policy_id: UUID,
        document_id: UUID,
    ) -> Optional[Dict[str, Any]]:
        """Fetch existing policy_attributes row for a specific (policy_id, document_id) pair."""
        with self.db.connection() as conn:
            reg = conn.execute("SELECT to_regclass('public.policy_attributes') AS tbl").fetchone()
            if not reg or reg.get("tbl") is None:
                return None
            row = conn.execute(
                """
                SELECT *
                FROM policy_attributes
                WHERE policy_id = %(policy_id)s
                  AND document_id = %(document_id)s
                LIMIT 1
                """,
                {"policy_id": policy_id, "document_id": document_id},
            ).fetchone()
            return dict(row) if row is not None else None

    def upsert_policy_attributes(
        self,
        *,
        policy_id: UUID,
        document_id: UUID,
        attributes: Any,
        is_catalog_eligible: Optional[bool] = None,
        conn: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Idempotently persist extracted versioned policy attributes inside DB transactions."""
        if hasattr(attributes, "model_dump"):
            data = attributes.model_dump()
        elif isinstance(attributes, dict):
            data = dict(attributes)
        else:
            raise DatabaseValidationError("attributes must be a PolicyAttributeExtractionResult or dict.")

        def _exec_upsert(tx_conn: Any) -> Dict[str, Any]:
            if is_catalog_eligible is None:
                pol_row = tx_conn.execute(
                    "SELECT visibility FROM policies WHERE id = %(policy_id)s LIMIT 1",
                    {"policy_id": policy_id},
                ).fetchone()
                if pol_row is None:
                    raise DatabaseNotFoundError(f"Policy '{policy_id}' not found.")
                effective_catalog_eligible = bool(pol_row.get("visibility") == "catalog")
            else:
                effective_catalog_eligible = bool(is_catalog_eligible)

            params = {
                "policy_id": policy_id,
                "document_id": document_id,
                "is_catalog_eligible": effective_catalog_eligible,
                "extraction_status": data.get("extraction_status", "insufficient_data"),
                "extraction_source": data.get("extraction_source", "ingestion_llm"),
                "extraction_confidence": data.get("extraction_confidence"),
                "extraction_error": data.get("extraction_error"),
                "source_chunk_ids": Jsonb(list(data.get("source_chunk_ids") or [])),
                "field_provenance": Jsonb(dict(data.get("field_provenance") or {})),
                "min_entry_age": data.get("min_entry_age"),
                "max_entry_age": data.get("max_entry_age"),
                "max_renewal_age": data.get("max_renewal_age"),
                "lifelong_renewal": data.get("lifelong_renewal"),
                "coverage_mode": data.get("coverage_mode"),
                "max_family_members": data.get("max_family_members"),
                "eligible_applicant_types": Jsonb(list(data.get("eligible_applicant_types") or [])),
                "annual_premium_min": data.get("annual_premium_min"),
                "annual_premium_max": data.get("annual_premium_max"),
                "sum_insured_min": data.get("sum_insured_min"),
                "sum_insured_max": data.get("sum_insured_max"),
                "deductible_amount": data.get("deductible_amount"),
                "copay_pct": data.get("copay_pct"),
                "room_rent_limit_type": data.get("room_rent_limit_type"),
                "room_rent_limit_value": data.get("room_rent_limit_value"),
                "initial_waiting_period_days": data.get("initial_waiting_period_days"),
                "ped_covered": data.get("ped_covered"),
                "ped_waiting_period_months": data.get("ped_waiting_period_months"),
                "specific_disease_waiting_months": data.get("specific_disease_waiting_months"),
                "maternity_waiting_months": data.get("maternity_waiting_months"),
                "network_hospitals_count": data.get("network_hospitals_count"),
                "cashless_available": data.get("cashless_available"),
                "pan_india_coverage": data.get("pan_india_coverage"),
                "available_regions": Jsonb(list(data.get("available_regions") or [])),
                "min_tenure_years": data.get("min_tenure_years"),
                "max_tenure_years": data.get("max_tenure_years"),
                "covered_benefits": Jsonb(list(data.get("covered_benefits") or [])),
                "exclusions": Jsonb(list(data.get("exclusions") or [])),
                "claim_settlement_ratio": data.get("claim_settlement_ratio"),
                "no_claim_bonus_pct": data.get("no_claim_bonus_pct"),
                "restoration_benefit": data.get("restoration_benefit"),
                "type_specific_metadata": Jsonb(dict(data.get("type_specific_metadata") or {})),
            }

            row = tx_conn.execute(
                """
                INSERT INTO policy_attributes (
                    policy_id, document_id, is_catalog_eligible,
                    extraction_status, extraction_source, extraction_confidence, extraction_error,
                    source_chunk_ids, field_provenance,
                    min_entry_age, max_entry_age, max_renewal_age, lifelong_renewal,
                    coverage_mode, max_family_members, eligible_applicant_types,
                    annual_premium_min, annual_premium_max, sum_insured_min, sum_insured_max,
                    deductible_amount, copay_pct, room_rent_limit_type, room_rent_limit_value,
                    initial_waiting_period_days, ped_covered, ped_waiting_period_months,
                    specific_disease_waiting_months, maternity_waiting_months,
                    network_hospitals_count, cashless_available, pan_india_coverage, available_regions,
                    min_tenure_years, max_tenure_years, covered_benefits, exclusions,
                    claim_settlement_ratio, no_claim_bonus_pct, restoration_benefit, type_specific_metadata
                )
                VALUES (
                    %(policy_id)s, %(document_id)s, %(is_catalog_eligible)s,
                    %(extraction_status)s, %(extraction_source)s, %(extraction_confidence)s, %(extraction_error)s,
                    %(source_chunk_ids)s, %(field_provenance)s,
                    %(min_entry_age)s, %(max_entry_age)s, %(max_renewal_age)s, %(lifelong_renewal)s,
                    %(coverage_mode)s, %(max_family_members)s, %(eligible_applicant_types)s,
                    %(annual_premium_min)s, %(annual_premium_max)s, %(sum_insured_min)s, %(sum_insured_max)s,
                    %(deductible_amount)s, %(copay_pct)s, %(room_rent_limit_type)s, %(room_rent_limit_value)s,
                    %(initial_waiting_period_days)s, %(ped_covered)s, %(ped_waiting_period_months)s,
                    %(specific_disease_waiting_months)s, %(maternity_waiting_months)s,
                    %(network_hospitals_count)s, %(cashless_available)s, %(pan_india_coverage)s, %(available_regions)s,
                    %(min_tenure_years)s, %(max_tenure_years)s, %(covered_benefits)s, %(exclusions)s,
                    %(claim_settlement_ratio)s, %(no_claim_bonus_pct)s, %(restoration_benefit)s, %(type_specific_metadata)s
                )
                ON CONFLICT (policy_id, document_id) DO UPDATE SET
                    is_catalog_eligible = EXCLUDED.is_catalog_eligible,
                    extraction_status = EXCLUDED.extraction_status,
                    extraction_source = EXCLUDED.extraction_source,
                    extraction_confidence = EXCLUDED.extraction_confidence,
                    extraction_error = EXCLUDED.extraction_error,
                    source_chunk_ids = EXCLUDED.source_chunk_ids,
                    field_provenance = EXCLUDED.field_provenance,
                    min_entry_age = EXCLUDED.min_entry_age,
                    max_entry_age = EXCLUDED.max_entry_age,
                    max_renewal_age = EXCLUDED.max_renewal_age,
                    lifelong_renewal = EXCLUDED.lifelong_renewal,
                    coverage_mode = EXCLUDED.coverage_mode,
                    max_family_members = EXCLUDED.max_family_members,
                    eligible_applicant_types = EXCLUDED.eligible_applicant_types,
                    annual_premium_min = EXCLUDED.annual_premium_min,
                    annual_premium_max = EXCLUDED.annual_premium_max,
                    sum_insured_min = EXCLUDED.sum_insured_min,
                    sum_insured_max = EXCLUDED.sum_insured_max,
                    deductible_amount = EXCLUDED.deductible_amount,
                    copay_pct = EXCLUDED.copay_pct,
                    room_rent_limit_type = EXCLUDED.room_rent_limit_type,
                    room_rent_limit_value = EXCLUDED.room_rent_limit_value,
                    initial_waiting_period_days = EXCLUDED.initial_waiting_period_days,
                    ped_covered = EXCLUDED.ped_covered,
                    ped_waiting_period_months = EXCLUDED.ped_waiting_period_months,
                    specific_disease_waiting_months = EXCLUDED.specific_disease_waiting_months,
                    maternity_waiting_months = EXCLUDED.maternity_waiting_months,
                    network_hospitals_count = EXCLUDED.network_hospitals_count,
                    cashless_available = EXCLUDED.cashless_available,
                    pan_india_coverage = EXCLUDED.pan_india_coverage,
                    available_regions = EXCLUDED.available_regions,
                    min_tenure_years = EXCLUDED.min_tenure_years,
                    max_tenure_years = EXCLUDED.max_tenure_years,
                    covered_benefits = EXCLUDED.covered_benefits,
                    exclusions = EXCLUDED.exclusions,
                    claim_settlement_ratio = EXCLUDED.claim_settlement_ratio,
                    no_claim_bonus_pct = EXCLUDED.no_claim_bonus_pct,
                    restoration_benefit = EXCLUDED.restoration_benefit,
                    type_specific_metadata = EXCLUDED.type_specific_metadata,
                    updated_at = NOW()
                RETURNING *
                """,
                params,
            ).fetchone()
            return dict(row)

        if conn is not None:
            return _exec_upsert(conn)
        with self.db.transaction() as tx_conn:
            return _exec_upsert(tx_conn)

    def complete_document_ingestion(
        self,
        *,
        job_id: UUID,
        policy_id: UUID,
        document_id: UUID,
        embedded_chunks: Sequence[PolicyChunk],
        model_name: str,
        model_revision: str,
        page_count: int,
        document_metadata: Optional[Dict[str, Any]],
        job_metadata: Optional[Dict[str, Any]],
        scope: AuthorizedScope,
        is_normalized: bool = True,
        activate_document: bool = True,
        extracted_attributes: Optional[Any] = None,
    ) -> Tuple[List[PersistedChunkRecord], List[ChunkEmbeddingRecord], IngestionJobRecord]:
        """
        Atomically persist chunks + 1024-dim embeddings and mark document + job COMPLETED.
        Prevents partially processed documents from ever appearing completed.
        """
        auth = validate_authorized_scope(scope)
        if not embedded_chunks:
            raise DatabaseValidationError("Cannot complete ingestion with zero extracted chunks.")
        if not model_name.strip() or not model_revision.strip():
            raise DatabaseValidationError("model_name and model_revision must be non-empty.")

        with self.db.transaction() as conn:
            self._assert_document_writable(conn, policy_id, document_id, auth)
            persisted_chunks: List[PersistedChunkRecord] = []
            persisted_embeddings: List[ChunkEmbeddingRecord] = []

            for idx, chunk in enumerate(embedded_chunks):
                if chunk.embedding is None:
                    raise DatabaseValidationError(
                        f"Chunk '{chunk.chunk_id}' is missing its 1024-dim embedding."
                    )
                pc = self._upsert_chunk_in_tx(
                    conn,
                    policy_id=policy_id,
                    document_id=document_id,
                    chunk=chunk,
                    chunk_index=idx,
                )
                ce = self._upsert_embedding_in_tx(
                    conn,
                    chunk_uuid=pc.id,
                    policy_id=policy_id,
                    document_id=document_id,
                    vector_values=chunk.embedding,
                    model_name=model_name,
                    model_revision=model_revision,
                    is_normalized=is_normalized,
                )
                persisted_chunks.append(pc)
                persisted_embeddings.append(ce)

            conn.execute(
                """
                UPDATE policy_documents
                SET ingestion_status = 'completed',
                    page_count = %(page_count)s,
                    document_metadata = document_metadata || %(doc_meta)s,
                    updated_at = NOW()
                WHERE id = %(document_id)s
                  AND policy_id = %(policy_id)s
                """,
                {
                    "document_id": document_id,
                    "policy_id": policy_id,
                    "page_count": max(0, int(page_count)),
                    "doc_meta": Jsonb(dict(document_metadata or {})),
                },
            )

            if activate_document:
                conn.execute(
                    """
                    UPDATE policies
                    SET active_version_id = %(document_id)s,
                        status = CASE WHEN status = 'draft' THEN 'active' ELSE status END,
                        updated_at = NOW()
                    WHERE id = %(policy_id)s
                      AND owner_scope = %(owner_scope)s
                    """,
                    {
                        "policy_id": policy_id,
                        "document_id": document_id,
                        "owner_scope": auth.owner_scope,
                    },
                )

            if extracted_attributes is not None:
                self.upsert_policy_attributes(
                    policy_id=policy_id,
                    document_id=document_id,
                    attributes=extracted_attributes,
                    conn=conn,
                )

            job_row = conn.execute(
                """
                UPDATE ingestion_jobs
                SET status = 'completed',
                    current_stage = 'COMPLETED',
                    total_chunks = %(count)s,
                    processed_chunks = %(count)s,
                    error_code = NULL,
                    error_message = NULL,
                    completed_at = NOW(),
                    metadata = metadata || %(job_meta)s,
                    updated_at = NOW()
                WHERE id = %(job_id)s
                  AND owner_scope = %(owner_scope)s
                RETURNING *
                """,
                {
                    "job_id": job_id,
                    "owner_scope": auth.owner_scope,
                    "count": len(persisted_chunks),
                    "job_meta": Jsonb(dict(job_metadata or {})),
                },
            ).fetchone()
            if job_row is None:
                raise DatabaseNotFoundError("Ingestion job not found during completion commit.")

            return (
                persisted_chunks,
                persisted_embeddings,
                IngestionJobRecord.model_validate(job_row),
            )

    # -------------------------------------------------------------------------
    # Authenticated Internal Enqueue Validation & Dispatch Recording
    # -------------------------------------------------------------------------
    def prepare_job_for_enqueue(
        self,
        *,
        job_id: UUID,
        policy_id: UUID,
        document_id: UUID,
    ) -> Tuple[str, IngestionJobRecord, PolicyRecord, PolicyDocumentRecord, AuthorizedScope]:
        """
        Verify persisted job/policy/document records inside a row lock before Celery dispatch.

        Resolves owner_scope strictly from persisted database rows rather than trusting
        any caller-supplied scope.
        """
        with self.db.transaction() as conn:
            job_row = conn.execute(
                "SELECT * FROM ingestion_jobs WHERE id = %(job_id)s FOR UPDATE",
                {"job_id": job_id},
            ).fetchone()
            if job_row is None:
                raise DatabaseNotFoundError(f"Ingestion job '{job_id}' does not exist.")

            pol_row = conn.execute(
                "SELECT * FROM policies WHERE id = %(policy_id)s",
                {"policy_id": policy_id},
            ).fetchone()
            if pol_row is None:
                raise DatabaseNotFoundError(f"Policy '{policy_id}' does not exist.")

            doc_row = conn.execute(
                "SELECT * FROM policy_documents WHERE id = %(document_id)s AND policy_id = %(policy_id)s",
                {"document_id": document_id, "policy_id": policy_id},
            ).fetchone()
            if doc_row is None:
                raise DatabaseNotFoundError(
                    f"Document '{document_id}' does not exist for policy '{policy_id}'."
                )

            if (
                job_row["policy_id"] != policy_id
                or job_row["document_id"] != document_id
                or job_row["owner_scope"] != pol_row["owner_scope"]
            ):
                raise DatabaseAuthorizationError(
                    "Enqueue request identifiers conflict with persisted policy ownership."
                )

            if pol_row["status"] == "archived" or doc_row["is_archived"]:
                raise DatabaseAuthorizationError(
                    "Cannot enqueue an archived policy or archived document version."
                )

            if (
                doc_row["file_hash_sha256"]
                and job_row["file_hash_sha256"]
                and doc_row["file_hash_sha256"] != job_row["file_hash_sha256"]
            ):
                raise DatabaseValidationError(
                    "Document checksum does not match persisted ingestion job snapshot."
                )

            job = IngestionJobRecord.model_validate(job_row)
            policy = PolicyRecord.model_validate(pol_row)
            doc = PolicyDocumentRecord.model_validate(doc_row)

            trusted_scope = AuthorizedScope(
                user_id=policy.owner_id,
                role="user",
                owner_scope=policy.owner_scope,
                allow_catalog_read=False,
                trusted_gateway_context=True,
            )

            if job.status == "completed":
                return "already_completed", job, policy, doc, trusted_scope

            if job.status in ("failed", "cancelled") or doc.ingestion_status in ("failed", "archived"):
                raise DatabaseValidationError(
                    f"Ingestion job '{job_id}' is in non-queueable state '{job.status}'."
                )

            if job.status == "processing" or (job.status == "queued" and job.celery_task_id is not None):
                return "already_enqueued", job, policy, doc, trusted_scope

            if doc.ingestion_status not in ("pending", "queued"):
                raise DatabaseValidationError(
                    f"Policy document '{document_id}' has unexpected ingestion status '{doc.ingestion_status}'."
                )

            return "ready_to_enqueue", job, policy, doc, trusted_scope

    def record_enqueue_dispatch(
        self,
        *,
        job_id: UUID,
        celery_task_id: str,
        scope: AuthorizedScope,
        request_id: Optional[str] = None,
    ) -> IngestionJobRecord:
        """Persist Celery dispatch identity atomically without clobbering faster worker progress."""
        auth = validate_authorized_scope(scope)
        patch = {
            "dispatch_state": "dispatched",
            "dispatch_task_id": celery_task_id,
            "dispatched_at": datetime.now(timezone.utc).isoformat(),
        }
        if request_id:
            patch["request_id"] = request_id

        with self.db.transaction() as conn:
            row = conn.execute(
                """
                UPDATE ingestion_jobs
                SET celery_task_id = COALESCE(celery_task_id, %(celery_task_id)s),
                    current_stage = CASE
                        WHEN status = 'queued' AND (current_stage IS NULL OR current_stage IN ('QUEUED', 'ENQUEUE_RETRY_PENDING'))
                            THEN 'QUEUED'
                        ELSE current_stage
                    END,
                    error_code = CASE WHEN status = 'queued' THEN NULL ELSE error_code END,
                    error_message = CASE WHEN status = 'queued' THEN NULL ELSE error_message END,
                    metadata = metadata || %(patch)s,
                    updated_at = NOW()
                WHERE id = %(job_id)s
                  AND owner_scope = %(owner_scope)s
                RETURNING *
                """,
                {
                    "job_id": job_id,
                    "owner_scope": auth.owner_scope,
                    "celery_task_id": celery_task_id,
                    "patch": Jsonb(patch),
                },
            ).fetchone()
            if row is None:
                raise DatabaseNotFoundError("Ingestion job not found while recording enqueue dispatch.")
            return IngestionJobRecord.model_validate(row)

    def record_enqueue_failure(
        self,
        *,
        job_id: UUID,
        scope: AuthorizedScope,
        error_code: str,
        error_message: str,
        request_id: Optional[str] = None,
    ) -> IngestionJobRecord:
        """Record sanitized broker dispatch failure so reconciliation picks up recovery cleanly."""
        auth = validate_authorized_scope(scope)
        patch = {
            "dispatch_state": "enqueue_failed",
            "last_enqueue_error_at": datetime.now(timezone.utc).isoformat(),
        }
        if request_id:
            patch["request_id"] = request_id

        with self.db.transaction() as conn:
            row = conn.execute(
                """
                UPDATE ingestion_jobs
                SET current_stage = CASE
                        WHEN status = 'queued' THEN 'ENQUEUE_RETRY_PENDING'
                        ELSE current_stage
                    END,
                    error_code = CASE WHEN status = 'queued' THEN %(error_code)s ELSE error_code END,
                    error_message = CASE WHEN status = 'queued' THEN %(error_message)s ELSE error_message END,
                    metadata = metadata || %(patch)s,
                    updated_at = NOW()
                WHERE id = %(job_id)s
                  AND owner_scope = %(owner_scope)s
                RETURNING *
                """,
                {
                    "job_id": job_id,
                    "owner_scope": auth.owner_scope,
                    "error_code": error_code,
                    "error_message": error_message,
                    "patch": Jsonb(patch),
                },
            ).fetchone()
            if row is None:
                raise DatabaseNotFoundError("Ingestion job not found while recording enqueue failure.")
            return IngestionJobRecord.model_validate(row)
