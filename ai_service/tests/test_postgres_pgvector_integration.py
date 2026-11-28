import os
from pathlib import Path
import socket
import subprocess
import uuid

from fastapi.testclient import TestClient
from pgvector import Vector
import psycopg
import pytest

from ai_service.persistence import (
    AuthorizedScope,
    DatabaseAuthorizationError,
    DatabaseConnectionError,
    DatabaseConstraintError,
    DatabaseValidationError,
    PolicyKnowledgeRepository,
    PostgresPoolManager,
    close_db_pool,
    get_db_pool_manager,
)
from ai_service.embeddings import OfflineHashEmbedder
from ai_service.ingestion.semantic_chunker import SemanticChunker
from ai_service.main import app
from ai_service.schemas.ingestion import (
    BoundingBox,
    DocumentMetadata,
    ExtractedDocument,
    NodeType,
    StructuredDocument,
    StructuredSection,
)
from ai_service.schemas.structured_output import ChunkCategory, PolicyChunk

_DEFAULT_RUNTIME_DSN = "postgresql://coverageiq_runtime:change_me_runtime@127.0.0.1:55432/coverageiq_db"
_DEFAULT_MIGRATION_DSN = "postgresql://coverageiq_migrator:change_me_migrator@127.0.0.1:55432/coverageiq_db"


def _resolve_test_dsns() -> tuple[str, str]:
    runtime_dsn = os.getenv("TEST_DATABASE_URL") or _DEFAULT_RUNTIME_DSN
    migration_dsn = os.getenv("TEST_MIGRATION_DATABASE_URL") or _DEFAULT_MIGRATION_DSN
    return runtime_dsn, migration_dsn


def _is_pg_reachable(dsn: str) -> bool:
    try:
        from urllib.parse import urlsplit

        parsed = urlsplit(dsn)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 5432
        with socket.create_connection((host, port), timeout=1.5):
            pass
        with psycopg.connect(dsn, connect_timeout=2) as conn:
            ver = conn.execute("SELECT current_setting('server_version_num')::int").fetchone()
            return bool(ver and ver[0] >= 160000)
    except Exception:
        return False


def _scope(user_id: str, owner_scope: str, role: str = "user", *, allow_catalog_read: bool = False) -> AuthorizedScope:
    return AuthorizedScope(
        user_id=user_id,
        role=role,  # type: ignore[arg-type]
        owner_scope=owner_scope,
        allow_catalog_read=allow_catalog_read,
        trusted_gateway_context=True,
    )


@pytest.fixture(scope="module")
def pg16_env():
    runtime_dsn, migration_dsn = _resolve_test_dsns()
    if not _is_pg_reachable(migration_dsn):
        pytest.skip(
            "Real PostgreSQL 16 + pgvector test instance is not reachable; "
            "start pgvector/pgvector:pg16 and run server migrations before running integration tests."
        )

    repo_root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "MIGRATION_DATABASE_URL": migration_dsn}
    # Ensure schema is cleanly up via node-pg-migrate before tests begin
    subprocess.run(
        ["npm", "--prefix", str(repo_root / "server"), "run", "migrate:up"],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    return {"runtime_dsn": runtime_dsn, "migration_dsn": migration_dsn, "repo_root": repo_root}


@pytest.fixture()
def pg_repo(pg16_env):
    # Clean tables between tests using migration owner (runtime role has no TRUNCATE/DDL privileges)
    with psycopg.connect(pg16_env["migration_dsn"]) as conn:
        with conn.transaction():
            conn.execute(
                "TRUNCATE TABLE chunk_embeddings, policy_chunks, ingestion_jobs, policy_documents, policies CASCADE"
            )

    pool_mgr = PostgresPoolManager(dsn=pg16_env["runtime_dsn"], min_size=1, max_size=4, timeout=10.0)
    pool_mgr.open(wait=True)
    try:
        yield PolicyKnowledgeRepository(pool_mgr), pool_mgr, pg16_env
    finally:
        pool_mgr.close()


def test_migration_up_down_redo_and_schema_indexes(pg_repo):
    _, pool_mgr, env = pg_repo
    schema_info = pool_mgr.verify_schema()
    assert schema_info["server_version_num"] >= 160000
    assert set(schema_info["tables"]) == {
        "policies",
        "policy_documents",
        "policy_chunks",
        "chunk_embeddings",
        "ingestion_jobs",
    }

    with pool_mgr.connection() as conn:
        idx_rows = conn.execute(
            """
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND tablename IN ('chunk_embeddings', 'policy_chunks', 'policies', 'policy_documents', 'ingestion_jobs')
            """
        ).fetchall()
    idx_map = {r["indexname"]: r["indexdef"] for r in idx_rows}
    assert "USING hnsw" in idx_map["idx_chunk_embeddings_hnsw_cosine"]
    assert "vector_cosine_ops" in idx_map["idx_chunk_embeddings_hnsw_cosine"]
    assert "USING gin" in idx_map["idx_policy_chunks_search_tsv_gin"]

    # Execute actual node-pg-migrate down & up CLI cycle against real PG16
    mig_env = {**os.environ, "MIGRATION_DATABASE_URL": env["migration_dsn"]}
    subprocess.run(
        ["npm", "--prefix", str(env["repo_root"] / "server"), "run", "migrate:down"],
        check=True,
        env=mig_env,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["npm", "--prefix", str(env["repo_root"] / "server"), "run", "migrate:up"],
        check=True,
        env=mig_env,
        capture_output=True,
        text=True,
    )
    assert set(pool_mgr.verify_schema()["tables"]) == {
        "policies",
        "policy_documents",
        "policy_chunks",
        "chunk_embeddings",
        "ingestion_jobs",
    }


def test_separate_migration_and_runtime_roles_enforced(pg_repo):
    _, pool_mgr, _ = pg_repo
    # Runtime role cannot run DDL or access pgmigrations
    with pytest.raises(DatabaseAuthorizationError):
        with pool_mgr.transaction() as conn:
            conn.execute("CREATE TABLE unauthorized_runtime_table (id INT)")

    with pytest.raises(DatabaseAuthorizationError):
        with pool_mgr.connection() as conn:
            conn.execute("SELECT * FROM pgmigrations")


def test_active_version_cross_policy_fk_and_version_uniqueness(pg_repo):
    repo, pool_mgr, _ = pg_repo
    alice = _scope("alice", "user:alice")

    pol_a = repo.upsert_policy(
        external_policy_key="pol_alpha",
        title="Alpha Health",
        owner_id="alice",
        scope=alice,
        policy_type="health",
    )
    pol_b = repo.upsert_policy(
        external_policy_key="pol_beta",
        title="Beta Health",
        owner_id="alice",
        scope=alice,
        policy_type="health",
    )

    doc_a_v1 = repo.create_document(
        policy_id=pol_a.id,
        version_number=1,
        file_name="alpha_v1.pdf",
        storage_path="/uploads/alpha_v1.pdf",
        file_size_bytes=1024,
        file_hash_sha256="a" * 64,
        scope=alice,
    )
    doc_b_v1 = repo.create_document(
        policy_id=pol_b.id,
        version_number=1,
        file_name="beta_v1.pdf",
        storage_path="/uploads/beta_v1.pdf",
        file_size_bytes=2048,
        file_hash_sha256="b" * 64,
        scope=alice,
    )

    # Duplicate version_number within same policy is rejected
    with pytest.raises(DatabaseConstraintError):
        repo.create_document(
            policy_id=pol_a.id,
            version_number=1,
            file_name="alpha_dup.pdf",
            storage_path="/uploads/alpha_dup.pdf",
            file_size_bytes=1024,
            file_hash_sha256="c" * 64,
            scope=alice,
        )

    # Second version for same policy succeeds
    doc_a_v2 = repo.create_document(
        policy_id=pol_a.id,
        version_number=2,
        file_name="alpha_v2.pdf",
        storage_path="/uploads/alpha_v2.pdf",
        file_size_bytes=1500,
        file_hash_sha256="d" * 64,
        scope=alice,
    )
    assert doc_a_v2.version_number == 2

    # Setting active_version_id to another policy's document is blocked at the DB FK constraint level
    with pytest.raises(DatabaseConstraintError):
        repo.set_active_document_version(policy_id=pol_a.id, document_id=doc_b_v1.id, scope=alice)

    with pytest.raises(DatabaseConstraintError):
        with pool_mgr.transaction() as conn:
            conn.execute(
                "UPDATE policies SET active_version_id = %(doc_id)s WHERE id = %(pol_id)s",
                {"doc_id": doc_b_v1.id, "pol_id": pol_a.id},
            )

    # Same-policy document succeeds
    updated_pol_a = repo.set_active_document_version(policy_id=pol_a.id, document_id=doc_a_v1.id, scope=alice)
    assert updated_pol_a.active_version_id == doc_a_v1.id


def test_deterministic_chunk_ids_no_collision_and_chunker_decoupled(pg_repo):
    repo, _, _ = pg_repo
    alice = _scope("alice", "user:alice")

    # SemanticChunker produces PolicyChunks without embeddings
    chunker = SemanticChunker(tokenizer_name="cl100k_base")
    sdoc = StructuredDocument(
        doc_id="health_pol_shared",
        document_title="Gold Health Shield",
        metadata=DocumentMetadata(
            file_size_bytes=5000,
            file_hash_sha256="1" * 64,
            page_count=2,
        ),
        sections=[
            StructuredSection(
                section_id="sec_2_1",
                title="In-Patient Hospitalization",
                clause_reference="Section 2.1",
                node_type=NodeType.SECTION,
                category=ChunkCategory.COVERAGE,
                content="Hospitalization for minimum 24 consecutive hours is covered.",
                page_start=1,
                page_end=1,
                source_block_ids=["blk_1"],
                bounding_boxes=[BoundingBox(x0=10, y0=10, x1=100, y1=40)],
            )
        ],
        source_page_count=2,
    )
    chunks_from_chunker = chunker.chunk_document(sdoc, policy_id="pol_shared", policy_version="v_const")
    assert len(chunks_from_chunker) == 1
    assert chunks_from_chunker[0].embedding is None  # Chunker remains decoupled

    # Create two separate policies AND two document versions that share identical chunk_key values
    pol_1 = repo.upsert_policy(external_policy_key="pol_1", title="Policy One", owner_id="alice", scope=alice)
    pol_2 = repo.upsert_policy(external_policy_key="pol_2", title="Policy Two", owner_id="alice", scope=alice)
    doc_1_v1 = repo.create_document(
        policy_id=pol_1.id,
        version_number=1,
        file_name="p1v1.pdf",
        storage_path="/uploads/p1v1.pdf",
        file_size_bytes=500,
        file_hash_sha256="1" * 64,
        scope=alice,
    )
    doc_1_v2 = repo.create_document(
        policy_id=pol_1.id,
        version_number=2,
        file_name="p1v2.pdf",
        storage_path="/uploads/p1v2.pdf",
        file_size_bytes=550,
        file_hash_sha256="2" * 64,
        scope=alice,
    )
    doc_2_v1 = repo.create_document(
        policy_id=pol_2.id,
        version_number=1,
        file_name="p2v1.pdf",
        storage_path="/uploads/p2v1.pdf",
        file_size_bytes=600,
        file_hash_sha256="3" * 64,
        scope=alice,
    )

    r1 = repo.persist_chunks(policy_id=pol_1.id, document_id=doc_1_v1.id, chunks=chunks_from_chunker, scope=alice)
    r2 = repo.persist_chunks(policy_id=pol_1.id, document_id=doc_1_v2.id, chunks=chunks_from_chunker, scope=alice)
    r3 = repo.persist_chunks(policy_id=pol_2.id, document_id=doc_2_v1.id, chunks=chunks_from_chunker, scope=alice)

    assert len({r1[0].id, r2[0].id, r3[0].id}) == 3
    assert r1[0].chunk_key == r2[0].chunk_key == r3[0].chunk_key


def test_embedding_dimension_validation_rollback_and_idempotency(pg_repo):
    repo, pool_mgr, _ = pg_repo
    alice = _scope("alice", "user:alice")
    embedder = OfflineHashEmbedder(dimension=1024, normalize=True)

    pol = repo.upsert_policy(
        external_policy_key="pol_tx_test",
        title="Tx Validation Policy",
        owner_id="alice",
        scope=alice,
        policy_type="health",
    )
    doc = repo.create_document(
        policy_id=pol.id,
        version_number=1,
        file_name="tx.pdf",
        storage_path="/uploads/tx.pdf",
        file_size_bytes=1200,
        file_hash_sha256="e" * 64,
        scope=alice,
    )

    c1 = PolicyChunk(
        chunk_id="chk_tx_01",
        policy_id="pol_tx_test",
        clause_ref="Section 2.1",
        page_number=12,
        category=ChunkCategory.COVERAGE,
        content_text="Hospitalization for 24 hours is covered up to Sum Insured.",
    )
    c2 = PolicyChunk(
        chunk_id="chk_tx_02",
        policy_id="pol_tx_test",
        clause_ref="Section 4.1",
        page_number=22,
        category=ChunkCategory.EXCLUSIONS,
        content_text="Cosmetic surgery is excluded unless accident reconstruction.",
    )

    embedded_c1 = embedder.embed_chunks([c1])[0]
    bad_dim_c2 = c2.model_copy(update={"embedding": [0.1] * 768})  # Wrong dimension (768 vs 1024)

    # 1. Atomicity / Rollback: failing on c2 must roll back c1 chunk & embedding writes completely
    with pytest.raises(DatabaseValidationError):
        repo.persist_embedded_chunks(
            policy_id=pol.id,
            document_id=doc.id,
            embedded_chunks=[embedded_c1, bad_dim_c2],
            model_name="BAAI/bge-m3",
            model_revision="test-rev",
            scope=alice,
        )

    with pool_mgr.connection() as conn:
        counts = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM policy_chunks WHERE document_id = %(doc_id)s) AS chunks_cnt,
                (SELECT COUNT(*) FROM chunk_embeddings WHERE document_id = %(doc_id)s) AS emb_cnt
            """,
            {"doc_id": doc.id},
        ).fetchone()
    assert counts["chunks_cnt"] == 0
    assert counts["emb_cnt"] == 0

    # 2. Idempotent persistence with valid 1024-dim embeddings
    both_embedded = embedder.embed_chunks([c1, c2])
    chunks_res1, embs_res1 = repo.persist_embedded_chunks(
        policy_id=pol.id,
        document_id=doc.id,
        embedded_chunks=both_embedded,
        model_name="BAAI/bge-m3",
        model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        scope=alice,
    )
    chunks_res2, embs_res2 = repo.persist_embedded_chunks(
        policy_id=pol.id,
        document_id=doc.id,
        embedded_chunks=both_embedded,
        model_name="BAAI/bge-m3",
        model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        scope=alice,
    )
    assert [c.id for c in chunks_res1] == [c.id for c in chunks_res2]
    assert [e.id for e in embs_res1] == [e.id for e in embs_res2]

    # 3. Direct DB-level vector(1024) enforcement rejects non-1024 vectors even bypassing Python helpers
    with pytest.raises((DatabaseConstraintError, DatabaseValidationError)):
        with pool_mgr.transaction() as conn:
            conn.execute(
                """
                INSERT INTO chunk_embeddings (
                    chunk_id, policy_id, document_id, model_name, model_revision, embedding_dim, embedding
                )
                VALUES (%(cid)s, %(pid)s, %(did)s, 'BAAI/bge-m3', 'raw-bad-dim', 1024, %(vec)s)
                """,
                {
                    "cid": chunks_res1[0].id,
                    "pid": pol.id,
                    "did": doc.id,
                    "vec": Vector([0.2] * 768),
                },
            )


def test_pool_lifecycle_connection_reuse_and_fastapi_lifespan(pg_repo, monkeypatch):
    _, pool_mgr, env = pg_repo

    # Repeated queries reuse pooled connections (never spawning 12 distinct connections)
    pids = set()
    for _ in range(12):
        with pool_mgr.connection() as conn:
            row = conn.execute("SELECT pg_backend_pid() AS pid").fetchone()
            pids.add(row["pid"])
    assert len(pids) <= pool_mgr.max_size
    assert pool_mgr.get_stats()["is_open"] is True

    # Closing a separate pool prevents further usage cleanly
    temp_pool = PostgresPoolManager(dsn=env["runtime_dsn"], min_size=1, max_size=2)
    temp_pool.open(wait=True)
    assert temp_pool.is_open is True
    temp_pool.close()
    assert temp_pool.is_open is False
    with pytest.raises(DatabaseConnectionError):
        with temp_pool.connection():
            pass

    # FastAPI lifespan with USE_MOCK_STORAGE=false opens and closes singleton pool cleanly
    close_db_pool()
    monkeypatch.setenv("USE_MOCK_STORAGE", "false")
    monkeypatch.setenv("DATABASE_URL", env["runtime_dsn"])
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")

    with TestClient(app) as client:
        resp = client.get("/ai/health")
        assert resp.status_code == 200
        assert resp.json()["database_mode"] == "postgresql_pgvector"
        assert get_db_pool_manager().is_open is True

    assert get_db_pool_manager().is_open is False


def test_policy_isolation_vector_hnsw_fts_and_job_history_preservation(pg_repo):
    repo, _, _ = pg_repo
    alice = _scope("alice", "user:alice")
    bob = _scope("bob", "user:bob", allow_catalog_read=True)
    bob_strict = _scope("bob", "user:bob", allow_catalog_read=False)
    embedder = OfflineHashEmbedder(dimension=1024, normalize=True)

    # Alice owns a private policy with sensitive chunks & embeddings
    pol_alice = repo.upsert_policy(
        external_policy_key="pol_alice_private",
        title="Alice Private Health Policy",
        owner_id="alice",
        scope=alice,
        policy_type="health",
        visibility="private",
        status="active",
    )
    doc_alice = repo.create_document(
        policy_id=pol_alice.id,
        version_number=1,
        file_name="alice_private.pdf",
        storage_path="/uploads/alice_private.pdf",
        file_size_bytes=4096,
        file_hash_sha256="f" * 64,
        page_count=10,
        ingestion_status="processing",
        scope=alice,
    )
    job = repo.create_ingestion_job(
        external_policy_key="pol_alice_private",
        policy_id=pol_alice.id,
        document_id=doc_alice.id,
        document_version_snapshot=1,
        file_hash_sha256="f" * 64,
        celery_task_id=f"celery-{uuid.uuid4()}",
        status="processing",
        current_stage="embed",
        scope=alice,
    )

    chunks = [
        PolicyChunk(
            chunk_id="chk_priv_01",
            policy_id="pol_alice_private",
            clause_ref="Section 2.1 - Hospitalization",
            page_number=4,
            category=ChunkCategory.COVERAGE,
            content_text="In-patient hospitalization for minimum 24 consecutive hours is covered.",
            source_block_ids=["b_1"],
        ),
        PolicyChunk(
            chunk_id="chk_priv_02",
            policy_id="pol_alice_private",
            clause_ref="Section 5.3 - Maternity Waiting Period",
            page_number=9,
            category=ChunkCategory.CONDITIONS,
            content_text="Maternity delivery expenses require 24 consecutive months continuous coverage.",
            source_block_ids=["b_2"],
        ),
    ]
    embedded = embedder.embed_chunks(chunks)
    repo.persist_embedded_chunks(
        policy_id=pol_alice.id,
        document_id=doc_alice.id,
        embedded_chunks=embedded,
        model_name="BAAI/bge-m3",
        model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        scope=alice,
    )
    repo.set_active_document_version(policy_id=pol_alice.id, document_id=doc_alice.id, scope=alice)
    repo.update_ingestion_job_status(
        job_id=job.id,
        status="completed",
        current_stage="done",
        total_chunks=2,
        processed_chunks=2,
        increment_attempt=True,
        scope=alice,
    )

    # Scoped vector search (HNSW) and PostgreSQL FTS search (GIN) succeed for Alice
    q_vec = embedder.embed_query("maternity delivery 24 months waiting period")
    vec_hits = repo.vector_search(policy_id=pol_alice.id, query_embedding=q_vec, scope=alice, top_n=2)
    assert len(vec_hits) == 2
    assert vec_hits[0].chunk.chunk_id == "chk_priv_02"

    fts_hits = repo.fts_search(
        policy_id=pol_alice.id,
        query_text="maternity waiting period",
        scope=alice,
        top_n=2,
    )
    assert len(fts_hits) == 1
    assert fts_hits[0].chunk.chunk_id == "chk_priv_02"

    # Strict cross-tenant policy isolation: Bob cannot read or mutate Alice's private data
    assert repo.get_policy(pol_alice.id, scope=bob) is None
    assert repo.get_policy_chunks(policy_id=pol_alice.id, scope=bob) == []
    assert repo.vector_search(policy_id=pol_alice.id, query_embedding=q_vec, scope=bob, top_n=5) == []
    assert repo.fts_search(policy_id=pol_alice.id, query_text="maternity", scope=bob, top_n=5) == []
    assert repo.get_ingestion_job(job.id, scope=bob) is None

    with pytest.raises(DatabaseAuthorizationError):
        repo.upsert_policy(
            external_policy_key="pol_alice_private",
            title="Hijack Attempt",
            owner_id="bob",
            scope=bob,
        )
    with pytest.raises(DatabaseAuthorizationError):
        repo.delete_document(doc_alice.id, scope=bob)

    # Catalog visibility works only when active AND allow_catalog_read=True
    org_scope = _scope("org_1", "org:acme", role="organization")
    cat_pol = repo.upsert_policy(
        external_policy_key="pol_catalog_active",
        title="Acme Catalog Plan",
        owner_id="org_1",
        scope=org_scope,
        visibility="catalog",
        status="active",
    )
    assert repo.get_policy(cat_pol.id, scope=bob) is not None
    assert repo.get_policy(cat_pol.id, scope=bob_strict) is None

    # Ingestion job history is preserved when document is archived and even when hard-deleted
    archived_doc = repo.archive_document(doc_alice.id, scope=alice)
    assert archived_doc.is_archived is True
    job_after_archive = repo.get_ingestion_job(job.id, scope=alice)
    assert job_after_archive is not None
    assert job_after_archive.status == "completed"
    assert job_after_archive.document_id == doc_alice.id

    assert repo.delete_document(doc_alice.id, scope=alice) is True
    assert repo.get_document(doc_alice.id, scope=alice) is None
    assert repo.get_policy_chunks(policy_id=pol_alice.id, scope=alice, document_id=doc_alice.id) == []

    # active_version_id cleared safely without breaking policy row, and ingestion job survived with audit snapshot
    pol_after_del = repo.get_policy(pol_alice.id, scope=alice)
    assert pol_after_del is not None
    assert pol_after_del.active_version_id is None

    job_after_delete = repo.get_ingestion_job(job.id, scope=alice)
    assert job_after_delete is not None
    assert job_after_delete.document_id is None
    assert job_after_delete.policy_id == pol_alice.id
    assert job_after_delete.external_policy_key == "pol_alice_private"
    assert job_after_delete.document_version_snapshot == 1
    assert job_after_delete.file_hash_sha256 == "f" * 64
    assert job_after_delete.status == "completed"


def test_schema_check_constraints_and_policy_delete_preserves_jobs(pg_repo):
    repo, pool_mgr, _ = pg_repo
    alice = _scope("alice", "user:alice")

    pol = repo.upsert_policy(
        external_policy_key="pol_checks",
        title="Constraint Check Policy",
        owner_id="alice",
        scope=alice,
    )
    doc = repo.create_document(
        policy_id=pol.id,
        version_number=1,
        file_name="checks.pdf",
        storage_path="/uploads/checks.pdf",
        file_size_bytes=512,
        file_hash_sha256="9" * 64,
        scope=alice,
    )
    repo.set_active_document_version(policy_id=pol.id, document_id=doc.id, scope=alice)
    failed_job = repo.create_ingestion_job(
        external_policy_key="pol_checks",
        policy_id=pol.id,
        document_id=doc.id,
        document_version_snapshot=1,
        file_hash_sha256="9" * 64,
        status="failed",
        current_stage="ocr",
        scope=alice,
    )
    failed_job = repo.update_ingestion_job_status(
        job_id=failed_job.id,
        status="failed",
        error_code="OCR_ENGINE_TIMEOUT",
        error_message="Page 4 OCR timed out safely.",
        increment_attempt=True,
        scope=alice,
    )

    # DB-level CHECK constraints enforce SHA-256 regex, page range, category, and job status
    with pytest.raises(DatabaseConstraintError):
        with pool_mgr.transaction() as conn:
            conn.execute(
                """
                INSERT INTO policy_documents (
                    policy_id, version_number, file_name, storage_path, file_size_bytes, file_hash_sha256
                )
                VALUES (%(pid)s, 2, 'bad.pdf', '/bad.pdf', 100, 'NOT_HEX_64')
                """,
                {"pid": pol.id},
            )

    with pytest.raises(DatabaseConstraintError):
        with pool_mgr.transaction() as conn:
            conn.execute(
                """
                INSERT INTO policy_chunks (
                    policy_id, document_id, chunk_key, chunk_index, clause_ref,
                    category, page_number, page_start, page_end, content_text
                )
                VALUES (%(pid)s, %(did)s, 'bad_page_range', 0, 'Sec 1', 'Coverage', 5, 5, 2, 'Text')
                """,
                {"pid": pol.id, "did": doc.id},
            )

    # Deleting parent policy cascades document & active_version FK cleanly while preserving job error audit
    with pool_mgr.transaction() as conn:
        conn.execute("DELETE FROM policies WHERE id = %(pid)s", {"pid": pol.id})

    persisted_job = repo.get_ingestion_job(failed_job.id, scope=alice)
    assert persisted_job is not None
    assert persisted_job.policy_id is None
    assert persisted_job.document_id is None
    assert persisted_job.status == "failed"
    assert persisted_job.error_code == "OCR_ENGINE_TIMEOUT"
    assert persisted_job.error_message == "Page 4 OCR timed out safely."
    assert persisted_job.attempt_count == 1

