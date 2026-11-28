from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import socket
import sys
from unittest.mock import MagicMock
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi.testclient import TestClient
import fitz
import numpy as np
import psycopg
import pytest

from ai_service.config import Settings
from ai_service.embeddings import (
    BgeM3Embedder,
    EmbeddingConfigurationError,
    EmbeddingCudaOutOfMemoryError,
    EmbeddingError,
    EmbeddingInferenceError,
    EmbeddingModelLoadError,
    EmbeddingOutputValidationError,
    create_embedder,
)
from ai_service.ingestion import (
    CorruptedPDFError,
    EmptyDocumentError,
    InvalidPDFSignatureError,
    PaddleOCREngine,
    PasswordProtectedPDFError,
    PathTraversalError,
    PyMuPDFLoader,
)
from ai_service.main import app
from ai_service.persistence import (
    AuthorizedScope,
    DatabaseConstraintError,
    DatabaseError,
    PolicyKnowledgeRepository,
    PostgresPoolManager,
)
from ai_service.storage import LocalFilesystemDocumentStorage
from ai_service.workers import (
    IngestionPermanentError,
    IngestionTaskPayload,
    WorkerResourceManager,
    execute_ingestion_task,
)

_DEFAULT_RUNTIME_DSN = "postgresql://coverageiq_runtime:change_me_runtime@127.0.0.1:55432/coverageiq_db"
_DEFAULT_MIGRATION_DSN = "postgresql://coverageiq_migrator:change_me_migrator@127.0.0.1:55432/coverageiq_db"


def _resolve_pg_dsns() -> tuple[str, str]:
    return (
        os.getenv("TEST_DATABASE_URL") or _DEFAULT_RUNTIME_DSN,
        os.getenv("TEST_MIGRATION_DATABASE_URL") or _DEFAULT_MIGRATION_DSN,
    )


def _is_pg_reachable(dsn: str) -> bool:
    try:
        parsed = urlsplit(dsn)
        with socket.create_connection((parsed.hostname or "127.0.0.1", parsed.port or 5432), timeout=1.5):
            pass
        with psycopg.connect(dsn, connect_timeout=2) as conn:
            row = conn.execute("SELECT current_setting('server_version_num')::int").fetchone()
            return bool(row and row[0] >= 160000)
    except Exception:
        return False


def _cached_bge_m3_dir() -> Path | None:
    hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-m3"
    if not hub.exists():
        return None
    snapshots = list((hub / "snapshots").glob("*")) if (hub / "snapshots").exists() else []
    return snapshots[0] if snapshots else hub


# =============================================================================
# 1. FastAPI CORS & Input / Internal Token Security Tests
# =============================================================================
def test_fastapi_cors_rejects_untrusted_origin_and_no_wildcard_credentials():
    client = TestClient(app)
    res = client.get("/ai/health", headers={"Origin": "https://evil-attacker.example"})
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") is None
    assert res.headers.get("access-control-allow-credentials") is None


def test_internal_enqueue_rejects_missing_or_invalid_service_token(monkeypatch):
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "expected-secret-token")
    client = TestClient(app)
    payload = {
        "job_id": str(uuid4()),
        "policy_id": str(uuid4()),
        "document_id": str(uuid4()),
    }
    res_missing = client.post("/internal/ingestion/enqueue", json=payload)
    assert res_missing.status_code == 403

    res_wrong = client.post(
        "/internal/ingestion/enqueue",
        json=payload,
        headers={"X-Service-Token": "wrong-token"},
    )
    assert res_wrong.status_code == 403


def test_qa_endpoint_rejects_invalid_and_empty_inputs():
    client = TestClient(app)
    res_empty = client.post("/ai/ask", json={"policy_id": "pol_1", "question": ""})
    assert res_empty.status_code == 422

    res_missing_policy = client.post("/ai/ask", json={"question": "What is covered?"})
    assert res_missing_policy.status_code == 422


# =============================================================================
# 2. PDF Security & OCR Failure-Path Resilience Tests
# =============================================================================
def test_pdf_loader_rejects_empty_corrupt_encrypted_and_traversal_paths(tmp_path):
    loader = PyMuPDFLoader(allowed_root=tmp_path)

    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")
    with pytest.raises(EmptyDocumentError):
        loader.load_document(empty_pdf)

    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"NOT A REAL PDF FILE CONTENT")
    with pytest.raises(InvalidPDFSignatureError):
        loader.load_document(fake_pdf)

    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.7\n%%CORRUPTED-TRAILER-MISSING\n")
    with pytest.raises(CorruptedPDFError):
        loader.load_document(corrupt_pdf)

    enc_pdf = tmp_path / "encrypted.pdf"
    d = fitz.open()
    p = d.new_page()
    p.insert_text((50, 72), "Protected Clause")
    d.save(str(enc_pdf), encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="adm", user_pw="usr")
    d.close()
    with pytest.raises(PasswordProtectedPDFError):
        loader.load_document(enc_pdf)

    outside_file = tmp_path.parent / f"outside_{uuid4().hex}.pdf"
    outside_file.write_bytes(b"%PDF-1.7\n")
    try:
        with pytest.raises(PathTraversalError):
            loader.load_document(outside_file)
    finally:
        outside_file.unlink(missing_ok=True)


def test_paddle_ocr_engine_reports_unavailable_and_handles_empty_or_failing_output():
    engine = PaddleOCREngine()
    doc = fitz.open()
    page = doc.new_page(width=400, height=600)
    try:
        if not engine.is_available():
            res = engine.process_page(page, page_number=1)
            assert res.success is False
            assert res.boxes == []
            assert res.full_text == ""
            assert res.error_message

        # Simulate initialized PaddleOCR returning empty blocks (blank scan)
        empty_stub = MagicMock()
        empty_stub.ocr.return_value = [[]]
        engine_empty = PaddleOCREngine()
        engine_empty._ocr = empty_stub
        engine_empty._is_available = True
        res_empty = engine_empty.process_page(page, page_number=1)
        assert res_empty.success is True
        assert res_empty.boxes == []
        assert res_empty.full_text == ""

        # Simulate runtime OCR failure cleanly trapped without uncaught exception
        boom_stub = MagicMock()
        boom_stub.ocr.side_effect = RuntimeError("Paddle native segfault guard")
        engine_boom = PaddleOCREngine()
        engine_boom._ocr = boom_stub
        engine_boom._is_available = True
        res_boom = engine_boom.process_page(page, page_number=1)
        assert res_boom.success is False
        assert res_boom.error_message and ("Paddle" in res_boom.error_message or "error" in res_boom.error_message.lower())
    finally:
        doc.close()


# =============================================================================
# 3. BGE-M3 Embedding Strictness & Real Cached GPU/CPU Inference Tests
# =============================================================================
def test_bge_m3_strict_validation_no_fallback_and_oom_cache_cleanup(monkeypatch):
    # Empty text rejected
    embedder = BgeM3Embedder(device="cpu", model=MagicMock())
    with pytest.raises((EmbeddingConfigurationError, EmbeddingError, ValueError)):
        embedder.embed_documents(["   "])

    # Wrong vector dimensions rejected strictly
    bad_dim_model = MagicMock()
    bad_dim_model.encode.return_value = {"dense_vecs": np.ones((1, 768), dtype=np.float32)}
    embedder_bad_dim = BgeM3Embedder(device="cpu", model=bad_dim_model)
    with pytest.raises(EmbeddingOutputValidationError):
        embedder_bad_dim.embed_query("Coverage waiting period")

    # Non-finite NaN output rejected strictly
    nan_model = MagicMock()
    arr = np.zeros((1, 1024), dtype=np.float32)
    arr[0, 0] = np.nan
    nan_model.encode.return_value = {"dense_vecs": arr}
    embedder_nan = BgeM3Embedder(device="cpu", model=nan_model)
    with pytest.raises(EmbeddingOutputValidationError):
        embedder_nan.embed_query("Coverage waiting period")

    # CUDA OOM calls torch.cuda.empty_cache() and raises EmbeddingCudaOutOfMemoryError
    oom_model = MagicMock()
    oom_model.encode.side_effect = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    embedder_oom = BgeM3Embedder(device="cpu", model=oom_model)
    empty_cache_calls = []
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(1))
    with pytest.raises(EmbeddingCudaOutOfMemoryError):
        embedder_oom.embed_query("Coverage waiting period")
    assert len(empty_cache_calls) == 1

    # No silent fallback to OfflineHashEmbedder when provider is 'bge-m3'
    monkeypatch.setenv("EMBEDDING_PROVIDER", "bge-m3")
    monkeypatch.setenv("EMBEDDING_LOCAL_FILES_ONLY", "true")
    monkeypatch.setitem(sys.modules, "FlagEmbedding", None)
    strict_embedder = create_embedder(app_settings=Settings())
    assert isinstance(strict_embedder, BgeM3Embedder)
    with pytest.raises(EmbeddingModelLoadError):
        strict_embedder.embed_query("Hospitalization room rent sublimit")


def test_bge_m3_real_cached_cuda_inference_batch_size_1():
    cached = _cached_bge_m3_dir()
    if cached is None:
        pytest.skip("Local BAAI/bge-m3 HuggingFace cache not present.")

    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = BgeM3Embedder(
        model_name="BAAI/bge-m3",
        device=target_device,
        batch_size=1,
        normalize=True,
        local_files_only=True,
    )
    try:
        texts = [
            "Section 2.1 In-patient hospitalization for minimum 24 hours is covered.",
            "Section 4.1 Cosmetic and plastic surgery is permanently excluded.",
        ]
        vecs_run1 = embedder.embed_documents(texts)
        vecs_run2 = embedder.embed_documents(texts)
        q_vec = embedder.embed_query("Is 24 hour hospitalization covered?")

        assert len(vecs_run1) == 2
        assert len(vecs_run1[0]) == 1024
        assert len(q_vec) == 1024

        v0 = np.asarray(vecs_run1[0], dtype=np.float32)
        v1 = np.asarray(vecs_run1[1], dtype=np.float32)
        v0_repeat = np.asarray(vecs_run2[0], dtype=np.float32)
        qv = np.asarray(q_vec, dtype=np.float32)

        assert np.all(np.isfinite(v0)) and np.all(np.isfinite(v1)) and np.all(np.isfinite(qv))
        assert abs(float(np.linalg.norm(v0)) - 1.0) < 1e-4
        assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-4
        assert abs(float(np.linalg.norm(qv)) - 1.0) < 1e-4
        # eval() mode deterministic across identical calls
        assert float(np.max(np.abs(v0 - v0_repeat))) < 1e-4
        # Semantic cosine relevance: hospitalization query closer to clause 0 than exclusion clause 1
        assert float(np.dot(qv, v0)) > float(np.dot(qv, v1))
    finally:
        embedder.close()
        assert embedder.is_loaded is False


# =============================================================================
# 4. Live PostgreSQL 16 + pgvector + Celery Job Claim Concurrency & Tamper Guards
# =============================================================================
def _scope(owner_id: str) -> AuthorizedScope:
    return AuthorizedScope(
        user_id=owner_id,
        role="user",
        owner_scope=f"user:{owner_id}",
        allow_catalog_read=False,
        trusted_gateway_context=True,
    )


def test_live_pg16_concurrent_claim_idempotency_dim_guard_and_tamper_failure(tmp_path):
    runtime_dsn, migration_dsn = _resolve_pg_dsns()
    if not _is_pg_reachable(runtime_dsn):
        pytest.skip("Live PostgreSQL 16 + pgvector not reachable for hardening DB tests.")

    owner_id = f"hardening_{uuid4().hex[:10]}"
    scope = _scope(owner_id)
    cfg = Settings(
        USE_MOCK_STORAGE=False,
        DATABASE_URL=runtime_dsn,
        EMBEDDING_PROVIDER="offline",
        POLICY_UPLOAD_ROOT=str(tmp_path),
    )
    pool_mgr = PostgresPoolManager(dsn=runtime_dsn, min_size=2, max_size=6)
    pool_mgr.open()
    pool_mgr.verify_schema()
    repo = PolicyKnowledgeRepository(pool_mgr)
    policy_id = None

    try:
        policy = repo.upsert_policy(
            external_policy_key=f"pol-hardening-{uuid4().hex[:8]}",
            title="Hardening DB Test Policy",
            owner_id=owner_id,
            scope=scope,
            policy_type="health",
            visibility="private",
            status="draft",
        )
        policy_id = policy.id

        # 1. Valid PDF written + checksum computed
        pdf_path = tmp_path / "policy_v1.pdf"
        d = fitz.open()
        p = d.new_page()
        p.insert_text((50, 72), "Section 1 Hospitalization Coverage is active after waiting period.")
        d.save(str(pdf_path))
        d.close()
        raw_bytes = pdf_path.read_bytes()
        sha_hex = hashlib.sha256(raw_bytes).hexdigest()

        storage = LocalFilesystemDocumentStorage(root_dir=tmp_path)
        storage_key = f"{owner_id}/pol_hardening/v1.pdf"
        storage.put_bytes(storage_key, raw_bytes)

        doc_rec = repo.create_document(
            policy_id=policy.id,
            version_number=1,
            file_name="policy_v1.pdf",
            storage_path=storage_key,
            file_size_bytes=len(raw_bytes),
            file_hash_sha256=sha_hex,
            ingestion_status="queued",
            scope=scope,
        )
        job_rec = repo.create_ingestion_job(
            policy_id=policy.id,
            document_id=doc_rec.id,
            external_policy_key=policy.external_policy_key,
            document_version_snapshot=1,
            file_hash_sha256=sha_hex,
            scope=scope,
        )

        # 2. Concurrent worker claims with two different Celery task IDs -> 1 claimed, 1 duplicate_in_progress
        def _race_claim(task_id: str) -> str:
            status, _, _ = repo.claim_ingestion_job_for_execution(
                job_id=job_rec.id,
                document_id=doc_rec.id,
                task_id=task_id,
                scope=scope,
            )
            return status

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(_race_claim, "celery-task-alpha")
            f2 = pool.submit(_race_claim, "celery-task-beta")
            outcomes = sorted([f1.result(), f2.result()])

        assert outcomes == ["claimed", "duplicate_in_progress"]
        locked_job = repo.get_ingestion_job(job_rec.id, scope=scope)
        assert locked_job is not None and locked_job.attempt_count == 1

        # 3. Completing job then calling claim again returns 'already_completed' without bumping attempts
        repo.transition_ingestion_job_stage(
            job_id=job_rec.id,
            document_id=doc_rec.id,
            scope=scope,
            status="completed",
            stage="COMPLETED",
            doc_status="completed",
            mark_completed_at=True,
        )
        reclaim_status, final_job, _ = repo.claim_ingestion_job_for_execution(
            job_id=job_rec.id,
            document_id=doc_rec.id,
            task_id="celery-task-gamma",
            scope=scope,
        )
        assert reclaim_status == "already_completed"
        assert final_job.attempt_count == 1

        # 4. pgvector column & CHECK constraint reject wrong dimension (!= 1024)
        with pytest.raises((DatabaseConstraintError, DatabaseError, psycopg.Error)):
            with pool_mgr.transaction() as conn:
                chunk_row = conn.execute(
                    """
                    INSERT INTO policy_chunks (
                        policy_id, document_id, chunk_key, chunk_index, clause_ref,
                        category, page_number, page_start, page_end, content_text, token_count
                    ) VALUES (
                        %(pid)s, %(did)s, 'chk-dim-test-0', 0, 'Section 1',
                        'Coverage', 1, 1, 1, 'Sample clause content for vector check', 7
                    )
                    RETURNING id
                    """,
                    {"pid": policy.id, "did": doc_rec.id},
                ).fetchone()
                assert chunk_row is not None
                conn.execute(
                    """
                    INSERT INTO chunk_embeddings (
                        chunk_id, policy_id, document_id, model_name, embedding_dim, is_normalized, embedding
                    ) VALUES (%(cid)s, %(pid)s, %(did)s, 'BAAI/bge-m3', 1024, TRUE, %(vec)s::vector)
                    """,
                    {
                        "cid": chunk_row["id"],
                        "pid": policy.id,
                        "did": doc_rec.id,
                        "vec": "[" + ",".join(["0.1"] * 768) + "]",
                    },
                )

        # 5. Tampered checksum PDF triggers permanent CHECKSUM_MISMATCH failure without crashing worker
        tamper_doc = repo.create_document(
            policy_id=policy.id,
            version_number=2,
            file_name="tampered_v2.pdf",
            storage_path=storage_key,
            file_size_bytes=len(raw_bytes),
            file_hash_sha256="0" * 64,  # deliberately mismatched expected SHA-256
            ingestion_status="queued",
            scope=scope,
        )
        tamper_job = repo.create_ingestion_job(
            policy_id=policy.id,
            document_id=tamper_doc.id,
            external_policy_key=policy.external_policy_key,
            document_version_snapshot=2,
            file_hash_sha256="0" * 64,
            scope=scope,
        )
        res_mgr = WorkerResourceManager(app_settings=cfg)
        task_payload = IngestionTaskPayload(
            job_id=tamper_job.id,
            policy_id=policy.id,
            document_id=tamper_doc.id,
            owner_scope=scope.owner_scope,
            storage_ref=storage_key,
            expected_sha256="0" * 64,
        )
        with pytest.raises(IngestionPermanentError, match="SHA256_MISMATCH"):
            execute_ingestion_task(
                task=MagicMock(request=MagicMock(id="celery-tamper-check", retries=0)),
                raw_payload=task_payload.model_dump(mode="json"),
                resources=res_mgr,
                repository=repo,
            )
        res_mgr.shutdown()

        failed_job = repo.get_ingestion_job(tamper_job.id, scope=scope)
        assert failed_job is not None
        assert failed_job.status == "failed"
        assert failed_job.current_stage == "FAILED"
        assert failed_job.error_code == "SHA256_MISMATCH"
        assert runtime_dsn not in (failed_job.error_message or "")
    finally:
        if policy_id is not None:
            with psycopg.connect(migration_dsn) as admin_conn:
                admin_conn.execute("DELETE FROM ingestion_jobs WHERE policy_id = %s", (policy_id,))
                admin_conn.execute("DELETE FROM policies WHERE id = %s", (policy_id,))
                admin_conn.commit()
        pool_mgr.close()
