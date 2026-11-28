from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urlsplit
from uuid import uuid4

from celery import Celery
import fitz
import httpx
import psycopg
import pytest
import redis

from ai_service.config import Settings
from ai_service.embeddings import BgeM3Embedder, OfflineHashEmbedder
from ai_service.evaluator.decision_gate import DeterministicDecisionGate
from ai_service.evaluator.retry_controller import ClosedRetryLoopOrchestrator
from ai_service.evaluator.verifier import EvaluatorAgent
from ai_service.llm.client import LLMClient
from ai_service.llm.generator import GroundedGenerator
from ai_service.llm.providers.nvidia_nemotron import NvidiaNemotronProvider
from ai_service.persistence import (
    AuthorizedScope,
    PolicyKnowledgeRepository,
    PostgresPoolManager,
    close_db_pool,
)
from ai_service.retrieval import (
    BM25Retriever,
    DenseVectorRetriever,
    PersistentHybridRetriever,
)
from ai_service.schemas.structured_output import DecisionType, PolicyChunk
from ai_service.storage import LocalFilesystemDocumentStorage
from ai_service.workers import IngestionStage

_DEFAULT_RUNTIME_DSN = "postgresql://coverageiq_runtime:change_me_runtime@127.0.0.1:55432/coverageiq_db"
_DEFAULT_MIGRATION_DSN = "postgresql://coverageiq_migrator:change_me_migrator@127.0.0.1:55432/coverageiq_db"
_DEFAULT_REDIS_URL = "redis://127.0.0.1:56379/0"


def _resolve_test_endpoints() -> tuple[str, str, str]:
    runtime_dsn = os.getenv("TEST_DATABASE_URL") or _DEFAULT_RUNTIME_DSN
    migration_dsn = os.getenv("TEST_MIGRATION_DATABASE_URL") or _DEFAULT_MIGRATION_DSN
    redis_url = os.getenv("TEST_REDIS_URL") or _DEFAULT_REDIS_URL
    return runtime_dsn, migration_dsn, redis_url


def _is_pg_reachable(dsn: str) -> bool:
    try:
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


def _is_redis_reachable(redis_url: str) -> bool:
    try:
        client = redis.Redis.from_url(redis_url, socket_connect_timeout=1.5, socket_timeout=1.5)
        try:
            return bool(client.ping())
        finally:
            client.close()
    except Exception:
        return False


def _scope(user_id: str, owner_scope: str) -> AuthorizedScope:
    return AuthorizedScope(
        user_id=user_id,
        role="user",
        owner_scope=owner_scope,
        allow_catalog_read=False,
        trusted_gateway_context=True,
    )


def _make_policy_pdf_bytes() -> bytes:
    doc = fitz.open()
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text(
        (50, 72),
        "COVERAGEIQ HEALTHCARE POLICY - SECTION 2 BENEFITS\n"
        "Section 2.1 In-Patient Hospitalization\n"
        "In-patient hospitalization expenses for medically necessary treatment requiring at least "
        "twenty-four consecutive hours of hospital confinement are covered up to the Sum Insured, "
        "including nursing care, surgeon fees, diagnostic procedures, and prescribed pharmaceuticals.\n\n"
        "Section 2.2 Room Rent and ICU Sub-Limits\n"
        "Standard room rent expenses are covered up to 1% of Sum Insured per day and Intensive "
        "Care Unit charges are covered up to 2% of Sum Insured per day.",
    )
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text(
        (50, 72),
        "SECTION 4 GENERAL EXCLUSIONS\n"
        "Section 4.1 Cosmetic Treatment Exclusion\n"
        "Cosmetic or aesthetic plastic surgery is strictly excluded unless necessitated by "
        "accidental traumatic injury or medically certified burn reconstruction.\n\n"
        "SECTION 5 WAITING PERIODS\n"
        "Section 5.3 Maternity Benefit\n"
        "Maternity delivery expenses are covered subject to a continuous 24 month waiting period.",
    )
    data = doc.tobytes()
    doc.close()
    return data


def _make_scanned_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 400, 600), 0)
    pix.set_rect(pix.irect, (238, 238, 238))
    page.insert_image(fitz.Rect(50, 50, 545, 792), stream=pix.tobytes())
    data = doc.tobytes()
    doc.close()
    return data


def _make_celery_client(redis_url: str) -> Celery:
    app = Celery("coverageiq_test_client", broker=redis_url, backend=redis_url)
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_always_eager=False,
    )
    return app


@contextmanager
def _running_celery_worker(
    *,
    repo_root: Path,
    env_overrides: Dict[str, str],
    redis_url: str,
    worker_name: str,
    bootstrap_code: Optional[str] = None,
) -> Iterator[subprocess.Popen[str]]:
    env = {**os.environ, **env_overrides}
    if bootstrap_code is None:
        cmd = [
            sys.executable,
            "-m",
            "celery",
            "-A",
            "ai_service.workers.celery_app:celery_app",
            "worker",
            "-P",
            "solo",
            "-c",
            "1",
            "-Q",
            "ingestion,ocr",
            "--loglevel=INFO",
            "-n",
            worker_name,
        ]
    else:
        cmd = [sys.executable, "-c", bootstrap_code]

    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    client = _make_celery_client(redis_url)
    deadline = time.monotonic() + 30.0
    ready = False
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                output = proc.stdout.read() if proc.stdout else ""
                raise RuntimeError(
                    f"Celery worker '{worker_name}' exited prematurely with code {proc.returncode}:\n{output}"
                )
            try:
                reply = client.control.inspect(destination=[worker_name], timeout=1.0).ping()
                if reply and worker_name in reply:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.4)

        if not ready:
            raise TimeoutError(f"Celery worker '{worker_name}' did not respond to ping within 30s.")

        yield proc
    finally:
        client.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        if proc.stdout is not None:
            proc.stdout.close()


def _verdict_json(*, verdict: str, confidence: float, evidence_supported: bool, claim_verdicts: list[dict]) -> str:
    return json.dumps(
        {
            "verdict": verdict,
            "confidence_score": confidence,
            "evidence_supported": evidence_supported,
            "contradiction_found": False,
            "completeness_passed": True,
            "claim_verdicts": claim_verdicts,
            "issues": [],
            "summary": f"Verifier verdict: {verdict}",
        }
    )


@pytest.fixture(scope="module")
def pg_redis_env():
    runtime_dsn, migration_dsn, redis_url = _resolve_test_endpoints()
    if not _is_pg_reachable(migration_dsn):
        pytest.skip(
            "Real PostgreSQL 16 + pgvector test instance is not reachable; start pgvector/pgvector:pg16 before running."
        )
    if not _is_redis_reachable(redis_url):
        pytest.skip(
            "Real Redis broker/backend is not reachable; start redis:7-alpine before running."
        )

    repo_root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "MIGRATION_DATABASE_URL": migration_dsn}
    subprocess.run(
        ["npm", "--prefix", str(repo_root / "server"), "run", "migrate:up"],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    return {
        "runtime_dsn": runtime_dsn,
        "migration_dsn": migration_dsn,
        "redis_url": redis_url,
        "repo_root": repo_root,
    }


@pytest.fixture()
def pg_redis_ctx(pg_redis_env, tmp_path: Path):
    with psycopg.connect(pg_redis_env["migration_dsn"]) as conn:
        with conn.transaction():
            conn.execute(
                "TRUNCATE TABLE chunk_embeddings, policy_chunks, ingestion_jobs, policy_documents, policies CASCADE"
            )

    r_client = redis.Redis.from_url(pg_redis_env["redis_url"])
    r_client.flushdb()

    pool_mgr = PostgresPoolManager(dsn=pg_redis_env["runtime_dsn"], min_size=1, max_size=4, timeout=10.0)
    pool_mgr.open(wait=True)
    repo = PolicyKnowledgeRepository(pool_mgr)
    upload_root = tmp_path / "uploads"
    upload_root.mkdir(parents=True, exist_ok=True)
    storage = LocalFilesystemDocumentStorage(root_dir=upload_root)
    try:
        yield repo, pool_mgr, storage, upload_root, pg_redis_env
    finally:
        r_client.flushdb()
        r_client.close()
        close_db_pool()
        pool_mgr.close()


def _seed_queued_document_and_job(
    repo: PolicyKnowledgeRepository,
    *,
    external_policy_key: str,
    owner_id: str = "alice",
    storage_path: str,
    file_hash_sha256: str,
    file_size_bytes: int,
):
    scope = _scope(owner_id, f"user:{owner_id}")
    policy = repo.upsert_policy(
        external_policy_key=external_policy_key,
        title=f"{external_policy_key} Policy",
        owner_id=owner_id,
        scope=scope,
        policy_type="health",
        visibility="private",
        status="draft",
    )
    doc = repo.create_document(
        policy_id=policy.id,
        version_number=1,
        file_name=Path(storage_path).name,
        storage_path=storage_path,
        file_size_bytes=file_size_bytes,
        file_hash_sha256=file_hash_sha256,
        ingestion_status="queued",
        scope=scope,
    )
    job = repo.create_ingestion_job(
        policy_id=policy.id,
        document_id=doc.id,
        external_policy_key=external_policy_key,
        document_version_snapshot=1,
        file_hash_sha256=file_hash_sha256,
        status="queued",
        current_stage=IngestionStage.QUEUED.value,
        scope=scope,
    )
    return policy, doc, job, scope


# =============================================================================
# 1. Real Celery Worker + Redis + PostgreSQL End-to-End Ingestion & Q&A Query
# =============================================================================
def test_real_celery_worker_digital_pdf_end_to_end_persists_and_enables_qa(pg_redis_ctx):
    repo, _, storage, upload_root, env = pg_redis_ctx
    pdf_bytes = _make_policy_pdf_bytes()
    sha = hashlib.sha256(pdf_bytes).hexdigest()
    storage_key = "alice/pol_async_health/v1.pdf"
    storage.put_bytes(storage_key, pdf_bytes)

    policy, doc, job, alice = _seed_queued_document_and_job(
        repo,
        external_policy_key="pol_async_health",
        owner_id="alice",
        storage_path=storage_key,
        file_hash_sha256=sha,
        file_size_bytes=len(pdf_bytes),
    )

    worker_name = f"ciq-digital-{uuid4().hex[:6]}@localhost"
    worker_env = {
        "APP_ENV": "development",
        "DEBUG": "true",
        "USE_MOCK_STORAGE": "false",
        "DATABASE_URL": env["runtime_dsn"],
        "REDIS_URL": env["redis_url"],
        "CELERY_BROKER_URL": env["redis_url"],
        "CELERY_RESULT_BACKEND": env["redis_url"],
        "POLICY_UPLOAD_ROOT": str(upload_root),
        "EMBEDDING_PROVIDER": "offline",
        "LLM_PROVIDER": "offline",
        "CELERY_WORKER_CONCURRENCY": "1",
        "CELERY_INGESTION_EMBEDDING_BATCH_SIZE": "1",
        "CELERY_TASK_MAX_RETRIES": "2",
    }

    payload = {
        "job_id": str(job.id),
        "policy_id": str(policy.id),
        "document_id": str(doc.id),
        "owner_scope": "user:alice",
        "storage_ref": storage_key,
        "expected_sha256": sha,
        "request_id": "req-e2e-001",
    }

    with _running_celery_worker(
        repo_root=env["repo_root"],
        env_overrides=worker_env,
        redis_url=env["redis_url"],
        worker_name=worker_name,
    ):
        client = _make_celery_client(env["redis_url"])
        try:
            async_res = client.send_task(
                "coverageiq.ingestion.ingest_policy_document",
                args=[payload],
                queue="ingestion",
            )
            result = async_res.get(timeout=45)

            assert result["status"] == "completed"
            assert result["stage"] == "COMPLETED"
            assert result["page_count"] == 2
            assert result["ocr_pages_count"] == 0
            assert result["chunk_count"] >= 2
            assert result["embedding_dim"] == 1024
            assert result["model_name"] == "offline-sha256-hash"

            # Verify PostgreSQL persistence state
            db_job = repo.get_ingestion_job(job.id, scope=alice)
            assert db_job is not None
            assert db_job.status == "completed"
            assert db_job.current_stage == "COMPLETED"
            assert db_job.attempt_count == 1
            assert db_job.celery_task_id == async_res.id
            assert db_job.processed_chunks == db_job.total_chunks == result["chunk_count"]
            assert db_job.error_code is None
            assert db_job.started_at is not None and db_job.completed_at is not None

            db_doc = repo.get_document(doc.id, scope=alice)
            assert db_doc is not None
            assert db_doc.ingestion_status == "completed"
            assert db_doc.page_count == 2

            db_policy = repo.get_policy(policy.id, scope=alice)
            assert db_policy is not None
            assert db_policy.active_version_id == doc.id
            assert db_policy.status == "active"

            persisted_chunks = repo.get_policy_chunks(policy_id=policy.id, scope=alice)
            assert len(persisted_chunks) == result["chunk_count"]

            # Re-dispatch identical task over Redis -> idempotent no-op, zero duplicate chunks
            dup_res = client.send_task(
                "coverageiq.ingestion.ingest_policy_document",
                args=[payload],
                queue="ingestion",
            ).get(timeout=30)
            assert dup_res["already_completed"] is True
            assert len(repo.get_policy_chunks(policy_id=policy.id, scope=alice)) == result["chunk_count"]
        finally:
            client.close()

    # Immediately query the Celery-ingested policy through PersistentHybridRetriever + ClosedRetryLoopOrchestrator
    embedder = OfflineHashEmbedder(dimension=1024, normalize=True, model_name="offline-sha256-hash")
    persistent = PersistentHybridRetriever(
        repository=repo,
        dense_retriever=DenseVectorRetriever(embedder=embedder),
        bm25_retriever=BM25Retriever(),
        embedding_model_name="offline-sha256-hash",
    )

    retrieved_chunk_id = [persisted_chunks[0].chunk_id]

    class _CapturingGenerator(GroundedGenerator):
        def generate_draft(self, query: str, evidence: List[PolicyChunk]):
            if evidence:
                retrieved_chunk_id[0] = evidence[0].chunk_id
            return super().generate_draft(query, evidence)

    def gen_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "decision": "COVERED",
                            "reason": "In-patient hospitalization for minimum 24 consecutive hours is covered.",
                            "conditions": ["Minimum 24 consecutive hours confinement"],
                            "claims": [{
                                "text": "In-patient hospitalization for 24 consecutive hours is covered.",
                                "chunk_ids": [retrieved_chunk_id[0]],
                                "claim_type": "coverage",
                            }],
                        }),
                    }
                }]
            },
        )

    def ver_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": _verdict_json(
                            verdict="PASS",
                            confidence=0.95,
                            evidence_supported=True,
                            claim_verdicts=[{
                                "claim_idx": 0,
                                "status": "SUPPORTED",
                                "supported": True,
                                "cited_chunk_ids_valid": True,
                                "reason": "Confirmed by ingested policy chunk.",
                            }],
                        ),
                    }
                }]
            },
        )

    gen_http = httpx.Client(transport=httpx.MockTransport(gen_handler))
    ver_http = httpx.Client(transport=httpx.MockTransport(ver_handler))
    try:
        orch = ClosedRetryLoopOrchestrator(
            generator=_CapturingGenerator(
                client=LLMClient(
                    provider="nvidia",
                    nvidia_provider=NvidiaNemotronProvider(api_key="nvapi-test", http_client=gen_http),
                )
            ),
            evaluator=EvaluatorAgent(
                client=LLMClient(
                    provider="nvidia",
                    nvidia_provider=NvidiaNemotronProvider(api_key="nvapi-test", http_client=ver_http),
                )
            ),
            decision_gate=DeterministicDecisionGate(),
            persistent_retriever=persistent,
            app_settings=Settings(),
        )
        qa = orch.execute_qa("Is in-patient hospitalization covered?", str(policy.id), scope=alice)
        assert qa.decision == DecisionType.COVERED
        assert qa.evaluator_status == "VERIFIED_PASS"
        assert qa.citations[0].chunk_id == retrieved_chunk_id[0]
        assert qa.metadata is not None and qa.metadata["storage_mode"] == "postgresql_pgvector"
    finally:
        gen_http.close()
        ver_http.close()


# =============================================================================
# 2. Permanent Failures & Cross-Tenant Tampering Over Real Redis + PG16
# =============================================================================
def test_real_celery_worker_permanent_failures_mark_failed_without_retry(pg_redis_ctx):
    repo, _, storage, upload_root, env = pg_redis_ctx
    worker_name = f"ciq-perm-{uuid4().hex[:6]}@localhost"
    worker_env = {
        "APP_ENV": "development",
        "DEBUG": "true",
        "USE_MOCK_STORAGE": "false",
        "DATABASE_URL": env["runtime_dsn"],
        "REDIS_URL": env["redis_url"],
        "CELERY_BROKER_URL": env["redis_url"],
        "CELERY_RESULT_BACKEND": env["redis_url"],
        "POLICY_UPLOAD_ROOT": str(upload_root),
        "EMBEDDING_PROVIDER": "offline",
        "LLM_PROVIDER": "offline",
        "CELERY_TASK_MAX_RETRIES": "3",
        "CELERY_TASK_RETRY_BACKOFF_BASE": "1",
    }

    with _running_celery_worker(
        repo_root=env["repo_root"],
        env_overrides=worker_env,
        redis_url=env["redis_url"],
        worker_name=worker_name,
    ):
        client = _make_celery_client(env["redis_url"])
        try:
            # A. Invalid PDF signature (never retries -> attempt_count == 1)
            bad_bytes = b"not a pdf signature at all"
            bad_sha = hashlib.sha256(bad_bytes).hexdigest()
            storage.put_bytes("alice/bad_sig.pdf", bad_bytes)
            pol_a, doc_a, job_a, alice = _seed_queued_document_and_job(
                repo,
                external_policy_key="pol_bad_sig",
                owner_id="alice",
                storage_path="alice/bad_sig.pdf",
                file_hash_sha256=bad_sha,
                file_size_bytes=len(bad_bytes),
            )
            res_a = client.send_task(
                "coverageiq.ingestion.ingest_policy_document",
                args=[{
                    "job_id": str(job_a.id),
                    "policy_id": str(pol_a.id),
                    "document_id": str(doc_a.id),
                    "owner_scope": "user:alice",
                }],
                queue="ingestion",
            )
            with pytest.raises(Exception):
                res_a.get(timeout=30)

            db_job_a = repo.get_ingestion_job(job_a.id, scope=alice)
            assert db_job_a is not None
            assert db_job_a.status == "failed"
            assert db_job_a.current_stage == "FAILED"
            assert db_job_a.error_code == "INVALID_PDF_SIGNATURE"
            assert db_job_a.attempt_count == 1
            assert repo.get_policy_chunks(policy_id=pol_a.id, scope=alice) == []

            # B. Missing file in storage
            good_bytes = _make_policy_pdf_bytes()
            good_sha = hashlib.sha256(good_bytes).hexdigest()
            pol_b, doc_b, job_b, _ = _seed_queued_document_and_job(
                repo,
                external_policy_key="pol_missing_file",
                owner_id="alice",
                storage_path="alice/never_written.pdf",
                file_hash_sha256=good_sha,
                file_size_bytes=len(good_bytes),
            )
            res_b = client.send_task(
                "coverageiq.ingestion.ingest_policy_document",
                args=[{
                    "job_id": str(job_b.id),
                    "policy_id": str(pol_b.id),
                    "document_id": str(doc_b.id),
                    "owner_scope": "user:alice",
                }],
                queue="ingestion",
            )
            with pytest.raises(Exception):
                res_b.get(timeout=30)
            db_job_b = repo.get_ingestion_job(job_b.id, scope=alice)
            assert db_job_b is not None
            assert db_job_b.status == "failed"
            assert db_job_b.error_code == "DOCUMENT_FILE_MISSING"
            assert db_job_b.attempt_count == 1

            # C. SHA-256 mismatch on disk
            storage.put_bytes("alice/tampered_disk.pdf", good_bytes)
            pol_c, doc_c, job_c, _ = _seed_queued_document_and_job(
                repo,
                external_policy_key="pol_sha_mismatch",
                owner_id="alice",
                storage_path="alice/tampered_disk.pdf",
                file_hash_sha256="0" * 64,
                file_size_bytes=len(good_bytes),
            )
            res_c = client.send_task(
                "coverageiq.ingestion.ingest_policy_document",
                args=[{
                    "job_id": str(job_c.id),
                    "policy_id": str(pol_c.id),
                    "document_id": str(doc_c.id),
                    "owner_scope": "user:alice",
                }],
                queue="ingestion",
            )
            with pytest.raises(Exception):
                res_c.get(timeout=30)
            db_job_c = repo.get_ingestion_job(job_c.id, scope=alice)
            assert db_job_c is not None
            assert db_job_c.status == "failed"
            assert db_job_c.error_code == "SHA256_MISMATCH"
            assert db_job_c.attempt_count == 1

            # D. Forged cross-tenant scope rejected BEFORE touching Alice's queued job
            storage.put_bytes("alice/protected.pdf", good_bytes)
            pol_d, doc_d, job_d, _ = _seed_queued_document_and_job(
                repo,
                external_policy_key="pol_cross_tenant",
                owner_id="alice",
                storage_path="alice/protected.pdf",
                file_hash_sha256=good_sha,
                file_size_bytes=len(good_bytes),
            )
            res_d = client.send_task(
                "coverageiq.ingestion.ingest_policy_document",
                args=[{
                    "job_id": str(job_d.id),
                    "policy_id": str(pol_d.id),
                    "document_id": str(doc_d.id),
                    "owner_scope": "user:bob",
                }],
                queue="ingestion",
            )
            with pytest.raises(Exception):
                res_d.get(timeout=30)

            untouched = repo.get_ingestion_job(job_d.id, scope=alice)
            assert untouched is not None
            assert untouched.status == "queued"
            assert untouched.attempt_count == 0
        finally:
            client.close()


# =============================================================================
# 3. Scanned PDF OCR Path & Transient Retry Recovery Over Real Redis + PG16
# =============================================================================
def test_real_celery_worker_scanned_ocr_and_transient_retry_recovery_over_redis(pg_redis_ctx, tmp_path: Path):
    repo, _, storage, upload_root, env = pg_redis_ctx
    scanned_pdf = _make_scanned_pdf_bytes()
    sha = hashlib.sha256(scanned_pdf).hexdigest()
    storage_key = "alice/pol_scanned_retry/v1.pdf"
    storage.put_bytes(storage_key, scanned_pdf)

    policy, doc, job, alice = _seed_queued_document_and_job(
        repo,
        external_policy_key="pol_scanned_retry",
        owner_id="alice",
        storage_path=storage_key,
        file_hash_sha256=sha,
        file_size_bytes=len(scanned_pdf),
    )

    fault_marker = tmp_path / "transient_fault_triggered.flag"
    worker_name = f"ciq-retry-{uuid4().hex[:6]}@localhost"
    bootstrap_code = f"""
from pathlib import Path
from ai_service.ingestion import MockOCREngine
from ai_service.persistence import DatabaseConnectionError, PolicyKnowledgeRepository
from ai_service.workers.celery_app import WorkerResourceManager, celery_app

_marker = Path({str(fault_marker)!r})
_orig_ocr = WorkerResourceManager.get_ocr_engine
_orig_complete = PolicyKnowledgeRepository.complete_document_ingestion

def _test_ocr(self):
    return MockOCREngine(mock_boxes=[
        ("Section 3.1 Scanned Coverage Clause", 0.93, (60.0, 140.0, 520.0, 170.0)),
        (
            "Accidental hospitalization medical expenses are covered up to Sum Insured after emergency admission for twenty-four hours.",
            0.95,
            (60.0, 190.0, 520.0, 260.0),
        ),
    ])

def _fault_once(self, *args, **kwargs):
    if not _marker.exists():
        _marker.write_text("tripped", encoding="utf-8")
        raise DatabaseConnectionError("Simulated transient PostgreSQL disconnect during commit")
    return _orig_complete(self, *args, **kwargs)

WorkerResourceManager.get_ocr_engine = _test_ocr
PolicyKnowledgeRepository.complete_document_ingestion = _fault_once
celery_app.worker_main(["worker", "-P", "solo", "-c", "1", "-Q", "ingestion,ocr", "--loglevel=INFO", "-n", {worker_name!r}])
"""

    worker_env = {
        "APP_ENV": "development",
        "DEBUG": "true",
        "USE_MOCK_STORAGE": "false",
        "DATABASE_URL": env["runtime_dsn"],
        "REDIS_URL": env["redis_url"],
        "CELERY_BROKER_URL": env["redis_url"],
        "CELERY_RESULT_BACKEND": env["redis_url"],
        "POLICY_UPLOAD_ROOT": str(upload_root),
        "EMBEDDING_PROVIDER": "offline",
        "LLM_PROVIDER": "offline",
        "CELERY_WORKER_CONCURRENCY": "1",
        "CELERY_INGESTION_EMBEDDING_BATCH_SIZE": "1",
        "CELERY_TASK_MAX_RETRIES": "2",
        "CELERY_TASK_RETRY_BACKOFF_BASE": "1",
        "CELERY_TASK_RETRY_BACKOFF_MAX": "2",
    }

    with _running_celery_worker(
        repo_root=env["repo_root"],
        env_overrides=worker_env,
        redis_url=env["redis_url"],
        worker_name=worker_name,
        bootstrap_code=bootstrap_code,
    ):
        client = _make_celery_client(env["redis_url"])
        try:
            async_res = client.send_task(
                "coverageiq.ingestion.ingest_policy_document",
                args=[{
                    "job_id": str(job.id),
                    "policy_id": str(policy.id),
                    "document_id": str(doc.id),
                    "owner_scope": "user:alice",
                }],
                queue="ingestion",
            )
            result = async_res.get(timeout=45)
        finally:
            client.close()

    assert fault_marker.exists()
    assert result["status"] == "completed"
    assert result["stage"] == "COMPLETED"
    assert result["ocr_pages_count"] == 1
    assert result["chunk_count"] >= 1

    db_job = repo.get_ingestion_job(job.id, scope=alice)
    assert db_job is not None
    assert db_job.status == "completed"
    assert db_job.current_stage == "COMPLETED"
    assert db_job.attempt_count == 2  # Failed attempt 1 -> Redis retry -> Completed attempt 2

    chunks = repo.get_policy_chunks(policy_id=policy.id, scope=alice)
    assert len(chunks) == result["chunk_count"]
    assert any("Accidental hospitalization" in c.content_text for c in chunks)


# =============================================================================
# 4. Real Cached BAAI/bge-m3 Celery Worker Ingestion Smoke (When Cached)
# =============================================================================
def _is_bge_m3_cached() -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download("BAAI/bge-m3", local_files_only=True)
        return True
    except Exception:
        return False


def test_real_bge_m3_worker_ingestion_smoke_when_cached(pg_redis_ctx):
    if not _is_bge_m3_cached():
        pytest.skip("BAAI/bge-m3 model snapshot is not cached locally; skipping live BGE-M3 worker smoke test.")

    repo, pool_mgr, storage, upload_root, env = pg_redis_ctx
    pdf_bytes = _make_policy_pdf_bytes()
    sha = hashlib.sha256(pdf_bytes).hexdigest()
    storage_key = "alice/pol_bge_m3_live/v1.pdf"
    storage.put_bytes(storage_key, pdf_bytes)

    policy, doc, job, alice = _seed_queued_document_and_job(
        repo,
        external_policy_key="pol_bge_m3_live",
        owner_id="alice",
        storage_path=storage_key,
        file_hash_sha256=sha,
        file_size_bytes=len(pdf_bytes),
    )

    worker_name = f"ciq-bgem3-{uuid4().hex[:6]}@localhost"
    worker_env = {
        "APP_ENV": "development",
        "DEBUG": "true",
        "USE_MOCK_STORAGE": "false",
        "DATABASE_URL": env["runtime_dsn"],
        "REDIS_URL": env["redis_url"],
        "CELERY_BROKER_URL": env["redis_url"],
        "CELERY_RESULT_BACKEND": env["redis_url"],
        "POLICY_UPLOAD_ROOT": str(upload_root),
        "EMBEDDING_PROVIDER": "bge-m3",
        "EMBEDDING_MODEL_NAME": "BAAI/bge-m3",
        "EMBEDDING_DIM": "1024",
        "EMBEDDING_DEVICE": "auto",
        "EMBEDDING_LOCAL_FILES_ONLY": "true",
        "LLM_PROVIDER": "offline",
        "CELERY_WORKER_CONCURRENCY": "1",
        "CELERY_INGESTION_EMBEDDING_BATCH_SIZE": "1",
    }

    with _running_celery_worker(
        repo_root=env["repo_root"],
        env_overrides=worker_env,
        redis_url=env["redis_url"],
        worker_name=worker_name,
    ):
        client = _make_celery_client(env["redis_url"])
        try:
            async_res = client.send_task(
                "coverageiq.ingestion.ingest_policy_document",
                args=[{
                    "job_id": str(job.id),
                    "policy_id": str(policy.id),
                    "document_id": str(doc.id),
                    "owner_scope": "user:alice",
                }],
                queue="ingestion",
            )
            result = async_res.get(timeout=120)
        finally:
            client.close()

    assert result["status"] == "completed"
    assert result["model_name"] == "BAAI/bge-m3"
    assert result["embedding_dim"] == 1024
    assert result["chunk_count"] >= 2

    # Verify stored pgvector vectors directly and query via real BgeM3Embedder
    with pool_mgr.connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS cnt, MIN(embedding_dim) AS min_d, MAX(embedding_dim) AS max_d
            FROM chunk_embeddings
            WHERE document_id = %(doc_id)s AND model_name = 'BAAI/bge-m3'
            """,
            {"doc_id": doc.id},
        ).fetchone()
        assert row is not None
        assert int(row["cnt"]) == result["chunk_count"]
        assert int(row["min_d"]) == 1024 and int(row["max_d"]) == 1024

    bge_emb = BgeM3Embedder(local_files_only=True, batch_size=1)
    try:
        retriever = PersistentHybridRetriever(
            repository=repo,
            dense_retriever=DenseVectorRetriever(embedder=bge_emb),
            bm25_retriever=BM25Retriever(),
            embedding_model_name="BAAI/bge-m3",
        )
        hits = retriever.retrieve(query="Is cosmetic surgery excluded?", policy_id=str(policy.id), scope=alice, top_k=3)
        assert len(hits.chunks) >= 1
        assert any("Cosmetic" in c.content_text or "cosmetic" in c.content_text for c in hits.chunks)
    finally:
        try:
            bge_emb.close()
        except Exception:
            pass
        del bge_emb
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
