from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Dict, Iterator
from urllib.parse import urlsplit
from uuid import uuid4

from celery import Celery
from fastapi.testclient import TestClient
import fitz
import psycopg
import pytest
import redis

from ai_service.main import app
from ai_service.persistence import (
    AuthorizedScope,
    PolicyKnowledgeRepository,
    PostgresPoolManager,
    close_db_pool,
)
from ai_service.storage import LocalFilesystemDocumentStorage
from ai_service.workers import IngestionStage

_DEFAULT_RUNTIME_DSN = "postgresql://coverageiq_runtime:change_me_runtime@127.0.0.1:55432/coverageiq_db"
_DEFAULT_MIGRATION_DSN = "postgresql://coverageiq_migrator:change_me_migrator@127.0.0.1:55432/coverageiq_db"
_DEFAULT_REDIS_URL = "redis://127.0.0.1:56379/0"


def _resolve_endpoints() -> tuple[str, str, str]:
    return (
        os.getenv("TEST_DATABASE_URL") or _DEFAULT_RUNTIME_DSN,
        os.getenv("TEST_MIGRATION_DATABASE_URL") or _DEFAULT_MIGRATION_DSN,
        os.getenv("TEST_REDIS_URL") or _DEFAULT_REDIS_URL,
    )


def _is_pg_reachable(dsn: str) -> bool:
    try:
        parsed = urlsplit(dsn)
        with socket.create_connection((parsed.hostname or "127.0.0.1", parsed.port or 5432), timeout=1.5):
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


def _sample_pdf_bytes() -> bytes:
    doc = fitz.open()
    p = doc.new_page(width=595, height=842)
    p.insert_text(
        (50, 72),
        "SECTION 2 COVERAGE BENEFITS\n"
        "Section 2.1 In-Patient Hospitalization\n"
        "Medically necessary hospitalization for at least 24 consecutive hours is covered up to Sum Insured.",
    )
    data = doc.tobytes()
    doc.close()
    return data


@contextmanager
def _running_worker(repo_root: Path, env_overrides: Dict[str, str], redis_url: str, worker_name: str) -> Iterator[None]:
    env = {**os.environ, **env_overrides}
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
        "--loglevel=WARNING",
        "-n",
        worker_name,
    ]
    proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    c = Celery("ciq_enq_probe", broker=redis_url, backend=redis_url)
    deadline = time.monotonic() + 25.0
    try:
        ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                raise RuntimeError(f"Worker exited prematurely ({proc.returncode}): {out}")
            try:
                reply = c.control.inspect(destination=[worker_name], timeout=1.0).ping()
                if reply and worker_name in reply:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.35)
        if not ready:
            raise TimeoutError("Celery worker ping timed out.")
        yield
    finally:
        c.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        if proc.stdout:
            proc.stdout.close()


@pytest.fixture(scope="module")
def pg_redis_env():
    runtime_dsn, migration_dsn, redis_url = _resolve_endpoints()
    if not _is_pg_reachable(migration_dsn):
        pytest.skip("Real PostgreSQL 16 + pgvector is required for internal enqueue integration tests.")
    if not _is_redis_reachable(redis_url):
        pytest.skip("Real Redis is required for internal enqueue integration tests.")

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
def enqueue_ctx(pg_redis_env, tmp_path: Path, monkeypatch):
    with psycopg.connect(pg_redis_env["migration_dsn"]) as conn:
        with conn.transaction():
            conn.execute(
                "TRUNCATE TABLE audit_logs, refresh_tokens, organization_members, organizations, users, "
                "chunk_embeddings, policy_chunks, ingestion_jobs, policy_documents, policies CASCADE"
            )

    r_client = redis.Redis.from_url(pg_redis_env["redis_url"])
    r_client.flushdb()

    upload_root = tmp_path / "uploads"
    upload_root.mkdir(parents=True, exist_ok=True)
    svc_token = "ciq-internal-token-45"

    monkeypatch.setenv("USE_MOCK_STORAGE", "false")
    monkeypatch.setenv("DATABASE_URL", pg_redis_env["runtime_dsn"])
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", svc_token)
    monkeypatch.setenv("REDIS_URL", pg_redis_env["redis_url"])
    monkeypatch.setenv("CELERY_BROKER_URL", pg_redis_env["redis_url"])
    monkeypatch.setenv("CELERY_RESULT_BACKEND", pg_redis_env["redis_url"])
    monkeypatch.setenv("CELERY_INGESTION_QUEUE", "ingestion")
    monkeypatch.setenv("POLICY_UPLOAD_ROOT", str(upload_root))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")

    close_db_pool()
    pool_mgr = PostgresPoolManager(dsn=pg_redis_env["runtime_dsn"], min_size=1, max_size=4, timeout=10.0)
    pool_mgr.open(wait=True)
    repo = PolicyKnowledgeRepository(pool_mgr)
    storage = LocalFilesystemDocumentStorage(root_dir=upload_root)
    try:
        yield repo, storage, upload_root, svc_token, r_client, pg_redis_env
    finally:
        r_client.flushdb()
        r_client.close()
        close_db_pool()
        pool_mgr.close()


def _seed_record_set(
    repo: PolicyKnowledgeRepository,
    storage: LocalFilesystemDocumentStorage,
    *,
    owner_id: str,
    policy_key: str,
    pdf_bytes: bytes,
    status: str = "queued",
    policy_status: str = "draft",
):
    scope = _scope(owner_id, f"user:{owner_id}")
    sha = hashlib.sha256(pdf_bytes).hexdigest()
    storage_key = f"{owner_id}/{policy_key}/v1.pdf"
    storage.put_bytes(storage_key, pdf_bytes)
    policy = repo.upsert_policy(
        external_policy_key=policy_key,
        title=f"{policy_key} Title",
        owner_id=owner_id,
        scope=scope,
        policy_type="health",
        status=policy_status,
    )
    doc = repo.create_document(
        policy_id=policy.id,
        version_number=1,
        file_name="policy.pdf",
        storage_path=storage_key,
        file_size_bytes=len(pdf_bytes),
        file_hash_sha256=sha,
        ingestion_status=status if status in ("pending", "queued", "processing", "completed", "failed", "archived") else "queued",
        scope=scope,
    )
    job = repo.create_ingestion_job(
        policy_id=policy.id,
        document_id=doc.id,
        external_policy_key=policy_key,
        document_version_snapshot=1,
        file_hash_sha256=sha,
        status=status,
        current_stage=IngestionStage.QUEUED.value,
        scope=scope,
    )
    return policy, doc, job, scope


def test_internal_enqueue_auth_and_payload_contract(enqueue_ctx, monkeypatch):
    repo, storage, _, svc_token, _, _ = enqueue_ctx
    pol, doc, job, _ = _seed_record_set(repo, storage, owner_id="alice", policy_key="pol_enq_auth", pdf_bytes=_sample_pdf_bytes())

    with TestClient(app) as client:
        # Missing token -> 403
        r_no_tok = client.post(
            "/internal/ingestion/enqueue",
            json={"job_id": str(job.id), "policy_id": str(pol.id), "document_id": str(doc.id)},
        )
        assert r_no_tok.status_code == 403

        # Wrong token -> 403
        r_bad_tok = client.post(
            "/internal/ingestion/enqueue",
            headers={"X-Service-Token": "wrong-token"},
            json={"job_id": str(job.id), "policy_id": str(pol.id), "document_id": str(doc.id)},
        )
        assert r_bad_tok.status_code == 403

        # Unconfigured server token fails closed -> 403
        monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "")
        r_unset = client.post(
            "/internal/ingestion/enqueue",
            headers={"X-Service-Token": svc_token},
            json={"job_id": str(job.id), "policy_id": str(pol.id), "document_id": str(doc.id)},
        )
        assert r_unset.status_code == 403
        monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", svc_token)

        # Reject extra/forbidden client fields (owner_scope, raw bytes, embeddings, text)
        for bad_field in (
            {"owner_scope": "user:spoofed"},
            {"raw_pdf_bytes": "abc"},
            {"pdf_base64": "abc"},
            {"storage_path": "/etc/passwd"},
            {"embeddings": [[0.1]]},
            {"text": "raw text"},
        ):
            r_extra = client.post(
                "/internal/ingestion/enqueue",
                headers={"X-Service-Token": svc_token},
                json={
                    "job_id": str(job.id),
                    "policy_id": str(pol.id),
                    "document_id": str(doc.id),
                    **bad_field,
                },
            )
            assert r_extra.status_code == 422


def test_internal_enqueue_rejects_missing_inconsistent_or_terminal_records(enqueue_ctx):
    repo, storage, _, svc_token, _, _ = enqueue_ctx
    pdf = _sample_pdf_bytes()
    pol_a, doc_a, job_a, _ = _seed_record_set(repo, storage, owner_id="alice", policy_key="pol_a", pdf_bytes=pdf)
    pol_b, doc_b, _, _ = _seed_record_set(repo, storage, owner_id="bob", policy_key="pol_b", pdf_bytes=pdf)
    pol_arch, doc_arch, job_arch, _ = _seed_record_set(
        repo, storage, owner_id="alice", policy_key="pol_arch", pdf_bytes=pdf, policy_status="archived"
    )
    pol_fail, doc_fail, job_fail, _ = _seed_record_set(
        repo, storage, owner_id="alice", policy_key="pol_fail", pdf_bytes=pdf, status="failed"
    )

    with TestClient(app) as client:
        # Unknown job -> 404
        assert (
            client.post(
                "/internal/ingestion/enqueue",
                headers={"X-Service-Token": svc_token},
                json={"job_id": str(uuid4()), "policy_id": str(pol_a.id), "document_id": str(doc_a.id)},
            ).status_code
            == 404
        )

        # Cross-tenant policy_id mismatch for Alice's job -> 403
        assert (
            client.post(
                "/internal/ingestion/enqueue",
                headers={"X-Service-Token": svc_token},
                json={"job_id": str(job_a.id), "policy_id": str(pol_b.id), "document_id": str(doc_b.id)},
            ).status_code
            == 403
        )

        # Archived policy -> 403
        assert (
            client.post(
                "/internal/ingestion/enqueue",
                headers={"X-Service-Token": svc_token},
                json={"job_id": str(job_arch.id), "policy_id": str(pol_arch.id), "document_id": str(doc_arch.id)},
            ).status_code
            == 403
        )

        # Failed terminal job -> 409
        assert (
            client.post(
                "/internal/ingestion/enqueue",
                headers={"X-Service-Token": svc_token},
                json={"job_id": str(job_fail.id), "policy_id": str(pol_fail.id), "document_id": str(doc_fail.id)},
            ).status_code
            == 409
        )


def test_internal_enqueue_idempotency_and_real_celery_execution(enqueue_ctx):
    repo, storage, upload_root, svc_token, r_client, env = enqueue_ctx
    pol, doc, job, alice = _seed_record_set(
        repo, storage, owner_id="alice", policy_key="pol_enq_e2e", pdf_bytes=_sample_pdf_bytes()
    )

    with TestClient(app) as client:
        req_body = {
            "job_id": str(job.id),
            "policy_id": str(pol.id),
            "document_id": str(doc.id),
            "request_id": "req-py-enq-1",
        }

        # 1st enqueue pushes exactly ONE Celery task to Redis 'ingestion' queue
        r1 = client.post("/internal/ingestion/enqueue", headers={"X-Service-Token": svc_token}, json=req_body)
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["enqueued"] is True
        assert d1["already_enqueued"] is False
        assert d1["task_id"] == f"ingest-{job.id}"
        assert d1["queue"] == "ingestion"
        assert r_client.llen("ingestion") == 1

        # 2nd enqueue is strictly idempotent: zero extra Redis tasks queued
        r2 = client.post("/internal/ingestion/enqueue", headers={"X-Service-Token": svc_token}, json=req_body)
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["enqueued"] is False
        assert d2["already_enqueued"] is True
        assert r_client.llen("ingestion") == 1

        # Start real Celery worker subprocess and let it finish PG ingestion
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
        }
        with _running_worker(env["repo_root"], worker_env, env["redis_url"], f"ciq-enq-w-{uuid4().hex[:6]}@localhost"):
            deadline = time.monotonic() + 30.0
            final_job = None
            while time.monotonic() < deadline:
                cur = repo.get_ingestion_job(job.id, scope=alice)
                if cur and cur.status == "completed":
                    final_job = cur
                    break
                time.sleep(0.3)
            assert final_job is not None and final_job.status == "completed"
            assert final_job.current_stage == "COMPLETED"
            assert len(repo.get_policy_chunks(policy_id=pol.id, scope=alice)) >= 1

        # 3rd call after completion returns idempotent already_completed=True
        r3 = client.post("/internal/ingestion/enqueue", headers={"X-Service-Token": svc_token}, json=req_body)
        assert r3.status_code == 200
        d3 = r3.json()
        assert d3["enqueued"] is False
        assert d3["already_completed"] is True
        assert d3["status"] == "completed"


def test_internal_enqueue_handles_broker_outage_and_recovers_cleanly(enqueue_ctx, monkeypatch):
    repo, storage, _, svc_token, _, env = enqueue_ctx
    pol, doc, job, alice = _seed_record_set(
        repo, storage, owner_id="alice", policy_key="pol_enq_outage", pdf_bytes=_sample_pdf_bytes()
    )

    with TestClient(app) as client:
        req_body = {
            "job_id": str(job.id),
            "policy_id": str(pol.id),
            "document_id": str(doc.id),
            "request_id": "req-outage-1",
        }

        # Point Celery broker at unreachable Redis port
        monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:59998/0")
        monkeypatch.setenv("CELERY_BROKER_URL", "redis://127.0.0.1:59998/0")
        monkeypatch.setenv("CELERY_RESULT_BACKEND", "redis://127.0.0.1:59998/0")

        r_fail = client.post("/internal/ingestion/enqueue", headers={"X-Service-Token": svc_token}, json=req_body)
        assert r_fail.status_code == 503

        outage_job = repo.get_ingestion_job(job.id, scope=alice)
        assert outage_job is not None
        assert outage_job.status == "queued"
        assert outage_job.current_stage == "ENQUEUE_RETRY_PENDING"
        assert outage_job.error_code == "ENQUEUE_BROKER_UNAVAILABLE"
        assert outage_job.celery_task_id is None  # Critical: not falsely marked dispatched!

        # Restore working Redis broker and re-enqueue idempotently
        monkeypatch.setenv("REDIS_URL", env["redis_url"])
        monkeypatch.setenv("CELERY_BROKER_URL", env["redis_url"])
        monkeypatch.setenv("CELERY_RESULT_BACKEND", env["redis_url"])

        r_recover = client.post("/internal/ingestion/enqueue", headers={"X-Service-Token": svc_token}, json=req_body)
        assert r_recover.status_code == 200
        assert r_recover.json()["enqueued"] is True

        recovered_job = repo.get_ingestion_job(job.id, scope=alice)
        assert recovered_job is not None
        assert recovered_job.status == "queued"
        assert recovered_job.current_stage == "QUEUED"
        assert recovered_job.celery_task_id == f"ingest-{job.id}"
        assert recovered_job.error_code is None
