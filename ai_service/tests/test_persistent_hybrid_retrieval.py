import json
import logging
import os
from pathlib import Path
import socket
import subprocess
from typing import List

from fastapi.testclient import TestClient
import httpx
import psycopg
import pytest

from ai_service.api import routes_qa
from ai_service.config import Settings
from ai_service.embeddings import OfflineHashEmbedder
from ai_service.evaluator.decision_gate import DeterministicDecisionGate
from ai_service.evaluator.retry_controller import ClosedRetryLoopOrchestrator
from ai_service.evaluator.verifier import EvaluatorAgent
from ai_service.llm.client import LLMClient
from ai_service.llm.generator import GroundedGenerator
from ai_service.llm.providers.nvidia_nemotron import NvidiaNemotronProvider
from ai_service.main import app
from ai_service.persistence import (
    AuthorizedScope,
    DatabaseAuthorizationError,
    PolicyKnowledgeRepository,
    PostgresPoolManager,
    close_db_pool,
    get_db_pool_manager,
)
from ai_service.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseVectorRetriever,
    PersistentHybridRetriever,
    RRFFusion,
)
from ai_service.schemas.structured_output import ChunkCategory, DecisionType, PolicyChunk

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


def _scope(
    user_id: str,
    owner_scope: str,
    role: str = "user",
    *,
    allow_catalog_read: bool = False,
) -> AuthorizedScope:
    return AuthorizedScope(
        user_id=user_id,
        role=role,  # type: ignore[arg-type]
        owner_scope=owner_scope,
        allow_catalog_read=allow_catalog_read,
        trusted_gateway_context=True,
    )


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
def pg16_env():
    runtime_dsn, migration_dsn = _resolve_test_dsns()
    if not _is_pg_reachable(migration_dsn):
        pytest.skip(
            "Real PostgreSQL 16 + pgvector test instance is not reachable; "
            "start pgvector/pgvector:pg16 and run server migrations before running integration tests."
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
    return {"runtime_dsn": runtime_dsn, "migration_dsn": migration_dsn, "repo_root": repo_root}


@pytest.fixture()
def pg_ctx(pg16_env):
    with psycopg.connect(pg16_env["migration_dsn"]) as conn:
        with conn.transaction():
            conn.execute(
                "TRUNCATE TABLE chunk_embeddings, policy_chunks, ingestion_jobs, policy_documents, policies CASCADE"
            )

    pool_mgr = PostgresPoolManager(dsn=pg16_env["runtime_dsn"], min_size=1, max_size=4, timeout=10.0)
    pool_mgr.open(wait=True)
    repo = PolicyKnowledgeRepository(pool_mgr)
    try:
        yield repo, pool_mgr, pg16_env
    finally:
        close_db_pool()
        routes_qa.close_orchestrator()
        pool_mgr.close()


def _seed_sample_policy(
    repo: PolicyKnowledgeRepository,
    *,
    external_policy_key: str = "pol_health_pg",
    owner_id: str = "alice",
    scope: AuthorizedScope | None = None,
    visibility: str = "private",
    status: str = "active",
    model_name: str = "BAAI/bge-m3",
    chunks: List[PolicyChunk] | None = None,
):
    eff_scope = scope or _scope(owner_id, f"user:{owner_id}")
    embedder = OfflineHashEmbedder(dimension=1024, normalize=True, model_name=model_name)
    sample_chunks = chunks or [
        PolicyChunk(
            chunk_id="chk_pg_hosp_01",
            policy_id=external_policy_key,
            clause_ref="Section 2.1 - In-Patient Hospitalization",
            page_number=12,
            page_start=12,
            page_end=13,
            source_block_ids=["blk_p12_1", "blk_p13_2"],
            table_id=None,
            category=ChunkCategory.COVERAGE,
            content_text=(
                "Section 2.1 In-Patient Hospitalization Expenses: Medical expenses incurred for in-patient "
                "hospitalization for minimum 24 consecutive hours are covered up to Sum Insured."
            ),
            metadata={"section_id": "sec_2_1", "extraction_method": "DIGITAL", "token_count": 32},
        ),
        PolicyChunk(
            chunk_id="chk_pg_cosmetic_02",
            policy_id=external_policy_key,
            clause_ref="Section 4.1 - Cosmetic Exclusion",
            page_number=22,
            page_start=22,
            page_end=22,
            source_block_ids=["blk_p22_4"],
            table_id=None,
            category=ChunkCategory.EXCLUSIONS,
            content_text=(
                "Section 4.1 Exclusions: Cosmetic and aesthetic plastic surgery is strictly NOT COVERED "
                "except when medically necessary for burn reconstruction or cancer treatment."
            ),
            metadata={"section_id": "sec_4_1", "extraction_method": "DIGITAL", "token_count": 30},
        ),
        PolicyChunk(
            chunk_id="chk_pg_maternity_03",
            policy_id=external_policy_key,
            clause_ref="Section 5.3 - Maternity Waiting Period",
            page_number=29,
            page_start=29,
            page_end=31,
            source_block_ids=["blk_p29_1", "blk_p30_2", "blk_p31_1"],
            table_id="tbl_maternity_limits",
            category=ChunkCategory.CONDITIONS,
            content_text=(
                "Section 5.3 Maternity Benefit: Normal delivery limit INR 50,000 and caesarean section limit INR 75,000, "
                "subject to a mandatory waiting period of 24 consecutive months continuous coverage."
            ),
            metadata={"section_id": "sec_5_3", "extraction_method": "DIGITAL", "token_count": 38},
        ),
    ]

    pol = repo.upsert_policy(
        external_policy_key=external_policy_key,
        title=f"Policy {external_policy_key}",
        owner_id=owner_id,
        scope=eff_scope,
        policy_type="health",
        visibility=visibility,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
    )
    doc = repo.create_document(
        policy_id=pol.id,
        version_number=1,
        file_name=f"{external_policy_key}.pdf",
        storage_path=f"/uploads/{external_policy_key}.pdf",
        file_size_bytes=4096,
        file_hash_sha256="a" * 64,
        page_count=32,
        ingestion_status="completed",
        scope=eff_scope,
    )
    embedded = embedder.embed_chunks(sample_chunks)
    repo.persist_embedded_chunks(
        policy_id=pol.id,
        document_id=doc.id,
        embedded_chunks=embedded,
        model_name=model_name,
        model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        scope=eff_scope,
    )
    repo.set_active_document_version(policy_id=pol.id, document_id=doc.id, scope=eff_scope)
    return pol, doc, sample_chunks, eff_scope, embedder


# =============================================================================
# 1–4. Dense (pgvector HNSW), Sparse (Python BM25), RRF, Rerank & Provenance
# =============================================================================
def test_persistent_hybrid_components_dense_bm25_rrf_rerank_and_provenance(pg_ctx):
    repo, _, _ = pg_ctx
    pol, doc, _, alice, embedder = _seed_sample_policy(repo)

    active_tx_depth = 0
    tx_entered_during_embed = False
    orig_embed_query = embedder.embed_query
    embed_calls: List[str] = []

    class SpyEmbedder(OfflineHashEmbedder):
        def embed_query(self, text: str) -> List[float]:
            nonlocal tx_entered_during_embed
            if active_tx_depth > 0:
                tx_entered_during_embed = True
            embed_calls.append(text)
            return orig_embed_query(text)

    spy_embedder = SpyEmbedder(dimension=1024, normalize=True, model_name="BAAI/bge-m3")
    dense_ret = DenseVectorRetriever(embedder=spy_embedder)
    bm25_ret = BM25Retriever()
    rrf = RRFFusion(k=60, weight_vector=0.6, weight_bm25=0.4)
    reranker = CrossEncoderReranker()

    fts_called = False
    orig_fts = repo.fts_search

    def forbid_fts(*args, **kwargs):
        nonlocal fts_called
        fts_called = True
        return orig_fts(*args, **kwargs)

    repo.fts_search = forbid_fts  # type: ignore[method-assign]

    bm25_prepare_calls = 0
    orig_prepare = bm25_ret.prepare_corpus
    orig_search_prepared = bm25_ret.search_prepared
    bm25_scored_calls = 0

    def spy_prepare(chunks):
        nonlocal bm25_prepare_calls
        bm25_prepare_calls += 1
        return orig_prepare(chunks)

    def spy_search_prepared(q, chunks, idx, top_n=10):
        nonlocal bm25_scored_calls
        bm25_scored_calls += 1
        return orig_search_prepared(q, chunks, idx, top_n=top_n)

    bm25_ret.prepare_corpus = spy_prepare  # type: ignore[method-assign]
    bm25_ret.search_prepared = spy_search_prepared  # type: ignore[method-assign]

    rrf_calls = 0
    orig_fuse = rrf.fuse

    def spy_fuse(vec_res, bm25_res, top_m=25):
        nonlocal rrf_calls
        rrf_calls += 1
        return orig_fuse(vec_res, bm25_res, top_m=top_m)

    rrf.fuse = spy_fuse  # type: ignore[method-assign]

    rerank_calls = 0
    orig_rerank = reranker.rerank

    def spy_rerank(q, candidates, top_k=5):
        nonlocal rerank_calls
        rerank_calls += 1
        return orig_rerank(q, candidates, top_k=top_k)

    reranker.rerank = spy_rerank  # type: ignore[method-assign]

    from contextlib import contextmanager

    orig_tx = repo.db.transaction

    @contextmanager
    def tracking_tx():
        nonlocal active_tx_depth
        active_tx_depth += 1
        try:
            with orig_tx() as conn:
                yield conn
        finally:
            active_tx_depth -= 1

    repo.db.transaction = tracking_tx  # type: ignore[method-assign]

    retriever = PersistentHybridRetriever(
        repository=repo,
        dense_retriever=dense_ret,
        bm25_retriever=bm25_ret,
        rrf_fusion=rrf,
        reranker=reranker,
        embedding_model_name="BAAI/bge-m3",
        hnsw_ef_search=64,
        hnsw_iterative_scan="relaxed_order",
    )

    result = retriever.retrieve(
        query="Is Section 5.3 maternity delivery 24 months waiting period covered?",
        policy_id=str(pol.id),
        scope=alice,
        top_k=2,
    )

    assert tx_entered_during_embed is False
    assert fts_called is False  # Python BM25Okapi used instead of SQL ts_rank_cd
    assert bm25_prepare_calls == 1
    assert bm25_scored_calls == 1
    assert rrf_calls == 1
    assert rerank_calls == 1
    assert len(embed_calls) == 1
    assert len(result.chunks) == 2
    assert len({c.chunk_id for c in result.chunks}) == 2
    assert result.chunks[0].chunk_id == "chk_pg_maternity_03"
    top_chunk = result.chunks[0]
    assert top_chunk.page_number == 29
    assert top_chunk.page_start == 29
    assert top_chunk.page_end == 31
    assert top_chunk.source_block_ids == ["blk_p29_1", "blk_p30_2", "blk_p31_1"]
    assert top_chunk.table_id == "tbl_maternity_limits"
    assert top_chunk.metadata is not None
    assert top_chunk.metadata["document_id"] == str(doc.id)
    assert "db_chunk_uuid" in top_chunk.metadata
    assert top_chunk.similarity_score is not None
    assert top_chunk.bm25_score is not None
    assert top_chunk.rrf_score is not None
    assert top_chunk.rerank_score is not None
    assert result.latency.total_ms > 0.0
    assert result.latency.dense_pgvector_ms >= 0.0
    assert result.latency.sparse_bm25_ms >= 0.0
    assert result.latency.rrf_fusion_ms >= 0.0
    assert result.latency.rerank_ms >= 0.0


# =============================================================================
# 5 & 6. Full Q&A Pipeline & Retry Expansion Using PostgreSQL
# =============================================================================
def test_orchestrator_qa_pipeline_and_retry_expansion_over_postgres(pg_ctx, monkeypatch):
    repo, pool_mgr, _ = pg_ctx
    alice = _scope("alice", "user:alice")

    ten_chunks = [
        PolicyChunk(
            chunk_id=f"chk_pg_exp_{i:02d}",
            policy_id="pol_pg_expand",
            clause_ref=f"Section {i}.1",
            page_number=i,
            page_start=i,
            page_end=i + 1,
            source_block_ids=[f"blk_{i}"],
            category=ChunkCategory.COVERAGE,
            content_text=(
                f"Section {i}.1 Inpatient treatment clause {i}: Hospitalization for 24 consecutive hours "
                "including surgical ward and room rent is covered."
            ),
        )
        for i in range(1, 11)
    ]
    pol, _, _, _, _ = _seed_sample_policy(
        repo,
        external_policy_key="pol_pg_expand",
        owner_id="alice",
        scope=alice,
        chunks=ten_chunks,
    )

    monkeypatch.setenv("USE_MOCK_STORAGE", "false")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")

    get_chunks_calls = 0
    orig_get_chunks = repo.get_policy_chunks

    def spy_get_chunks(**kwargs):
        nonlocal get_chunks_calls
        get_chunks_calls += 1
        return orig_get_chunks(**kwargs)

    repo.get_policy_chunks = spy_get_chunks  # type: ignore[method-assign]

    bm25_ret = BM25Retriever()
    prepare_calls = 0
    orig_prepare = bm25_ret.prepare_corpus

    def spy_prepare(chunks):
        nonlocal prepare_calls
        prepare_calls += 1
        return orig_prepare(chunks)

    bm25_ret.prepare_corpus = spy_prepare  # type: ignore[method-assign]

    dense_ret = DenseVectorRetriever(embedder=OfflineHashEmbedder(dimension=1024, normalize=True, model_name="BAAI/bge-m3"))
    persistent = PersistentHybridRetriever(
        repository=repo,
        dense_retriever=dense_ret,
        bm25_retriever=bm25_ret,
        embedding_model_name="BAAI/bge-m3",
    )

    evidence_counts: List[int] = []
    latest_chunk_id = ["chk_pg_exp_01"]

    class RecordingGenerator(GroundedGenerator):
        def generate_draft(self, query: str, evidence: List[PolicyChunk]):
            evidence_counts.append(len(evidence))
            if evidence:
                latest_chunk_id[0] = evidence[0].chunk_id
            return super().generate_draft(query, evidence)

    attempt_num = {"val": 0}

    def gen_handler(request: httpx.Request) -> httpx.Response:
        attempt_num["val"] += 1
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "decision": "COVERED",
                            "reason": "Inpatient hospitalization for 24 hours is covered under the policy.",
                            "conditions": ["Minimum 24 hours hospitalization"],
                            "claims": [{
                                "text": "Inpatient hospitalization for 24 hours is covered.",
                                "chunk_ids": [latest_chunk_id[0]],
                                "claim_type": "coverage",
                            }],
                        }),
                    }
                }]
            },
        )

    ver_calls = {"val": 0}

    def ver_handler(request: httpx.Request) -> httpx.Response:
        ver_calls["val"] += 1
        if ver_calls["val"] < 2:
            # Force one verification retry to prove PostgreSQL retry expansion 5 -> 8
            return httpx.Response(
                200,
                json={
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": _verdict_json(
                                verdict="FAIL",
                                confidence=0.35,
                                evidence_supported=False,
                                claim_verdicts=[{
                                    "claim_idx": 0,
                                    "status": "UNSUPPORTED",
                                    "supported": False,
                                    "cited_chunk_ids_valid": True,
                                    "reason": "Need wider evidence context.",
                                }],
                            ),
                        }
                    }]
                },
            )
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
                                "reason": "Verified against retrieved chunk.",
                            }],
                        ),
                    }
                }]
            },
        )

    gen_http = httpx.Client(transport=httpx.MockTransport(gen_handler))
    ver_http = httpx.Client(transport=httpx.MockTransport(ver_handler))
    gen_prov = NvidiaNemotronProvider(api_key="nvapi-test", http_client=gen_http)
    ver_prov = NvidiaNemotronProvider(api_key="nvapi-test", http_client=ver_http)
    orch = ClosedRetryLoopOrchestrator(
        generator=RecordingGenerator(client=LLMClient(provider="nvidia", nvidia_provider=gen_prov)),
        evaluator=EvaluatorAgent(client=LLMClient(provider="nvidia", nvidia_provider=ver_prov)),
        decision_gate=DeterministicDecisionGate(),
        persistent_retriever=persistent,
        app_settings=Settings(),
    )

    try:
        assert orch.kb is None  # MockKnowledgeBase is NOT wired in persistent mode
        res = orch.execute_qa("Is inpatient hospitalization covered?", str(pol.id), max_retries=2, scope=alice)
        assert res.decision == DecisionType.COVERED
        assert res.evaluator_status == "VERIFIED_PASS"
        assert res.retry_count == 1
        assert evidence_counts == [5, 8]
        assert get_chunks_calls == 1  # Avoided repeated chunk corpus SQL fetches during retry
        assert prepare_calls == 1  # Avoided rebuilding BM25Okapi corpus index during retry
        assert res.metadata is not None
        assert res.metadata["storage_mode"] == "postgresql_pgvector"
        assert len(res.metadata["retrieval_attempts"]) == 2
        assert res.metadata["retrieval_attempts"][0]["corpus_cache_hit"] is False
        assert res.metadata["retrieval_attempts"][1]["corpus_cache_hit"] is True
        assert len(res.citations) == 1
        assert "-" in (res.citations[0].page_range or "")
    finally:
        gen_http.close()
        ver_http.close()


# =============================================================================
# 7–9. Provenance, Cross-Tenant Isolation & Unauthorized Access via FastAPI
# =============================================================================
def test_cross_tenant_isolation_and_unauthorized_api_access_blocked_before_retrieval(pg_ctx, monkeypatch, caplog):
    repo, _, env = pg_ctx
    alice = _scope("alice", "user:alice")
    bob = _scope("bob", "user:bob", allow_catalog_read=True)
    bob_no_catalog = _scope("bob", "user:bob", allow_catalog_read=False)
    org_acme = _scope("acme", "org:acme", role="organization")

    pol_private, _, _, _, _ = _seed_sample_policy(
        repo,
        external_policy_key="pol_alice_priv",
        owner_id="alice",
        scope=alice,
        model_name="offline-sha256-hash",
    )
    pol_cat_active, _, _, _, _ = _seed_sample_policy(
        repo,
        external_policy_key="pol_acme_cat_active",
        owner_id="acme",
        scope=org_acme,
        visibility="catalog",
        status="active",
        model_name="offline-sha256-hash",
    )
    pol_cat_draft, _, _, _, _ = _seed_sample_policy(
        repo,
        external_policy_key="pol_acme_cat_draft",
        owner_id="acme",
        scope=org_acme,
        visibility="catalog",
        status="draft",
        model_name="offline-sha256-hash",
    )

    embedder = OfflineHashEmbedder(dimension=1024, normalize=True, model_name="BAAI/bge-m3")
    embed_calls = 0
    orig_embed = embedder.embed_query

    def spy_embed(q: str):
        nonlocal embed_calls
        embed_calls += 1
        return orig_embed(q)

    embedder.embed_query = spy_embed  # type: ignore[method-assign]

    vec_search_calls = 0
    orig_vec_search = repo.vector_search

    def spy_vec_search(**kwargs):
        nonlocal vec_search_calls
        vec_search_calls += 1
        return orig_vec_search(**kwargs)

    repo.vector_search = spy_vec_search  # type: ignore[method-assign]

    retriever = PersistentHybridRetriever(
        repository=repo,
        dense_retriever=DenseVectorRetriever(embedder=embedder),
        embedding_model_name="offline-sha256-hash",
    )

    # Bob cannot access Alice's private policy or Acme's draft catalog BEFORE any retrieval runs
    with pytest.raises(DatabaseAuthorizationError):
        retriever.retrieve(query="Is hospitalization covered?", policy_id=str(pol_private.id), scope=bob, top_k=3)
    with pytest.raises(DatabaseAuthorizationError):
        retriever.retrieve(query="Is hospitalization covered?", policy_id=str(pol_cat_draft.id), scope=bob, top_k=3)
    with pytest.raises(DatabaseAuthorizationError):
        retriever.retrieve(query="Is hospitalization covered?", policy_id=str(pol_cat_active.id), scope=bob_no_catalog, top_k=3)
    assert embed_calls == 0
    assert vec_search_calls == 0

    # Active catalog policy is accessible ONLY when allow_catalog_read=True
    cat_hits = retriever.retrieve(
        query="Is maternity waiting period covered?",
        policy_id=str(pol_cat_active.id),
        scope=bob,
        top_k=3,
    )
    assert len(cat_hits.chunks) > 0
    assert embed_calls == 1
    assert vec_search_calls == 1

    # End-to-end FastAPI /ai/ask authorization & provenance verification
    secret_svc_token = "gateway-service-token-43"
    sensitive_question = "Can Bob view maternity benefits?"
    monkeypatch.setenv("USE_MOCK_STORAGE", "false")
    monkeypatch.setenv("DATABASE_URL", env["runtime_dsn"])
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", secret_svc_token)
    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-43")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")

    def api_gen_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "decision": "CONDITIONAL",
                            "reason": "Maternity delivery benefits are covered after a 24-month waiting period.",
                            "conditions": ["24 consecutive months continuous coverage waiting period"],
                            "claims": [{
                                "text": "Maternity delivery benefits require a 24 consecutive months waiting period.",
                                "chunk_ids": ["chk_pg_maternity_03"],
                                "claim_type": "condition",
                            }],
                        }),
                    }
                }]
            },
        )

    def api_ver_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": _verdict_json(
                            verdict="PASS",
                            confidence=0.96,
                            evidence_supported=True,
                            claim_verdicts=[{
                                "claim_idx": 0,
                                "status": "SUPPORTED",
                                "supported": True,
                                "cited_chunk_ids_valid": True,
                                "reason": "Confirmed by Section 5.3.",
                            }],
                        ),
                    }
                }]
            },
        )

    api_gen_http = httpx.Client(transport=httpx.MockTransport(api_gen_handler))
    api_ver_http = httpx.Client(transport=httpx.MockTransport(api_ver_handler))
    cfg = Settings()
    api_orch = ClosedRetryLoopOrchestrator(
        generator=GroundedGenerator(
            client=LLMClient(
                provider="nvidia",
                nvidia_provider=NvidiaNemotronProvider(api_key="nvapi-test-43", http_client=api_gen_http),
                app_settings=cfg,
            )
        ),
        evaluator=EvaluatorAgent(
            client=LLMClient(
                provider="nvidia",
                nvidia_provider=NvidiaNemotronProvider(api_key="nvapi-test-43", http_client=api_ver_http),
                app_settings=cfg,
            ),
            app_settings=cfg,
        ),
        decision_gate=DeterministicDecisionGate(),
        app_settings=cfg,
    )
    assert api_orch.kb is None

    close_db_pool()
    routes_qa.close_orchestrator()
    app.dependency_overrides[routes_qa.get_orchestrator] = lambda: api_orch

    try:
        with TestClient(app) as client:
            # 1. Missing X-Service-Token blocked BEFORE retrieval
            unauth_resp = client.post(
                "/ai/ask",
                json={
                    "policy_id": str(pol_private.id),
                    "question": sensitive_question,
                    "scope": {"user_id": "alice", "role": "user", "owner_scope": "user:alice"},
                },
            )
            assert unauth_resp.status_code == 403
            assert secret_svc_token not in unauth_resp.text

            # 2. Forged X-Service-Token blocked
            forged_resp = client.post(
                "/ai/ask",
                headers={"X-Service-Token": "wrong-token"},
                json={
                    "policy_id": str(pol_private.id),
                    "question": sensitive_question,
                    "scope": {"user_id": "alice", "role": "user", "owner_scope": "user:alice"},
                },
            )
            assert forged_resp.status_code == 403

            # 3. Cross-tenant access blocked even with valid gateway token
            cross_resp = client.post(
                "/ai/ask",
                headers={"X-Service-Token": secret_svc_token},
                json={
                    "policy_id": str(pol_private.id),
                    "question": sensitive_question,
                    "scope": {"user_id": "bob", "role": "user", "owner_scope": "user:bob", "allow_catalog_read": True},
                },
            )
            assert cross_resp.status_code == 403

            # 4. Authorized owner query succeeds end-to-end with multi-page provenance preserved
            with caplog.at_level(logging.INFO):
                ok_resp = client.post(
                    "/ai/ask",
                    headers={"X-Service-Token": secret_svc_token},
                    json={
                        "policy_id": str(pol_private.id),
                        "question": "Is maternity delivery covered after waiting period?",
                        "scope": {"user_id": "alice", "role": "user", "owner_scope": "user:alice"},
                    },
                )
            assert ok_resp.status_code == 200
            body = ok_resp.json()
            assert body["decision"] == "CONDITIONAL"
            assert body["evaluator_status"] == "VERIFIED_PASS"
            assert body["metadata"]["storage_mode"] == "postgresql_pgvector"
            assert len(body["citations"]) > 0
            cit = body["citations"][0]
            assert cit["chunk_id"] == "chk_pg_maternity_03"
            assert cit["page"] == 29
            assert cit["page_end"] == 31
            assert cit["page_range"] == "29-31"
            assert secret_svc_token not in caplog.text
            assert "change_me_runtime" not in caplog.text
    finally:
        app.dependency_overrides.clear()
        api_gen_http.close()
        api_ver_http.close()


# =============================================================================
# 10. Mock Mode Regression
# =============================================================================
def test_mock_mode_regression_preserves_mock_kb(monkeypatch):
    close_db_pool()
    routes_qa.close_orchestrator()
    monkeypatch.setenv("USE_MOCK_STORAGE", "true")
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    # Even with an unreachable PostgreSQL URL, mock mode never opens DB pool
    monkeypatch.setenv("DATABASE_URL", "postgresql://bad:bad@127.0.0.1:59999/nope")

    with TestClient(app) as client:
        health = client.get("/ai/health")
        assert health.status_code == 200
        assert health.json()["database_mode"] == "mock_memory"
        assert get_db_pool_manager().is_open is False

        ask_resp = client.post(
            "/ai/ask",
            json={
                "policy_id": "policy_health_01",
                "question": "Is in-patient hospitalization covered?",
            },
        )
        assert ask_resp.status_code == 200
        data = ask_resp.json()
        assert data["decision"] == "COVERED"
        assert data["evaluator_status"] == "VERIFIED_PASS"
        assert data["metadata"]["storage_mode"] == "mock_memory"
        assert len(data["citations"]) > 0
