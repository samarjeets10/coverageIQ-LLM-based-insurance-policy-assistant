import logging
import time
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel, Field

from ai_service.config import Settings, settings
from ai_service.persistence import (
    AuthorizedPolicyTarget,
    AuthorizedScope,
    PolicyKnowledgeRepository,
    PostgresPoolManager,
    get_db_pool_manager,
    validate_authorized_scope,
)
from ai_service.retrieval.bm25_retriever import BM25Retriever
from ai_service.retrieval.dense_retriever import DenseVectorRetriever
from ai_service.retrieval.query_analyzer import QueryAnalyzer
from ai_service.retrieval.reranker import CrossEncoderReranker
from ai_service.retrieval.rrf_fusion import RRFFusion
from ai_service.schemas.structured_output import PolicyChunk

logger = logging.getLogger("coverageiq.retrieval.persistent")


class RetrievalLatencyMetrics(BaseModel):
    auth_and_chunk_load_ms: float = 0.0
    query_embed_ms: float = 0.0
    dense_pgvector_ms: float = 0.0
    sparse_bm25_ms: float = 0.0
    rrf_fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0
    query_embed_cache_hit: bool = False
    corpus_cache_hit: bool = False


class PersistentRetrievalResult(BaseModel):
    authorized_target: Optional[AuthorizedPolicyTarget] = None
    chunks: List[PolicyChunk] = Field(default_factory=list)
    total_policy_chunks: int = 0
    dense_hits_count: int = 0
    sparse_hits_count: int = 0
    fused_candidates_count: int = 0
    latency: RetrievalLatencyMetrics = Field(default_factory=RetrievalLatencyMetrics)


class PersistentRetrievalSession:
    """
    Request-scoped PostgreSQL 16 + pgvector hybrid retrieval session.

    Reuses the authorized chunk corpus, prepared BM25Okapi index, query embeddings,
    and identical HNSW searches within a single Q&A request (initial pass + retries)
    while keeping all embedding inference strictly outside DB transactions.
    """

    def __init__(
        self,
        *,
        retriever: "PersistentHybridRetriever",
        policy_id: str,
        scope: AuthorizedScope,
        document_id: Optional[UUID] = None,
    ):
        self._retriever = retriever
        self._scope = validate_authorized_scope(scope)
        self._target: Optional[AuthorizedPolicyTarget] = None
        self._all_chunks: Optional[List[PolicyChunk]] = None
        self._bm25_index = None
        self._query_vec_cache: Dict[str, List[float]] = {}
        self._dense_hits_cache: Dict[str, Tuple[int, List[Tuple[PolicyChunk, float]]]] = {}

        t_auth0 = time.perf_counter()
        self._target = self._retriever.repository.authorize_policy_access(
            policy_id,
            scope=self._scope,
            document_id=document_id,
        )
        self._auth_latency_ms = (time.perf_counter() - t_auth0) * 1000.0

    @property
    def authorized_target(self) -> Optional[AuthorizedPolicyTarget]:
        return self._target

    @property
    def has_retrievable_document(self) -> bool:
        return self._target is not None and self._target.document_id is not None

    def _ensure_corpus_loaded(self) -> tuple[List[PolicyChunk], bool, float]:
        if self._target is None or self._target.document_id is None:
            return [], True, 0.0

        if self._all_chunks is not None:
            return self._all_chunks, True, 0.0

        t0 = time.perf_counter()
        chunks = self._retriever.repository.get_policy_chunks(
            policy_id=self._target.policy_id,
            document_id=self._target.document_id,
            scope=self._scope,
        )
        self._all_chunks = chunks
        self._bm25_index = self._retriever.bm25_retriever.prepare_corpus(chunks)
        load_ms = (time.perf_counter() - t0) * 1000.0
        return chunks, False, load_ms

    def _get_query_embedding(self, query_text: str) -> tuple[List[float], bool, float]:
        cached = self._query_vec_cache.get(query_text)
        if cached is not None:
            return cached, True, 0.0

        t0 = time.perf_counter()
        # Model inference executes strictly outside database connections/transactions
        embedder = self._retriever.dense_retriever._get_embedder()
        vec = embedder.embed_query(query_text)
        if len(vec) != 1024:
            raise ValueError(f"Expected 1024-dim query embedding, got {len(vec)}.")
        embed_ms = (time.perf_counter() - t0) * 1000.0
        self._query_vec_cache[query_text] = vec
        return vec, False, embed_ms

    @staticmethod
    def _merge_fused_provenance_and_deduplicate(
        fused: List[PolicyChunk],
        vector_results: List[Tuple[PolicyChunk, float]],
        bm25_results: List[Tuple[PolicyChunk, float]],
    ) -> List[PolicyChunk]:
        sim_by_id = {c.chunk_id: score for c, score in vector_results}
        bm25_by_id = {c.chunk_id: score for c, score in bm25_results}
        seen_ids = set()
        merged: List[PolicyChunk] = []

        for chunk in fused:
            if chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(chunk.chunk_id)
            updates: dict[str, float] = {}
            if chunk.similarity_score is None and chunk.chunk_id in sim_by_id:
                updates["similarity_score"] = sim_by_id[chunk.chunk_id]
            if chunk.bm25_score is None and chunk.chunk_id in bm25_by_id:
                updates["bm25_score"] = bm25_by_id[chunk.chunk_id]
            merged.append(chunk.model_copy(update=updates) if updates else chunk)
        return merged

    def retrieve(self, query_text: str, *, top_k: int) -> PersistentRetrievalResult:
        t_total0 = time.perf_counter()

        if self._target is None or self._target.document_id is None:
            return PersistentRetrievalResult(
                authorized_target=self._target,
                chunks=[],
                total_policy_chunks=0,
                latency=RetrievalLatencyMetrics(
                    auth_and_chunk_load_ms=round(self._auth_latency_ms, 3),
                    total_ms=round((time.perf_counter() - t_total0) * 1000.0, 3),
                ),
            )

        all_chunks, corpus_cached, chunk_load_ms = self._ensure_corpus_loaded()
        total_chunks = len(all_chunks)
        if not all_chunks:
            return PersistentRetrievalResult(
                authorized_target=self._target,
                chunks=[],
                total_policy_chunks=0,
                latency=RetrievalLatencyMetrics(
                    auth_and_chunk_load_ms=round((0.0 if corpus_cached else self._auth_latency_ms) + chunk_load_ms, 3),
                    corpus_cache_hit=corpus_cached,
                    total_ms=round((time.perf_counter() - t_total0) * 1000.0, 3),
                ),
            )

        analyzed = self._retriever.query_analyzer.analyze(query_text)
        rerank_k = min(max(1, top_k), total_chunks)
        pool_cap = min(total_chunks, 100)
        fusion_pool_m = min(pool_cap, max(20, rerank_k * 2))
        retriever_top_n = min(pool_cap, max(15, rerank_k * 2))

        # 1. Dense pgvector HNSW search (using existing BGE-M3 embedder outside DB transactions)
        dense_key = analyzed.cleaned_query or query_text.strip()
        q_vec, q_cache_hit, q_embed_ms = self._get_query_embedding(dense_key)

        t_dense0 = time.perf_counter()
        cached_dense = self._dense_hits_cache.get(dense_key)
        if cached_dense is not None and cached_dense[0] >= retriever_top_n:
            vector_results = cached_dense[1][:retriever_top_n]
        else:
            embedder = self._retriever.dense_retriever._get_embedder()
            hits = self._retriever.repository.vector_search(
                policy_id=self._target.policy_id,
                document_id=self._target.document_id,
                query_embedding=q_vec,
                scope=self._scope,
                top_n=retriever_top_n,
                model_name=self._retriever.embedding_model_name or embedder.model_name,
                ef_search=self._retriever.hnsw_ef_search,
                iterative_scan=self._retriever.hnsw_iterative_scan,
            )
            vector_results = [(h.chunk, h.cosine_similarity) for h in hits]
            self._dense_hits_cache[dense_key] = (retriever_top_n, vector_results)
        dense_ms = (time.perf_counter() - t_dense0) * 1000.0

        # 2. Sparse Python BM25 search over authorized PostgreSQL chunks (never SQL ts_rank_cd)
        t_sparse0 = time.perf_counter()
        bm25_query = " ".join(analyzed.expanded_terms) if analyzed.expanded_terms else query_text
        bm25_results = self._retriever.bm25_retriever.search_prepared(
            bm25_query,
            all_chunks,
            self._bm25_index,
            top_n=retriever_top_n,
        )
        sparse_ms = (time.perf_counter() - t_sparse0) * 1000.0

        # 3. Reciprocal Rank Fusion (RRF) + deduplication + full provenance merge
        t_fuse0 = time.perf_counter()
        raw_fused = self._retriever.rrf_fusion.fuse(vector_results, bm25_results, top_m=fusion_pool_m)
        fused_candidates = self._merge_fused_provenance_and_deduplicate(raw_fused, vector_results, bm25_results)
        fusion_ms = (time.perf_counter() - t_fuse0) * 1000.0

        # 4. Existing CrossEncoderReranker
        t_rerank0 = time.perf_counter()
        top_evidence = self._retriever.reranker.rerank(query_text, fused_candidates, top_k=rerank_k)
        rerank_ms = (time.perf_counter() - t_rerank0) * 1000.0

        total_ms = (time.perf_counter() - t_total0) * 1000.0
        auth_load_total = (0.0 if corpus_cached else self._auth_latency_ms) + chunk_load_ms

        metrics = RetrievalLatencyMetrics(
            auth_and_chunk_load_ms=round(auth_load_total, 3),
            query_embed_ms=round(q_embed_ms, 3),
            dense_pgvector_ms=round(dense_ms, 3),
            sparse_bm25_ms=round(sparse_ms, 3),
            rrf_fusion_ms=round(fusion_ms, 3),
            rerank_ms=round(rerank_ms, 3),
            total_ms=round(total_ms, 3),
            query_embed_cache_hit=q_cache_hit,
            corpus_cache_hit=corpus_cached,
        )
        logger.info(
            "Persistent hybrid retrieval completed: policy_id=%s corpus=%d dense=%d sparse=%d fused=%d top_k=%d total_ms=%.2f",
            self._target.external_policy_key,
            total_chunks,
            len(vector_results),
            len(bm25_results),
            len(fused_candidates),
            len(top_evidence),
            metrics.total_ms,
        )
        return PersistentRetrievalResult(
            authorized_target=self._target,
            chunks=top_evidence,
            total_policy_chunks=total_chunks,
            dense_hits_count=len(vector_results),
            sparse_hits_count=len(bm25_results),
            fused_candidates_count=len(fused_candidates),
            latency=metrics,
        )


class PersistentHybridRetriever:
    """
    PostgreSQL 16 + pgvector persistent hybrid retrieval adapter combining:
    - AuthorizedScope pre-checks + SQL policy/document isolation
    - BGE-M3 1024-dim dense query embedding + pgvector HNSW cosine similarity
    - Existing Python BM25Okapi sparse scoring over authorized PostgreSQL chunks
    - Existing RRF fusion + CrossEncoderReranker with full provenance preservation
    """

    def __init__(
        self,
        *,
        repository: Optional[PolicyKnowledgeRepository] = None,
        pool_manager: Optional[PostgresPoolManager] = None,
        query_analyzer: Optional[QueryAnalyzer] = None,
        dense_retriever: Optional[DenseVectorRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        rrf_fusion: Optional[RRFFusion] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        app_settings: Optional[Settings] = None,
        embedding_model_name: Optional[str] = None,
        hnsw_ef_search: Optional[int] = None,
        hnsw_iterative_scan: Optional[str] = None,
    ):
        cfg = app_settings or settings
        if repository is not None:
            self.repository = repository
        else:
            mgr = pool_manager or get_db_pool_manager(auto_open=True)
            self.repository = PolicyKnowledgeRepository(mgr)

        self.query_analyzer = query_analyzer or QueryAnalyzer()
        self.dense_retriever = dense_retriever or DenseVectorRetriever(app_settings=cfg)
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.rrf_fusion = rrf_fusion or RRFFusion(
            k=cfg.RRF_K,
            weight_vector=cfg.WEIGHT_VECTOR,
            weight_bm25=cfg.WEIGHT_BM25,
        )
        self.reranker = reranker or CrossEncoderReranker()
        self.embedding_model_name = embedding_model_name
        self.hnsw_ef_search = hnsw_ef_search if hnsw_ef_search is not None else cfg.PGVECTOR_HNSW_EF_SEARCH
        self.hnsw_iterative_scan = (
            hnsw_iterative_scan if hnsw_iterative_scan is not None else cfg.PGVECTOR_HNSW_ITERATIVE_SCAN
        )

    def open_session(
        self,
        *,
        policy_id: str,
        scope: AuthorizedScope,
        document_id: Optional[UUID] = None,
    ) -> PersistentRetrievalSession:
        return PersistentRetrievalSession(
            retriever=self,
            policy_id=policy_id,
            scope=scope,
            document_id=document_id,
        )

    def retrieve(
        self,
        *,
        query: str,
        policy_id: str,
        scope: AuthorizedScope,
        top_k: int = 5,
        document_id: Optional[UUID] = None,
    ) -> PersistentRetrievalResult:
        session = self.open_session(policy_id=policy_id, scope=scope, document_id=document_id)
        return session.retrieve(query, top_k=top_k)
