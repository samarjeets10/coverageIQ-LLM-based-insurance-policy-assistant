import logging
import os
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from ai_service.config import Settings, settings
from ai_service.database import get_knowledge_base
from ai_service.evaluator.decision_gate import DecisionGateResult, DeterministicDecisionGate
from ai_service.evaluator.verifier import EvaluatorAgent
from ai_service.llm.generator import GroundedGenerator
from ai_service.persistence import (
    AuthorizedScope,
    DatabaseAuthorizationError,
    validate_authorized_scope,
)
from ai_service.retrieval.bm25_retriever import BM25Retriever
from ai_service.retrieval.dense_retriever import DenseVectorRetriever
from ai_service.retrieval.persistent_retriever import (
    PersistentHybridRetriever,
    PersistentRetrievalSession,
)
from ai_service.retrieval.query_analyzer import QueryAnalyzer
from ai_service.retrieval.reranker import CrossEncoderReranker
from ai_service.retrieval.rrf_fusion import RRFFusion
from ai_service.schemas.structured_output import (
    DecisionType,
    StructuredAnswerResponse,
)

logger = logging.getLogger("coverageiq.retry_controller")


def _resolve_runtime_settings(explicit: Optional[Settings] = None) -> Settings:
    if explicit is not None:
        return explicit
    if os.getenv("USE_MOCK_STORAGE") is not None or os.getenv("DATABASE_URL") is not None:
        return Settings()
    return settings


class ClosedRetryLoopOrchestrator:
    """
    Closed Retry Loop Controller (Algorithm 4.5.3).
    Coordinates:
    Query Analyzer -> Hybrid Search (Vector + BM25) -> RRF Fusion ->
    Cross-Encoder Reranker -> Grounded LLM -> Nemotron Verifier ->
    Deterministic Decision Gate -> Bounded Retry Loop (max_retries = 2) ->
    Verified Structured Output.

    Storage selection:
    - USE_MOCK_STORAGE=false (or explicit persistent_retriever): uses PostgreSQL 16 + pgvector
      with strict AuthorizedScope enforcement, never falling back silently to MockKnowledgeBase.
    - USE_MOCK_STORAGE=true (default for local mock execution): uses MockKnowledgeBase.

    Provider failures (NvidiaProviderError) and malformed generation/verification JSON
    (LLMGenerationError) propagate directly and are NEVER converted into INSUFFICIENT EVIDENCE.
    """

    def __init__(
        self,
        query_analyzer: Optional[QueryAnalyzer] = None,
        dense_retriever: Optional[DenseVectorRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        rrf_fusion: Optional[RRFFusion] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        generator: Optional[GroundedGenerator] = None,
        evaluator: Optional[EvaluatorAgent] = None,
        decision_gate: Optional[DeterministicDecisionGate] = None,
        persistent_retriever: Optional[PersistentHybridRetriever] = None,
        app_settings: Optional[Settings] = None,
    ):
        cfg = _resolve_runtime_settings(app_settings)
        self._explicit_settings = app_settings
        self.query_analyzer = query_analyzer or QueryAnalyzer()
        self.dense_retriever = dense_retriever or DenseVectorRetriever(app_settings=cfg)
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.rrf_fusion = rrf_fusion or RRFFusion(
            k=cfg.RRF_K,
            weight_vector=cfg.WEIGHT_VECTOR,
            weight_bm25=cfg.WEIGHT_BM25,
        )
        self.reranker = reranker or CrossEncoderReranker()
        self.generator = generator or GroundedGenerator()
        self.evaluator = evaluator or EvaluatorAgent(confidence_threshold=cfg.CONFIDENCE_THRESHOLD)
        self.decision_gate = decision_gate or DeterministicDecisionGate(
            confidence_threshold=cfg.CONFIDENCE_THRESHOLD
        )
        self._persistent_retriever = persistent_retriever
        self.kb = get_knowledge_base() if (cfg.USE_MOCK_STORAGE and persistent_retriever is None) else None

    def close(self) -> None:
        if hasattr(self.generator, "close"):
            self.generator.close()
        if hasattr(self.evaluator, "close"):
            self.evaluator.close()

    def _get_persistent_retriever(self) -> Optional[PersistentHybridRetriever]:
        if self._persistent_retriever is not None:
            return self._persistent_retriever
        cfg = _resolve_runtime_settings(self._explicit_settings)
        if not cfg.USE_MOCK_STORAGE:
            self._persistent_retriever = PersistentHybridRetriever(
                query_analyzer=self.query_analyzer,
                dense_retriever=self.dense_retriever,
                bm25_retriever=self.bm25_retriever,
                rrf_fusion=self.rrf_fusion,
                reranker=self.reranker,
                app_settings=cfg,
            )
            return self._persistent_retriever
        return None

    def execute_qa(
        self,
        query: str,
        policy_id: str,
        max_retries: Optional[int] = None,
        *,
        scope: Optional[AuthorizedScope] = None,
        document_id: Optional[UUID] = None,
    ) -> StructuredAnswerResponse:
        cfg = _resolve_runtime_settings(self._explicit_settings)
        retries_limit = max(0, max_retries if max_retries is not None else cfg.MAX_RETRIES)
        attempt = 0
        current_top_k = cfg.INITIAL_TOP_K
        current_query = query

        persistent_retriever = self._get_persistent_retriever()
        session: Optional[PersistentRetrievalSession] = None
        all_chunks = []

        if persistent_retriever is not None:
            if scope is None:
                raise DatabaseAuthorizationError(
                    "Persistent PostgreSQL retrieval requires a validated AuthorizedScope."
                )
            auth_scope = validate_authorized_scope(scope)
            session = persistent_retriever.open_session(
                policy_id=policy_id,
                scope=auth_scope,
                document_id=document_id,
            )
            if not session.has_retrievable_document:
                return StructuredAnswerResponse(
                    query=query,
                    decision=DecisionType.INSUFFICIENT_EVIDENCE,
                    confidence_score=0.95,
                    conditions=[],
                    reason=f"No indexed policy documents or chunks found for policy_id '{policy_id}'.",
                    citations=[],
                    evaluator_status="NO_DATA",
                    retry_count=0,
                    metadata={"storage_mode": "postgresql_pgvector"},
                )
        else:
            if self.kb is None:
                self.kb = get_knowledge_base()
            all_chunks = self.kb.get_chunks(policy_id)
            if not all_chunks:
                return StructuredAnswerResponse(
                    query=query,
                    decision=DecisionType.INSUFFICIENT_EVIDENCE,
                    confidence_score=0.95,
                    conditions=[],
                    reason=f"No indexed policy documents or chunks found for policy_id '{policy_id}'.",
                    citations=[],
                    evaluator_status="NO_DATA",
                    retry_count=0,
                    metadata={"storage_mode": "mock_memory", "mock_mode": True},
                )

        last_gate_res: Optional[DecisionGateResult] = None
        last_draft: Optional[dict] = None
        attempt_latencies: List[Dict[str, Any]] = []

        while attempt <= retries_limit:
            logger.info(
                "QA pipeline attempt=%d/%d top_k=%d policy_id=%s mode=%s",
                attempt,
                retries_limit,
                current_top_k,
                policy_id,
                "postgresql" if session is not None else "mock",
            )

            if session is not None:
                retrieval_res = session.retrieve(current_query, top_k=current_top_k)
                top_evidence = retrieval_res.chunks
                attempt_latencies.append(
                    {
                        "attempt": attempt,
                        "total_ms": retrieval_res.latency.total_ms,
                        "dense_pgvector_ms": retrieval_res.latency.dense_pgvector_ms,
                        "sparse_bm25_ms": retrieval_res.latency.sparse_bm25_ms,
                        "rrf_fusion_ms": retrieval_res.latency.rrf_fusion_ms,
                        "rerank_ms": retrieval_res.latency.rerank_ms,
                        "query_embed_ms": retrieval_res.latency.query_embed_ms,
                        "query_embed_cache_hit": retrieval_res.latency.query_embed_cache_hit,
                        "corpus_cache_hit": retrieval_res.latency.corpus_cache_hit,
                    }
                )
            else:
                t_ret0 = time.perf_counter()
                analyzed = self.query_analyzer.analyze(current_query)
                total_policy_chunks = len(all_chunks)
                rerank_k = min(max(1, current_top_k), total_policy_chunks)
                pool_cap = min(total_policy_chunks, 100)
                fusion_pool_m = min(pool_cap, max(20, rerank_k * 2))
                retriever_top_n = min(pool_cap, max(15, rerank_k * 2))

                vector_results = self.dense_retriever.search(analyzed.cleaned_query, all_chunks, top_n=retriever_top_n)
                bm25_results = self.bm25_retriever.search(
                    " ".join(analyzed.expanded_terms), all_chunks, top_n=retriever_top_n
                )
                fused_candidates = self.rrf_fusion.fuse(vector_results, bm25_results, top_m=fusion_pool_m)
                top_evidence = self.reranker.rerank(current_query, fused_candidates, top_k=rerank_k)
                attempt_latencies.append(
                    {"attempt": attempt, "total_ms": round((time.perf_counter() - t_ret0) * 1000.0, 3)}
                )

            if not top_evidence:
                return StructuredAnswerResponse(
                    query=query,
                    decision=DecisionType.INSUFFICIENT_EVIDENCE,
                    confidence_score=0.95,
                    conditions=[],
                    reason="No relevant policy evidence chunks were retrieved for this query.",
                    citations=[],
                    evaluator_status="NO_DATA",
                    retry_count=attempt,
                    metadata={
                        "storage_mode": "postgresql_pgvector" if session is not None else "mock_memory",
                        "mock_mode": session is None,
                        "retrieval_attempts": attempt_latencies,
                    },
                )

            # 5. Grounded LLM Generation (provider / parse / schema errors propagate directly)
            draft = self.generator.generate_draft(current_query, top_evidence)
            last_draft = draft

            # 6. Independent Verification (provider / parse / schema errors propagate directly)
            eval_res = self.evaluator.verify(current_query, top_evidence, draft)

            # 7. Deterministic Decision Gate (kept strictly separate from verifier LLM call)
            gate_res = self.decision_gate.evaluate(current_query, top_evidence, draft, eval_res)
            last_gate_res = gate_res

            logger.info(
                "Decision gate verdict: passed=%s decision=%s confidence=%.2f issues=%d",
                gate_res.passed,
                gate_res.final_decision.value,
                gate_res.confidence_score,
                len(gate_res.issues),
            )

            if gate_res.passed:
                metadata: Dict[str, Any] = {
                    "storage_mode": "postgresql_pgvector" if session is not None else "mock_memory",
                    "mock_mode": session is None or draft.get("llm_metadata", {}).get("provider") == "offline",
                    "retrieved_chunks_count": len(top_evidence),
                    "issues_detected": len(gate_res.issues),
                    "verified_claims_count": len(gate_res.verified_claims),
                    "verifier_llm_skipped": "skipped verifier LLM call" in (eval_res.raw_evaluator_comment or ""),
                    "retrieval_attempts": attempt_latencies,
                }
                if "llm_metadata" in draft:
                    metadata["generation_llm"] = draft["llm_metadata"]

                return StructuredAnswerResponse(
                    query=query,
                    decision=gate_res.final_decision,
                    confidence_score=gate_res.confidence_score,
                    conditions=gate_res.verified_conditions,
                    reason=draft.get("reason", ""),
                    citations=gate_res.verified_citations,
                    evaluator_status="VERIFIED_PASS",
                    retry_count=attempt,
                    metadata=metadata,
                )

            # Verification failed -> bounded retry with widened retrieval and query refinement
            attempt += 1
            if attempt <= retries_limit:
                current_top_k += cfg.RETRY_TOP_K_INCREMENT
                if gate_res.contradiction_found:
                    current_query = f"{query} exclusion exception limitation condition"
                elif not gate_res.completeness_passed:
                    current_query = f"{query} waiting period terms criteria"
                else:
                    current_query = f"{query} coverage terms conditions exclusions"

        # Retry limit exhausted -> Enforce NO EVIDENCE = NO CLAIM with zero unverified claims/citations
        logger.warning(
            "Verification retry limit exhausted (%d) for policy_id=%s; returning INSUFFICIENT EVIDENCE.",
            retries_limit,
            policy_id,
        )

        contradiction_flag = bool(last_gate_res and last_gate_res.contradiction_found)
        issues_dump = [issue.model_dump() for issue in (last_gate_res.issues if last_gate_res else [])]

        if contradiction_flag:
            fallback_reason = (
                "Verification failed because contradictory policy evidence was detected across retrieved clauses "
                f"after {attempt} attempt(s). System adheres to NO EVIDENCE = NO CLAIM."
            )
        else:
            fallback_reason = (
                "Verification failed or insufficient conclusive policy evidence was retrieved "
                f"after {attempt} evaluation attempt(s). System adheres to NO EVIDENCE = NO CLAIM."
            )

        return StructuredAnswerResponse(
            query=query,
            decision=DecisionType.INSUFFICIENT_EVIDENCE,
            confidence_score=last_gate_res.confidence_score if last_gate_res else 0.50,
            conditions=[],
            reason=fallback_reason,
            citations=[],
            evaluator_status="VERIFIED_FALLBACK_EXHAUSTED",
            retry_count=attempt - 1,
            metadata={
                "storage_mode": "postgresql_pgvector" if session is not None else "mock_memory",
                "mock_mode": session is None,
                "contradiction_found": contradiction_flag,
                "evaluator_issues": issues_dump,
                "draft_decision_rejected": last_draft.get("decision") if last_draft else None,
                "retrieval_attempts": attempt_latencies,
            },
        )
