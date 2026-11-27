import logging
from typing import Optional

from ai_service.config import settings
from ai_service.database import get_knowledge_base
from ai_service.evaluator.decision_gate import DecisionGateResult, DeterministicDecisionGate
from ai_service.evaluator.verifier import EvaluatorAgent
from ai_service.llm.generator import GroundedGenerator
from ai_service.retrieval.bm25_retriever import BM25Retriever
from ai_service.retrieval.dense_retriever import DenseVectorRetriever
from ai_service.retrieval.query_analyzer import QueryAnalyzer
from ai_service.retrieval.reranker import CrossEncoderReranker
from ai_service.retrieval.rrf_fusion import RRFFusion
from ai_service.schemas.structured_output import (
    DecisionType,
    StructuredAnswerResponse,
)

logger = logging.getLogger("coverageiq.retry_controller")


class ClosedRetryLoopOrchestrator:
    """
    Closed Retry Loop Controller (Algorithm 4.5.3).
    Coordinates:
    Query Analyzer -> Hybrid Search (Vector + BM25) -> RRF Fusion ->
    Cross-Encoder Reranker -> Grounded LLM -> Nemotron Verifier ->
    Deterministic Decision Gate -> Bounded Retry Loop (max_retries = 2) ->
    Verified Structured Output.

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
    ):
        self.query_analyzer = query_analyzer or QueryAnalyzer()
        self.dense_retriever = dense_retriever or DenseVectorRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.rrf_fusion = rrf_fusion or RRFFusion(
            k=settings.RRF_K,
            weight_vector=settings.WEIGHT_VECTOR,
            weight_bm25=settings.WEIGHT_BM25,
        )
        self.reranker = reranker or CrossEncoderReranker()
        self.generator = generator or GroundedGenerator()
        self.evaluator = evaluator or EvaluatorAgent(confidence_threshold=settings.CONFIDENCE_THRESHOLD)
        self.decision_gate = decision_gate or DeterministicDecisionGate(
            confidence_threshold=settings.CONFIDENCE_THRESHOLD
        )
        self.kb = get_knowledge_base()

    def close(self) -> None:
        if hasattr(self.generator, "close"):
            self.generator.close()
        if hasattr(self.evaluator, "close"):
            self.evaluator.close()

    def execute_qa(
        self,
        query: str,
        policy_id: str,
        max_retries: Optional[int] = None,
    ) -> StructuredAnswerResponse:
        retries_limit = max(0, max_retries if max_retries is not None else settings.MAX_RETRIES)
        attempt = 0
        current_top_k = settings.INITIAL_TOP_K
        current_query = query
        all_chunks = self.kb.get_chunks(policy_id)

        # If no chunks exist for this policy in KB
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
            )

        last_gate_res: Optional[DecisionGateResult] = None
        last_draft: Optional[dict] = None

        while attempt <= retries_limit:
            logger.info(
                "QA pipeline attempt=%d/%d top_k=%d policy_id=%s",
                attempt,
                retries_limit,
                current_top_k,
                policy_id,
            )

            # 1. Query Analysis
            analyzed = self.query_analyzer.analyze(current_query)

            # Scale retrieval pool with current_top_k while bounding maximum pool size
            total_policy_chunks = len(all_chunks)
            rerank_k = min(max(1, current_top_k), total_policy_chunks)
            pool_cap = min(total_policy_chunks, 100)
            fusion_pool_m = min(pool_cap, max(20, rerank_k * 2))
            retriever_top_n = min(pool_cap, max(15, rerank_k * 2))

            # 2. Hybrid Retrieval (Dense Vector + Sparse BM25)
            vector_results = self.dense_retriever.search(analyzed.cleaned_query, all_chunks, top_n=retriever_top_n)
            bm25_results = self.bm25_retriever.search(
                " ".join(analyzed.expanded_terms), all_chunks, top_n=retriever_top_n
            )

            # 3. Reciprocal Rank Fusion (RRF)
            fused_candidates = self.rrf_fusion.fuse(vector_results, bm25_results, top_m=fusion_pool_m)

            # 4. Cross-Encoder Reranking
            top_evidence = self.reranker.rerank(current_query, fused_candidates, top_k=rerank_k)

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
                metadata = {
                    "retrieved_chunks_count": len(top_evidence),
                    "issues_detected": len(gate_res.issues),
                    "verified_claims_count": len(gate_res.verified_claims),
                    "verifier_llm_skipped": "skipped verifier LLM call" in (eval_res.raw_evaluator_comment or ""),
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
                current_top_k += settings.RETRY_TOP_K_INCREMENT
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
                "contradiction_found": contradiction_flag,
                "evaluator_issues": issues_dump,
                "draft_decision_rejected": last_draft.get("decision") if last_draft else None,
            },
        )
