import logging
from typing import Optional
from ai_service.schemas.structured_output import (
    StructuredAnswerResponse,
    DecisionType,
    PolicyChunk
)
from ai_service.retrieval.query_analyzer import QueryAnalyzer
from ai_service.retrieval.dense_retriever import DenseVectorRetriever
from ai_service.retrieval.bm25_retriever import BM25Retriever
from ai_service.retrieval.rrf_fusion import RRFFusion
from ai_service.retrieval.reranker import CrossEncoderReranker
from ai_service.llm.generator import GroundedGenerator
from ai_service.evaluator.verifier import EvaluatorAgent
from ai_service.database import get_knowledge_base
from ai_service.config import settings

logger = logging.getLogger("coverageiq.retry_controller")

class ClosedRetryLoopOrchestrator:
    """
    Closed Retry Loop Controller (Algorithm 4.5.3).
    Coordinates:
    Query Analyzer -> Hybrid Search (Vector + BM25) -> RRF Fusion ->
    Cross-Encoder Reranker -> Grounded LLM -> Evaluator Agent ->
    Conditional Retry Loop (max_retries = 2) -> Verified Structured Output.
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
    ):
        self.query_analyzer = query_analyzer or QueryAnalyzer()
        self.dense_retriever = dense_retriever or DenseVectorRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.rrf_fusion = rrf_fusion or RRFFusion(
            k=settings.RRF_K,
            weight_vector=settings.WEIGHT_VECTOR,
            weight_bm25=settings.WEIGHT_BM25
        )
        self.reranker = reranker or CrossEncoderReranker()
        self.generator = generator or GroundedGenerator()
        self.evaluator = evaluator or EvaluatorAgent(confidence_threshold=settings.CONFIDENCE_THRESHOLD)
        self.kb = get_knowledge_base()

    def execute_qa(
        self,
        query: str,
        policy_id: str,
        max_retries: int = None
    ) -> StructuredAnswerResponse:
        retries_limit = max_retries if max_retries is not None else settings.MAX_RETRIES
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
                retry_count=0
            )

        last_draft = None
        last_eval_res = None

        while attempt <= retries_limit:
            logger.info(f"QA pipeline attempt={attempt}/{retries_limit} | top_k={current_top_k} | query='{current_query}'")

            # 1. Query Analysis
            analyzed = self.query_analyzer.analyze(current_query)

            # 2. Hybrid Retrieval (Dense Vector + Sparse BM25)
            vector_results = self.dense_retriever.search(analyzed.cleaned_query, all_chunks, top_n=15)
            bm25_results = self.bm25_retriever.search(" ".join(analyzed.expanded_terms), all_chunks, top_n=15)

            # 3. Reciprocal Rank Fusion (RRF)
            fused_candidates = self.rrf_fusion.fuse(vector_results, bm25_results, top_m=20)

            # 4. Cross-Encoder Reranking
            top_evidence = self.reranker.rerank(current_query, fused_candidates, top_k=current_top_k)

            # 5. Grounded LLM Generation
            draft = self.generator.generate_draft(current_query, top_evidence)
            last_draft = draft

            # 6. Evaluator Agent Verification
            eval_res = self.evaluator.verify(current_query, top_evidence, draft)
            last_eval_res = eval_res

            logger.info(
                f"Evaluator Verdict: passed={eval_res.passed}, "
                f"confidence={eval_res.confidence_score}, issues={len(eval_res.issues)}"
            )

            # Check if verification passed
            if eval_res.passed and eval_res.confidence_score >= settings.CONFIDENCE_THRESHOLD:
                # Format final structured output
                dec = draft["decision"]
                decision_enum = (
                    DecisionType.COVERED if "COVERED" in dec and "NOT" not in dec
                    else DecisionType.NOT_COVERED if "NOT COVERED" in dec
                    else DecisionType.CONDITIONAL if "CONDITIONAL" in dec
                    else DecisionType.INSUFFICIENT_EVIDENCE
                )

                return StructuredAnswerResponse(
                    query=query,
                    decision=decision_enum,
                    confidence_score=eval_res.confidence_score,
                    conditions=draft.get("conditions", []),
                    reason=draft.get("reason", ""),
                    citations=draft.get("citations", []),
                    evaluator_status="VERIFIED_PASS",
                    retry_count=attempt,
                    metadata={
                        "retrieved_chunks_count": len(top_evidence),
                        "issues_detected": len(eval_res.issues)
                    }
                )

            # If verification failed, escalate parameters for retry
            attempt += 1
            if attempt <= retries_limit:
                current_top_k += settings.RETRY_TOP_K_INCREMENT
                # If contradiction was found, expand search query to include exclusion/condition concepts
                if eval_res.contradiction_found:
                    current_query = f"{query} exclusion exception limitation condition"
                elif not eval_res.completeness_passed:
                    current_query = f"{query} waiting period terms criteria"

        # Retry limit exhausted -> Enforce "NO EVIDENCE = NO CLAIM" -> Fallback to INSUFFICIENT EVIDENCE
        logger.warning(f"Retry limit exhausted ({retries_limit}). Falling back to INSUFFICIENT EVIDENCE.")
        return StructuredAnswerResponse(
            query=query,
            decision=DecisionType.INSUFFICIENT_EVIDENCE,
            confidence_score=last_eval_res.confidence_score if last_eval_res else 0.50,
            conditions=[],
            reason=(
                "Verification failed or insufficient conclusive policy evidence was retrieved "
                f"after {attempt} evaluation retries. System adheres to NO EVIDENCE = NO CLAIM."
            ),
            citations=last_draft.get("citations", []) if last_draft else [],
            evaluator_status="VERIFIED_FALLBACK_EXHAUSTED",
            retry_count=attempt - 1,
            metadata={
                "evaluator_issues": [issue.model_dump() for issue in (last_eval_res.issues if last_eval_res else [])]
            }
        )
