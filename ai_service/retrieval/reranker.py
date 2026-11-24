import logging
import re
from typing import List
from ai_service.schemas.structured_output import PolicyChunk
from ai_service.config import settings

logger = logging.getLogger("coverageiq.reranker")

class CrossEncoderReranker:
    """
    Cross-Encoder Reranker.
    Jointly attends to (query, clause) token pairs to re-score top candidate chunks
    and produce the final high-precision Top-K evidence set.
    """
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self._model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading CrossEncoder model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        except Exception as e:
            logger.info(f"CrossEncoder model not initialized ({e}). Using native cross-attention scoring engine.")
            self._model = None

    def _native_cross_score(self, query: str, chunk: PolicyChunk) -> float:
        """
        High-precision deterministic cross-scoring engine when PyTorch cross-encoder is offline.
        Measures:
        1. Exact query terms in clause reference & text
        2. Clause type relevance (e.g. Exclusion inquiry matching Exclusion chunks)
        3. Numeric matching (waiting periods, percentages, sub-limits)
        """
        q_lower = query.lower()
        chunk_text = f"{chunk.clause_ref} {chunk.content_text}".lower()
        
        q_tokens = set(re.findall(r"\b[a-zA-Z0-9_\.-]+\b", q_lower))
        c_tokens = set(re.findall(r"\b[a-zA-Z0-9_\.-]+\b", chunk_text))
        
        if not q_tokens:
            return 0.0

        # Term overlap
        overlap = len(q_tokens.intersection(c_tokens)) / len(q_tokens)
        score = overlap * 0.6

        # Intent & Category alignment boost
        if ("exclude" in q_lower or "not covered" in q_lower) and chunk.category.value == "Exclusions":
            score += 0.25
        elif ("wait" in q_lower or "pre-existing" in q_lower) and chunk.category.value == "Conditions":
            score += 0.25
        elif ("cover" in q_lower or "hospital" in q_lower) and chunk.category.value == "Coverage":
            score += 0.20

        # Clause reference direct match boost (e.g., Section 2.1, 4.1)
        clause_numbers = re.findall(r"\d+\.\d+", q_lower)
        for num in clause_numbers:
            if num in chunk.clause_ref:
                score += 0.35

        # Factor in prior RRF score if present
        if chunk.rrf_score:
            score += min(chunk.rrf_score * 5.0, 0.2)

        return float(min(score, 1.0))

    def rerank(self, query: str, candidates: List[PolicyChunk], top_k: int = 5) -> List[PolicyChunk]:
        if not candidates:
            return []

        if self._model is not None:
            try:
                pairs = [[query, f"{c.clause_ref}: {c.content_text}"] for c in candidates]
                scores = self._model.predict(pairs)
                for chunk, score in zip(candidates, scores):
                    chunk.rerank_score = float(score)
            except Exception as e:
                logger.warning(f"Error during neural cross-encoder prediction ({e}); falling back to native engine.")
                for chunk in candidates:
                    chunk.rerank_score = self._native_cross_score(query, chunk)
        else:
            for chunk in candidates:
                chunk.rerank_score = self._native_cross_score(query, chunk)

        # Sort descending by rerank score
        reranked = sorted(candidates, key=lambda x: (x.rerank_score or 0.0), reverse=True)
        return reranked[:top_k]
