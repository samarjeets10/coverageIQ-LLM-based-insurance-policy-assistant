from typing import List, Dict, Tuple
from ai_service.schemas.structured_output import PolicyChunk
from ai_service.config import settings

class RRFFusion:
    """
    Reciprocal Rank Fusion (RRF) for Hybrid Retrieval.
    Combines dense vector ranks and sparse BM25 ranks according to the formula:
    RRF_Score(d) = w_vec * (1 / (k + rank_vector(d))) + w_bm25 * (1 / (k + rank_bm25(d)))
    """
    def __init__(self, k: int = 60, weight_vector: float = 0.6, weight_bm25: float = 0.4):
        self.k = k
        self.weight_vector = weight_vector
        self.weight_bm25 = weight_bm25

    def fuse(
        self,
        vector_results: List[Tuple[PolicyChunk, float]],
        bm25_results: List[Tuple[PolicyChunk, float]],
        top_m: int = 25
    ) -> List[PolicyChunk]:
        """
        Fuses two ranked lists into a single candidate list ordered by descending RRF score.
        """
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, PolicyChunk] = {}

        # Process vector results (1-indexed rank)
        for rank, (chunk, _) in enumerate(vector_results, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            scores[cid] = scores.get(cid, 0.0) + self.weight_vector * (1.0 / (self.k + rank))

        # Process BM25 results (1-indexed rank)
        for rank, (chunk, _) in enumerate(bm25_results, start=1):
            cid = chunk.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = chunk
            scores[cid] = scores.get(cid, 0.0) + self.weight_bm25 * (1.0 / (self.k + rank))

        # Sort chunks by fused RRF score descending
        sorted_cids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        fused_chunks: List[PolicyChunk] = []
        for cid in sorted_cids[:top_m]:
            chunk = chunk_map[cid].model_copy()
            chunk.rrf_score = scores[cid]
            fused_chunks.append(chunk)

        return fused_chunks
