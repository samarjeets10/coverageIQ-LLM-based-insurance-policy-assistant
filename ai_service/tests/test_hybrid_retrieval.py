from ai_service.retrieval.dense_retriever import DenseVectorRetriever
from ai_service.retrieval.bm25_retriever import BM25Retriever
from ai_service.retrieval.rrf_fusion import RRFFusion
from ai_service.database import get_knowledge_base

def test_hybrid_search_and_rrf():
    kb = get_knowledge_base()
    chunks = kb.get_chunks("policy_health_01")
    assert len(chunks) >= 4

    dense = DenseVectorRetriever()
    bm25 = BM25Retriever()
    rrf = RRFFusion(k=60, weight_vector=0.6, weight_bm25=0.4)

    query = "maternity delivery expenses waiting period"
    
    vec_results = dense.search(query, chunks, top_n=5)
    bm25_results = bm25.search(query, chunks, top_n=5)

    assert len(vec_results) > 0
    assert len(bm25_results) > 0

    # BM25 should rank maternity clause (chunk_h_004) at the top
    assert bm25_results[0][0].chunk_id == "chunk_h_004"

    # Fused candidates
    fused = rrf.fuse(vec_results, bm25_results, top_m=5)
    assert len(fused) > 0
    assert fused[0].chunk_id == "chunk_h_004"
    assert fused[0].rrf_score is not None
    assert fused[0].rrf_score > 0.0
