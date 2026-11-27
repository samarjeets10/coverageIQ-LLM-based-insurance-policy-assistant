from ai_service.retrieval.reranker import CrossEncoderReranker
from ai_service.database import get_knowledge_base

def test_cross_encoder_reranking():
    kb = get_knowledge_base()
    chunks = kb.get_chunks("policy_health_01")
    reranker = CrossEncoderReranker()

    query = "Is cosmetic surgery covered or excluded?"
    reranked = reranker.rerank(query, chunks, top_k=3)

    assert len(reranked) == 3
    # Exclusion chunk (chunk_h_003) should be ranked first
    assert reranked[0].chunk_id == "chunk_h_003"
    assert reranked[0].rerank_score is not None
    assert reranked[0].rerank_score > reranked[1].rerank_score
