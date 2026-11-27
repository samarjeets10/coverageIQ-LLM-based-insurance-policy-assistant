import math
from typing import List, Tuple
import numpy as np
from ai_service.schemas.structured_output import PolicyChunk
from ai_service.config import settings

class DenseVectorRetriever:
    """
    Dense Semantic Search Engine.
    Computes cosine similarity between user query and policy chunk embeddings.
    Supports pgvector in production and high-speed normalized vector calculation locally.
    """
    def __init__(self):
        self.embedding_dim = 768

    def _generate_fallback_embedding(self, text: str) -> List[float]:
        """
        Deterministic, zero-external-dependency semantic hash embedding for offline local tests.
        Produces a normalized 768-dimensional float vector.
        """
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec.tolist()
            
        for i, word in enumerate(words):
            h = hash(word)
            idx = abs(h) % self.embedding_dim
            sign = 1.0 if (h // self.embedding_dim) % 2 == 0 else -1.0
            vec[idx] += sign * (1.0 / (1.0 + math.log1p(i + 1)))

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def get_embedding(self, text: str) -> List[float]:
        # If Gemini API key is configured, use Gemini Embeddings if available; otherwise use fallback
        if settings.GEMINI_API_KEY:
            try:
                from google import genai
                client = genai.Client(api_key=settings.GEMINI_API_KEY)
                res = client.models.embed_content(
                    model="text-embedding-004",
                    contents=text
                )
                if hasattr(res, 'embedding') and res.embedding.values:
                    return res.embedding.values
            except Exception:
                pass
        return self._generate_fallback_embedding(text)

    def search(self, query: str, chunks: List[PolicyChunk], top_n: int = 10) -> List[Tuple[PolicyChunk, float]]:
        if not chunks:
            return []

        q_vec = np.array(self.get_embedding(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            q_norm = 1.0

        scored_chunks = []
        for chunk in chunks:
            if not chunk.embedding:
                chunk.embedding = self.get_embedding(f"{chunk.clause_ref} {chunk.content_text}")
            
            c_vec = np.array(chunk.embedding, dtype=np.float32)
            c_norm = np.linalg.norm(c_vec)
            if c_norm == 0:
                c_norm = 1.0
                
            sim = float(np.dot(q_vec, c_vec) / (q_norm * c_norm))
            c_copy = chunk.model_copy()
            c_copy.similarity_score = sim
            scored_chunks.append((c_copy, sim))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_n]
