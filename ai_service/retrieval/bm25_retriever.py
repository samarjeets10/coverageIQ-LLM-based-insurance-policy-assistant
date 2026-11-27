import re
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from ai_service.schemas.structured_output import PolicyChunk

class BM25Retriever:
    """
    Sparse Lexical Search using BM25Okapi.
    Specializes in finding exact policy keywords, numeric clause references
    (e.g., 'Section 2.1', '24 consecutive hours', 'Exclusion 4.1'), and specific medical terminology.
    """
    def __init__(self):
        pass

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase and split on non-alphanumeric (preserves numbers, section identifiers)
        tokens = re.findall(r"\b[a-zA-Z0-9_\.-]+\b", text.lower())
        return tokens

    def search(self, query: str, chunks: List[PolicyChunk], top_n: int = 10) -> List[Tuple[PolicyChunk, float]]:
        if not chunks:
            return []

        tokenized_corpus = [
            self._tokenize(f"{c.clause_ref} {c.category.value} {c.content_text}") 
            for c in chunks
        ]
        
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = self._tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            c_copy = chunk.model_copy()
            c_copy.bm25_score = float(score)
            scored_chunks.append((c_copy, float(score)))

        # Sort descending by BM25 score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_n]
