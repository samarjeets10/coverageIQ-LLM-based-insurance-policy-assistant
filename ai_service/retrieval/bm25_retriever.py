import re
from typing import List, Sequence, Tuple
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
        return re.findall(r"\b[a-zA-Z0-9_\.-]+\b", text.lower())

    def prepare_corpus(self, chunks: Sequence[PolicyChunk]) -> BM25Okapi | None:
        """Build a BM25Okapi index once so retry expansion avoids re-tokenizing the same corpus."""
        if not chunks:
            return None
        tokenized_corpus = [
            self._tokenize(f"{c.clause_ref} {c.category.value} {c.content_text}")
            for c in chunks
        ]
        return BM25Okapi(tokenized_corpus)

    def search_prepared(
        self,
        query: str,
        chunks: Sequence[PolicyChunk],
        index: BM25Okapi | None,
        top_n: int = 10,
    ) -> List[Tuple[PolicyChunk, float]]:
        if not chunks or index is None or top_n <= 0:
            return []

        tokenized_query = self._tokenize(query)
        scores = index.get_scores(tokenized_query)

        scored_chunks: List[Tuple[PolicyChunk, float]] = []
        for chunk, score in zip(chunks, scores):
            c_copy = chunk.model_copy()
            c_copy.bm25_score = float(score)
            scored_chunks.append((c_copy, float(score)))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_n]

    def search(self, query: str, chunks: List[PolicyChunk], top_n: int = 10) -> List[Tuple[PolicyChunk, float]]:
        if not chunks:
            return []
        index = self.prepare_corpus(chunks)
        return self.search_prepared(query, chunks, index, top_n=top_n)
