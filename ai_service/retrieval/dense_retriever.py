import os
from typing import List, Optional, Tuple

import numpy as np

from ai_service.config import Settings, settings
from ai_service.embeddings import (
    BaseEmbedder,
    EmbeddingOutputValidationError,
    create_embedder,
)
from ai_service.schemas.structured_output import PolicyChunk


class DenseVectorRetriever:
    """
    Dense Semantic Search Engine.
    Uses the configured 1024-dimensional BGE-M3 embedder (or explicit OfflineHashEmbedder)
    for consistent query and document embeddings with strict dimension validation.
    """

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        app_settings: Optional[Settings] = None,
    ):
        self._app_settings = app_settings
        self._explicit_embedder = embedder
        self._managed_embedder: Optional[BaseEmbedder] = None
        self._managed_provider: Optional[str] = None
        self.embedding_dim = embedder.dimension if embedder is not None else (app_settings or settings).EMBEDDING_DIM

    def _effective_provider_name(self) -> str:
        cfg = self._app_settings or settings
        return (os.getenv("EMBEDDING_PROVIDER") or cfg.EMBEDDING_PROVIDER).strip().lower()

    def _get_embedder(self) -> BaseEmbedder:
        if self._explicit_embedder is not None:
            self.embedding_dim = self._explicit_embedder.dimension
            return self._explicit_embedder

        expected_provider = self._effective_provider_name()
        if self._managed_embedder is None or self._managed_provider != expected_provider:
            cfg = self._app_settings or settings
            self._managed_embedder = create_embedder(provider=expected_provider, app_settings=cfg)
            self._managed_provider = expected_provider

        self.embedding_dim = self._managed_embedder.dimension
        return self._managed_embedder

    def _validate_vector(self, vector: List[float], label: str, expected_dim: int) -> np.ndarray:
        if not isinstance(vector, (list, tuple, np.ndarray)):
            raise EmbeddingOutputValidationError(f"{label} embedding must be a sequence of floats.")
        try:
            arr = np.asarray(vector, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise EmbeddingOutputValidationError(f"Invalid {label} embedding values: {exc}") from exc

        if arr.ndim != 1 or arr.shape[0] != expected_dim:
            actual = arr.shape[0] if arr.ndim == 1 else arr.shape
            raise EmbeddingOutputValidationError(
                f"Expected {expected_dim}-dim {label} embedding, got {actual}."
            )
        if not np.all(np.isfinite(arr)):
            raise EmbeddingOutputValidationError(f"{label} embedding contains NaN or Inf values.")
        return arr

    def get_embedding(self, text: str) -> List[float]:
        """Generate a single document embedding via the configured embedder."""
        return self._get_embedder().embed_document(text)

    def search(self, query: str, chunks: List[PolicyChunk], top_n: int = 10) -> List[Tuple[PolicyChunk, float]]:
        if not chunks:
            return []

        if not isinstance(query, str) or not query.strip():
            return [(c.model_copy(update={"similarity_score": 0.0}), 0.0) for c in chunks[:top_n]]

        embedder = self._get_embedder()
        expected_dim = embedder.dimension

        q_vec = self._validate_vector(embedder.embed_query(query), "query", expected_dim)
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm == 0.0:
            q_norm = 1.0

        # Batch-encode any chunks missing stored embeddings without mutating caller inputs
        missing_indices: List[int] = []
        missing_chunks: List[PolicyChunk] = []
        for idx, chunk in enumerate(chunks):
            if chunk.embedding is None or len(chunk.embedding) == 0:
                missing_indices.append(idx)
                missing_chunks.append(chunk)

        encoded_map: dict[int, List[float]] = {}
        if missing_chunks:
            embedded_copies = embedder.embed_chunks(missing_chunks)
            for idx, emb_chunk in zip(missing_indices, embedded_copies):
                assert emb_chunk.embedding is not None
                encoded_map[idx] = emb_chunk.embedding

        scored_chunks: List[Tuple[PolicyChunk, float]] = []
        for idx, chunk in enumerate(chunks):
            raw_vec = chunk.embedding if idx not in encoded_map else encoded_map[idx]
            assert raw_vec is not None
            c_vec = self._validate_vector(raw_vec, f"chunk '{chunk.chunk_id}'", expected_dim)
            c_norm = float(np.linalg.norm(c_vec))
            if c_norm == 0.0:
                c_norm = 1.0

            sim = float(np.dot(q_vec, c_vec) / (q_norm * c_norm))
            c_copy = chunk.model_copy(update={"similarity_score": sim})
            scored_chunks.append((c_copy, sim))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_n]
