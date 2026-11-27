import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import pytest

from ai_service.config import Settings
from ai_service.database import get_knowledge_base
from ai_service.embeddings import (
    BgeM3Embedder,
    EmbeddingConfigurationError,
    EmbeddingCudaOutOfMemoryError,
    EmbeddingInferenceError,
    EmbeddingModelLoadError,
    EmbeddingOutputValidationError,
    OfflineHashEmbedder,
    create_embedder,
)
from ai_service.retrieval.bm25_retriever import BM25Retriever
from ai_service.retrieval.dense_retriever import DenseVectorRetriever
from ai_service.retrieval.reranker import CrossEncoderReranker
from ai_service.retrieval.rrf_fusion import RRFFusion
from ai_service.schemas.structured_output import ChunkCategory, PolicyChunk


class _FakeInnerPyTorchModel:
    def __init__(self) -> None:
        self.eval_calls = 0

    def eval(self) -> None:
        self.eval_calls += 1


class _FakeFlagM3Model:
    """Deterministic lightweight mock replicating FlagEmbedding's BGEM3FlagModel API."""

    def __init__(self, *, unnormalized: bool = True, vector_fn=None) -> None:
        self.model = _FakeInnerPyTorchModel()
        self.encode_calls: List[dict] = []
        self.unnormalized = unnormalized
        self.vector_fn = vector_fn

    def encode(
        self,
        sentences,
        batch_size: int = 12,
        max_length: int = 8192,
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert_vecs: bool = False,
    ):
        self.encode_calls.append(
            {
                "sentences": list(sentences),
                "batch_size": batch_size,
                "max_length": max_length,
                "return_dense": return_dense,
                "return_sparse": return_sparse,
                "return_colbert_vecs": return_colbert_vecs,
            }
        )
        if self.vector_fn is not None:
            return self.vector_fn(sentences)

        rows = []
        for text in sentences:
            base = np.zeros(1024, dtype=np.float32)
            # Deterministic directional signal from text tokens
            for i, word in enumerate(text.lower().split()):
                idx = sum(ord(ch) for ch in word) % 1024
                base[idx] += 3.5 / (1.0 + i)
            if float(np.linalg.norm(base)) == 0.0:
                base[0] = 5.0
            if self.unnormalized:
                base = base * 4.25  # deliberately non-unit to test L2 normalization
            rows.append(base)
        return {"dense_vecs": np.stack(rows, axis=0)}


def _sample_chunks() -> List[PolicyChunk]:
    return [
        PolicyChunk(
            chunk_id="chk_bge_01",
            policy_id="pol_bge",
            clause_ref="Section 2.1 - Hospitalization",
            page_number=12,
            category=ChunkCategory.COVERAGE,
            content_text="In-patient hospitalization expenses for minimum 24 hours are covered.",
        ),
        PolicyChunk(
            chunk_id="chk_bge_02",
            policy_id="pol_bge",
            clause_ref="Section 4.1 - Cosmetic Exclusion",
            page_number=22,
            category=ChunkCategory.EXCLUSIONS,
            content_text="Cosmetic and aesthetic surgery is strictly excluded unless medically necessary.",
        ),
    ]


# =============================================================================
# 1. Output Dimension, Normalization & Batch Equivalence Tests
# =============================================================================
def test_bge_m3_lazy_load_eval_mode_1024_dim_and_l2_normalization():
    fake_model = _FakeFlagM3Model(unnormalized=True)
    factory_calls = 0

    def factory(model_name, device, use_fp16, local_files_only):
        nonlocal factory_calls
        factory_calls += 1
        return fake_model

    embedder = BgeM3Embedder(
        model_name="BAAI/bge-m3",
        dimension=1024,
        batch_size=4,
        device="cpu",
        normalize=True,
        model_factory=factory,
    )
    assert embedder.is_loaded is False
    assert factory_calls == 0

    q_vec = embedder.embed_query("Is hospitalization covered?")
    assert factory_calls == 1
    assert embedder.is_loaded is True
    assert fake_model.model.eval_calls >= 1
    assert len(q_vec) == 1024
    assert all(isinstance(v, float) and np.isfinite(v) for v in q_vec)
    assert abs(float(np.linalg.norm(np.array(q_vec, dtype=np.float32))) - 1.0) < 1e-5

    d_vec = embedder.embed_document("Is hospitalization covered?")
    assert q_vec == d_vec

    # Unnormalized setting preserves raw vector magnitude
    embedder_raw = BgeM3Embedder(device="cpu", normalize=False, model=fake_model)
    raw_vec = embedder_raw.embed_query("Is hospitalization covered?")
    assert float(np.linalg.norm(np.array(raw_vec, dtype=np.float32))) > 2.0


def test_bge_m3_single_versus_batch_equivalence_and_chunk_embedding():
    fake_model = _FakeFlagM3Model(unnormalized=True)
    embedder = BgeM3Embedder(batch_size=2, device="cpu", normalize=True, model=fake_model)

    texts = [
        "Section 2.1 Hospitalization for 24 consecutive hours is covered.",
        "Section 4.1 Cosmetic surgery is excluded unless accident or burn reconstruction.",
        "Section 5.3 Maternity delivery has a 24 month waiting period.",
    ]
    singles = [embedder.embed_document(t) for t in texts]
    batches = embedder.embed_documents(texts)

    assert len(batches) == 3
    for s_vec, b_vec in zip(singles, batches):
        assert np.allclose(s_vec, b_vec, atol=1e-6)

    # Confirm FlagEmbedding batched encode() interface used cleanly with configured batch_size
    last_call = fake_model.encode_calls[-1]
    assert last_call["batch_size"] == 2
    assert last_call["return_dense"] is True
    assert last_call["return_sparse"] is False
    assert last_call["return_colbert_vecs"] is False

    chunks = _sample_chunks()
    embedded = embedder.embed_chunks(chunks)
    assert [c.chunk_id for c in embedded] == ["chk_bge_01", "chk_bge_02"]
    assert all(c.embedding is not None and len(c.embedding) == 1024 for c in embedded)
    # Input chunks must remain untouched
    assert all(c.embedding is None for c in chunks)


# =============================================================================
# 2. Empty Input, Invalid Outputs, NaN/Inf Rejection
# =============================================================================
def test_empty_input_handling():
    fake_model = _FakeFlagM3Model()
    embedder = BgeM3Embedder(device="cpu", model=fake_model)

    assert embedder.embed_documents([]) == []
    assert embedder.embed_chunks([]) == []
    assert fake_model.encode_calls == []

    for bad in ("", "   "):
        with pytest.raises(EmbeddingConfigurationError):
            embedder.embed_query(bad)
        with pytest.raises(EmbeddingConfigurationError):
            embedder.embed_document(bad)

    with pytest.raises(EmbeddingConfigurationError):
        embedder.embed_documents(["valid clause", "   "])


@pytest.mark.parametrize(
    "bad_output",
    [
        None,
        {"wrong_key": np.ones((1, 1024), dtype=np.float32)},
        {"dense_vecs": None},
        {"dense_vecs": np.ones(1024, dtype=np.float32)},  # 1D instead of 2D
        {"dense_vecs": np.ones((2, 1024), dtype=np.float32)},  # mismatched batch row count
        {"dense_vecs": np.ones((1, 768), dtype=np.float32)},  # wrong dimension (768 vs 1024)
        {"dense_vecs": np.zeros((1, 1024), dtype=np.float32)},  # zero-norm cannot normalize
    ],
)
def test_invalid_model_output_rejected(bad_output):
    bad_model = _FakeFlagM3Model(vector_fn=lambda sentences: bad_output)
    embedder = BgeM3Embedder(device="cpu", normalize=True, model=bad_model)
    with pytest.raises(EmbeddingOutputValidationError):
        embedder.embed_query("Hospitalization query")


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_nan_and_infinity_rejected(bad_val):
    vec = np.ones((1, 1024), dtype=np.float32)
    vec[0, 10] = bad_val
    bad_model = _FakeFlagM3Model(vector_fn=lambda sentences: {"dense_vecs": vec})
    embedder = BgeM3Embedder(device="cpu", normalize=True, model=bad_model)
    with pytest.raises(EmbeddingOutputValidationError):
        embedder.embed_query("Hospitalization query")


# =============================================================================
# 3. Device Selection, Fallback, Model Load Failures & CUDA OOM
# =============================================================================
def test_cpu_and_cuda_device_selection_and_auto_fallback(monkeypatch):
    import torch

    recorded = []

    def recording_factory(model_name, device, use_fp16, local_files_only):
        recorded.append((device, use_fp16))
        return _FakeFlagM3Model()

    # 1. Explicit CPU ignores CUDA availability
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    e_cpu = BgeM3Embedder(device="cpu", model_factory=recording_factory)
    e_cpu.embed_query("test cpu")
    assert recorded[-1] == ("cpu", False)

    # 2. Auto uses CUDA + fp16 when available
    e_auto_cuda = BgeM3Embedder(device="auto", model_factory=recording_factory)
    e_auto_cuda.embed_query("test cuda auto")
    assert recorded[-1] == ("cuda:0", True)

    # 3. Auto falls back to CPU + fp32 when CUDA unavailable
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    e_auto_cpu = BgeM3Embedder(device="auto", model_factory=recording_factory)
    e_auto_cpu.embed_query("test cpu fallback")
    assert recorded[-1] == ("cpu", False)

    # 4. Explicit CUDA fails clearly when CUDA unavailable (no silent fake vectors)
    e_force_cuda = BgeM3Embedder(device="cuda", model_factory=recording_factory)
    with pytest.raises(EmbeddingModelLoadError):
        e_force_cuda.embed_query("test force cuda")


def test_model_load_failure_never_falls_back_to_fake_embeddings(caplog):
    def broken_factory(*args):
        raise OSError("Simulated missing HuggingFace weight file")

    embedder = BgeM3Embedder(device="cpu", model_factory=broken_factory)
    sensitive_query = "SECRET_POLICYHOLDER_CANCER_CLAIM"
    with caplog.at_level(logging.ERROR):
        with pytest.raises(EmbeddingModelLoadError):
            embedder.embed_query(sensitive_query)
    assert sensitive_query not in caplog.text


def test_cuda_oom_handled_explicitly_without_fallback(monkeypatch, caplog):
    import torch

    empty_cache_calls = 0

    def fake_empty_cache():
        nonlocal empty_cache_calls
        empty_cache_calls += 1

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", fake_empty_cache)

    class _OomModel:
        def encode(self, sentences, **kwargs):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 512 MiB.")

    embedder = BgeM3Embedder(device="cuda:0", model=_OomModel())
    embedder._resolved_device = "cuda:0"
    sensitive_doc = "CONFIDENTIAL_MEDICAL_CLAUSE_TEXT"

    with caplog.at_level(logging.ERROR):
        with pytest.raises(EmbeddingCudaOutOfMemoryError):
            embedder.embed_document(sensitive_doc)

    assert empty_cache_calls == 1
    assert sensitive_doc not in caplog.text


def test_generic_inference_failure_raises_typed_error():
    class _BrokenEncodeModel:
        def encode(self, sentences, **kwargs):
            raise RuntimeError("Tensor runtime mismatch")

    embedder = BgeM3Embedder(device="cpu", model=_BrokenEncodeModel())
    with pytest.raises(EmbeddingInferenceError):
        embedder.embed_query("test")


# =============================================================================
# 4. Cross-Process Determinism & DenseVectorRetriever Integration
# =============================================================================
def test_offline_hash_embedder_deterministic_across_separate_processes():
    repo_root = str(Path(__file__).resolve().parents[2])
    code = (
        "import json; "
        "from ai_service.embeddings import OfflineHashEmbedder; "
        "e = OfflineHashEmbedder(dimension=1024); "
        "print(json.dumps(e.embed_query('In-patient hospitalization Section 2.1 covered 24 hours.')))"
    )
    env_a = {**os.environ, "PYTHONHASHSEED": "101", "PYTHONPATH": repo_root}
    env_b = {**os.environ, "PYTHONHASHSEED": "90909", "PYTHONPATH": repo_root}

    out_a = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env_a, check=True)
    out_b = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env_b, check=True)

    assert out_a.stdout == out_b.stdout
    import json

    vec = json.loads(out_a.stdout)
    assert len(vec) == 1024
    assert abs(float(np.linalg.norm(np.array(vec, dtype=np.float32))) - 1.0) < 1e-5


def test_dense_retriever_uses_injected_1024_embedder_and_validates_dimensions(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    fake_model = _FakeFlagM3Model()
    embedder = BgeM3Embedder(device="cpu", normalize=True, model=fake_model)
    retriever = DenseVectorRetriever(embedder=embedder)

    assert retriever.embedding_dim == 1024
    chunks = _sample_chunks()
    results = retriever.search("cosmetic aesthetic surgery exclusion", chunks, top_n=2)
    assert len(results) == 2
    assert results[0][0].chunk_id == "chk_bge_02"
    assert results[0][0].similarity_score is not None
    # Chunks encoded via single batch call, original list unmutated
    assert len(fake_model.encode_calls) == 2  # 1 query + 1 batched chunks call
    assert all(c.embedding is None for c in chunks)

    # Pre-populated stale 768-dim embedding must be rejected strictly
    bad_dim_chunk = chunks[0].model_copy(update={"embedding": [0.1] * 768})
    with pytest.raises(EmbeddingOutputValidationError, match="1024-dim"):
        retriever.search("hospitalization", [bad_dim_chunk], top_n=1)

    # Non-finite pre-populated embedding must be rejected strictly
    nan_chunk = chunks[0].model_copy(update={"embedding": [float("nan")] + [0.0] * 1023})
    with pytest.raises(EmbeddingOutputValidationError):
        retriever.search("hospitalization", [nan_chunk], top_n=1)


def test_dense_retriever_never_silently_uses_offline_hash_in_live_mode(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "bge-m3")
    monkeypatch.setenv("EMBEDDING_LOCAL_FILES_ONLY", "true")

    cfg = Settings()
    retriever = DenseVectorRetriever(app_settings=cfg)
    assert retriever.embedding_dim == 1024
    # Calling get_embedding without mocking or installed weights must fail explicitly, not silently fake
    monkeypatch.setitem(sys.modules, "FlagEmbedding", None)
    with pytest.raises(EmbeddingModelLoadError):
        retriever.search("hospitalization coverage", _sample_chunks(), top_n=1)


def test_offline_mode_only_when_configured_and_retrieval_pipeline_compatible(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    embedder = create_embedder()
    assert isinstance(embedder, OfflineHashEmbedder)
    assert embedder.dimension == 1024

    kb = get_knowledge_base()
    chunks = kb.get_chunks("policy_health_01")
    dense = DenseVectorRetriever()
    bm25 = BM25Retriever()
    rrf = RRFFusion(k=60, weight_vector=0.6, weight_bm25=0.4)
    reranker = CrossEncoderReranker()

    assert dense.embedding_dim == 1024
    vec_res = dense.search("maternity delivery 24 months waiting period", chunks, top_n=5)
    bm25_res = bm25.search("maternity delivery 24 months waiting period", chunks, top_n=5)
    fused = rrf.fuse(vec_res, bm25_res, top_m=5)
    reranked = reranker.rerank("maternity delivery 24 months waiting period", fused, top_k=3)

    assert len(reranked) == 3
    assert reranked[0].chunk_id == "chunk_h_004"
    assert reranked[0].rerank_score is not None
