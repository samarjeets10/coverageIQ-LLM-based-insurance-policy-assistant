from abc import ABC, abstractmethod
import hashlib
import inspect
import logging
import math
import os
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ai_service.config import Settings, settings
from ai_service.schemas.structured_output import PolicyChunk

logger = logging.getLogger("coverageiq.embeddings.bge_m3")

_TOKEN_RE = re.compile(r"\b[a-z0-9_.-]+\b")


class EmbeddingError(Exception):
    """Base exception for CoverageIQ embedding initialization, inference, and validation failures."""


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when embedding configuration or input parameters are invalid."""


class EmbeddingModelLoadError(EmbeddingError):
    """Raised when BGE-M3 weights or runtime libraries cannot be loaded."""


class EmbeddingInferenceError(EmbeddingError):
    """Raised when dense embedding forward execution fails."""


class EmbeddingCudaOutOfMemoryError(EmbeddingInferenceError):
    """Raised explicitly when CUDA VRAM is exhausted during embedding inference."""


class EmbeddingOutputValidationError(EmbeddingError):
    """Raised when produced embeddings fail shape, dimension, finite-value, or norm checks."""


class BaseEmbedder(ABC):
    """Abstract base interface for CoverageIQ dense embedding providers."""

    def __init__(self, model_name: str, expected_dim: int = 1024, normalize: bool = True):
        if expected_dim <= 0:
            raise EmbeddingConfigurationError("Embedding dimension must be a positive integer.")
        self._model_name = model_name
        self._expected_dim = int(expected_dim)
        self._normalize = bool(normalize)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._expected_dim

    @property
    def normalize(self) -> bool:
        return self._normalize

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Encode a single search query into a fixed-dimension embedding vector."""

    @abstractmethod
    def embed_document(self, text: str) -> List[float]:
        """Encode a single document passage into a fixed-dimension embedding vector."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode a sequence of document passages into fixed-dimension embedding vectors."""

    @staticmethod
    def _format_chunk_text(chunk: PolicyChunk) -> str:
        combined = f"{chunk.clause_ref} {chunk.content_text}".strip()
        if not combined:
            raise EmbeddingConfigurationError(
                f"PolicyChunk '{chunk.chunk_id}' has empty clause_ref and content_text."
            )
        return combined

    def embed_chunks(self, chunks: Sequence[PolicyChunk]) -> List[PolicyChunk]:
        """
        Encode PolicyChunk instances in batches and return updated chunk copies
        with populated embeddings, keeping SemanticChunker decoupled from inference.
        """
        if not chunks:
            return []
        texts = [self._format_chunk_text(chunk) for chunk in chunks]
        vectors = self.embed_documents(texts)
        if len(vectors) != len(chunks):
            raise EmbeddingOutputValidationError(
                f"Expected {len(chunks)} chunk embeddings, received {len(vectors)}."
            )
        return [
            chunk.model_copy(update={"embedding": vec})
            for chunk, vec in zip(chunks, vectors)
        ]

    @staticmethod
    def _validate_non_empty_text(text: str, role: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingConfigurationError(f"Embedding {role} text must be a non-empty string.")
        return text

    def _validate_texts_batch(self, texts: Sequence[str]) -> List[str]:
        if isinstance(texts, (str, bytes)):
            raise EmbeddingConfigurationError("embed_documents expects a sequence of strings, not a single string.")
        return [self._validate_non_empty_text(item, "document") for item in texts]

    def _validate_and_finalize_vectors(self, raw_vectors: Any, expected_count: int) -> List[List[float]]:
        if raw_vectors is None:
            raise EmbeddingOutputValidationError("Embedding model returned None instead of vectors.")
        try:
            arr = np.asarray(raw_vectors, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise EmbeddingOutputValidationError(f"Embedding output could not be converted to float32 array: {exc}") from exc

        if arr.ndim != 2:
            raise EmbeddingOutputValidationError(
                f"Expected 2D embedding array of shape ({expected_count}, {self._expected_dim}), got ndim={arr.ndim}."
            )
        if arr.shape[0] != expected_count:
            raise EmbeddingOutputValidationError(
                f"Expected {expected_count} embedding rows, got {arr.shape[0]}."
            )
        if arr.shape[1] != self._expected_dim:
            raise EmbeddingOutputValidationError(
                f"Expected {self._expected_dim}-dimensional embeddings, got {arr.shape[1]}."
            )
        if not np.all(np.isfinite(arr)):
            raise EmbeddingOutputValidationError("Embedding output contains NaN or Inf values.")

        if self._normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
                raise EmbeddingOutputValidationError("Embedding vector has zero or non-finite norm before L2 normalization.")
            arr = (arr / norms).astype(np.float32)
            if not np.all(np.isfinite(arr)):
                raise EmbeddingOutputValidationError("Normalized embedding contains NaN or Inf values.")

        return [[float(x) for x in row] for row in arr]


class OfflineHashEmbedder(BaseEmbedder):
    """
    Cross-process deterministic hash-projection embedder for explicit offline test mode.
    Uses SHA-256 token hashing rather than Python's process-salted hash().
    """

    def __init__(self, dimension: int = 1024, normalize: bool = True, model_name: str = "offline-sha256-hash"):
        super().__init__(model_name=model_name, expected_dim=dimension, normalize=normalize)

    def _encode_single(self, text: str) -> np.ndarray:
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            tokens = [text.strip().lower()]

        vec = np.zeros(self._expected_dim, dtype=np.float32)
        for idx, token in enumerate(tokens):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._expected_dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[bucket] += np.float32(sign * (1.0 / (1.0 + math.log1p(idx + 1))))

        if float(np.linalg.norm(vec)) == 0.0:
            # Deterministic non-zero fallback projection if opposing weighted signs cancel
            digest = hashlib.sha256(text.strip().lower().encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._expected_dim
            vec[bucket] = np.float32(1.0)
        return vec

    def embed_query(self, text: str) -> List[float]:
        cleaned = self._validate_non_empty_text(text, "query")
        return self._validate_and_finalize_vectors([self._encode_single(cleaned)], 1)[0]

    def embed_document(self, text: str) -> List[float]:
        cleaned = self._validate_non_empty_text(text, "document")
        return self._validate_and_finalize_vectors([self._encode_single(cleaned)], 1)[0]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        validated = self._validate_texts_batch(texts)
        if not validated:
            return []
        matrix = np.stack([self._encode_single(item) for item in validated], axis=0)
        return self._validate_and_finalize_vectors(matrix, len(validated))


class BgeM3Embedder(BaseEmbedder):
    """
    Lazy-loaded BAAI/bge-m3 dense embedding service using FlagEmbedding's BGEM3FlagModel.
    Supports CUDA acceleration with automatic CPU fallback (when device='auto'), batched
    inference, strict 1024-dim finite float32 validation, and explicit CUDA OOM errors.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        normalize: Optional[bool] = None,
        local_files_only: Optional[bool] = None,
        max_length: int = 8192,
        app_settings: Optional[Settings] = None,
        model: Optional[Any] = None,
        model_factory: Optional[Callable[[str, str, bool, bool], Any]] = None,
    ):
        cfg = app_settings or settings
        eff_name = (model_name if model_name is not None else cfg.EMBEDDING_MODEL_NAME).strip()
        eff_dim = int(dimension if dimension is not None else cfg.EMBEDDING_DIM)
        eff_batch = int(batch_size if batch_size is not None else cfg.EMBEDDING_BATCH_SIZE)
        eff_device = (device if device is not None else cfg.EMBEDDING_DEVICE).strip().lower()
        eff_norm = cfg.EMBEDDING_NORMALIZE if normalize is None else bool(normalize)
        eff_local_only = cfg.EMBEDDING_LOCAL_FILES_ONLY if local_files_only is None else bool(local_files_only)

        if not eff_name:
            raise EmbeddingConfigurationError("EMBEDDING_MODEL_NAME cannot be empty.")
        if eff_dim != 1024:
            raise EmbeddingConfigurationError("BgeM3Embedder requires EMBEDDING_DIM=1024.")
        if eff_batch <= 0:
            raise EmbeddingConfigurationError("EMBEDDING_BATCH_SIZE must be a positive integer.")
        if max_length <= 0:
            raise EmbeddingConfigurationError("max_length must be a positive integer.")

        is_cuda_idx = eff_device.startswith("cuda:") and eff_device[5:].isdigit()
        if eff_device not in ("auto", "cpu", "cuda") and not is_cuda_idx:
            raise EmbeddingConfigurationError("EMBEDDING_DEVICE must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'.")

        super().__init__(model_name=eff_name, expected_dim=eff_dim, normalize=eff_norm)
        self.batch_size = eff_batch
        self.requested_device = eff_device
        self.local_files_only = eff_local_only
        self.max_length = int(max_length)

        self._model = model
        self._model_factory = model_factory
        self._resolved_device: Optional[str] = "injected" if model is not None else None
        self._use_fp16: bool = False
        self._load_lock = threading.Lock()

    @property
    def resolved_device(self) -> Optional[str]:
        return self._resolved_device

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @staticmethod
    def _import_torch() -> Any:
        try:
            import torch
            return torch
        except ImportError as exc:
            raise EmbeddingModelLoadError("PyTorch is required to load BGE-M3 embeddings.") from exc

    def _resolve_device_and_precision(self) -> tuple[str, bool]:
        torch = self._import_torch()
        cuda_ok = bool(torch.cuda.is_available())
        if self.requested_device == "cpu":
            return "cpu", False
        if self.requested_device == "auto":
            if cuda_ok:
                return "cuda:0", True
            return "cpu", False
        if not cuda_ok:
            raise EmbeddingModelLoadError(
                f"Requested CUDA device '{self.requested_device}' is unavailable on this host."
            )
        resolved = "cuda:0" if self.requested_device == "cuda" else self.requested_device
        return resolved, True

    @staticmethod
    def _build_flag_model(
        model_name: str,
        device: str,
        use_fp16: bool,
        local_files_only: bool,
        *,
        normalize: bool = True,
        batch_size: int = 4,
        max_length: int = 8192,
    ) -> Any:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise EmbeddingModelLoadError(
                "FlagEmbedding package is not installed; install FlagEmbedding to run BGE-M3 inference."
            ) from exc

        sig = inspect.signature(BGEM3FlagModel.__init__)
        params = sig.parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        kwargs: Dict[str, Any] = {"use_fp16": use_fp16}
        if "devices" in params:
            kwargs["devices"] = device
        elif "device" in params or has_var_kw:
            kwargs["device"] = device
        if "normalize_embeddings" in params:
            kwargs["normalize_embeddings"] = normalize
        if "batch_size" in params:
            kwargs["batch_size"] = batch_size
        if "query_max_length" in params:
            kwargs["query_max_length"] = min(max_length, 8192)
        if "passage_max_length" in params:
            kwargs["passage_max_length"] = min(max_length, 8192)

        prev_hf_offline = os.environ.get("HF_HUB_OFFLINE")
        prev_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
        if local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            if "model_name_or_path" in params:
                return BGEM3FlagModel(model_name_or_path=model_name, **kwargs)
            return BGEM3FlagModel(model_name, **kwargs)
        finally:
            if local_files_only:
                if prev_hf_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = prev_hf_offline
                if prev_tf_offline is None:
                    os.environ.pop("TRANSFORMERS_OFFLINE", None)
                else:
                    os.environ["TRANSFORMERS_OFFLINE"] = prev_tf_offline

    @staticmethod
    def _ensure_eval_mode(model: Any) -> None:
        if callable(getattr(model, "eval", None)):
            model.eval()
        inner = getattr(model, "model", None)
        if inner is not None and callable(getattr(inner, "eval", None)):
            inner.eval()

    def _get_model(self) -> Any:
        if self._model is not None:
            self._ensure_eval_mode(self._model)
            return self._model

        with self._load_lock:
            if self._model is not None:
                self._ensure_eval_mode(self._model)
                return self._model

            device, use_fp16 = self._resolve_device_and_precision()
            logger.info(
                "Loading BGE-M3 embedding model lazily: model=%s device=%s fp16=%s",
                self.model_name,
                device,
                use_fp16,
            )
            try:
                if self._model_factory is not None:
                    loaded = self._model_factory(self.model_name, device, use_fp16, self.local_files_only)
                else:
                    loaded = self._build_flag_model(
                        self.model_name,
                        device,
                        use_fp16,
                        self.local_files_only,
                        normalize=self._normalize,
                        batch_size=self.batch_size,
                        max_length=self.max_length,
                    )
            except EmbeddingError:
                raise
            except Exception as exc:
                logger.error(
                    "Failed to load BGE-M3 model: model=%s device=%s error_type=%s",
                    self.model_name,
                    device,
                    type(exc).__name__,
                )
                raise EmbeddingModelLoadError(
                    f"Failed to load BGE-M3 model '{self.model_name}' on device '{device}': {type(exc).__name__}"
                ) from exc

            self._ensure_eval_mode(loaded)
            self._model = loaded
            self._resolved_device = device
            self._use_fp16 = use_fp16
            return self._model

    @staticmethod
    def _is_cuda_oom_error(exc: Exception, torch_mod: Optional[Any]) -> bool:
        if torch_mod is not None:
            cuda_mod = getattr(torch_mod, "cuda", None)
            oom_cls = getattr(cuda_mod, "OutOfMemoryError", None)
            if oom_cls is not None and isinstance(exc, oom_cls):
                return True
        msg = str(exc).lower()
        return "out of memory" in msg and ("cuda" in msg or "cublas" in msg or "hip" in msg)

    def _encode_batch_raw(self, texts: Sequence[str]) -> Any:
        model = self._get_model()
        torch = None
        try:
            import torch as _torch
            torch = _torch
        except ImportError:
            pass

        encode_fn = getattr(model, "encode", None)
        if not callable(encode_fn):
            raise EmbeddingInferenceError("Loaded BGE-M3 model object does not expose an encode() method.")

        sig = inspect.signature(encode_fn)
        params = sig.parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        kwargs: Dict[str, Any] = {
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "return_dense": True,
            "return_sparse": False,
            "return_colbert_vecs": False,
        }
        # Only forward optional flags accepted by the installed FlagEmbedding signature
        filtered_kwargs = kwargs if has_var_kw else {k: v for k, v in kwargs.items() if k in params}

        ctx = torch.inference_mode() if (torch is not None and hasattr(torch, "inference_mode")) else None
        try:
            if ctx is not None:
                with ctx:
                    output = encode_fn(list(texts), **filtered_kwargs)
            else:
                output = encode_fn(list(texts), **filtered_kwargs)
        except Exception as exc:
            if self._is_cuda_oom_error(exc, torch):
                if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                logger.error(
                    "CUDA OOM during BGE-M3 inference: batch_len=%d batch_size=%d device=%s",
                    len(texts),
                    self.batch_size,
                    self._resolved_device,
                )
                raise EmbeddingCudaOutOfMemoryError(
                    f"CUDA out of memory while encoding {len(texts)} text(s) with batch_size={self.batch_size}."
                ) from exc

            logger.error(
                "BGE-M3 inference failed: batch_len=%d device=%s error_type=%s",
                len(texts),
                self._resolved_device,
                type(exc).__name__,
            )
            raise EmbeddingInferenceError(
                f"BGE-M3 embedding inference failed ({type(exc).__name__})."
            ) from exc

        if isinstance(output, dict):
            if "dense_vecs" not in output or output["dense_vecs"] is None:
                raise EmbeddingOutputValidationError("BGE-M3 encode() dictionary output is missing 'dense_vecs'.")
            return output["dense_vecs"]

        return output

    def embed_query(self, text: str) -> List[float]:
        cleaned = self._validate_non_empty_text(text, "query")
        raw = self._encode_batch_raw([cleaned])
        return self._validate_and_finalize_vectors(raw, 1)[0]

    def embed_document(self, text: str) -> List[float]:
        cleaned = self._validate_non_empty_text(text, "document")
        raw = self._encode_batch_raw([cleaned])
        return self._validate_and_finalize_vectors(raw, 1)[0]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        validated = self._validate_texts_batch(texts)
        if not validated:
            return []
        raw = self._encode_batch_raw(validated)
        return self._validate_and_finalize_vectors(raw, len(validated))

    def close(self) -> None:
        """Release loaded model weights and reclaim CUDA cache if allocated."""
        import gc

        with self._load_lock:
            model = self._model
            self._model = None
            self._resolved_device = None

        if model is not None:
            try:
                inner = getattr(model, "model", None)
                if inner is not None and callable(getattr(inner, "to", None)):
                    inner.to("cpu")
            except Exception:
                pass
            del model

        gc.collect()
        try:
            torch = self._import_torch()
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def create_embedder(
    *,
    provider: Optional[str] = None,
    app_settings: Optional[Settings] = None,
) -> BaseEmbedder:
    """Instantiate the configured dense embedder (BgeM3Embedder or OfflineHashEmbedder)."""
    cfg = app_settings or settings
    eff_provider = (
        provider if provider is not None else (os.getenv("EMBEDDING_PROVIDER") or cfg.EMBEDDING_PROVIDER)
    ).strip().lower()

    if eff_provider == "offline":
        return OfflineHashEmbedder(
            dimension=cfg.EMBEDDING_DIM,
            normalize=cfg.EMBEDDING_NORMALIZE,
        )
    if eff_provider == "bge-m3":
        return BgeM3Embedder(app_settings=cfg)

    raise EmbeddingConfigurationError(
        f"Unsupported EMBEDDING_PROVIDER '{eff_provider}'. Expected 'bge-m3' or 'offline'."
    )
