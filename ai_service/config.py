import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit
from pydantic import BaseModel, Field, model_validator


def _load_local_env_file() -> None:
    """Load variables from ai_service/.env into os.environ without overwriting existing values."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        if key:
            os.environ.setdefault(key, val)


_load_local_env_file()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    val = raw.strip().lower()
    if val in ("true", "1", "yes", "on"):
        return True
    if val in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean value for {name}: '{raw}'.")


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _env_optional_str(name: str) -> Optional[str]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip()


def _redact_url_credentials(url: str) -> str:
    try:
        parsed = urlsplit(url)
        if parsed.password:
            user = parsed.username or ""
            host = parsed.hostname or ""
            port = f":{parsed.port}" if parsed.port else ""
            netloc = f"{user}:***@{host}{port}" if user else f"***@{host}{port}"
            return parsed._replace(netloc=netloc).geturl()
    except Exception:
        pass
    return url


_NON_TEST_ENVS = ("production", "staging", "prod", "stage")


def _resolve_app_env() -> str:
    raw = os.getenv("APP_ENV") or os.getenv("ENVIRONMENT") or os.getenv("NODE_ENV") or "development"
    return raw.strip().lower() or "development"


class Settings(BaseModel):
    # Service settings (Node gateway runs on 8000; internal FastAPI AI service runs on 8001)
    SERVICE_NAME: str = "coverageIQ-ai-service"
    APP_ENV: str = Field(default_factory=_resolve_app_env)
    PORT: int = Field(default_factory=lambda: _env_int("PORT", 8001))
    HOST: str = Field(default_factory=lambda: _env_str("HOST", "127.0.0.1"))
    DEBUG: bool = Field(default_factory=lambda: _env_bool("DEBUG", True))

    # Internal Service Integration & Upload Storage
    INTERNAL_SERVICE_TOKEN: Optional[str] = Field(
        default_factory=lambda: _env_optional_str("INTERNAL_SERVICE_TOKEN")
    )
    POLICY_UPLOAD_ROOT: str = Field(
        default_factory=lambda: _env_str("POLICY_UPLOAD_ROOT", "./uploads/policies")
    )

    # Database settings (Node.js owns PostgreSQL migrations via node-pg-migrate)
    DATABASE_URL: str = Field(
        default_factory=lambda: _env_str(
            "DATABASE_URL",
            "postgresql://coverageiq:coverageiq@localhost:5432/coverageiq_db",
        )
    )
    USE_MOCK_STORAGE: bool = Field(default_factory=lambda: _env_bool("USE_MOCK_STORAGE", True))
    POSTGRES_POOL_MIN_SIZE: int = Field(default_factory=lambda: _env_int("POSTGRES_POOL_MIN_SIZE", 1))
    POSTGRES_POOL_MAX_SIZE: int = Field(default_factory=lambda: _env_int("POSTGRES_POOL_MAX_SIZE", 10))
    POSTGRES_POOL_TIMEOUT: float = Field(default_factory=lambda: _env_float("POSTGRES_POOL_TIMEOUT", 30.0))

    # Celery & Redis Async Ingestion Settings
    REDIS_URL: str = Field(default_factory=lambda: _env_str("REDIS_URL", "redis://localhost:6379/0"))
    CELERY_BROKER_URL: str = Field(
        default_factory=lambda: _env_optional_str("CELERY_BROKER_URL")
        or _env_str("REDIS_URL", "redis://localhost:6379/0")
    )
    CELERY_RESULT_BACKEND: str = Field(
        default_factory=lambda: _env_optional_str("CELERY_RESULT_BACKEND")
        or _env_str("REDIS_URL", "redis://localhost:6379/0")
    )
    CELERY_INGESTION_QUEUE: str = Field(default_factory=lambda: _env_str("CELERY_INGESTION_QUEUE", "ingestion"))
    CELERY_OCR_QUEUE: str = Field(default_factory=lambda: _env_str("CELERY_OCR_QUEUE", "ocr"))
    CELERY_WORKER_CONCURRENCY: int = Field(
        default_factory=lambda: _env_int("CELERY_WORKER_CONCURRENCY", 1)
    )
    CELERY_INGESTION_EMBEDDING_BATCH_SIZE: int = Field(
        default_factory=lambda: _env_int("CELERY_INGESTION_EMBEDDING_BATCH_SIZE", 1)
    )
    CELERY_TASK_MAX_RETRIES: int = Field(default_factory=lambda: _env_int("CELERY_TASK_MAX_RETRIES", 3))
    CELERY_TASK_RETRY_BACKOFF_BASE: int = Field(
        default_factory=lambda: _env_int("CELERY_TASK_RETRY_BACKOFF_BASE", 5)
    )
    CELERY_TASK_RETRY_BACKOFF_MAX: int = Field(
        default_factory=lambda: _env_int("CELERY_TASK_RETRY_BACKOFF_MAX", 300)
    )
    CELERY_TASK_SOFT_TIME_LIMIT: int = Field(
        default_factory=lambda: _env_int("CELERY_TASK_SOFT_TIME_LIMIT", 300)
    )
    CELERY_TASK_TIME_LIMIT: int = Field(default_factory=lambda: _env_int("CELERY_TASK_TIME_LIMIT", 360))

    # Embedding Settings (BAAI/bge-m3, 1024-dim; weights loaded lazily by embedding service only)
    EMBEDDING_PROVIDER: str = Field(default_factory=lambda: _env_str("EMBEDDING_PROVIDER", "bge-m3"))
    EMBEDDING_MODEL_NAME: str = Field(
        default_factory=lambda: _env_str("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    )
    EMBEDDING_DIM: int = Field(default_factory=lambda: _env_int("EMBEDDING_DIM", 1024))
    EMBEDDING_BATCH_SIZE: int = Field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 4))
    EMBEDDING_DEVICE: str = Field(default_factory=lambda: _env_str("EMBEDDING_DEVICE", "auto"))
    EMBEDDING_NORMALIZE: bool = Field(default_factory=lambda: _env_bool("EMBEDDING_NORMALIZE", True))
    EMBEDDING_LOCAL_FILES_ONLY: bool = Field(
        default_factory=lambda: _env_bool("EMBEDDING_LOCAL_FILES_ONLY", False)
    )

    # LLM Settings
    LLM_PROVIDER: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER") or "nvidia")
    GEMINI_API_KEY: Optional[str] = Field(default_factory=lambda: _env_optional_str("GEMINI_API_KEY"))
    GEMINI_MODEL: str = Field(default_factory=lambda: _env_str("GEMINI_MODEL", "gemini-2.0-flash"))
    LLM_TEMPERATURE: float = 0.0  # Zero temperature for deterministic grounding

    # NVIDIA Nemotron Provider Settings
    NVIDIA_API_KEY: Optional[str] = Field(default_factory=lambda: _env_optional_str("NVIDIA_API_KEY"))
    NVIDIA_BASE_URL: str = Field(
        default_factory=lambda: os.getenv("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1"
    )
    NVIDIA_MODEL: str = Field(
        default_factory=lambda: os.getenv("NVIDIA_MODEL") or "nvidia/nemotron-3.5-lightning-30b-a3b"
    )
    NVIDIA_TIMEOUT_SECONDS: float = Field(default_factory=lambda: _env_float("NVIDIA_TIMEOUT_SECONDS", 60.0))
    NVIDIA_MAX_RETRIES: int = Field(default_factory=lambda: _env_int("NVIDIA_MAX_RETRIES", 2))
    NVIDIA_MAX_TOKENS: int = Field(default_factory=lambda: _env_int("NVIDIA_MAX_TOKENS", 8192))
    NVIDIA_TEMPERATURE: float = Field(default_factory=lambda: _env_float("NVIDIA_TEMPERATURE", 1.0))

    # Retrieval & Reranker Settings
    RRF_K: int = 60
    WEIGHT_VECTOR: float = 0.6
    WEIGHT_BM25: float = 0.4
    INITIAL_TOP_K: int = 5
    RETRY_TOP_K_INCREMENT: int = 3
    MAX_RETRIES: int = 2
    CONFIDENCE_THRESHOLD: float = 0.70
    PGVECTOR_HNSW_EF_SEARCH: int = Field(default_factory=lambda: _env_int("PGVECTOR_HNSW_EF_SEARCH", 40))
    PGVECTOR_HNSW_ITERATIVE_SCAN: str = Field(
        default_factory=lambda: _env_str("PGVECTOR_HNSW_ITERATIVE_SCAN", "relaxed_order")
    )

    # Cross-encoder model
    RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Ingestion & PDF Extraction Settings
    MAX_PDF_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB
    ALLOWED_PDF_EXTENSIONS: tuple = (".pdf",)

    # OCR & Scanned Detection Settings
    OCR_MIN_DIGITAL_CHARS: int = 100
    OCR_IMAGE_COVERAGE_THRESHOLD: float = 0.40
    OCR_MIN_BLOCK_COUNT: int = 3
    OCR_DPI: int = 200
    OCR_LANG: str = os.getenv("OCR_LANG", "en")
    OCR_USE_ANGLE_CLS: bool = True

    @property
    def is_non_test_env(self) -> bool:
        """Return True when running in production/staging or with DEBUG explicitly disabled."""
        return self.APP_ENV in _NON_TEST_ENVS or not self.DEBUG

    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        """Validate and normalize service, database, and embedding configuration values."""
        self.APP_ENV = (self.APP_ENV or "development").strip().lower() or "development"

        if self.PORT < 1 or self.PORT > 65535:
            raise ValueError("PORT must be between 1 and 65535.")

        if self.is_non_test_env:
            explicit_mock_env = os.getenv("USE_MOCK_STORAGE") is not None or "USE_MOCK_STORAGE" in self.model_fields_set
            if not explicit_mock_env:
                raise ValueError(
                    "USE_MOCK_STORAGE must be explicitly set to 'false' in non-test environments."
                )
            if self.USE_MOCK_STORAGE:
                raise ValueError(
                    "USE_MOCK_STORAGE=true is forbidden in non-test environments; PostgreSQL persistence is required."
                )
            if self.LLM_PROVIDER.strip().lower() in ("offline", "mock"):
                raise ValueError(
                    f"LLM_PROVIDER={self.LLM_PROVIDER!r} is forbidden in non-test environments."
                )
            if self.EMBEDDING_PROVIDER.strip().lower() == "offline":
                raise ValueError(
                    "EMBEDDING_PROVIDER='offline' is forbidden in non-test environments."
                )

        provider = self.EMBEDDING_PROVIDER.strip().lower()
        if provider not in ("bge-m3", "offline"):
            raise ValueError("EMBEDDING_PROVIDER must be 'bge-m3' or 'offline'.")
        self.EMBEDDING_PROVIDER = provider

        model_name = self.EMBEDDING_MODEL_NAME.strip()
        if not model_name:
            raise ValueError("EMBEDDING_MODEL_NAME must not be empty.")
        self.EMBEDDING_MODEL_NAME = model_name

        if self.EMBEDDING_DIM <= 0:
            raise ValueError("EMBEDDING_DIM must be a positive integer.")
        if self.EMBEDDING_PROVIDER == "bge-m3" and self.EMBEDDING_DIM != 1024:
            raise ValueError("EMBEDDING_DIM must be 1024 when EMBEDDING_PROVIDER is 'bge-m3'.")

        if self.EMBEDDING_BATCH_SIZE <= 0:
            raise ValueError("EMBEDDING_BATCH_SIZE must be a positive integer.")

        device = self.EMBEDDING_DEVICE.strip().lower()
        is_cuda_indexed = device.startswith("cuda:") and device[5:].isdigit()
        if device not in ("auto", "cpu", "cuda") and not is_cuda_indexed:
            raise ValueError("EMBEDDING_DEVICE must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'.")
        self.EMBEDDING_DEVICE = device

        if self.POSTGRES_POOL_MIN_SIZE < 1:
            raise ValueError("POSTGRES_POOL_MIN_SIZE must be at least 1.")
        if self.POSTGRES_POOL_MAX_SIZE < self.POSTGRES_POOL_MIN_SIZE:
            raise ValueError("POSTGRES_POOL_MAX_SIZE cannot be less than POSTGRES_POOL_MIN_SIZE.")
        if self.POSTGRES_POOL_TIMEOUT <= 0:
            raise ValueError("POSTGRES_POOL_TIMEOUT must be positive.")

        if self.PGVECTOR_HNSW_EF_SEARCH < 1:
            raise ValueError("PGVECTOR_HNSW_EF_SEARCH must be at least 1.")

        iter_scan = self.PGVECTOR_HNSW_ITERATIVE_SCAN.strip().lower()
        if iter_scan not in ("off", "relaxed_order", "strict_order"):
            raise ValueError("PGVECTOR_HNSW_ITERATIVE_SCAN must be 'off', 'relaxed_order', or 'strict_order'.")
        self.PGVECTOR_HNSW_ITERATIVE_SCAN = iter_scan

        if self.CELERY_WORKER_CONCURRENCY < 1:
            raise ValueError("CELERY_WORKER_CONCURRENCY must be at least 1.")
        if self.CELERY_INGESTION_EMBEDDING_BATCH_SIZE < 1:
            raise ValueError("CELERY_INGESTION_EMBEDDING_BATCH_SIZE must be at least 1.")
        if self.CELERY_TASK_MAX_RETRIES < 0:
            raise ValueError("CELERY_TASK_MAX_RETRIES must be non-negative.")
        if self.CELERY_TASK_RETRY_BACKOFF_BASE < 0 or self.CELERY_TASK_RETRY_BACKOFF_MAX < 0:
            raise ValueError("Celery retry backoff settings must be non-negative.")
        if self.CELERY_TASK_SOFT_TIME_LIMIT < 1 or self.CELERY_TASK_TIME_LIMIT < self.CELERY_TASK_SOFT_TIME_LIMIT:
            raise ValueError("CELERY_TASK_TIME_LIMIT must be >= CELERY_TASK_SOFT_TIME_LIMIT >= 1.")

        return self

    def __repr__(self) -> str:
        nvidia_state = "configured" if self.NVIDIA_API_KEY else "unset"
        gemini_state = "configured" if self.GEMINI_API_KEY else "unset"
        token_state = "configured" if self.INTERNAL_SERVICE_TOKEN else "unset"
        return (
            f"Settings(SERVICE_NAME={self.SERVICE_NAME!r}, HOST={self.HOST!r}, PORT={self.PORT}, "
            f"USE_MOCK_STORAGE={self.USE_MOCK_STORAGE}, DATABASE_URL={_redact_url_credentials(self.DATABASE_URL)!r}, "
            f"EMBEDDING_PROVIDER={self.EMBEDDING_PROVIDER!r}, EMBEDDING_MODEL_NAME={self.EMBEDDING_MODEL_NAME!r}, "
            f"EMBEDDING_DIM={self.EMBEDDING_DIM}, EMBEDDING_BATCH_SIZE={self.EMBEDDING_BATCH_SIZE}, "
            f"EMBEDDING_DEVICE={self.EMBEDDING_DEVICE!r}, LLM_PROVIDER={self.LLM_PROVIDER!r}, "
            f"NVIDIA_API_KEY={nvidia_state}, GEMINI_API_KEY={gemini_state}, INTERNAL_SERVICE_TOKEN={token_state})"
        )


settings = Settings()
