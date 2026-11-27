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


class Settings(BaseModel):
    # Service settings (Node gateway runs on 8000; internal FastAPI AI service runs on 8001)
    SERVICE_NAME: str = "coverageIQ-ai-service"
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

    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        """Validate and normalize service, database, and embedding configuration values."""
        if self.PORT < 1 or self.PORT > 65535:
            raise ValueError("PORT must be between 1 and 65535.")

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
