import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


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


class Settings(BaseModel):
    # Service settings
    SERVICE_NAME: str = "coverageIQ-ai-service"
    PORT: int = 8000
    HOST: str = "127.0.0.1"
    DEBUG: bool = True

    # Database settings
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://coverageiq:coverageiq@localhost:5432/coverageiq_db"
    )
    USE_MOCK_STORAGE: bool = os.getenv("USE_MOCK_STORAGE", "true").lower() in ("true", "1", "yes")

    # LLM Settings
    LLM_PROVIDER: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER") or "nvidia")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", None)
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    LLM_TEMPERATURE: float = 0.0  # Zero temperature for deterministic grounding

    # NVIDIA Nemotron Provider Settings
    NVIDIA_API_KEY: Optional[str] = Field(default_factory=lambda: os.getenv("NVIDIA_API_KEY") or None)
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

settings = Settings()
