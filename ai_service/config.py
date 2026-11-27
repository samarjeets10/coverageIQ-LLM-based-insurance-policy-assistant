import os
from typing import Optional
from pydantic import BaseModel

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
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", None)
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    LLM_TEMPERATURE: float = 0.0  # Zero temperature for deterministic grounding
    
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
