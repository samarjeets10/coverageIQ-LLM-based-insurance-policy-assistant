from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from ai_service.config import Settings


_OPTIONAL_CONFIG_VARS = (
    "PORT",
    "HOST",
    "DEBUG",
    "INTERNAL_SERVICE_TOKEN",
    "POLICY_UPLOAD_ROOT",
    "DATABASE_URL",
    "USE_MOCK_STORAGE",
    "POSTGRES_POOL_MIN_SIZE",
    "POSTGRES_POOL_MAX_SIZE",
    "POSTGRES_POOL_TIMEOUT",
    "REDIS_URL",
    "CELERY_BROKER_URL",
    "CELERY_RESULT_BACKEND",
    "CELERY_INGESTION_QUEUE",
    "CELERY_OCR_QUEUE",
    "CELERY_WORKER_CONCURRENCY",
    "CELERY_INGESTION_EMBEDDING_BATCH_SIZE",
    "CELERY_TASK_MAX_RETRIES",
    "CELERY_TASK_RETRY_BACKOFF_BASE",
    "CELERY_TASK_RETRY_BACKOFF_MAX",
    "CELERY_TASK_SOFT_TIME_LIMIT",
    "CELERY_TASK_TIME_LIMIT",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL_NAME",
    "EMBEDDING_DIM",
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_DEVICE",
    "EMBEDDING_NORMALIZE",
    "EMBEDDING_LOCAL_FILES_ONLY",
    "LLM_PROVIDER",
    "NVIDIA_API_KEY",
    "NVIDIA_BASE_URL",
    "NVIDIA_MODEL",
    "NVIDIA_TIMEOUT_SECONDS",
    "NVIDIA_MAX_RETRIES",
    "NVIDIA_MAX_TOKENS",
    "NVIDIA_TEMPERATURE",
    "GEMINI_API_KEY",
    "PGVECTOR_HNSW_EF_SEARCH",
    "PGVECTOR_HNSW_ITERATIVE_SCAN",
)


def _clear_config_env(monkeypatch) -> None:
    for var in _OPTIONAL_CONFIG_VARS:
        monkeypatch.delenv(var, raising=False)


def test_phase4_configuration_defaults_without_loading_model(monkeypatch):
    _clear_config_env(monkeypatch)

    cfg = Settings()

    assert cfg.PORT == 8001
    assert cfg.HOST == "127.0.0.1"
    assert cfg.DEBUG is True
    assert cfg.INTERNAL_SERVICE_TOKEN is None
    assert cfg.POLICY_UPLOAD_ROOT == "./uploads/policies"

    # Database & Pooling
    assert cfg.USE_MOCK_STORAGE is True
    assert cfg.DATABASE_URL == "postgresql://coverageiq:coverageiq@localhost:5432/coverageiq_db"
    assert cfg.POSTGRES_POOL_MIN_SIZE == 1
    assert cfg.POSTGRES_POOL_MAX_SIZE == 10
    assert cfg.POSTGRES_POOL_TIMEOUT == 30.0

    # Celery & Redis
    assert cfg.REDIS_URL == "redis://localhost:6379/0"
    assert cfg.CELERY_BROKER_URL == "redis://localhost:6379/0"
    assert cfg.CELERY_RESULT_BACKEND == "redis://localhost:6379/0"

    # Embeddings
    assert cfg.EMBEDDING_PROVIDER == "bge-m3"
    assert cfg.EMBEDDING_MODEL_NAME == "BAAI/bge-m3"
    assert cfg.EMBEDDING_DIM == 1024
    assert cfg.EMBEDDING_BATCH_SIZE == 4
    assert cfg.EMBEDDING_DEVICE == "auto"
    assert cfg.EMBEDDING_NORMALIZE is True
    assert cfg.EMBEDDING_LOCAL_FILES_ONLY is False
    assert cfg.PGVECTOR_HNSW_EF_SEARCH == 40
    assert cfg.PGVECTOR_HNSW_ITERATIVE_SCAN == "relaxed_order"

    # Config initialization in a clean process must never import FlagEmbedding weights
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from ai_service.config import Settings; Settings(); assert 'FlagEmbedding' not in sys.modules",
        ],
        check=True,
    )


def test_phase4_env_variable_overrides_and_secret_redaction(monkeypatch):
    _clear_config_env(monkeypatch)
    monkeypatch.setenv("PORT", "8005")
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "internal-secret-token-xyz")
    monkeypatch.setenv("POLICY_UPLOAD_ROOT", "/var/lib/coverageiq/uploads")
    monkeypatch.setenv("DATABASE_URL", "postgresql://ciq_user:supersecretpass@db-host:5432/ciq_prod")
    monkeypatch.setenv("USE_MOCK_STORAGE", "false")
    monkeypatch.setenv("POSTGRES_POOL_MIN_SIZE", "2")
    monkeypatch.setenv("POSTGRES_POOL_MAX_SIZE", "24")
    monkeypatch.setenv("POSTGRES_POOL_TIMEOUT", "15.5")
    monkeypatch.setenv("REDIS_URL", "redis://redis-host:6379/2")
    monkeypatch.setenv("CELERY_BROKER_URL", "redis://redis-host:6379/3")
    monkeypatch.setenv("CELERY_RESULT_BACKEND", "redis://redis-host:6379/4")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "custom/test-model")
    monkeypatch.setenv("EMBEDDING_DIM", "256")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "16")
    monkeypatch.setenv("EMBEDDING_DEVICE", "CPU")
    monkeypatch.setenv("EMBEDDING_NORMALIZE", "false")
    monkeypatch.setenv("EMBEDDING_LOCAL_FILES_ONLY", "true")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-dummy-secret-check")

    cfg = Settings()

    assert cfg.PORT == 8005
    assert cfg.INTERNAL_SERVICE_TOKEN == "internal-secret-token-xyz"
    assert cfg.POLICY_UPLOAD_ROOT == "/var/lib/coverageiq/uploads"
    assert cfg.USE_MOCK_STORAGE is False
    assert cfg.POSTGRES_POOL_MIN_SIZE == 2
    assert cfg.POSTGRES_POOL_MAX_SIZE == 24
    assert cfg.POSTGRES_POOL_TIMEOUT == 15.5
    assert cfg.REDIS_URL == "redis://redis-host:6379/2"
    assert cfg.CELERY_BROKER_URL == "redis://redis-host:6379/3"
    assert cfg.CELERY_RESULT_BACKEND == "redis://redis-host:6379/4"
    assert cfg.EMBEDDING_PROVIDER == "offline"
    assert cfg.EMBEDDING_MODEL_NAME == "custom/test-model"
    assert cfg.EMBEDDING_DIM == 256
    assert cfg.EMBEDDING_BATCH_SIZE == 16
    assert cfg.EMBEDDING_DEVICE == "cpu"
    assert cfg.EMBEDDING_NORMALIZE is False
    assert cfg.EMBEDDING_LOCAL_FILES_ONLY is True

    rendered = repr(cfg)
    assert "supersecretpass" not in rendered
    assert "internal-secret-token-xyz" not in rendered
    assert "nvapi-dummy-secret-check" not in rendered
    assert "configured" in rendered


@pytest.mark.parametrize("bad_dim", [0, -1, -1024, 768])
def test_invalid_embedding_dimensions_rejected(monkeypatch, bad_dim):
    _clear_config_env(monkeypatch)
    with pytest.raises(ValidationError):
        Settings(EMBEDDING_PROVIDER="bge-m3", EMBEDDING_DIM=bad_dim)


def test_invalid_embedding_dim_from_env(monkeypatch):
    _clear_config_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_DIM", "not-an-integer")
    with pytest.raises(ValueError):
        Settings()
    monkeypatch.setenv("EMBEDDING_DIM", "0")
    with pytest.raises(ValidationError):
        Settings()
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize("bad_batch", [0, -1, -8])
def test_invalid_embedding_batch_sizes_rejected(monkeypatch, bad_batch):
    _clear_config_env(monkeypatch)
    with pytest.raises(ValidationError):
        Settings(EMBEDDING_BATCH_SIZE=bad_batch)


def test_invalid_embedding_batch_size_from_env(monkeypatch):
    _clear_config_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "abc")
    with pytest.raises(ValueError):
        Settings()
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "0")
    with pytest.raises(ValidationError):
        Settings()


def test_invalid_pool_device_and_port_rejected(monkeypatch):
    _clear_config_env(monkeypatch)
    with pytest.raises(ValidationError):
        Settings(EMBEDDING_DEVICE="tpu")
    with pytest.raises(ValidationError):
        Settings(POSTGRES_POOL_MIN_SIZE=10, POSTGRES_POOL_MAX_SIZE=5)
    with pytest.raises(ValidationError):
        Settings(POSTGRES_POOL_TIMEOUT=0.0)
    with pytest.raises(ValidationError):
        Settings(PORT=0)
    with pytest.raises(ValidationError):
        Settings(PGVECTOR_HNSW_EF_SEARCH=0)
    with pytest.raises(ValidationError):
        Settings(PGVECTOR_HNSW_ITERATIVE_SCAN="invalid_mode")
    monkeypatch.setenv("EMBEDDING_NORMALIZE", "definitely")
    with pytest.raises(ValueError):
        Settings()


def test_missing_optional_configuration_values(monkeypatch):
    _clear_config_env(monkeypatch)
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "   ")
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("REDIS_URL", "redis://cache:6379/5")

    cfg = Settings()
    assert cfg.INTERNAL_SERVICE_TOKEN is None
    assert cfg.NVIDIA_API_KEY is None
    assert cfg.GEMINI_API_KEY is None
    assert cfg.CELERY_BROKER_URL == "redis://cache:6379/5"
    assert cfg.CELERY_RESULT_BACKEND == "redis://cache:6379/5"


def test_node_gateway_and_python_ai_service_port_separation(monkeypatch):
    _clear_config_env(monkeypatch)
    cfg = Settings()
    assert cfg.PORT == 8001

    repo_root = Path(__file__).resolve().parents[2]
    server_js = (repo_root / "server" / "server.js").read_text(encoding="utf-8")
    assert "process.env.SERVER_PORT || 8000" in server_js

    node_env_ex = (repo_root / "server" / ".env.example").read_text(encoding="utf-8")
    ai_env_ex = (repo_root / "ai_service" / ".env.example").read_text(encoding="utf-8")
    assert "SERVER_PORT=8000" in node_env_ex
    assert "AI_SERVICE_URL=http://127.0.0.1:8001" in node_env_ex
    assert "PORT=8001" in ai_env_ex


def test_phase4_celery_worker_configuration_defaults_overrides_and_validation(monkeypatch):
    _clear_config_env(monkeypatch)
    cfg = Settings()
    assert cfg.CELERY_INGESTION_QUEUE == "ingestion"
    assert cfg.CELERY_OCR_QUEUE == "ocr"
    assert cfg.CELERY_WORKER_CONCURRENCY == 1
    assert cfg.CELERY_INGESTION_EMBEDDING_BATCH_SIZE == 1
    assert cfg.CELERY_TASK_MAX_RETRIES == 3
    assert cfg.CELERY_TASK_RETRY_BACKOFF_BASE == 5
    assert cfg.CELERY_TASK_RETRY_BACKOFF_MAX == 300
    assert cfg.CELERY_TASK_SOFT_TIME_LIMIT == 300
    assert cfg.CELERY_TASK_TIME_LIMIT == 360

    monkeypatch.setenv("CELERY_INGESTION_QUEUE", "custom_ingestion")
    monkeypatch.setenv("CELERY_OCR_QUEUE", "custom_ocr")
    monkeypatch.setenv("CELERY_WORKER_CONCURRENCY", "2")
    monkeypatch.setenv("CELERY_INGESTION_EMBEDDING_BATCH_SIZE", "4")
    monkeypatch.setenv("CELERY_TASK_MAX_RETRIES", "5")
    monkeypatch.setenv("CELERY_TASK_RETRY_BACKOFF_BASE", "10")
    monkeypatch.setenv("CELERY_TASK_RETRY_BACKOFF_MAX", "120")
    monkeypatch.setenv("CELERY_TASK_SOFT_TIME_LIMIT", "180")
    monkeypatch.setenv("CELERY_TASK_TIME_LIMIT", "240")
    custom = Settings()
    assert custom.CELERY_INGESTION_QUEUE == "custom_ingestion"
    assert custom.CELERY_OCR_QUEUE == "custom_ocr"
    assert custom.CELERY_WORKER_CONCURRENCY == 2
    assert custom.CELERY_INGESTION_EMBEDDING_BATCH_SIZE == 4
    assert custom.CELERY_TASK_MAX_RETRIES == 5
    assert custom.CELERY_TASK_RETRY_BACKOFF_BASE == 10
    assert custom.CELERY_TASK_RETRY_BACKOFF_MAX == 120
    assert custom.CELERY_TASK_SOFT_TIME_LIMIT == 180
    assert custom.CELERY_TASK_TIME_LIMIT == 240

    with pytest.raises(ValidationError):
        Settings(CELERY_WORKER_CONCURRENCY=0)
    with pytest.raises(ValidationError):
        Settings(CELERY_INGESTION_EMBEDDING_BATCH_SIZE=0)
    with pytest.raises(ValidationError):
        Settings(CELERY_TASK_MAX_RETRIES=-1)
    with pytest.raises(ValidationError):
        Settings(CELERY_TASK_SOFT_TIME_LIMIT=120, CELERY_TASK_TIME_LIMIT=60)


def test_existing_nemotron_configuration_regression(monkeypatch):
    _clear_config_env(monkeypatch)
    cfg = Settings()
    assert cfg.LLM_PROVIDER == "nvidia"
    assert cfg.NVIDIA_BASE_URL == "https://integrate.api.nvidia.com/v1"
    assert cfg.NVIDIA_MODEL == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert cfg.NVIDIA_TIMEOUT_SECONDS == 60.0
    assert cfg.NVIDIA_MAX_RETRIES == 2
    assert cfg.NVIDIA_MAX_TOKENS == 8192
    assert cfg.NVIDIA_TEMPERATURE == 1.0


def test_requirements_and_migration_docs_match_phase4_decisions():
    repo_root = Path(__file__).resolve().parents[2]
    req_text = (repo_root / "ai_service" / "requirements.txt").read_text(encoding="utf-8")
    for dep in ("FlagEmbedding", "psycopg[binary]", "psycopg_pool", "pgvector", "celery", "redis"):
        assert dep in req_text

    roadmap = (repo_root / "doc" / "roadmap.md").read_text(encoding="utf-8")
    assert "node-pg-migrate" in roadmap
    assert "Alembic" not in roadmap
