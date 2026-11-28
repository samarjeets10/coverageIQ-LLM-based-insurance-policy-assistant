"""Application settings loaded from environment variables and ai_service/.env."""

from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL, make_url

_SERVICE_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_DEV_ENVS: frozenset[str] = frozenset({"development", "dev", "local", "test", "testing"})


class Settings(BaseSettings):
    """Typed runtime settings for the FastAPI AI Service (Module M5 + Unified App)."""

    model_config = SettingsConfigDict(
        env_file=_SERVICE_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    ai_service_host: str = Field(default="0.0.0.0")
    ai_service_port: int = Field(default=8001, ge=1, le=65535)
    ai_service_env: str = Field(default="development")
    app_env: str | None = Field(default=None)
    ai_service_token: SecretStr | None = Field(default=None)
    internal_service_token: SecretStr | None = Field(default=None)
    ai_service_require_auth: bool | None = Field(default=None)
    max_request_body_bytes: int = Field(default=65_536, ge=1_024, le=1_048_576)

    database_url: str | None = Field(default=None)
    db_host: str = Field(default="localhost")
    db_port: int = Field(default=5432, ge=1, le=65535)
    db_name: str = Field(default="coverageiq_db")
    db_user: str = Field(default="postgres")
    db_password: SecretStr | None = Field(default=None)
    db_ssl: bool = Field(default=False)
    db_pool_size: int = Field(default=5, ge=1, le=50)
    db_max_overflow: int = Field(default=5, ge=0, le=50)
    db_connect_timeout_seconds: int = Field(default=5, ge=1, le=60)
    db_statement_timeout_ms: int = Field(default=10_000, ge=500, le=300_000)
    db_pool_timeout_seconds: float = Field(default=10.0, ge=1.0, le=120.0)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    log_json: bool = Field(default=True)

    model_artifacts_dir: Path = Field(default=_SERVICE_ROOT / "artifacts" / "models")

    @field_validator("ai_service_env", mode="before")
    @classmethod
    def _normalize_service_env(cls, value: Any) -> Any:
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if not cleaned:
                raise ValueError("AI_SERVICE_ENV cannot be empty.")
            return cleaned
        return value

    @field_validator("ai_service_token", "internal_service_token", mode="before")
    @classmethod
    def _normalize_empty_token(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("database_url", "app_env", mode="before")
    @classmethod
    def _normalize_optional_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @property
    def is_development_env(self) -> bool:
        """Return True only when both AI_SERVICE_ENV and APP_ENV are local dev/test names."""
        env_ok = self.ai_service_env.strip().lower() in _LOCAL_DEV_ENVS
        if self.app_env is not None:
            return env_ok and self.app_env.strip().lower() in _LOCAL_DEV_ENVS
        return env_ok

    @property
    def effective_service_token(self) -> str:
        """Resolve shared internal service token from AI_SERVICE_TOKEN or INTERNAL_SERVICE_TOKEN."""
        if self.ai_service_token is not None:
            val = self.ai_service_token.get_secret_value().strip()
            if val:
                return val
        if self.internal_service_token is not None:
            val = self.internal_service_token.get_secret_value().strip()
            if val:
                return val
        return (os.getenv("INTERNAL_SERVICE_TOKEN") or "").strip()

    @property
    def auth_enforced(self) -> bool:
        """Return True when X-Service-Token verification is required for requests."""
        return (
            not self.is_development_env
            or self.ai_service_require_auth is True
            or bool(self.effective_service_token)
        )

    @model_validator(mode="after")
    def _validate_security_configuration(self) -> "Settings":
        token_text = (
            self.ai_service_token.get_secret_value().strip()
            if self.ai_service_token is not None
            else (
                self.internal_service_token.get_secret_value().strip()
                if self.internal_service_token is not None
                else ""
            )
        )
        if not self.is_development_env:
            if self.ai_service_require_auth is False:
                raise ValueError(
                    "Authentication cannot be disabled when AI_SERVICE_ENV is non-development."
                )
            if not token_text:
                raise ValueError(
                    "AI_SERVICE_TOKEN must be configured with a non-empty secret when "
                    "AI_SERVICE_ENV is non-development."
                )
        elif self.ai_service_token is not None and not token_text:
            raise ValueError(
                "AI_SERVICE_TOKEN cannot be empty or whitespace-only when explicitly provided."
            )
        elif self.ai_service_require_auth is True and not token_text:
            raise ValueError(
                "AI_SERVICE_TOKEN must be configured when AI_SERVICE_REQUIRE_AUTH is True."
            )
        return self

    def build_database_url(self) -> URL:
        """Construct a safe SQLAlchemy PostgreSQL URL from DB_* fields or DATABASE_URL."""
        if self.db_password is None:
            raw_dsn = (self.database_url or os.getenv("DATABASE_URL") or "").strip()
            if raw_dsn:
                parsed = make_url(raw_dsn)
                if parsed.drivername in ("postgres", "postgresql"):
                    parsed = parsed.set(drivername="postgresql+psycopg")
                return parsed

        query = {"sslmode": "require"} if self.db_ssl else {}
        password = self.db_password.get_secret_value() if self.db_password else None
        return URL.create(
            drivername="postgresql+psycopg",
            username=self.db_user,
            password=password,
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
            query=query,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance for dependency injection."""
    return Settings()
