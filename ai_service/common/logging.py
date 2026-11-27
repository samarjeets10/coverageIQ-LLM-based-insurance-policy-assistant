"""Structured JSON logging with strict secret and PII scrubbing (NFR-11)."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

_SENSITIVE_KEY_SUBSTRINGS: tuple[str, ...] = (
    "password",
    "secret",
    "token",
    "dsn",
    "authorization",
    "cookie",
    "credential",
    "api_key",
)

_PII_KEY_NAMES: frozenset[str] = frozenset(
    {
        "profile",
        "annual_income_inr",
        "annual_income",
        "pre_existing_conditions",
        "smoking_status",
        "location",
        "owner_scope",
        "user_id",
        "age",
        "family_size",
        "vehicle_age_years",
        "email",
        "phone",
        "full_name",
    }
)

_DB_URL_CREDENTIAL_PATTERN = re.compile(
    r"(postgres(?:ql)?(?:\+\w+)?://[^:\s/@]+:)([^@\s]+)(@)",
    re.IGNORECASE,
)


def _redact_string(value: str) -> str:
    """Scrub embedded database URL passwords from diagnostic strings."""
    return _DB_URL_CREDENTIAL_PATTERN.sub(r"\1[REDACTED]\3", value)


def _is_sensitive_key(key: str | None) -> bool:
    """Return True when a field key may hold secrets or customer PII."""
    if not key:
        return False
    lowered = key.strip().lower()
    if lowered in _PII_KEY_NAMES:
        return True
    return any(sub in lowered for sub in _SENSITIVE_KEY_SUBSTRINGS)


def sanitize_log_value(value: Any, *, key: str | None = None) -> Any:
    """Recursively scrub credentials, tokens, and user PII from log payloads."""
    if _is_sensitive_key(key):
        return "[REDACTED]"

    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, Mapping):
        return {str(k): sanitize_log_value(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [sanitize_log_value(item) for item in value]
    return _redact_string(str(value))


class StructuredJsonFormatter(logging.Formatter):
    """Render Python LogRecords as compact single-line JSON without PII or secrets."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize LogRecord metadata and sanitized structured context to JSON."""
        payload: dict[str, Any] = {
            "timestamp_utc": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": _redact_string(record.getMessage()),
        }

        event = getattr(record, "event", None)
        if event is not None:
            payload["event"] = sanitize_log_value(event)

        structured = getattr(record, "structured", None)
        if isinstance(structured, Mapping):
            sanitized_context = sanitize_log_value(structured)
            if isinstance(sanitized_context, Mapping):
                payload.update(sanitized_context)

        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def emit_structured_event(
    target_logger: logging.Logger,
    *,
    level: int,
    event: str,
    message: str,
    **fields: Any,
) -> None:
    """Emit a structured operational log event after scrubbing sensitive fields."""
    sanitized_fields = sanitize_log_value(fields)
    target_logger.log(
        level,
        _redact_string(message),
        extra={
            "event": event,
            "structured": sanitized_fields if isinstance(sanitized_fields, dict) else {},
        },
    )


def configure_logging(*, level: str = "INFO", json_format: bool = True) -> None:
    """Configure service loggers safely without breaking pytest capture handlers."""
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    namespace_loggers = (
        "ai_service",
        "ai_service.recommendation",
        "ai_service.common",
        "ai_service.main",
        "recommendation",
        "common",
        "main",
    )

    for name in namespace_loggers:
        logging.getLogger(name).setLevel(resolved_level)

    root_logger = logging.getLogger()
    if root_logger.level == logging.NOTSET or root_logger.level > resolved_level:
        root_logger.setLevel(resolved_level)

    formatter: logging.Formatter = (
        StructuredJsonFormatter()
        if json_format
        else logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )

    tagged_handler: logging.Handler | None = None
    for handler in root_logger.handlers:
        if getattr(handler, "_coverageiq_structured_handler", False):
            tagged_handler = handler
            break

    if tagged_handler is None:
        stream_handler = logging.StreamHandler()
        stream_handler._coverageiq_structured_handler = True  # type: ignore[attr-defined]
        stream_handler.setLevel(resolved_level)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
    else:
        tagged_handler.setLevel(resolved_level)
        tagged_handler.setFormatter(formatter)
