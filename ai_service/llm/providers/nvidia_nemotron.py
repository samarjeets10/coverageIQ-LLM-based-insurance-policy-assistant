import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

import httpx
from pydantic import BaseModel

from ai_service.config import Settings, settings

logger = logging.getLogger("coverageiq.llm.providers.nvidia_nemotron")


class NvidiaUsage(BaseModel):
    """Token usage statistics returned by the NVIDIA Chat Completions endpoint."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class NvidiaCompletionResult(BaseModel):
    """Structured result for a single non-streaming NVIDIA Nemotron completion."""

    content: str
    model: str
    usage: Optional[NvidiaUsage] = None
    finish_reason: Optional[str] = None
    request_id: Optional[str] = None


class NvidiaProviderError(Exception):
    """Base exception for NVIDIA Nemotron provider failures."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class NvidiaConfigurationError(NvidiaProviderError):
    """Raised when required NVIDIA provider configuration is missing or invalid."""


class NvidiaAuthenticationError(NvidiaProviderError):
    """Raised on HTTP 401 authentication failure."""


class NvidiaPermissionError(NvidiaProviderError):
    """Raised on HTTP 403 permission or model access denial."""


class NvidiaRateLimitError(NvidiaProviderError):
    """Raised on HTTP 429 rate-limit responses."""


class NvidiaServerError(NvidiaProviderError):
    """Raised on HTTP 5xx server failures."""


class NvidiaClientError(NvidiaProviderError):
    """Raised on non-retryable HTTP 4xx request failures."""


class NvidiaTimeoutError(NvidiaProviderError):
    """Raised when the NVIDIA API request times out."""


class NvidiaConnectionError(NvidiaProviderError):
    """Raised when a network or transport error prevents contacting the NVIDIA API."""


class NvidiaResponseError(NvidiaProviderError):
    """Raised when the NVIDIA API returns an empty, missing, truncated, or malformed response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        retryable: bool = False,
        finish_reason: Optional[str] = None,
        request_id: Optional[str] = None,
        usage: Optional[NvidiaUsage] = None,
    ):
        super().__init__(message, status_code=status_code, retryable=retryable)
        self.finish_reason = finish_reason
        self.request_id = request_id
        self.usage = usage

    def safe_metadata(self) -> Dict[str, Any]:
        """Return redacted response metadata suitable for diagnostics and logs."""
        return {
            "status_code": self.status_code,
            "finish_reason": self.finish_reason,
            "request_id": self.request_id,
            "usage": self.usage.model_dump() if self.usage is not None else None,
        }


class NvidiaTokenLimitError(NvidiaResponseError):
    """Raised when NVIDIA generation terminates due to max_tokens exhaustion (finish_reason='length')."""


class NvidiaNemotronProvider:
    """
    Minimal, reusable adapter for NVIDIA-hosted Chat Completions API.
    Uses httpx against the OpenAI-compatible /chat/completions endpoint with
    Bearer authentication, bounded retries for transient errors, and strict
    credential/payload redaction in logs and exceptions.
    """

    DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
    DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        backoff_base_seconds: float = 1.0,
        app_settings: Optional[Settings] = None,
        http_client: Optional[httpx.Client] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ):
        cfg = app_settings or settings

        self._api_key = self._resolve_secret(api_key, "NVIDIA_API_KEY", cfg.NVIDIA_API_KEY)
        resolved_base = self._resolve_str(base_url, "NVIDIA_BASE_URL", cfg.NVIDIA_BASE_URL, self.DEFAULT_BASE_URL)
        self.base_url = resolved_base.rstrip("/")
        self.model = self._resolve_str(model, "NVIDIA_MODEL", cfg.NVIDIA_MODEL, self.DEFAULT_MODEL)
        self.timeout_seconds = self._resolve_float(
            timeout_seconds, "NVIDIA_TIMEOUT_SECONDS", cfg.NVIDIA_TIMEOUT_SECONDS
        )
        self.max_retries = self._resolve_int(max_retries, "NVIDIA_MAX_RETRIES", cfg.NVIDIA_MAX_RETRIES)
        self.max_tokens = self._resolve_int(max_tokens, "NVIDIA_MAX_TOKENS", cfg.NVIDIA_MAX_TOKENS)
        self.temperature = self._resolve_float(temperature, "NVIDIA_TEMPERATURE", cfg.NVIDIA_TEMPERATURE)
        self.backoff_base_seconds = max(0.0, float(backoff_base_seconds))

        if not self.base_url.startswith(("https://", "http://")):
            raise NvidiaConfigurationError("NVIDIA_BASE_URL must start with http:// or https://.")
        if self.timeout_seconds <= 0:
            raise NvidiaConfigurationError("NVIDIA_TIMEOUT_SECONDS must be positive.")
        if self.max_retries < 0:
            raise NvidiaConfigurationError("NVIDIA_MAX_RETRIES cannot be negative.")
        if self.max_tokens <= 0:
            raise NvidiaConfigurationError("NVIDIA_MAX_TOKENS must be positive.")

        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=self.timeout_seconds)
        self._sleep = sleep_fn or time.sleep

    def __repr__(self) -> str:
        key_state = "configured" if self._api_key else "unset"
        return (
            f"NvidiaNemotronProvider(model={self.model!r}, base_url={self.base_url!r}, "
            f"api_key={key_state}, timeout_seconds={self.timeout_seconds}, "
            f"max_retries={self.max_retries}, max_tokens={self.max_tokens}, temperature={self.temperature})"
        )

    @property
    def is_closed(self) -> bool:
        return self._client is None or getattr(self._client, "is_closed", False)

    def close(self) -> None:
        if self._owns_client and self._client is not None and not getattr(self._client, "is_closed", False):
            self._client.close()

    def __enter__(self) -> "NvidiaNemotronProvider":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @staticmethod
    def _resolve_secret(explicit: Optional[str], env_name: str, cfg_value: Optional[str]) -> Optional[str]:
        if explicit is not None:
            val = explicit.strip()
            return val or None
        env_val = os.getenv(env_name)
        if env_val is not None:
            val = env_val.strip()
            return val or None
        if cfg_value is not None:
            val = cfg_value.strip()
            return val or None
        return None

    @staticmethod
    def _resolve_str(explicit: Optional[str], env_name: str, cfg_value: Optional[str], default: str) -> str:
        if explicit is not None and explicit.strip():
            return explicit.strip()
        env_val = os.getenv(env_name)
        if env_val is not None and env_val.strip():
            return env_val.strip()
        if cfg_value is not None and cfg_value.strip():
            return cfg_value.strip()
        return default

    @staticmethod
    def _resolve_float(explicit: Optional[float], env_name: str, cfg_value: float) -> float:
        if explicit is not None:
            return float(explicit)
        env_val = os.getenv(env_name)
        if env_val is not None and env_val.strip():
            try:
                return float(env_val)
            except ValueError as exc:
                raise NvidiaConfigurationError(f"Invalid float value for {env_name}.") from exc
        return float(cfg_value)

    @staticmethod
    def _resolve_int(explicit: Optional[int], env_name: str, cfg_value: int) -> int:
        if explicit is not None:
            return int(explicit)
        env_val = os.getenv(env_name)
        if env_val is not None and env_val.strip():
            try:
                return int(env_val)
            except ValueError as exc:
                raise NvidiaConfigurationError(f"Invalid integer value for {env_name}.") from exc
        return int(cfg_value)

    def _redact(self, text: str) -> str:
        if self._api_key and self._api_key in text:
            return text.replace(self._api_key, "[REDACTED]")
        return text

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Sends a non-streaming chat completion request and returns assistant text content.
        """
        result = self.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return result.content

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> NvidiaCompletionResult:
        """
        Sends a non-streaming /chat/completions request and returns content plus usage metadata.
        """
        if self.is_closed:
            raise NvidiaConfigurationError("NVIDIA provider HTTP client is closed.")
        if not self._api_key:
            raise NvidiaConfigurationError(
                "NVIDIA_API_KEY is missing or empty. Set NVIDIA_API_KEY before calling NvidiaNemotronProvider."
            )
        if not user_prompt or not user_prompt.strip():
            raise NvidiaConfigurationError("user_prompt must be a non-empty string.")

        eff_temperature = self.temperature if temperature is None else float(temperature)
        eff_max_tokens = self.max_tokens if max_tokens is None else int(max_tokens)
        if eff_max_tokens <= 0:
            raise NvidiaConfigurationError("max_tokens must be positive.")

        messages: List[Dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": eff_temperature,
            "max_tokens": eff_max_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        total_attempts = 1 + self.max_retries
        last_error: Optional[NvidiaProviderError] = None

        for attempt in range(total_attempts):
            try:
                logger.info(
                    "NVIDIA Chat Completions request started: model=%s attempt=%d/%d",
                    self.model,
                    attempt + 1,
                    total_attempts,
                )
                response = self._client.post(
                    self.chat_completions_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except httpx.TimeoutException as exc:
                last_error = NvidiaTimeoutError(
                    f"NVIDIA API request timed out after {self.timeout_seconds}s: {self._redact(str(exc))}",
                    retryable=True,
                )
                logger.warning(
                    "NVIDIA API timeout on attempt %d/%d for model=%s",
                    attempt + 1,
                    total_attempts,
                    self.model,
                )
                if attempt + 1 < total_attempts:
                    self._backoff(attempt)
                    continue
                raise last_error from exc
            except httpx.RequestError as exc:
                last_error = NvidiaConnectionError(
                    f"NVIDIA API connection error: {self._redact(str(exc))}",
                    retryable=True,
                )
                logger.warning(
                    "NVIDIA API connection failure on attempt %d/%d for model=%s: %s",
                    attempt + 1,
                    total_attempts,
                    self.model,
                    self._redact(type(exc).__name__),
                )
                if attempt + 1 < total_attempts:
                    self._backoff(attempt)
                    continue
                raise last_error from exc

            status = response.status_code
            if status == 200:
                return self._parse_success_response(response)

            error = self._map_http_error(response)
            if error.retryable and attempt + 1 < total_attempts:
                logger.warning(
                    "Transient NVIDIA API error (status=%d) on attempt %d/%d for model=%s; retrying.",
                    status,
                    attempt + 1,
                    total_attempts,
                    self.model,
                )
                last_error = error
                self._backoff(attempt)
                continue

            logger.error(
                "NVIDIA API request failed: status=%d model=%s retryable=%s",
                status,
                self.model,
                error.retryable,
            )
            raise error

        assert last_error is not None  # pragma: no cover
        raise last_error

    def _backoff(self, attempt: int) -> None:
        delay = self.backoff_base_seconds * (2 ** attempt)
        if delay > 0:
            self._sleep(delay)

    def _map_http_error(self, response: httpx.Response) -> NvidiaProviderError:
        status = response.status_code
        detail = self._extract_error_detail(response)

        if status == 401:
            return NvidiaAuthenticationError(
                f"NVIDIA API authentication failed (HTTP 401){detail}.",
                status_code=status,
                retryable=False,
            )
        if status == 403:
            return NvidiaPermissionError(
                f"NVIDIA API permission or model access denied (HTTP 403){detail}.",
                status_code=status,
                retryable=False,
            )
        if status == 429:
            return NvidiaRateLimitError(
                f"NVIDIA API rate limit exceeded (HTTP 429){detail}.",
                status_code=status,
                retryable=True,
            )
        if 500 <= status < 600:
            return NvidiaServerError(
                f"NVIDIA API server error (HTTP {status}){detail}.",
                status_code=status,
                retryable=True,
            )
        return NvidiaClientError(
            f"NVIDIA API request rejected (HTTP {status}){detail}.",
            status_code=status,
            retryable=False,
        )

    def _extract_error_detail(self, response: httpx.Response) -> str:
        try:
            body = response.json()
            if isinstance(body, dict):
                err = body.get("error")
                if isinstance(err, dict) and err.get("message"):
                    return f": {self._redact(str(err['message'])[:200])}"
                if body.get("detail"):
                    return f": {self._redact(str(body['detail'])[:200])}"
                if body.get("message"):
                    return f": {self._redact(str(body['message'])[:200])}"
        except Exception:
            pass
        return ""

    def _parse_success_response(self, response: httpx.Response) -> NvidiaCompletionResult:
        request_id_hdr = response.headers.get("x-request-id")
        try:
            data = response.json()
        except Exception as exc:
            raise NvidiaResponseError(
                f"NVIDIA API returned malformed JSON: {self._redact(str(exc))}",
                status_code=response.status_code,
                request_id=str(request_id_hdr) if request_id_hdr is not None else None,
            ) from exc

        if not isinstance(data, dict):
            raise NvidiaResponseError(
                "NVIDIA API returned a non-object JSON payload.",
                status_code=response.status_code,
                request_id=str(request_id_hdr) if request_id_hdr is not None else None,
            )

        usage_obj: Optional[NvidiaUsage] = None
        raw_usage = data.get("usage")
        if isinstance(raw_usage, dict):
            usage_obj = NvidiaUsage(
                prompt_tokens=raw_usage.get("prompt_tokens"),
                completion_tokens=raw_usage.get("completion_tokens"),
                total_tokens=raw_usage.get("total_tokens"),
            )

        request_id = request_id_hdr or data.get("id")
        req_id_str = str(request_id) if request_id is not None else None
        response_model = str(data.get("model") or self.model)

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise NvidiaResponseError(
                "NVIDIA API response missing non-empty 'choices' list.",
                status_code=response.status_code,
                request_id=req_id_str,
                usage=usage_obj,
            )

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise NvidiaResponseError(
                "NVIDIA API response choice[0] is malformed.",
                status_code=response.status_code,
                request_id=req_id_str,
                usage=usage_obj,
            )

        raw_finish = first_choice.get("finish_reason")
        finish_reason = str(raw_finish) if raw_finish is not None else None

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise NvidiaResponseError(
                "NVIDIA API response choice[0] missing 'message' object.",
                status_code=response.status_code,
                finish_reason=finish_reason,
                request_id=req_id_str,
                usage=usage_obj,
            )

        content = message.get("content")
        has_nonempty_content = isinstance(content, str) and bool(content.strip())

        if finish_reason == "length":
            logger.error(
                "NVIDIA Chat Completions hit max_tokens ceiling: model=%s finish_reason=%s request_id=%s total_tokens=%s has_content=%s",
                response_model,
                finish_reason,
                req_id_str,
                usage_obj.total_tokens if usage_obj else None,
                has_nonempty_content,
            )
            raise NvidiaTokenLimitError(
                "NVIDIA API stopped due to max_tokens exhaustion (finish_reason='length').",
                status_code=response.status_code,
                retryable=False,
                finish_reason=finish_reason,
                request_id=req_id_str,
                usage=usage_obj,
            )

        if content is None or not isinstance(content, str) or not has_nonempty_content:
            reason_tag = "missing" if content is None else "empty"
            logger.error(
                "NVIDIA Chat Completions returned %s assistant content: model=%s finish_reason=%s request_id=%s total_tokens=%s",
                reason_tag,
                response_model,
                finish_reason,
                req_id_str,
                usage_obj.total_tokens if usage_obj else None,
            )
            raise NvidiaResponseError(
                f"NVIDIA API returned {reason_tag} assistant content.",
                status_code=response.status_code,
                retryable=False,
                finish_reason=finish_reason,
                request_id=req_id_str,
                usage=usage_obj,
            )

        logger.info(
            "NVIDIA Chat Completions succeeded: model=%s finish_reason=%s total_tokens=%s",
            response_model,
            finish_reason,
            usage_obj.total_tokens if usage_obj else None,
        )

        return NvidiaCompletionResult(
            content=content,
            model=response_model,
            usage=usage_obj,
            finish_reason=finish_reason,
            request_id=req_id_str,
        )
