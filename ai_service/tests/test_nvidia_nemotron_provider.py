import json
import logging
from typing import List

import httpx
import pytest

from ai_service.config import Settings
from ai_service.llm.providers.nvidia_nemotron import (
    NvidiaAuthenticationError,
    NvidiaClientError,
    NvidiaConfigurationError,
    NvidiaConnectionError,
    NvidiaNemotronProvider,
    NvidiaPermissionError,
    NvidiaRateLimitError,
    NvidiaResponseError,
    NvidiaServerError,
    NvidiaTimeoutError,
)


def _make_provider(handler, *, api_key="nvapi-test-secret-12345", max_retries=2, sleep_calls=None, **kwargs):
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    sleeps: List[float] = sleep_calls if sleep_calls is not None else []
    provider = NvidiaNemotronProvider(
        api_key=api_key,
        max_retries=max_retries,
        backoff_base_seconds=0.5,
        http_client=client,
        sleep_fn=sleeps.append,
        **kwargs,
    )
    return provider, client, sleeps


def test_default_model_and_endpoint_configuration(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-env-secret-999")
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)
    monkeypatch.delenv("NVIDIA_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("NVIDIA_MAX_RETRIES", raising=False)
    monkeypatch.delenv("NVIDIA_MAX_TOKENS", raising=False)
    monkeypatch.delenv("NVIDIA_TEMPERATURE", raising=False)

    cfg = Settings()
    assert cfg.LLM_PROVIDER == "nvidia"
    assert cfg.NVIDIA_BASE_URL == "https://integrate.api.nvidia.com/v1"
    assert cfg.NVIDIA_MODEL == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert cfg.NVIDIA_TIMEOUT_SECONDS == 60.0
    assert cfg.NVIDIA_MAX_RETRIES == 2
    assert cfg.NVIDIA_MAX_TOKENS == 8192
    assert cfg.NVIDIA_TEMPERATURE == 1.0

    provider = NvidiaNemotronProvider(app_settings=cfg)
    try:
        assert provider.base_url == "https://integrate.api.nvidia.com/v1"
        assert provider.chat_completions_url == "https://integrate.api.nvidia.com/v1/chat/completions"
        assert provider.model == "nvidia/nemotron-3.5-lightning-30b-a3b"
        assert provider.timeout_seconds == 60.0
        assert provider.max_retries == 2
        assert provider.max_tokens == 8192
        assert provider.temperature == 1.0
    finally:
        provider.close()


def test_bearer_authentication_and_key_redaction_in_repr():
    captured_headers = {}
    captured_payload = {}
    captured_url = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_url
        captured_url = str(request.url)
        captured_headers.update(dict(request.headers))
        captured_payload.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test-1",
                "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Verified answer."}, "finish_reason": "stop"}],
            },
        )

    secret = "nvapi-super-secret-token-xyz"
    provider, client, _ = _make_provider(handler, api_key=secret)
    try:
        assert secret not in repr(provider)
        assert "configured" in repr(provider)

        text = provider.generate_text("You are an insurance assistant.", "Is hospitalization covered?")
        assert text == "Verified answer."
        assert captured_url == "https://integrate.api.nvidia.com/v1/chat/completions"
        assert captured_headers.get("authorization") == f"Bearer {secret}"
        assert captured_payload["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
        assert captured_payload["stream"] is False
        assert captured_payload["temperature"] == 1.0
        assert captured_payload["max_tokens"] == 8192
        assert captured_payload["messages"] == [
            {"role": "system", "content": "You are an insurance assistant."},
            {"role": "user", "content": "Is hospitalization covered?"},
        ]
    finally:
        client.close()


def test_successful_response_and_usage_metadata_extraction():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"x-request-id": "req-nvidia-42"},
            json={
                "id": "chatcmpl-99",
                "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"decision": "COVERED"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 25,
                    "total_tokens": 145,
                },
            },
        )

    provider, client, sleeps = _make_provider(handler)
    try:
        result = provider.complete("System prompt", "User question")
        assert result.content == '{"decision": "COVERED"}'
        assert result.model == "nvidia/nemotron-3.5-lightning-30b-a3b"
        assert result.finish_reason == "stop"
        assert result.request_id == "req-nvidia-42"
        assert result.usage is not None
        assert result.usage.prompt_tokens == 120
        assert result.usage.completion_tokens == 25
        assert result.usage.total_tokens == 145
        assert sleeps == []
    finally:
        client.close()


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {"choices": [{"message": {"role": "assistant", "content": ""}}]},
        {"choices": [{"message": {"role": "assistant", "content": "   "}}]},
        {"choices": [{"message": {}}]},
        {"unexpected": "shape"},
    ],
)
def test_empty_or_malformed_response_handling(payload):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=payload)

    provider, client, sleeps = _make_provider(handler, max_retries=2)
    try:
        with pytest.raises(NvidiaResponseError):
            provider.generate_text("sys", "question")
        assert calls == 1
        assert sleeps == []
    finally:
        client.close()


def test_malformed_json_response_handling():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-valid-json")

    provider, client, _ = _make_provider(handler)
    try:
        with pytest.raises(NvidiaResponseError):
            provider.generate_text("sys", "question")
    finally:
        client.close()


def test_missing_api_key_raises_without_retry(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    cfg = Settings(NVIDIA_API_KEY=None)
    provider, client, sleeps = _make_provider(handler, api_key="", app_settings=cfg)
    try:
        with pytest.raises(NvidiaConfigurationError):
            provider.generate_text("sys", "question")
        assert calls == 0
        assert sleeps == []
    finally:
        client.close()


def test_authentication_failure_401_not_retried():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": {"message": "Invalid API key"}})

    provider, client, sleeps = _make_provider(handler, max_retries=2)
    try:
        with pytest.raises(NvidiaAuthenticationError) as exc_info:
            provider.generate_text("sys", "question")
        assert exc_info.value.status_code == 401
        assert calls == 1
        assert sleeps == []
    finally:
        client.close()


def test_permission_failure_403_not_retried():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(403, json={"error": {"message": "Model access forbidden"}})

    provider, client, sleeps = _make_provider(handler, max_retries=2)
    try:
        with pytest.raises(NvidiaPermissionError) as exc_info:
            provider.generate_text("sys", "question")
        assert exc_info.value.status_code == 403
        assert calls == 1
        assert sleeps == []
    finally:
        client.close()


def test_rate_limit_429_retries_bounded_and_succeeds():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Recovered after retry"}}]},
        )

    provider, client, sleeps = _make_provider(handler, max_retries=2)
    try:
        out = provider.generate_text("sys", "question")
        assert out == "Recovered after retry"
        assert calls == 3
        assert sleeps == [0.5, 1.0]
    finally:
        client.close()


def test_rate_limit_429_exhausted_raises():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, json={"error": {"message": "Too many requests"}})

    provider, client, sleeps = _make_provider(handler, max_retries=2)
    try:
        with pytest.raises(NvidiaRateLimitError) as exc_info:
            provider.generate_text("sys", "question")
        assert exc_info.value.status_code == 429
        assert calls == 3
        assert sleeps == [0.5, 1.0]
    finally:
        client.close()


def test_server_error_5xx_retries_and_raises_after_bound():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"error": {"message": "Upstream unavailable"}})

    provider, client, sleeps = _make_provider(handler, max_retries=2)
    try:
        with pytest.raises(NvidiaServerError) as exc_info:
            provider.generate_text("sys", "question")
        assert exc_info.value.status_code == 503
        assert calls == 3
        assert sleeps == [0.5, 1.0]
    finally:
        client.close()


def test_network_timeout_retries_with_exponential_backoff():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out waiting for NVIDIA", request=request)

    provider, client, sleeps = _make_provider(handler, max_retries=2)
    try:
        with pytest.raises(NvidiaTimeoutError):
            provider.generate_text("sys", "question")
        assert calls == 3
        assert sleeps == [0.5, 1.0]
    finally:
        client.close()


def test_network_connection_error_retries_with_exponential_backoff():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("DNS resolution failed", request=request)

    provider, client, sleeps = _make_provider(handler, max_retries=1)
    try:
        with pytest.raises(NvidiaConnectionError):
            provider.generate_text("sys", "question")
        assert calls == 2
        assert sleeps == [0.5]
    finally:
        client.close()


def test_no_sensitive_data_in_error_logs_or_exceptions(caplog):
    secret = "nvapi-very-secret-key-007"
    policy_text = "CONFIDENTIAL POLICYHOLDER DATA: John Doe SSN 123-45-6789"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": f"Rejected token {secret}"}})

    provider, client, _ = _make_provider(handler, api_key=secret, max_retries=0)
    try:
        with caplog.at_level(logging.INFO, logger="coverageiq.llm.providers.nvidia_nemotron"):
            with pytest.raises(NvidiaServerError) as exc_info:
                provider.generate_text("System", policy_text)

        assert secret not in str(exc_info.value)
        assert policy_text not in str(exc_info.value)
        assert secret not in caplog.text
        assert policy_text not in caplog.text
        assert "Bearer" not in caplog.text
    finally:
        client.close()


def test_missing_api_key_none_and_per_call_overrides(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    captured_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "OK"}}]},
        )

    cfg = Settings(NVIDIA_API_KEY=None)
    provider_no_key, client_no_key, sleeps_no_key = _make_provider(handler, api_key=None, app_settings=cfg)
    try:
        with pytest.raises(NvidiaConfigurationError):
            provider_no_key.generate_text("sys", "question")
        assert sleeps_no_key == []
    finally:
        client_no_key.close()

    provider, client, _ = _make_provider(handler, api_key="nvapi-override-test")
    try:
        out = provider.generate_text("sys", "question", temperature=0.4, max_tokens=512)
        assert out == "OK"
        assert captured_payload["temperature"] == 0.4
        assert captured_payload["max_tokens"] == 512
    finally:
        client.close()


def test_client_error_400_not_retried():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": {"message": "Unsupported parameter"}})

    provider, client, sleeps = _make_provider(handler, max_retries=2)
    try:
        with pytest.raises(NvidiaClientError) as exc_info:
            provider.generate_text("sys", "question")
        assert exc_info.value.status_code == 400
        assert calls == 1
        assert sleeps == []
    finally:
        client.close()
