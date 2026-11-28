import pytest
from fastapi.testclient import TestClient

from ai_service.api import routes_qa
from ai_service.config import Settings
from ai_service.llm.client import (
    ExternalProviderConfigurationError,
    ExternalProviderError,
    LLMClient,
)
from ai_service.llm.providers.nvidia_nemotron import NvidiaConfigurationError
from ai_service.main import app


def test_non_test_environments_fail_closed_on_missing_or_true_mock_storage(monkeypatch):
    monkeypatch.delenv("USE_MOCK_STORAGE", raising=False)
    monkeypatch.setenv("APP_ENV", "production")
    with pytest.raises(ValueError, match="USE_MOCK_STORAGE must be explicitly set"):
        Settings()

    monkeypatch.setenv("USE_MOCK_STORAGE", "true")
    with pytest.raises(ValueError, match="USE_MOCK_STORAGE=true is forbidden"):
        Settings()

    monkeypatch.setenv("APP_ENV", "staging")
    monkeypatch.setenv("USE_MOCK_STORAGE", "false")
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    with pytest.raises(ValueError, match="LLM_PROVIDER='offline' is forbidden"):
        Settings()

    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER='offline' is forbidden"):
        Settings()

    monkeypatch.setenv("EMBEDDING_PROVIDER", "bge-m3")
    cfg = Settings()
    assert cfg.is_non_test_env is True
    assert cfg.USE_MOCK_STORAGE is False


def test_debug_false_also_enforces_fail_closed_mock_storage(monkeypatch):
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("USE_MOCK_STORAGE", raising=False)
    monkeypatch.setenv("DEBUG", "false")
    with pytest.raises(ValueError, match="USE_MOCK_STORAGE must be explicitly set"):
        Settings()


def test_gemini_provider_never_silently_falls_back_to_offline(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    cfg = Settings(APP_ENV="development", LLM_PROVIDER="gemini", USE_MOCK_STORAGE=True)
    with pytest.raises(ExternalProviderConfigurationError, match="GEMINI_API_KEY"):
        LLMClient(provider="gemini", app_settings=cfg)

    class FailingModels:
        @staticmethod
        def generate_content(**_kwargs):
            raise RuntimeError("Upstream Gemini 503 with fake-key-secret")

    class FakeGenAiClient:
        models = FailingModels()

    client = LLMClient(provider="gemini", api_key="test-key-only", app_settings=cfg)
    client._genai_client = FakeGenAiClient()
    monkeypatch.setattr(
        client,
        "_offline_grounded_synthesis",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("Offline fallback must NEVER execute")),
    )

    with pytest.raises(ExternalProviderError) as exc_info:
        client.generate_json("sys", "USER QUERY: hospitalized?\nRETRIEVED POLICY EVIDENCE CHUNKS:\n")
    assert "fake-key-secret" not in str(exc_info.value)


def test_offline_llm_forbidden_in_production_and_routes_qa_blocks_invalid_config(monkeypatch):
    prod_cfg = Settings(
        APP_ENV="production",
        USE_MOCK_STORAGE=False,
        LLM_PROVIDER="nvidia",
        EMBEDDING_PROVIDER="bge-m3",
    )
    with pytest.raises(NvidiaConfigurationError):
        LLMClient(provider="offline", app_settings=prod_cfg)

    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("USE_MOCK_STORAGE", "true")
    routes_qa._orchestrator = None
    client = TestClient(app)
    resp = client.post(
        "/ai/ask",
        json={"policy_id": "policy_health_01", "question": "Is hospitalization covered?"},
    )
    assert resp.status_code == 503


def test_explicit_offline_test_mode_marks_mock_metadata(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("USE_MOCK_STORAGE", "true")
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    routes_qa._orchestrator = None

    client = TestClient(app)
    resp = client.post(
        "/ai/ask",
        json={"policy_id": "policy_health_01", "question": "Is hospitalization covered?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["metadata"]["storage_mode"] == "mock_memory"
    assert body["metadata"]["mock_mode"] is True
