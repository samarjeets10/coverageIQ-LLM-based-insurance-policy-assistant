"""Unit tests for Settings, FastAPI health endpoints, and token verification."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy.exc import OperationalError

from ai_service.main import create_app
from ai_service.recommendation.db import get_db_connection
from ai_service.recommendation.settings import Settings


def _make_mock_conn(db_name: str = "coverageiq_db") -> MagicMock:
    conn = MagicMock()
    return conn


def test_settings_builds_safe_postgresql_url() -> None:
    """Ensure SQLAlchemy URL creation handles SSL toggle and secret password safely."""
    cfg = Settings(
        _env_file=None,
        db_host="localhost",
        db_port=5432,
        db_name="coverageiq_db",
        db_user="postgres",
        db_password=SecretStr("p@ss:word/1"),
        db_ssl=True,
    )
    url = cfg.build_database_url()
    assert url.drivername == "postgresql+psycopg"
    assert url.host == "localhost"
    assert url.database == "coverageiq_db"
    assert url.password == "p@ss:word/1"
    assert url.query == {"sslmode": "require"}


def test_health_endpoints_return_200_when_db_ready(monkeypatch) -> None:
    """Verify /health and /ai/health report healthy when DB checks succeed."""
    cfg = Settings(_env_file=None)
    app = create_app(cfg)
    mock_conn = _make_mock_conn()

    def override_conn():
        yield mock_conn

    app.dependency_overrides[get_db_connection] = override_conn
    monkeypatch.setattr(
        "ai_service.main.check_database_connection",
        lambda _conn: {"database": "coverageiq_db", "username": "postgres"},
    )
    monkeypatch.setattr(
        "ai_service.main.inspect_database_readiness",
        lambda _conn: {
            "pgvector_installed": True,
            "tables_present": {
                "policies": True,
                "policy_versions": True,
                "organizations": True,
                "policy_attributes": False,
                "ranking_features": False,
            },
            "total_policy_count": 0,
            "active_catalog_policy_count": 0,
        },
    )

    with TestClient(app) as client:
        for route in ("/health", "/ai/health"):
            res = client.get(route)
            assert res.status_code == 200
            body = res.json()
            assert body["success"] is True
            assert body["status"] == "healthy"
            assert body["database"] == "coverageiq_db"
            assert body["readiness"]["tables_present"]["policies"] is True


def test_health_returns_503_on_database_error(monkeypatch) -> None:
    """Verify /health fails cleanly with HTTP 503 when PostgreSQL is unreachable."""
    app = create_app(Settings(_env_file=None))
    mock_conn = _make_mock_conn()

    def override_conn():
        yield mock_conn

    app.dependency_overrides[get_db_connection] = override_conn

    def _raise_db_err(_conn):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    monkeypatch.setattr("ai_service.main.check_database_connection", _raise_db_err)

    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 503
        body = res.json()
        assert body["success"] is False
        assert body["services"]["database"] == "down"


def test_service_token_enforced_when_configured(monkeypatch) -> None:
    """Verify X-Service-Token rejects unauthorized requests when configured."""
    cfg = Settings(_env_file=None, ai_service_token=SecretStr("secret-svc-token"))
    app = create_app(cfg)
    mock_conn = _make_mock_conn()

    def override_conn():
        yield mock_conn

    app.dependency_overrides[get_db_connection] = override_conn
    monkeypatch.setattr(
        "ai_service.recommendation.service.load_catalog_policies_from_db",
        lambda _conn, active_only=False: [],
    )

    valid_payload = {
        "policy_type": "health",
        "age": 30,
        "budget_max": 20000,
        "min_sum_insured": 500000,
    }

    with TestClient(app) as client:
        unauth = client.post("/api/v1/recommend", json=valid_payload)
        assert unauth.status_code == 401

        auth = client.post(
            "/api/v1/recommend",
            json=valid_payload,
            headers={"X-Service-Token": "secret-svc-token"},
        )
        assert auth.status_code == 200
        assert auth.json()["status"] == "NO_MATCHING_POLICIES"
