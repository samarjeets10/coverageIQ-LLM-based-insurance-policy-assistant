import logging
import pytest

from ai_service.config import Settings
from ai_service.persistence import (
    AuthorizedScope,
    DatabaseAuthorizationError,
    DatabaseConfigurationError,
    DatabaseConnectionError,
    DatabaseConstraintError,
    DatabaseMigrationError,
    DatabaseValidationError,
    PostgresPoolManager,
    PolicyKnowledgeRepository,
    redact_dsn,
    translate_psycopg_error,
    validate_authorized_scope,
)
from ai_service.persistence.repositories import _validate_embedding_vector, _validate_sha256
from ai_service.schemas.query import ScopeContext


def test_pool_manager_config_validation_and_secret_redaction(caplog):
    secret_pw = "super_secret_db_password_991"
    dsn = f"postgresql://coverageiq_runtime:{secret_pw}@db.internal:5432/coverageiq_db"

    mgr = PostgresPoolManager(dsn=dsn, min_size=1, max_size=4, timeout=10.0)
    rendered = repr(mgr)
    assert secret_pw not in rendered
    assert "coverageiq_runtime:***@db.internal:5432" in rendered
    assert secret_pw not in redact_dsn(dsn)
    assert mgr.is_open is False

    # Unopened pool rejects operations cleanly without leaking credentials
    with pytest.raises(DatabaseConnectionError) as exc_info:
        with mgr.connection():
            pass
    assert secret_pw not in str(exc_info.value)

    with pytest.raises(DatabaseConfigurationError):
        PostgresPoolManager(dsn="mysql://user:pw@localhost:3306/db")
    with pytest.raises(DatabaseConfigurationError):
        PostgresPoolManager(dsn="", min_size=1, max_size=2)
    with pytest.raises(DatabaseConfigurationError):
        PostgresPoolManager(dsn=dsn, min_size=0, max_size=2)
    with pytest.raises(DatabaseConfigurationError):
        PostgresPoolManager(dsn=dsn, min_size=5, max_size=2)
    with pytest.raises(DatabaseConfigurationError):
        PostgresPoolManager(dsn=dsn, min_size=1, max_size=2, timeout=0.0)


def test_authorized_scope_rejects_untrusted_client_contexts(monkeypatch):
    # Direct ScopeContext or dict without verified gateway token is rejected
    raw_scope = ScopeContext(user_id="user_1", role="user", owner_scope="user:user_1")
    with pytest.raises(DatabaseAuthorizationError, match="Untrusted client scope|validated gateway"):
        validate_authorized_scope(raw_scope)

    with pytest.raises(DatabaseAuthorizationError):
        AuthorizedScope(
            user_id="user_1",
            role="admin",
            owner_scope="user:user_1",
            trusted_gateway_context=False,
        )

    # Forged service token rejected
    cfg = Settings(INTERNAL_SERVICE_TOKEN="expected-gateway-secret-token")
    with pytest.raises(DatabaseAuthorizationError):
        AuthorizedScope.from_gateway_context(
            raw_scope,
            provided_service_token="forged-wrong-token",
            app_settings=cfg,
        )

    # Valid constant-time gateway token promotes trusted scope
    trusted = AuthorizedScope.from_gateway_context(
        raw_scope,
        provided_service_token="expected-gateway-secret-token",
        app_settings=cfg,
    )
    assert trusted.trusted_gateway_context is True
    assert trusted.owner_scope == "user:user_1"

    # Wildcard or SQL meta characters in owner_scope / user_id rejected
    for bad_owner in ("", "   ", "*", "user:*", "user%admin", "user;drop"):
        with pytest.raises(DatabaseAuthorizationError):
            AuthorizedScope(
                user_id="user_1",
                role="user",
                owner_scope=bad_owner,
                trusted_gateway_context=True,
            )

    with pytest.raises(DatabaseAuthorizationError):
        validate_authorized_scope(None)


def test_embedding_and_sha256_validation_helpers():
    valid_vec = [0.01] * 1024
    assert len(_validate_embedding_vector(valid_vec)) == 1024

    for bad_dim in ([0.01] * 768, [0.01] * 512, [0.01] * 1025, []):
        with pytest.raises(DatabaseValidationError, match="1024 dimensions"):
            _validate_embedding_vector(bad_dim)

    with pytest.raises(DatabaseValidationError, match="non-finite"):
        _validate_embedding_vector([float("nan")] + [0.0] * 1023)
    with pytest.raises(DatabaseValidationError, match="non-finite"):
        _validate_embedding_vector([float("inf")] + [0.0] * 1023)

    assert _validate_sha256("A" * 64) == "a" * 64
    with pytest.raises(DatabaseValidationError):
        _validate_sha256("not-a-sha256")


def test_psycopg_error_translation():
    import psycopg

    assert isinstance(translate_psycopg_error(psycopg.errors.UniqueViolation("dup")), DatabaseConstraintError)
    assert isinstance(translate_psycopg_error(psycopg.errors.ForeignKeyViolation("fk")), DatabaseConstraintError)
    assert isinstance(translate_psycopg_error(psycopg.errors.CheckViolation("chk")), DatabaseConstraintError)
    assert isinstance(translate_psycopg_error(psycopg.errors.UndefinedTable("missing")), DatabaseMigrationError)
    assert isinstance(
        translate_psycopg_error(psycopg.errors.InsufficientPrivilege("denied")),
        DatabaseAuthorizationError,
    )
    assert isinstance(translate_psycopg_error(psycopg.OperationalError("down")), DatabaseConnectionError)
    assert isinstance(_validate_sha256(None, allow_none=True), type(None))

    # Repository rejects untrusted scope before borrowing a pool connection
    mgr = PostgresPoolManager(dsn="postgresql://u:p@localhost:5432/db")
    repo = PolicyKnowledgeRepository(mgr)
    with pytest.raises(DatabaseAuthorizationError):
        repo.get_policy_chunks(
            policy_id="00000000-0000-0000-0000-000000000001",  # type: ignore[arg-type]
            scope=ScopeContext(user_id="attacker", role="admin", owner_scope="user:alice"),  # type: ignore[arg-type]
        )
