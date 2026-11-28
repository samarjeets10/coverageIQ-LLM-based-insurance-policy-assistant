import hmac
import re
from typing import Any, Literal, Optional, Tuple
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, model_validator

from ai_service.config import Settings, settings
from ai_service.persistence.exceptions import DatabaseAuthorizationError
from ai_service.schemas.query import ScopeContext

_VALID_ROLES = ("user", "organization", "admin")
_FORBIDDEN_SCOPE_CHARS_RE = re.compile(r"[*%;\s\x00]")


def redact_dsn(dsn: Optional[str]) -> str:
    """Mask passwords and userinfo secrets from PostgreSQL connection strings for safe logging/repr."""
    if not dsn:
        return "<unset>"
    try:
        parsed = urlsplit(dsn)
        if parsed.password or "@" in parsed.netloc:
            host = parsed.hostname or ""
            port = f":{parsed.port}" if parsed.port else ""
            user = parsed.username or ""
            netloc = f"{user}:***@{host}{port}" if user else f"***@{host}{port}"
            return parsed._replace(netloc=netloc).geturl()
        return dsn
    except Exception:
        return "<redacted-dsn>"


class AuthorizedScope(BaseModel):
    """
    Gateway-validated tenant and authorization context for PostgreSQL access.

    Repositories require an AuthorizedScope rather than trusting unvalidated
    client-supplied scope dictionaries or bare policy_id values.
    """

    model_config = ConfigDict(frozen=True)

    user_id: str
    role: Literal["user", "organization", "admin"]
    owner_scope: str
    allow_catalog_read: bool = False
    trusted_gateway_context: bool = False

    @model_validator(mode="after")
    def validate_scope_fields(self) -> "AuthorizedScope":
        if not isinstance(self.user_id, str) or not self.user_id.strip():
            raise DatabaseAuthorizationError("AuthorizedScope requires a non-empty user_id.")
        if _FORBIDDEN_SCOPE_CHARS_RE.search(self.user_id.strip()):
            raise DatabaseAuthorizationError("AuthorizedScope user_id contains invalid characters.")

        if not isinstance(self.owner_scope, str) or not self.owner_scope.strip():
            raise DatabaseAuthorizationError("AuthorizedScope requires a non-empty owner_scope.")
        if _FORBIDDEN_SCOPE_CHARS_RE.search(self.owner_scope.strip()):
            raise DatabaseAuthorizationError("AuthorizedScope owner_scope contains invalid characters.")

        if not self.trusted_gateway_context:
            raise DatabaseAuthorizationError(
                "Untrusted client scope context rejected; gateway authorization verification is required."
            )

        object.__setattr__(self, "user_id", self.user_id.strip())
        object.__setattr__(self, "owner_scope", self.owner_scope.strip())
        return self

    @classmethod
    def from_gateway_context(
        cls,
        context: ScopeContext | dict[str, Any],
        *,
        provided_service_token: Optional[str] = None,
        app_settings: Optional[Settings] = None,
        allow_catalog_read: bool = False,
        trusted_internal: bool = False,
    ) -> "AuthorizedScope":
        """
        Promote a Node.js gateway ScopeContext into a trusted AuthorizedScope.

        Untrusted client calls cannot bypass authentication: either `trusted_internal=True`
        must be set explicitly by a verified internal component or `provided_service_token`
        must match the configured `INTERNAL_SERVICE_TOKEN` via constant-time comparison.
        """
        cfg = app_settings or settings
        token_verified = False
        if not trusted_internal:
            expected_token = cfg.INTERNAL_SERVICE_TOKEN
            if expected_token and provided_service_token:
                token_verified = hmac.compare_digest(provided_service_token, expected_token)
            if not token_verified:
                raise DatabaseAuthorizationError(
                    "Client-supplied policy scope rejected without validated gateway authorization."
                )

        if isinstance(context, ScopeContext):
            user_id = context.user_id
            role = context.role or "user"
            owner_scope = context.owner_scope
        elif isinstance(context, dict):
            user_id = context.get("user_id")
            role = context.get("role") or "user"
            owner_scope = context.get("owner_scope")
        else:
            raise DatabaseAuthorizationError("Scope context must be a ScopeContext or mapping.")

        if role not in _VALID_ROLES:
            raise DatabaseAuthorizationError(f"Unsupported authorization role '{role}'.")

        return cls(
            user_id=user_id or "",
            role=role,  # type: ignore[arg-type]
            owner_scope=owner_scope or "",
            allow_catalog_read=allow_catalog_read,
            trusted_gateway_context=True,
        )


def validate_authorized_scope(
    scope: Any,
    *,
    provided_service_token: Optional[str] = None,
    app_settings: Optional[Settings] = None,
    allow_internal: bool = False,
) -> AuthorizedScope:
    """Validate an AuthorizedScope instance before executing any scoped SQL operation."""
    if scope is None:
        raise DatabaseAuthorizationError("Database access requires an AuthorizedScope.")

    if isinstance(scope, AuthorizedScope):
        if not scope.trusted_gateway_context:
            raise DatabaseAuthorizationError("AuthorizedScope lacks verified gateway trust.")
        return scope

    if isinstance(scope, (ScopeContext, dict)):
        return AuthorizedScope.from_gateway_context(
            scope,
            provided_service_token=provided_service_token,
            app_settings=app_settings,
            trusted_internal=allow_internal,
        )

    raise DatabaseAuthorizationError("Invalid scope object passed to database repository.")


def build_policy_scope_predicate(
    scope: AuthorizedScope,
    *,
    alias: str = "p",
    for_write: bool = False,
) -> Tuple[str, dict[str, Any]]:
    """
    Build a parameterized SQL predicate enforcing policy tenant isolation.

    - Writes require matching `owner_scope` (unless verified admin role).
    - Reads require matching `owner_scope` or active catalog visibility when
      `allow_catalog_read=True`. Private policies are strictly tenant-isolated.
    """
    validated = validate_authorized_scope(scope)
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", alias):
        raise DatabaseAuthorizationError("Invalid SQL table alias for scope predicate.")

    params: dict[str, Any] = {"scope_owner_scope": validated.owner_scope}

    if for_write:
        return f"{alias}.owner_scope = %(scope_owner_scope)s", params

    if validated.allow_catalog_read:
        return (
            f"({alias}.owner_scope = %(scope_owner_scope)s "
            f"OR ({alias}.visibility = 'catalog' AND {alias}.status = 'active'))",
            params,
        )

    return f"{alias}.owner_scope = %(scope_owner_scope)s", params
