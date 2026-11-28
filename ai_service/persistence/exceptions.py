from typing import Optional
import psycopg


class DatabaseError(Exception):
    """Base exception for all CoverageIQ PostgreSQL and pgvector database failures."""


class DatabaseConfigurationError(DatabaseError):
    """Raised when database URL, pool limits, or runtime settings are invalid."""


class DatabaseConnectionError(DatabaseError):
    """Raised when the database connection pool is unavailable, closed, or times out."""


class DatabaseMigrationError(DatabaseError):
    """Raised when required PostgreSQL schema tables or extensions are missing."""


class DatabaseConstraintError(DatabaseError):
    """Raised on PostgreSQL unique, foreign-key, not-null, or check constraint violations."""


class DatabaseNotFoundError(DatabaseError):
    """Raised when a requested record does not exist within the authorized scope."""


class DatabaseAuthorizationError(DatabaseError):
    """Raised when scope validation fails or a caller attempts unauthorized tenant access."""


class DatabaseValidationError(DatabaseError):
    """Raised when input parameters (e.g., 1024-dim vector, SHA-256 hash) fail validation."""


def translate_psycopg_error(exc: Exception, *, context: Optional[str] = None) -> DatabaseError:
    """Convert raw psycopg exceptions into safe typed DatabaseError instances without leaking DSN secrets."""
    if isinstance(exc, DatabaseError):
        return exc

    prefix = f"{context}: " if context else ""
    sqlstate = getattr(exc, "sqlstate", None)
    diag = getattr(exc, "diag", None)
    constraint_name = getattr(diag, "constraint_name", None) if diag is not None else None
    constraint_note = f" constraint={constraint_name}" if constraint_name else ""

    if isinstance(exc, psycopg.errors.InsufficientPrivilege):
        return DatabaseAuthorizationError(
            f"{prefix}Database role lacks required privilege ({type(exc).__name__})."
        )

    if isinstance(
        exc,
        (
            psycopg.errors.UniqueViolation,
            psycopg.errors.ForeignKeyViolation,
            psycopg.errors.CheckViolation,
            psycopg.errors.NotNullViolation,
            psycopg.errors.IntegrityError,
        ),
    ) or (isinstance(sqlstate, str) and sqlstate.startswith("23")):
        return DatabaseConstraintError(
            f"{prefix}Database integrity constraint violated ({type(exc).__name__}{constraint_note})."
        )

    if isinstance(exc, (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn, psycopg.errors.UndefinedObject)):
        return DatabaseMigrationError(
            f"{prefix}Database schema object missing; ensure Node.js node-pg-migrate migrations are applied ({type(exc).__name__})."
        )

    if isinstance(exc, psycopg.errors.DataError) or (isinstance(sqlstate, str) and sqlstate.startswith("22")):
        return DatabaseValidationError(
            f"{prefix}Invalid database parameter format ({type(exc).__name__})."
        )

    if isinstance(exc, psycopg.OperationalError):
        return DatabaseConnectionError(
            f"{prefix}PostgreSQL operational connection error ({type(exc).__name__})."
        )

    return DatabaseError(f"{prefix}Unexpected database error ({type(exc).__name__}).")
