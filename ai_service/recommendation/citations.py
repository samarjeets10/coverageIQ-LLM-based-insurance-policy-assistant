"""Citation provider interface for future M3 retrieval & M8 verification wiring.

Stage 9A intentionally defines ONLY the clean interface boundary and a safe
``NullCitationProvider`` that returns empty citations (``[]``) so no synthetic
or unverified chunk references are ever produced before M3 and M8 are integrated.
Stage 10A adds ``normalize_and_validate_citation`` so future M3/M8 providers
can return either ``dict`` or Pydantic models (e.g. ``CitationItem``) while
guaranteeing grounding fields (``chunk_id``, ``page >= 1``, ``clause``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ai_service.recommendation.schemas import TrustedScope


@dataclass(frozen=True)
class PolicyCitationRequest:
    """Metadata describing a ranked Top-N policy eligible for citation attachment.

    ``scope`` forwards the optional Node.js M1-attested ``TrustedScope`` context so
    future M3/M8 providers can apply tenant/policy visibility checks cleanly.
    """

    policy_id: str
    policy_version_id: str | None
    policy_name: str
    policy_type: str
    rank: int
    score: float
    model_used: str
    key_highlights: tuple[str, ...] = ()
    scope: TrustedScope | None = None


def normalize_and_validate_citation(raw_entry: Any) -> dict[str, Any]:
    """Normalize dict/Pydantic citation entries and enforce minimal grounding invariants."""
    if hasattr(raw_entry, "model_dump") and callable(raw_entry.model_dump):
        entry = raw_entry.model_dump()
    elif isinstance(raw_entry, Mapping):
        entry = dict(raw_entry)
    else:
        raise ValueError("Citation entry must be a mapping or Pydantic CitationItem model.")

    chunk_id = entry.get("chunk_id")
    clause = entry.get("clause")
    page = entry.get("page")

    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError("Citation entry requires a non-empty 'chunk_id' string.")
    if not isinstance(clause, str) or not clause.strip():
        raise ValueError("Citation entry requires a non-empty 'clause' string.")
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise ValueError("Citation entry requires a positive integer 'page' (>= 1).")

    entry["chunk_id"] = chunk_id.strip()
    entry["clause"] = clause.strip()
    return entry


@runtime_checkable
class RecommendationCitationProvider(Protocol):
    """Protocol for attaching verified chunk citations to Top-N recommendations."""

    def get_citations(
        self,
        requests: Sequence[PolicyCitationRequest],
        *,
        request_id: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return a mapping of ``policy_id`` -> list of verified citation objects."""
        ...


class NullCitationProvider:
    """Default Stage 9A citation provider returning empty citations without DB/LLM calls."""

    def get_citations(
        self,
        requests: Sequence[PolicyCitationRequest],
        *,
        request_id: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return empty citation lists until M3 retrieval & M8 verification are integrated."""
        del request_id
        return {req.policy_id: [] for req in requests}
