"""Citation provider interface for future M3 retrieval & M8 verification wiring.

Stage 9A intentionally defines ONLY the clean interface boundary and a safe
``NullCitationProvider`` that returns empty citations (``[]``) so no synthetic
or unverified chunk references are ever produced before M3 and M8 are integrated.
"""

from __future__ import annotations

from collections.abc import Sequence
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
