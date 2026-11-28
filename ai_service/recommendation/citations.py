"""Evidence-grounded Citation Providers for CoverageIQ Recommendations (Checkpoint 5.0).

Preserves the M5 ``RecommendationCitationProvider`` boundary and
``normalize_and_validate_citation`` invariants while providing a concrete
PostgreSQL-backed citation resolver querying ``policy_chunks`` joined with
``policy_documents`` and ``policies``.

Core invariant: NO EVIDENCE = NO CLAIM.
- Every returned citation must match an actual persisted ``policy_chunks`` row
  belonging to the authorized target ``policy_id`` and ``document_id``.
- Missing chunks never yield fabricated citations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
import re
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from ai_service.persistence.pool import PostgresPoolManager, get_db_pool_manager
from ai_service.recommendation.schemas import TrustedScope

logger = logging.getLogger("coverageiq.recommendation.citations")


@dataclass(frozen=True)
class PolicyCitationRequest:
    """Metadata describing a ranked Top-N policy eligible for citation attachment."""

    policy_id: str
    policy_version_id: str | None
    policy_name: str
    policy_type: str
    rank: int
    score: float
    model_used: str
    key_highlights: tuple[str, ...] = ()
    scope: TrustedScope | None = None
    document_id: str | None = None
    document_version: int | None = None
    field_provenance: Mapping[str, Any] | None = None
    required_benefits: tuple[str, ...] = ()
    pre_existing_conditions: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()


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
    """Explicit no-op citation provider returning empty lists when retrieval is disabled."""

    def get_citations(
        self,
        requests: Sequence[PolicyCitationRequest],
        *,
        request_id: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return empty citation lists without DB/LLM calls."""
        del request_id
        return {req.policy_id: [] for req in requests}


class PostgresEvidenceCitationProvider:
    """Concrete M3/M4-backed citation resolver over persisted PostgreSQL policy_chunks.

    Combines verified chunk IDs recorded in ``policy_attributes.field_provenance`` with
    ranked PostgreSQL full-text search (``search_tsv``) over ``policy_chunks`` joined
    strictly against active catalog or caller-owned non-archived ``policy_documents``.
    """

    def __init__(
        self,
        *,
        pool_manager: PostgresPoolManager | None = None,
        max_citations_per_policy: int = 4,
    ) -> None:
        self._pool_manager = pool_manager
        self._max_citations = max(1, int(max_citations_per_policy))

    def _get_pool(self) -> PostgresPoolManager | None:
        if self._pool_manager is not None:
            return self._pool_manager
        try:
            return get_db_pool_manager()
        except Exception:
            return None

    @staticmethod
    def _extract_provenance_chunk_ids(field_provenance: Mapping[str, Any] | None) -> list[str]:
        if not isinstance(field_provenance, Mapping):
            return []
        ordered_fields = (
            "covered_benefits",
            "exclusions",
            "sum_insured_max",
            "annual_premium_min",
            "min_entry_age",
            "ped_waiting_period_months",
            "copay_pct",
            "deductible_amount",
            "cashless_available",
        )
        seen: set[str] = set()
        chunk_ids: list[str] = []
        for fname in list(ordered_fields) + [k for k in field_provenance.keys() if k not in ordered_fields]:
            items = field_provenance.get(fname)
            if not isinstance(items, list):
                continue
            for entry in items:
                if isinstance(entry, Mapping):
                    cid = entry.get("chunk_id")
                    if isinstance(cid, str) and cid.strip() and cid.strip() not in seen:
                        seen.add(cid.strip())
                        chunk_ids.append(cid.strip())
        return chunk_ids

    @staticmethod
    def _build_fts_query_text(req: PolicyCitationRequest) -> str:
        terms: list[str] = [req.policy_type]
        for b in req.required_benefits:
            terms.append(b.replace("_", " "))
        for p in req.pre_existing_conditions:
            terms.append(p.replace("_", " "))
        for h in req.key_highlights[:3]:
            terms.extend(re.findall(r"[a-zA-Z]{4,}", h.lower())[:3])
        terms.extend(["coverage", "sum insured", "exclusion", "premium", "eligibility"])
        return " ".join(terms[:24])

    @staticmethod
    def _row_to_citation(row: Mapping[str, Any]) -> dict[str, Any]:
        p_start = int(row["page_start"] or row["page_number"])
        p_end = int(row["page_end"]) if row.get("page_end") is not None else p_start
        p_range = f"{p_start}-{p_end}" if p_end > p_start else str(p_start)
        snippet = " ".join(str(row["content_text"] or "").split())[:260]
        return {
            "policy_id": str(row["policy_id"]),
            "document_id": str(row["document_id"]),
            "document_version": int(row["version_number"]),
            "chunk_id": str(row["chunk_key"]).strip(),
            "page": int(row["page_number"]),
            "page_end": p_end if p_end != p_start else None,
            "page_range": p_range,
            "clause": str(row["clause_ref"]).strip(),
            "category": str(row["category"]) if row.get("category") else None,
            "snippet": snippet,
        }

    def get_citations(
        self,
        requests: Sequence[PolicyCitationRequest],
        *,
        request_id: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        del request_id
        results: dict[str, list[dict[str, Any]]] = {req.policy_id: [] for req in requests}
        if not requests:
            return results

        pool = self._get_pool()
        if pool is None:
            return results

        try:
            with pool.connection() as conn:
                for req in requests:
                    try:
                        policy_uuid = UUID(req.policy_id)
                    except ValueError:
                        continue

                    doc_uuid: UUID | None = None
                    raw_doc_id = req.document_id or req.policy_version_id
                    if raw_doc_id:
                        try:
                            doc_uuid = UUID(str(raw_doc_id))
                        except ValueError:
                            doc_uuid = None

                    allow_owner_scope = req.scope.owner_scope if req.scope and req.scope.owner_scope else None
                    provenance_cids = self._extract_provenance_chunk_ids(req.field_provenance)
                    picked_rows: list[Mapping[str, Any]] = []
                    seen_chunk_keys: set[str] = set()

                    # 1. Verify persisted provenance chunk IDs first against policy_chunks
                    if provenance_cids:
                        prov_rows = conn.execute(
                            """
                            SELECT pc.policy_id, pc.document_id, pd.version_number,
                                   pc.chunk_key, pc.page_number, pc.page_start, pc.page_end,
                                   pc.clause_ref, pc.category, pc.content_text
                            FROM policy_chunks pc
                            JOIN policy_documents pd
                              ON pd.id = pc.document_id AND pd.policy_id = pc.policy_id
                            JOIN policies p ON p.id = pc.policy_id
                            WHERE pc.policy_id = %(policy_id)s
                              AND (%(doc_id)s::uuid IS NULL OR pc.document_id = %(doc_id)s::uuid)
                              AND pd.is_archived = FALSE
                              AND pd.ingestion_status = 'completed'
                              AND p.status = 'active'
                              AND (p.visibility = 'catalog' OR (%(owner_scope)s::text IS NOT NULL AND p.owner_scope = %(owner_scope)s::text))
                              AND pc.chunk_key = ANY(%(chunk_keys)s::text[])
                            ORDER BY pc.page_number ASC, pc.chunk_index ASC
                            LIMIT %(limit)s
                            """,
                            {
                                "policy_id": policy_uuid,
                                "doc_id": doc_uuid,
                                "owner_scope": allow_owner_scope,
                                "chunk_keys": provenance_cids,
                                "limit": self._max_citations,
                            },
                        ).fetchall()
                        for r in prov_rows:
                            ck = str(r["chunk_key"])
                            if ck not in seen_chunk_keys:
                                seen_chunk_keys.add(ck)
                                picked_rows.append(r)

                    # 2. Fill remaining citation slots with FTS/structural chunks from same document
                    remaining = self._max_citations - len(picked_rows)
                    if remaining > 0:
                        fts_query = self._build_fts_query_text(req)
                        fts_rows = conn.execute(
                            """
                            SELECT pc.policy_id, pc.document_id, pd.version_number,
                                   pc.chunk_key, pc.page_number, pc.page_start, pc.page_end,
                                   pc.clause_ref, pc.category, pc.content_text
                            FROM policy_chunks pc
                            JOIN policy_documents pd
                              ON pd.id = pc.document_id AND pd.policy_id = pc.policy_id
                            JOIN policies p ON p.id = pc.policy_id
                            WHERE pc.policy_id = %(policy_id)s
                              AND (%(doc_id)s::uuid IS NULL OR pc.document_id = %(doc_id)s::uuid)
                              AND pd.is_archived = FALSE
                              AND pd.ingestion_status = 'completed'
                              AND p.status = 'active'
                              AND (p.visibility = 'catalog' OR (%(owner_scope)s::text IS NOT NULL AND p.owner_scope = %(owner_scope)s::text))
                              AND NOT (pc.chunk_key = ANY(%(exclude_keys)s::text[]))
                            ORDER BY ts_rank_cd(pc.search_tsv, plainto_tsquery('english', %(qtext)s)) DESC,
                                     CASE pc.category
                                         WHEN 'Coverage' THEN 0
                                         WHEN 'Exclusions' THEN 1
                                         WHEN 'Conditions' THEN 2
                                         ELSE 3
                                     END ASC,
                                     pc.page_number ASC,
                                     pc.chunk_index ASC
                            LIMIT %(limit)s
                            """,
                            {
                                "policy_id": policy_uuid,
                                "doc_id": doc_uuid,
                                "owner_scope": allow_owner_scope,
                                "exclude_keys": list(seen_chunk_keys),
                                "qtext": fts_query,
                                "limit": remaining,
                            },
                        ).fetchall()
                        for r in fts_rows:
                            ck = str(r["chunk_key"])
                            if ck not in seen_chunk_keys:
                                seen_chunk_keys.add(ck)
                                picked_rows.append(r)

                    results[req.policy_id] = [self._row_to_citation(r) for r in picked_rows]
        except Exception as exc:
            logger.warning("Evidence citation lookup failed gracefully: %s", type(exc).__name__)
            return results

        return results
