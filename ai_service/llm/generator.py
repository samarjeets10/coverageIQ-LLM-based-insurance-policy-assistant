from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ai_service.llm.client import LLMClient, LLMValidationError
from ai_service.llm.prompts import GROUNDED_SYSTEM_PROMPT, format_evidence_prompt
from ai_service.schemas.structured_output import AtomicClaim, CitationItem, DecisionType, PolicyChunk


class _RawClaimPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    chunk_ids: List[str]
    claim_type: Literal["coverage", "exclusion", "condition"] = "coverage"

    @field_validator("text")
    @classmethod
    def validate_text_non_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Claim text must be a non-empty string.")
        return v.strip()

    @field_validator("chunk_ids")
    @classmethod
    def validate_chunk_ids_non_empty(cls, v: List[str]) -> List[str]:
        cleaned = [cid.strip() for cid in v if isinstance(cid, str) and cid.strip()]
        if not cleaned:
            raise ValueError("Every claim must cite at least one valid chunk_id.")
        return cleaned


class _RawDraftPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    decision: DecisionType
    reason: str
    conditions: List[str]
    claims: List[_RawClaimPayload]
    citations: Optional[List[Dict[str, Any]]] = None

    @field_validator("reason")
    @classmethod
    def validate_reason_non_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("reason must be a non-empty string.")
        return v.strip()

    @field_validator("conditions")
    @classmethod
    def validate_conditions(cls, v: List[str]) -> List[str]:
        return [c.strip() for c in v if isinstance(c, str) and c.strip()]


class GroundedGenerator:
    """
    Module 5 (Stage 7): Grounded LLM Generation
    Enforces the principle: "NO EVIDENCE = NO CLAIM"
    Synthesizes a draft response with strictly validated chunk_id citations
    and provenance resolved directly from retrieved PolicyChunk objects.
    """

    def __init__(self, client: Optional[LLMClient] = None):
        self._owns_client = client is None
        self.client = client or LLMClient()

    def close(self) -> None:
        if self._owns_client and self.client is not None and hasattr(self.client, "close"):
            self.client.close()

    @staticmethod
    def _resolve_page_bounds(chunk: PolicyChunk) -> tuple[int, Optional[int]]:
        meta_start = chunk.metadata.get("page_start") if chunk.metadata else None
        meta_end = chunk.metadata.get("page_end") if chunk.metadata else None
        p_start = chunk.page_start if chunk.page_start is not None else (
            int(meta_start) if meta_start is not None else chunk.page_number
        )
        p_end = chunk.page_end if chunk.page_end is not None else (
            int(meta_end) if meta_end is not None else None
        )
        return p_start, p_end

    def _format_evidence_chunks(self, chunks: List[PolicyChunk]) -> str:
        lines = []
        for i, c in enumerate(chunks, start=1):
            p_start, p_end = self._resolve_page_bounds(c)
            p_display = f"{p_start}-{p_end}" if p_end is not None and p_end != p_start else str(p_start)
            lines.append(
                f"[{i}] CHUNK_ID: {c.chunk_id} | PAGE: {p_display} | "
                f"CLAUSE: {c.clause_ref} | CATEGORY: {c.category.value}\n"
                f"CONTENT: {c.content_text}\n"
            )
        return "\n".join(lines)

    def generate_draft(self, query: str, evidence: List[PolicyChunk]) -> Dict[str, Any]:
        """
        Generates and strictly validates a draft grounded answer from evidence chunks.
        Rejects malformed JSON, invalid DecisionType values, missing required fields,
        unsupported claims, and unknown chunk_id citations.
        """
        evidence_text = self._format_evidence_chunks(evidence)
        prompt = format_evidence_prompt(query, evidence_text)

        raw_output = self.client.generate_json(GROUNDED_SYSTEM_PROMPT, prompt)
        if not isinstance(raw_output, dict):
            raise LLMValidationError("LLM client output must be a JSON object.")

        try:
            validated = _RawDraftPayload.model_validate(raw_output)
        except ValidationError as exc:
            raise LLMValidationError(f"Generated output failed schema validation: {exc}") from exc

        if validated.decision != DecisionType.INSUFFICIENT_EVIDENCE and not validated.claims:
            raise LLMValidationError(
                f"Decision '{validated.decision.value}' requires at least one evidence-grounded claim."
            )

        chunk_lookup: Dict[str, PolicyChunk] = {c.chunk_id: c for c in evidence}

        # Reject any model-returned citations referencing unknown chunk IDs
        if validated.citations is not None:
            for item in validated.citations:
                cid = item.get("chunk_id") if isinstance(item, dict) else None
                if not isinstance(cid, str) or cid not in chunk_lookup:
                    raise LLMValidationError(
                        f"Model response citations contain unknown chunk_id '{cid}'."
                    )

        claims: List[AtomicClaim] = []
        cited_chunk_order: List[str] = []
        seen_cids = set()

        for raw_claim in validated.claims:
            for cid in raw_claim.chunk_ids:
                if cid not in chunk_lookup:
                    raise LLMValidationError(
                        f"Claim '{raw_claim.text[:60]}' references unknown chunk_id '{cid}'."
                    )
                if cid not in seen_cids:
                    seen_cids.add(cid)
                    cited_chunk_order.append(cid)

            claims.append(
                AtomicClaim(
                    text=raw_claim.text,
                    chunk_ids=raw_claim.chunk_ids,
                    claim_type=raw_claim.claim_type,
                )
            )

        # Build citations strictly from retrieved PolicyChunk objects (ignoring any model page numbers)
        citations: List[CitationItem] = []
        for cid in cited_chunk_order:
            chunk = chunk_lookup[cid]
            p_start, p_end = self._resolve_page_bounds(chunk)
            citations.append(
                CitationItem(
                    chunk_id=chunk.chunk_id,
                    page=p_start,
                    page_end=p_end if p_end is not None and p_end != p_start else None,
                    clause=chunk.clause_ref,
                    category=chunk.category.value,
                    snippet=chunk.content_text[:160] + "..." if len(chunk.content_text) > 160 else chunk.content_text,
                )
            )

        result: Dict[str, Any] = {
            "decision": validated.decision.value,
            "reason": validated.reason,
            "conditions": validated.conditions,
            "claims": claims,
            "citations": citations,
            "raw_output": raw_output,
        }
        client_meta = getattr(self.client, "last_metadata", None)
        if isinstance(client_meta, dict):
            result["llm_metadata"] = client_meta

        return result
