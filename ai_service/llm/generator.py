from typing import List, Dict, Any, Tuple
from ai_service.schemas.structured_output import PolicyChunk, AtomicClaim, CitationItem, DecisionType
from ai_service.llm.client import LLMClient
from ai_service.llm.prompts import GROUNDED_SYSTEM_PROMPT, format_evidence_prompt

class GroundedGenerator:
    """
    Module 5 (Stage 7): Grounded LLM Generation
    Enforces the principle: "NO EVIDENCE = NO CLAIM"
    Synthesizes a draft response with explicit chunk_id citations.
    """
    def __init__(self, client: LLMClient = None):
        self.client = client or LLMClient()

    def _format_evidence_chunks(self, chunks: List[PolicyChunk]) -> str:
        lines = []
        for i, c in enumerate(chunks, start=1):
            p_end = c.page_end or (c.metadata.get("page_end") if c.metadata else None)
            p_display = f"{c.page_number}-{p_end}" if p_end and p_end != c.page_number else str(c.page_number)
            lines.append(
                f"[{i}] CHUNK_ID: {c.chunk_id} | PAGE: {p_display} | "
                f"CLAUSE: {c.clause_ref} | CATEGORY: {c.category.value}\n"
                f"CONTENT: {c.content_text}\n"
            )
        return "\n".join(lines)

    def generate_draft(self, query: str, evidence: List[PolicyChunk]) -> Dict[str, Any]:
        """
        Generates a draft grounded answer from evidence chunks.
        """
        evidence_text = self._format_evidence_chunks(evidence)
        prompt = format_evidence_prompt(query, evidence_text)
        
        raw_output = self.client.generate_json(GROUNDED_SYSTEM_PROMPT, prompt)
        
        # Parse atomic claims
        claims: List[AtomicClaim] = []
        for c in raw_output.get("claims", []):
            claims.append(
                AtomicClaim(
                    text=c.get("text", ""),
                    chunk_ids=c.get("chunk_ids", []),
                    claim_type=c.get("claim_type", "coverage")
                )
            )

        # Build citations mapping from referenced chunk_ids
        chunk_lookup = {c.chunk_id: c for c in evidence}
        cited_chunks = set()
        for cl in claims:
            for cid in cl.chunk_ids:
                if cid in chunk_lookup:
                    cited_chunks.add(cid)

        citations: List[CitationItem] = []
        for cid in cited_chunks:
            chunk = chunk_lookup[cid]
            p_end = chunk.page_end or (chunk.metadata.get("page_end") if chunk.metadata else None)
            citations.append(
                CitationItem(
                    chunk_id=chunk.chunk_id,
                    page=chunk.page_number,
                    page_end=p_end if p_end and p_end != chunk.page_number else None,
                    clause=chunk.clause_ref,
                    category=chunk.category.value,
                    snippet=chunk.content_text[:160] + "..." if len(chunk.content_text) > 160 else chunk.content_text
                )
            )

        return {
            "decision": raw_output.get("decision", "INSUFFICIENT EVIDENCE"),
            "reason": raw_output.get("reason", "No verified reason provided."),
            "conditions": raw_output.get("conditions", []),
            "claims": claims,
            "citations": citations,
            "raw_output": raw_output
        }
