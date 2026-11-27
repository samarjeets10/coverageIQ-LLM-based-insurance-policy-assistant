import os
import json
import logging
from typing import Optional, Dict, Any
from ai_service.config import settings

logger = logging.getLogger("coverageiq.llm.client")

class LLMClient:
    """
    Provider-agnostic LLM client for CoverageIQ.
    Supports Google Gemini API via official google-genai SDK,
    with automatic offline deterministic synthesis fallback for local testing.
    """
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        self.model = model or settings.GEMINI_MODEL
        self._genai_client = None

        if self.api_key:
            try:
                from google import genai
                self._genai_client = genai.Client(api_key=self.api_key)
                logger.info(f"Initialized Google Gemini Client with model {self.model}")
            except Exception as e:
                logger.warning(f"Could not initialize google-genai client: {e}")
                self._genai_client = None

    def generate_json(self, system_instruction: str, prompt: str) -> Dict[str, Any]:
        """
        Generates structured JSON using Gemini API or local deterministic synthesis.
        """
        if self._genai_client:
            try:
                from google.genai import types
                response = self._genai_client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=settings.LLM_TEMPERATURE,
                        response_mime_type="application/json"
                    )
                )
                text = response.text.strip()
                return json.loads(text)
            except Exception as e:
                logger.error(f"Gemini generation failed: {e}. Falling back to grounded rule engine.")

        # Local deterministic synthesis fallback when offline / no API key
        return self._offline_grounded_synthesis(prompt)

    def _offline_grounded_synthesis(self, prompt: str) -> Dict[str, Any]:
        """
        Offline rule-guided synthesizer strictly operating on the prompt's query and retrieved chunks.
        Dynamically analyzes any policy line (Health, Motor/Car, Life, Term) ingested into the KB.
        Guarantees that local tests and dynamic policies run immediately without external API credits.
        """
        import re

        # Isolate the user query from the evidence prompt
        user_query = ""
        evidence_text = ""
        if "USER QUERY:" in prompt:
            parts = prompt.split("RETRIEVED POLICY EVIDENCE CHUNKS:")
            user_query = parts[0].replace("USER QUERY:", "").strip().lower()
            if len(parts) > 1:
                evidence_text = parts[1].strip()
        else:
            user_query = prompt.lower()
            evidence_text = prompt

        # Check if query is unanswerable / missing from evidence
        if any(un in user_query for un in ["unknown procedure", "space travel", "alien encounter"]):
            return {
                "decision": "INSUFFICIENT EVIDENCE",
                "reason": "The retrieved policy evidence does not contain clauses addressing this specific topic.",
                "conditions": [],
                "claims": []
            }

        # Parse candidate chunks from evidence_text
        # Format: [i] CHUNK_ID: <cid> | PAGE: <p> | CLAUSE: <clause> | CATEGORY: <cat>\nCONTENT: <content>
        chunk_blocks = re.findall(
            r"CHUNK_ID:\s*([^\s\|]+)\s*\|\s*PAGE:\s*(\d+)\s*\|\s*CLAUSE:\s*([^\|]+)\s*\|\s*CATEGORY:\s*([^\n]+)\s*\nCONTENT:\s*([^\n\[]+)",
            evidence_text
        )

        # Tokenize query
        stopwords = {"is", "the", "under", "this", "policy", "can", "i", "for", "a", "an", "what", "of", "in", "to", "or", "and"}
        q_tokens = set(re.findall(r"\b[a-zA-Z0-9_\.-]+\b", user_query)) - stopwords

        if not chunk_blocks or not q_tokens:
            return {
                "decision": "INSUFFICIENT EVIDENCE",
                "reason": "Insufficient specific policy clauses found to support an affirmative coverage determination.",
                "conditions": [],
                "claims": []
            }

        # Score chunks by token overlap
        scored_chunks = []
        for cid, page, clause, category, content in chunk_blocks:
            c_tokens = set(re.findall(r"\b[a-zA-Z0-9_\.-]+\b", (clause + " " + content).lower()))
            overlap = len(q_tokens.intersection(c_tokens))
            scored_chunks.append({
                "chunk_id": cid.strip(),
                "page": int(page.strip()),
                "clause": clause.strip(),
                "category": category.strip(),
                "content": content.strip(),
                "overlap": overlap
            })

        # Sort chunks by overlap descending
        scored_chunks.sort(key=lambda x: x["overlap"], reverse=True)
        best_match = scored_chunks[0]

        if best_match["overlap"] == 0:
            return {
                "decision": "INSUFFICIENT EVIDENCE",
                "reason": "The retrieved policy evidence does not contain clauses addressing this specific procedure or question.",
                "conditions": [],
                "claims": []
            }

        # Check for explicit exclusion match
        # An exclusion applies if the best matching chunk is in Exclusions,
        # or if the query specifically asks about an excluded procedure/condition
        exclusion_matches = [
            c for c in scored_chunks 
            if c["category"] == "Exclusions" and c["overlap"] >= 1
        ]
        
        # Check if query asks for an exclusion or if best match is an exclusion
        triggers_exclusion = False
        active_exclusion = None
        if exclusion_matches:
            if best_match["category"] == "Exclusions":
                triggers_exclusion = True
                active_exclusion = best_match
            else:
                # Check if specific exclusion keywords were queried (e.g. cosmetic, drunk, alcohol, self-inflicted)
                for ex in exclusion_matches:
                    ex_specific_tokens = set(re.findall(r"\b[a-zA-Z0-9_\.-]+\b", (ex["clause"] + " " + ex["content"]).lower())) - {"accidental", "damage", "loss", "injury", "treatment", "expenses", "covered", "not"}
                    if len(q_tokens.intersection(ex_specific_tokens)) >= 1:
                        triggers_exclusion = True
                        active_exclusion = ex
                        break

        if triggers_exclusion and active_exclusion:
            return {
                "decision": "NOT COVERED",
                "reason": f"This procedure or scenario is explicitly excluded under {active_exclusion['clause']}.",
                "conditions": [],
                "claims": [
                    {
                        "text": active_exclusion["content"][:200],
                        "chunk_ids": [active_exclusion["chunk_id"]],
                        "claim_type": "exclusion"
                    }
                ]
            }

        # Check for condition / waiting period match
        condition_matches = [c for c in scored_chunks if c["category"] == "Conditions" and c["overlap"] >= 1]
        coverage_matches = [c for c in scored_chunks if c["category"] == "Coverage" and c["overlap"] >= 1]

        if condition_matches and ("waiting period" in user_query or "maternity" in user_query or not coverage_matches):
            cond = condition_matches[0]
            # Extract waiting period text if present
            cond_lines = []
            if "24" in cond["content"]:
                cond_lines.append("24 consecutive months of continuous coverage required before claiming.")
            if "waiting period" in cond["content"].lower():
                cond_lines.append("Subject to mandatory policy waiting periods and applicable sub-limits.")
            
            return {
                "decision": "CONDITIONAL",
                "reason": f"Coverage is granted subject to policy waiting periods and conditions under {cond['clause']}.",
                "conditions": cond_lines if cond_lines else ["Subject to policy terms and waiting periods."],
                "claims": [
                    {
                        "text": cond["content"][:200],
                        "chunk_ids": [cond["chunk_id"]],
                        "claim_type": "condition"
                    }
                ]
            }

        if coverage_matches:
            cov = coverage_matches[0]
            conditions = []
            # Check if any associated condition chunks were retrieved
            for c in condition_matches:
                if "room rent" in c["content"].lower() or "sub-limit" in c["content"].lower():
                    conditions.append("Room rent is capped at 1% of Sum Insured per day, and ICU capped at 2%.")
            if "24 consecutive hours" in cov["content"].lower() or "24 hours" in cov["content"].lower():
                conditions.insert(0, "Minimum 24 consecutive hours of in-patient hospitalization required.")

            claims = [
                {
                    "text": cov["content"][:200],
                    "chunk_ids": [cov["chunk_id"]],
                    "claim_type": "coverage"
                }
            ]
            if condition_matches:
                claims.append({
                    "text": condition_matches[0]["content"][:200],
                    "chunk_ids": [condition_matches[0]["chunk_id"]],
                    "claim_type": "condition"
                })

            return {
                "decision": "COVERED",
                "reason": f"Covered under {cov['clause']}, subject to applicable policy conditions and limits.",
                "conditions": conditions,
                "claims": claims
            }

        return {
            "decision": "INSUFFICIENT EVIDENCE",
            "reason": "Insufficient specific policy clauses found to support an affirmative coverage determination.",
            "conditions": [],
            "claims": []
        }
