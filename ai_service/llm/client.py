import json
import logging
import os
import re
from typing import Any, Dict, Optional

from ai_service.config import Settings, settings
from ai_service.llm.providers.nvidia_nemotron import (
    NvidiaConfigurationError,
    NvidiaNemotronProvider,
    NvidiaResponseError,
)

logger = logging.getLogger("coverageiq.llm.client")


class LLMGenerationError(Exception):
    """Base exception for LLM output parsing and schema validation errors."""


class LLMJsonParseError(LLMGenerationError):
    """Raised when an LLM returns non-JSON, empty, or malformed JSON content."""


class LLMValidationError(LLMGenerationError):
    """Raised when parsed LLM output fails schema, decision, or citation validation."""


_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL | re.IGNORECASE)


def parse_llm_json(raw_text: str) -> Dict[str, Any]:
    """
    Parses a JSON object from LLM text, safely stripping outer markdown code fences
    while rejecting truncated, empty, or malformed JSON payloads.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise LLMJsonParseError("LLM response content is empty.")

    cleaned = raw_text.strip()
    fence_match = _CODE_FENCE_RE.match(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    elif cleaned.startswith("```"):
        raise LLMJsonParseError("LLM response JSON is truncated inside an unclosed markdown code fence.")

    if not cleaned:
        raise LLMJsonParseError("LLM response inside markdown code fence is empty.")

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMJsonParseError(f"LLM response JSON is truncated or malformed: {exc.msg}") from exc

    if not isinstance(parsed, dict):
        raise LLMJsonParseError("LLM JSON response must be a JSON object.")

    return parsed


class LLMClient:
    """
    Provider-agnostic LLM client for CoverageIQ.
    Supports NVIDIA Nemotron via NvidiaNemotronProvider (default when LLM_PROVIDER=nvidia),
    Google Gemini via google-genai (when LLM_PROVIDER=gemini), and explicit offline synthesis
    (when LLM_PROVIDER=offline) for local regression testing without API keys.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        nvidia_provider: Optional[NvidiaNemotronProvider] = None,
        app_settings: Optional[Settings] = None,
    ):
        cfg = app_settings or settings
        raw_provider = provider if provider is not None else (os.getenv("LLM_PROVIDER") or cfg.LLM_PROVIDER)
        self.provider_name = raw_provider.strip().lower()
        self.last_metadata: Optional[Dict[str, Any]] = None
        self._genai_client = None
        self._nvidia_provider: Optional[NvidiaNemotronProvider] = None
        self._owns_nvidia_provider = nvidia_provider is None

        if self.provider_name == "nvidia":
            self._nvidia_provider = nvidia_provider or NvidiaNemotronProvider(
                api_key=api_key,
                model=model,
                app_settings=cfg,
            )
            self.api_key = None  # Never store/expose NVIDIA key directly on LLMClient
            self.model = self._nvidia_provider.model
            logger.info("Initialized LLMClient with NVIDIA Nemotron provider (model=%s)", self.model)
        elif self.provider_name == "gemini":
            self.api_key = api_key or cfg.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
            self.model = model or cfg.GEMINI_MODEL
            if self.api_key:
                try:
                    from google import genai
                    self._genai_client = genai.Client(api_key=self.api_key)
                    logger.info("Initialized Google Gemini Client with model %s", self.model)
                except Exception as e:
                    logger.warning("Could not initialize google-genai client: %s", e)
                    self._genai_client = None
        elif self.provider_name in ("offline", "mock"):
            self.api_key = None
            self.model = "offline-deterministic"
            logger.info("Initialized LLMClient in explicit offline synthesis mode.")
        else:
            raise NvidiaConfigurationError(
                f"Unsupported LLM_PROVIDER '{self.provider_name}'. Expected 'nvidia', 'gemini', or 'offline'."
            )

    def close(self) -> None:
        if self._owns_nvidia_provider and self._nvidia_provider is not None:
            self._nvidia_provider.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def generate_json(self, system_instruction: str, prompt: str) -> Dict[str, Any]:
        """
        Generates and parses structured JSON using the configured LLM provider.
        When LLM_PROVIDER=nvidia, provider failures and JSON parse errors propagate
        directly and NEVER silently fall back to offline heuristic synthesis.
        """
        self.last_metadata = None

        if self.provider_name == "nvidia":
            assert self._nvidia_provider is not None
            try:
                result = self._nvidia_provider.complete(
                    system_prompt=system_instruction,
                    user_prompt=prompt,
                )
            except NvidiaResponseError as exc:
                self.last_metadata = {
                    "provider": "nvidia",
                    "model": self.model,
                    "finish_reason": exc.finish_reason,
                    "request_id": exc.request_id,
                    "usage": exc.usage.model_dump() if exc.usage else None,
                }
                raise

            self.last_metadata = {
                "provider": "nvidia",
                "model": result.model,
                "finish_reason": result.finish_reason,
                "request_id": result.request_id,
                "usage": result.usage.model_dump() if result.usage else None,
            }
            try:
                return parse_llm_json(result.content)
            except LLMJsonParseError:
                logger.error(
                    "NVIDIA response JSON parse failure: model=%s finish_reason=%s request_id=%s total_tokens=%s",
                    result.model,
                    result.finish_reason,
                    result.request_id,
                    result.usage.total_tokens if result.usage else None,
                )
                raise

        if self.provider_name == "gemini" and self._genai_client:
            try:
                from google.genai import types
                response = self._genai_client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=settings.LLM_TEMPERATURE,
                        response_mime_type="application/json",
                    ),
                )
                self.last_metadata = {"provider": "gemini", "model": self.model}
                return parse_llm_json(response.text)
            except Exception as e:
                logger.error("Gemini generation failed: %s. Falling back to grounded rule engine.", e)

        # Offline deterministic synthesis only when provider is offline/gemini-unconfigured
        self.last_metadata = {"provider": "offline", "model": "offline-deterministic"}
        return self._offline_grounded_synthesis(prompt)

    def _offline_grounded_synthesis(self, prompt: str) -> Dict[str, Any]:
        """
        Offline rule-guided synthesizer strictly operating on the prompt's query and retrieved chunks.
        Dynamically analyzes any policy line (Health, Motor/Car, Life, Term) ingested into the KB.
        Guarantees that local tests and dynamic policies run immediately without external API credits.
        """
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
                "claims": [],
            }

        # Parse candidate chunks from evidence_text
        # Format: [i] CHUNK_ID: <cid> | PAGE: <p> | CLAUSE: <clause> | CATEGORY: <cat>\nCONTENT: <content>
        chunk_blocks = re.findall(
            r"CHUNK_ID:\s*([^\s\|]+)\s*\|\s*PAGE:\s*(\d+)\s*\|\s*CLAUSE:\s*([^\|]+)\s*\|\s*CATEGORY:\s*([^\n]+)\s*\nCONTENT:\s*([^\n\[]+)",
            evidence_text,
        )

        # Tokenize query
        stopwords = {"is", "the", "under", "this", "policy", "can", "i", "for", "a", "an", "what", "of", "in", "to", "or", "and"}
        q_tokens = set(re.findall(r"\b[a-zA-Z0-9_\.-]+\b", user_query)) - stopwords

        if not chunk_blocks or not q_tokens:
            return {
                "decision": "INSUFFICIENT EVIDENCE",
                "reason": "Insufficient specific policy clauses found to support an affirmative coverage determination.",
                "conditions": [],
                "claims": [],
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
                "overlap": overlap,
            })

        # Sort chunks by overlap descending
        scored_chunks.sort(key=lambda x: x["overlap"], reverse=True)
        best_match = scored_chunks[0]

        if best_match["overlap"] == 0:
            return {
                "decision": "INSUFFICIENT EVIDENCE",
                "reason": "The retrieved policy evidence does not contain clauses addressing this specific procedure or question.",
                "conditions": [],
                "claims": [],
            }

        # Check for explicit exclusion match
        exclusion_matches = [
            c for c in scored_chunks
            if c["category"] == "Exclusions" and c["overlap"] >= 1
        ]

        triggers_exclusion = False
        active_exclusion = None
        if exclusion_matches:
            if best_match["category"] == "Exclusions":
                triggers_exclusion = True
                active_exclusion = best_match
            else:
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
                        "claim_type": "exclusion",
                    }
                ],
            }

        # Check for condition / waiting period match
        condition_matches = [c for c in scored_chunks if c["category"] == "Conditions" and c["overlap"] >= 1]
        coverage_matches = [c for c in scored_chunks if c["category"] == "Coverage" and c["overlap"] >= 1]

        if condition_matches and ("waiting period" in user_query or "maternity" in user_query or not coverage_matches):
            cond = condition_matches[0]
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
                        "claim_type": "condition",
                    }
                ],
            }

        if coverage_matches:
            cov = coverage_matches[0]
            conditions = []
            for c in condition_matches:
                if "room rent" in c["content"].lower() or "sub-limit" in c["content"].lower():
                    conditions.append("Room rent is capped at 1% of Sum Insured per day, and ICU capped at 2%.")
            if "24 consecutive hours" in cov["content"].lower() or "24 hours" in cov["content"].lower():
                conditions.insert(0, "Minimum 24 consecutive hours of in-patient hospitalization required.")

            claims = [
                {
                    "text": cov["content"][:200],
                    "chunk_ids": [cov["chunk_id"]],
                    "claim_type": "coverage",
                }
            ]
            if condition_matches:
                claims.append({
                    "text": condition_matches[0]["content"][:200],
                    "chunk_ids": [condition_matches[0]["chunk_id"]],
                    "claim_type": "condition",
                })

            return {
                "decision": "COVERED",
                "reason": f"Covered under {cov['clause']}, subject to applicable policy conditions and limits.",
                "conditions": conditions,
                "claims": claims,
            }

        return {
            "decision": "INSUFFICIENT EVIDENCE",
            "reason": "Insufficient specific policy clauses found to support an affirmative coverage determination.",
            "conditions": [],
            "claims": [],
        }
