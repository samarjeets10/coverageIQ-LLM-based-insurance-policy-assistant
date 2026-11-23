from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class DecisionType(str, Enum):
    COVERED = "COVERED"
    NOT_COVERED = "NOT COVERED"
    CONDITIONAL = "CONDITIONAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT EVIDENCE"


class ChunkCategory(str, Enum):
    COVERAGE = "Coverage"
    EXCLUSIONS = "Exclusions"
    CONDITIONS = "Conditions"
    DEFINITIONS = "Definitions"
    GENERAL = "General"


class PolicyChunk(BaseModel):
    chunk_id: str
    policy_id: str
    clause_ref: str
    page_number: int
    category: ChunkCategory = ChunkCategory.GENERAL
    content_text: str
    embedding: Optional[List[float]] = None
    similarity_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


class CitationItem(BaseModel):
    chunk_id: str
    page: int
    clause: str
    category: Optional[str] = None
    snippet: Optional[str] = None


class AtomicClaim(BaseModel):
    text: str
    chunk_ids: List[str] = Field(default_factory=list)
    claim_type: str = "coverage"  # coverage | exclusion | condition


class EvaluatorIssue(BaseModel):
    claim_idx: Optional[int] = None
    issue_type: str  # unsupported | contradiction | bad_citation | missing_exclusion
    description: str


class EvaluatorResult(BaseModel):
    passed: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    evidence_supported: bool
    contradiction_found: bool
    citation_valid: bool
    completeness_passed: bool
    issues: List[EvaluatorIssue] = Field(default_factory=list)
    raw_evaluator_comment: Optional[str] = None


class StructuredAnswerResponse(BaseModel):
    query: str
    decision: DecisionType
    confidence_score: float
    conditions: List[str] = Field(default_factory=list)
    reason: str
    citations: List[CitationItem] = Field(default_factory=list)
    evaluator_status: str
    retry_count: int
    metadata: Optional[Dict[str, Any]] = None
