from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class ScopeContext(BaseModel):
    user_id: Optional[str] = None
    role: Optional[str] = "user"
    owner_scope: Optional[str] = None
    allow_catalog_read: bool = False

class AskQueryRequest(BaseModel):
    policy_id: str = Field(..., description="Target policy ID to query against")
    question: str = Field(..., min_length=3, description="Natural language question from policyholder")
    session_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional conversational history context")
    scope: Optional[ScopeContext] = Field(default_factory=ScopeContext, description="Trusted security scope from Node.js gateway")

class IngestMockDataRequest(BaseModel):
    policy_id: str
    chunks: list
