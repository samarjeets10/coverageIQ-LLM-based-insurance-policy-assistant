import re
from typing import List, Dict, Any
from pydantic import BaseModel

class AnalyzedQuery(BaseModel):
    raw_query: str
    cleaned_query: str
    intent: str
    entities: List[str]
    expanded_terms: List[str]

class QueryAnalyzer:
    """
    Module 4 (6C): Query Analysis
    Parses natural language insurance queries to identify:
    - User Intent (Coverage, Exclusion, Condition, Claim, Waiting Period)
    - Domain Entities (medical conditions, procedures, clause codes)
    - Lexical Expansions for hybrid search
    """
    
    INTENT_KEYWORDS = {
        "EXCLUSION_INQUIRY": ["exclude", "not covered", "exclusion", "prohibited", "denied", "restriction"],
        "WAITING_PERIOD": ["waiting period", "wait", "months", "years", "ped", "pre-existing", "when can i claim"],
        "SUB_LIMIT_QUERY": ["sub-limit", "sub limit", "room rent", "cap", "limit", "maximum per day", "icu"],
        "CLAIM_PROCESS": ["claim", "reimbursement", "cashless", "settlement", "file claim"],
        "COVERAGE_CHECK": ["cover", "covered", "eligible", "pay for", "include", "can i claim"]
    }
    
    DOMAIN_SYNONYMS = {
        "hospital": ["hospitalization", "in-patient", "admission", "room rent"],
        "cosmetic": ["aesthetic", "plastic surgery", "beauty", "cosmetic surgery"],
        "maternity": ["delivery", "pregnancy", "caesarean", "childbirth", "infant"],
        "ped": ["pre-existing", "pre existing disease", "chronic", "hypertension", "diabetes"],
        "accident": ["emergency", "injury", "trauma", "casualty"],
        "teeth": ["dental", "dentistry", "oral"],
        "eye": ["cataract", "vision", "ophthalmic"]
    }

    def analyze(self, query: str) -> AnalyzedQuery:
        cleaned = re.sub(r"[^\w\s\.-]", " ", query.lower()).strip()
        tokens = cleaned.split()
        
        # Detect Intent
        intent = "COVERAGE_CHECK"
        for potential_intent, kw_list in self.INTENT_KEYWORDS.items():
            if any(kw in cleaned for kw in kw_list):
                intent = potential_intent
                break

        # Detect Entities and Clause Numbers (e.g. Clause 4.1, Section 2)
        clause_matches = re.findall(r"(?:section|clause|exclusion|rule)\s*\d+(?:\.\d+)?", cleaned)
        entities = list(set(clause_matches))
        
        for root_term, syns in self.DOMAIN_SYNONYMS.items():
            if root_term in cleaned or any(s in cleaned for s in syns):
                entities.append(root_term)

        # Lexical expansions
        expanded = list(tokens)
        for entity in entities:
            if entity in self.DOMAIN_SYNONYMS:
                expanded.extend(self.DOMAIN_SYNONYMS[entity])
                
        return AnalyzedQuery(
            raw_query=query,
            cleaned_query=cleaned,
            intent=intent,
            entities=list(set(entities)),
            expanded_terms=list(set(expanded))
        )
