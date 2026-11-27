import logging
from typing import List, Optional, Dict
import numpy as np
from ai_service.schemas.structured_output import PolicyChunk, ChunkCategory
from ai_service.config import settings

logger = logging.getLogger("coverageiq.database")

class MockKnowledgeBase:
    """
    In-memory knowledge base for local offline execution and automated test suites.
    Supports storing PolicyChunk instances, computing cosine similarity over embeddings,
    and retrieving by policy_id.
    """
    def __init__(self):
        self._chunks: Dict[str, List[PolicyChunk]] = {}
        self._init_default_insurance_corpus()

    def store_chunks(self, policy_id: str, chunks: List[PolicyChunk]):
        if policy_id not in self._chunks:
            self._chunks[policy_id] = []
        self._chunks[policy_id].extend(chunks)

    def get_chunks(self, policy_id: str) -> List[PolicyChunk]:
        return self._chunks.get(policy_id, [])

    def get_all_chunks(self) -> List[PolicyChunk]:
        all_c = []
        for c_list in self._chunks.values():
            all_c.extend(c_list)
        return all_c

    def _init_default_insurance_corpus(self):
        """
        Seeds standard gold-standard insurance policy clauses matching IEEE SRS test cases:
        - TC-06: Hospitalization (Coverage, Section 2.1, Page 12)
        - TC-07: Cosmetic surgery exclusion (Exclusion 4.1, Page 22)
        - TC-08: Maternity coverage with 24-month waiting period (Conditions, Section 5.3, Page 29)
        - Room rent sub-limits and Pre-existing disease waiting period (Clause 3.2, Page 15)
        """
        policy_health_01 = "policy_health_01"
        chunks = [
            PolicyChunk(
                chunk_id="chunk_h_001",
                policy_id=policy_health_01,
                clause_ref="Section 2.1 - In-Patient Hospitalization",
                page_number=12,
                category=ChunkCategory.COVERAGE,
                content_text=(
                    "Section 2.1 In-Patient Hospitalization Expenses: The company will indemnify medical expenses "
                    "incurred for in-patient hospitalization treatment for illness or injury, provided the hospitalization "
                    "is for a minimum period of 24 consecutive hours. Coverage includes room rent, nursing expenses, "
                    "surgeon and specialist fees, and operation theatre charges up to the Sum Insured."
                ),
            ),
            PolicyChunk(
                chunk_id="chunk_h_002",
                policy_id=policy_health_01,
                clause_ref="Section 2.4 - Room Rent and ICU Sub-Limits",
                page_number=15,
                category=ChunkCategory.CONDITIONS,
                content_text=(
                    "Section 2.4 Room Rent Sub-limit: Room rent is capped at 1% of the Sum Insured per day, "
                    "and Intensive Care Unit (ICU) charges are capped at 2% of the Sum Insured per day. If the insured occupies "
                    "a room with higher tariff, proportionate deduction shall apply to associated medical charges."
                ),
            ),
            PolicyChunk(
                chunk_id="chunk_h_003",
                policy_id=policy_health_01,
                clause_ref="Section 4.1 - General Exclusions (Cosmetic & Aesthetic Surgery)",
                page_number=22,
                category=ChunkCategory.EXCLUSIONS,
                content_text=(
                    "Section 4.1 Exclusions: Cosmetic, aesthetic or plastic surgery is strictly NOT COVERED under any circumstances, "
                    "except when medically necessary arising out of an accident, burn reconstruction, or as part of cancer treatment."
                ),
            ),
            PolicyChunk(
                chunk_id="chunk_h_004",
                policy_id=policy_health_01,
                clause_ref="Section 5.3 - Maternity Benefit & Waiting Periods",
                page_number=29,
                category=ChunkCategory.CONDITIONS,
                content_text=(
                    "Section 5.3 Maternity Cover: Medical expenses relating to delivery (normal or caesarean) are covered subject to "
                    "a mandatory waiting period of 24 consecutive months of continuous coverage under this policy. "
                    "A maximum limit of INR 50,000 for normal delivery and INR 75,000 for caesarean section applies."
                ),
            ),
            PolicyChunk(
                chunk_id="chunk_h_005",
                policy_id=policy_health_01,
                clause_ref="Section 5.1 - Pre-existing Disease (PED) Waiting Period",
                page_number=28,
                category=ChunkCategory.CONDITIONS,
                content_text=(
                    "Section 5.1 Pre-Existing Diseases: Any pre-existing condition, ailment or injury declared by the insured "
                    "shall be covered only after 36 continuous months of uninterrupted coverage from the policy inception date."
                ),
            ),
            PolicyChunk(
                chunk_id="chunk_h_006",
                policy_id=policy_health_01,
                clause_ref="Section 4.5 - Self-Inflicted Injury and Substance Abuse",
                page_number=24,
                category=ChunkCategory.EXCLUSIONS,
                content_text=(
                    "Section 4.5 Exclusions: Expenses directly or indirectly caused by intentional self-injury, suicide attempt, "
                    "or the abuse of alcohol, intoxicating drugs, or controlled substances are NOT COVERED."
                ),
            ),
        ]
        self.store_chunks(policy_health_01, chunks)


# Global Knowledge Base instance
kb = MockKnowledgeBase()

def get_knowledge_base() -> MockKnowledgeBase:
    return kb
