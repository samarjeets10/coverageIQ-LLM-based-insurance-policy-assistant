# CoverageIQ — Local Execution & Verification Guide

**Module:** Core Intelligence & Verification Engine (6C Query Retrieval & Evaluator)  
**Assigned Developer:** Mr. Yash Shankar Nimbalkar (Roll No.: 251)  
**Project:** CoverageIQ – An Evidence-Grounded Insurance Policy Intelligence System  
**Core Architectural Mandate:** **`NO EVIDENCE = NO CLAIM`**  

---

## 1. What This Module Does

This module bridges raw insurance policy clauses and final policyholder answers. It ensures that no claim of coverage is ever presented to a user without verified, citation-backed evidence from the source policy document.

### Pipeline Overview
```
User Query ──► 1. Query Analyzer (intent + entities)
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
2a. Dense Vector Search     2b. Sparse BM25 Search
  (Cosine similarity)          (Exact clause & code match)
        │                           │
        └─────────────┬─────────────┘
                      ▼
        3. Reciprocal Rank Fusion (RRF)
                      ▼
        4. Cross-Encoder Reranker (Top-K evidence)
                      ▼
        5. Grounded LLM ("NO EVIDENCE = NO CLAIM")
                      ▼
        6. Evaluator Agent (5-point verification)
            │ PASS                  │ FAIL (up to 2 retries)
            ▼                       ▼
      Structured Output       Closed Retry Loop (widen K, refine terms)
  (COVERED / NOT COVERED /          │ (if exhausted)
   CONDITIONAL)                     ▼
                             INSUFFICIENT EVIDENCE
```

### Component Breakdown
| Component | File Path | What It Does |
|---|---|---|
| **Query Analyzer** | `ai_service/retrieval/query_analyzer.py` | Detects user intent (`COVERAGE_CHECK`, `EXCLUSION_INQUIRY`, `WAITING_PERIOD`, etc.), extracts clause references (e.g. `Section 2.1`), and expands medical synonyms. |
| **Dense Retriever** | `ai_service/retrieval/dense_retriever.py` | Computes cosine similarity between query embeddings and policy clauses (supports 768-dim embeddings). |
| **BM25 Retriever** | `ai_service/retrieval/bm25_retriever.py` | Uses `BM25Okapi` to find exact numeric clauses, waiting period numbers, and exclusion terms. |
| **RRF Fusion** | `ai_service/retrieval/rrf_fusion.py` | Merges dense and sparse rankings using Reciprocal Rank Fusion ($k = 60, w_{vec} = 0.6, w_{bm25} = 0.4$). |
| **Cross-Encoder Reranker** | `ai_service/retrieval/reranker.py` | Jointly scores (query, clause) pairs using cross-attention neural models to pick the Top-K high-precision evidence chunks. |
| **Grounded LLM Generator** | `ai_service/llm/generator.py` | Prompts the LLM with strict grounding constraints. Every claim must have an associated `chunk_id` citation. |
| **Evaluator Agent (Verifier)** | `ai_service/evaluator/verifier.py` | Independently audits: (1) Evidence support, (2) Contradictions across clauses, (3) Citation validity, (4) Completeness, and (5) Confidence score ($\ge 0.70$). |
| **Closed Retry Loop** | `ai_service/evaluator/retry_controller.py` | Coordinates the entire pipeline. Automatically expands retrieval depth ($K = K + 3$) and refines terms on verification failure up to 2 retries. |
| **REST API Server** | `ai_service/main.py` | FastAPI application exposing `GET /ai/health` and `POST /ai/ask`. |

---

## 2. Technologies & Libraries Used

| Technology / Library | Purpose in this Module |
|---|---|
| **Python 3.10+** | Core programming language for all AI and retrieval microservices. |
| **FastAPI & Uvicorn** | High-performance asynchronous REST API framework and ASGI server. |
| **Pydantic v2** | Strict data validation and schema enforcement for the JSON output contract. |
| **rank-bm25** | BM25Okapi probabilistic lexical retrieval algorithm for exact clause and keyword matching. |
| **sentence-transformers & PyTorch** | Cross-Encoder neural reranking model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for query-clause relevance scoring. |
| **google-genai** | Official Google GenAI SDK for Gemini 2.0 Flash generation and text embeddings. |
| **NumPy** | High-performance vector operations and cosine similarity computation. |
| **pytest** | Automated testing framework for unit, integration, and hallucination tests. |

---

## 3. Step-by-Step Local Execution & Verification

### Prerequisites
Make sure dependencies are installed in your Python environment:
```powershell
pip install -r ai_service/requirements.txt
```

---

### Method 1: Run the Automated Pytest Suite (Recommended)
This runs all 14 unit and integration tests across query analysis, hybrid search, RRF fusion, cross-encoder reranking, evaluator contradiction checks, and retry loop fallbacks:

```powershell
python -m pytest ai_service/tests/ -v
```

---

### Method 2: Run Standalone Python Scenarios
You can test all 4 insurance decision scenarios directly via Python without starting a server:

```powershell
python -c "from ai_service.evaluator.retry_controller import ClosedRetryLoopOrchestrator; o = ClosedRetryLoopOrchestrator(); print('1. HOSPITALIZATION:'); print(o.execute_qa('Is in-patient hospitalization covered?', 'policy_health_01').model_dump_json(indent=2)); print('\n2. COSMETIC SURGERY:'); print(o.execute_qa('Can I claim for cosmetic surgery?', 'policy_health_01').model_dump_json(indent=2)); print('\n3. MATERNITY:'); print(o.execute_qa('Is maternity delivery covered?', 'policy_health_01').model_dump_json(indent=2)); print('\n4. UNANSWERABLE TOPIC:'); print(o.execute_qa('Is dental checkup covered?', 'policy_health_01').model_dump_json(indent=2))"
```

---

### Method 3: Start the Live FastAPI Server & Send Queries

#### 1. Start the Server:
```powershell
python -m uvicorn ai_service.main:app --host 127.0.0.1 --port 8000 --reload
```
The interactive Swagger API documentation will be available at:
👉 **`http://127.0.0.1:8000/docs`**

#### 2. Check Server Health:
Open a separate PowerShell window and run:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/ai/health" -Method Get | ConvertTo-Json
```

#### 3. Send a Coverage Query:
```powershell
$body = @{
    policy_id = "policy_health_01"
    question = "Is hospitalization covered under this policy?"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/ai/ask" -Method Post -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
```

---

## 4. Structured Output Contract

Every verified answer adheres to this strict JSON format:

```json
{
  "query": "Is in-patient hospitalization covered?",
  "decision": "COVERED",
  "confidence_score": 1.0,
  "conditions": [
    "Minimum 24 consecutive hours of in-patient hospitalization required.",
    "Room rent is capped at 1% of Sum Insured per day, and ICU capped at 2%."
  ],
  "reason": "In-patient hospitalization expenses for medical treatment are covered for admissions of at least 24 consecutive hours, subject to room rent limits.",
  "citations": [
    {
      "chunk_id": "chunk_h_001",
      "page": 12,
      "clause": "Section 2.1 - In-Patient Hospitalization",
      "category": "Coverage",
      "snippet": "Section 2.1 In-Patient Hospitalization Expenses: The company will indemnify medical expenses incurred for in-patient hospitalization..."
    }
  ],
  "evaluator_status": "VERIFIED_PASS",
  "retry_count": 0
}
```
