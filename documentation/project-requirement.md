# CoverageIQ — Project Requirements

> Defines **what** the system must do and how well. Architecture says *how*; this says *what and how good*.
> Priority key: **P0** = MVP must-have, **P1** = should-have, **P2** = nice-to-have.
> Every requirement has an ID. Reference IDs in PRs and issues (e.g., "implements FR-ING-03").

---

## 1. Problem Statement

Users cannot reliably understand insurance policy documents. They need verifiable answers about coverage, exclusions and suitability, with proof pointing to the exact page and clause.

## 2. Goals and Non-Goals

**Goals**
- Recommend suitable policies from an organization-managed catalog.
- Summarize a policy faithfully.
- Answer coverage questions with a decision and a citation.
- Never state a coverage claim without evidence.

**Non-Goals (v1)**
- Not a legal or financial advice product; no policy purchase or payment.
- No claims processing.
- No real-time insurer API integrations.
- No mobile app (responsive web only).
- Languages: English only in v1 (multilingual is P2).

## 3. Users and Roles

| Role | Description | Key needs |
|---|---|---|
| User | Person comparing or understanding policies | Upload PDF, ask questions, get recommendations |
| Organization | Insurer / agent managing a catalog | Add, edit, delete policies; see usage |
| Admin | Platform owner | Manage organizations and roles |

---

## 4. Functional Requirements

### 4.1 Authentication and RBAC (AUTH)
| ID | Requirement | Pri |
|---|---|---|
| FR-AUTH-01 | Register and log in with email + password; passwords hashed (argon2/bcrypt) | P0 |
| FR-AUTH-02 | JWT access + refresh tokens | P0 |
| FR-AUTH-03 | Three roles (User, Organization, Admin) enforced on every Node endpoint; Node passes a trusted scope (user_id, role, owner_scope) to the Python service | P0 |
| FR-AUTH-04 | Admin approves Organization accounts before they can publish policies | P1 |
| FR-AUTH-05 | Password reset by email | P2 |

### 4.2 Ingestion (ING)
| ID | Requirement | Pri |
|---|---|---|
| FR-ING-01 | Accept PDF upload for Car, Health, Life, Term policy types (handled by the Node API, processed by the Python AI service) | P0 |
| FR-ING-02 | Reject non-PDF, oversized (configurable, default 25 MB) and corrupt files with clear errors | P0 |
| FR-ING-03 | Extract text from digital PDFs (PyMuPDF) | P0 |
| FR-ING-04 | OCR scanned pages (PaddleOCR) | P1 |
| FR-ING-05 | Preserve page number for every extracted block | P0 |
| FR-ING-06 | Detect sections, clauses and tables | P0 |
| FR-ING-07 | Semantic chunking with labels: Coverage / Exclusions / Conditions / Definitions | P0 |
| FR-ING-08 | Generate BGE-M3 embeddings and BM25 `tsvector` per chunk | P0 |
| FR-ING-09 | Run asynchronously; show job status (queued, processing, done, failed) | P0 |
| FR-ING-10 | Detect duplicate PDFs by file hash | P1 |
| FR-ING-11 | Extract structured policy attributes (premium, sum insured, waiting period, room-rent limit, co-pay, etc.) for filtering | P0 (org catalog) |
| FR-ING-12 | User-uploaded PDFs are private to that user and excluded from the shared catalog | P0 |

### 4.3 Catalog Management (CAT)
| ID | Requirement | Pri |
|---|---|---|
| FR-CAT-01 | Organization can create, read, update, delete its policies | P0 |
| FR-CAT-02 | Updating a policy PDF re-runs ingestion and bumps its version | P1 |
| FR-CAT-03 | Organization can only modify its own policies | P0 |
| FR-CAT-04 | Admin can view and deactivate any policy | P1 |

### 4.4 Recommendation (REC, Module 6A)
| ID | Requirement | Pri |
|---|---|---|
| FR-REC-01 | Collect profile and requirements: insurance type, budget, age, applicant/family, sum insured, required benefits, pre-existing diseases, location, duration | P0 |
| FR-REC-02 | Apply hard filters (the full list in Architecture §3.4) before ranking | P0 |
| FR-REC-03 | Rank remaining policies and return Top N (default 5) | P0 |
| FR-REC-04 | Each recommendation has an evidence-backed explanation with citations | P0 |
| FR-REC-05 | If zero policies pass filters, explain which filter eliminated them | P1 |
| FR-REC-06 | User can select a recommended policy to open summary or Q&A | P0 |
| FR-REC-07 | Side-by-side comparison of Top N | P2 |

### 4.5 Summarization (SUM, Module 6B)
| ID | Requirement | Pri |
|---|---|---|
| FR-SUM-01 | Summarize an uploaded or selected policy | P0 |
| FR-SUM-02 | Summary sections: coverage, exclusions, conditions, waiting periods, limits, claim process | P0 |
| FR-SUM-03 | Every summary statement carries a citation (page and clause) | P0 |
| FR-SUM-04 | Show "Not found in document" for sections with no evidence rather than inventing | P0 |

### 4.6 Query Retrieval / Q&A (QA, Module 6C)
| ID | Requirement | Pri |
|---|---|---|
| FR-QA-01 | Free-text question against one selected policy | P0 |
| FR-QA-02 | Query analysis extracts intent and entities | P1 |
| FR-QA-03 | Hybrid retrieval (vector + BM25) followed by cross-encoder reranking | P0 |
| FR-QA-04 | Answer returns decision: COVERED / NOT_COVERED / CONDITIONAL / INSUFFICIENT_EVIDENCE | P0 |
| FR-QA-05 | Answer includes confidence, conditions, reason, page/clause citations | P0 |
| FR-QA-06 | Chat history within a session | P1 |
| FR-QA-07 | Follow-up questions resolve references ("what about for my wife?") | P2 |

### 4.7 Grounding and Verification (GRD)
| ID | Requirement | Pri |
|---|---|---|
| FR-GRD-01 | LLM outputs atomic claims as JSON with chunk-ID citations | P0 |
| FR-GRD-02 | Claim with no supporting chunk must not be emitted | P0 |
| FR-GRD-03 | Evaluator checks: evidence support, contradictions, citation accuracy, completeness/hallucination, confidence | P0 |
| FR-GRD-04 | On FAIL, retry up to 2 times (widen k, rewrite query, re-rank, re-answer with feedback) | P0 |
| FR-GRD-05 | After retries exhausted, return INSUFFICIENT_EVIDENCE | P0 |
| FR-GRD-06 | Persist every evaluator verdict for audit | P1 |

### 4.8 Frontend (UI)
| ID | Requirement | Pri |
|---|---|---|
| FR-UI-01 | Login, registration, role-based redirect to dashboards | P0 |
| FR-UI-02 | Upload screen with progress and processing status | P0 |
| FR-UI-03 | Recommendation form and results | P0 |
| FR-UI-04 | Summary view | P0 |
| FR-UI-05 | Q&A chat with decision badge (color-coded) | P0 |
| FR-UI-06 | Expandable evidence panel under every answer | P0 |
| FR-UI-07 | PDF viewer that opens at the cited page and highlights the clause | P1 |
| FR-UI-08 | Organization dashboard for catalog CRUD | P0 |
| FR-UI-09 | Admin dashboard for org approval and user management | P1 |

---

## 5. Non-Functional Requirements

| ID | Category | Requirement | Target |
|---|---|---|---|
| NFR-01 | Accuracy | Citation accuracy (cited chunk really supports claim) | ≥ 90% on gold set |
| NFR-02 | Accuracy | Hallucination rate (unsupported claims reaching user) | ≤ 5% on gold set |
| NFR-03 | Accuracy | Decision accuracy on Q&A gold set | ≥ 80% |
| NFR-04 | Performance | Q&A answer latency (including evaluator, no retry) | ≤ 15 s p95 |
| NFR-05 | Performance | Ingestion of a 50-page digital PDF | ≤ 2 min |
| NFR-06 | Security | Passwords hashed, JWT expiry, HTTPS only, per-user data isolation | Mandatory |
| NFR-07 | Privacy | User PDFs not used for training or shared; deletable on request | Mandatory |
| NFR-08 | Reliability | Failed ingestion jobs retry and surface a readable error | Mandatory |
| NFR-09 | Scalability | Support ~50 concurrent users on one VM (demo scale) | Target |
| NFR-10 | Maintainability | Each module runs independently with its own entry point; shared code in `common/` (per service). Node and Python services are independently buildable and testable | Mandatory |
| NFR-11 | Observability | Structured logs with request ID, retrieved chunk IDs, retries, token cost | Mandatory |
| NFR-12 | Usability | Core flow (upload → ask → see evidence) in ≤ 3 clicks | Target |
| NFR-13 | Cost | Per-query LLM cost tracked; hard cap on retries | Mandatory |
| NFR-14 | Integration | Node↔Python and Browser↔Node contracts defined in OpenAPI, validated at runtime (Zod in Node, Pydantic in Python), with contract tests in CI | Mandatory |
| NFR-15 | Integration | Node→Python internal calls carry a service token and request ID; Python is unreachable from the public internet | Mandatory |
| NFR-16 | Resilience | If the AI service is down or times out, Node returns a clear error (not a hang); ingestion jobs resume or fail visibly | Mandatory |

> Targets are proposals. Adjust after the first gold-set measurement and record changes in the Decision Log.

---

## 6. Data Requirements

- **Gold evaluation set:** minimum 10 real policies (mix of Car, Health, Life, Term) and 100 hand-written Q&A pairs with correct decision and citation. Created in Phase 1, used for all later measurement.
- **Ranking data:** profile-to-policy relevance labels (synthetic at first; see risk R1 in Architecture).
- **Policy sources:** publicly available insurer brochures/policy wordings only. No private customer data.
- **Retention:** user uploads deletable by the user; ingestion artifacts deleted with them.

---

## 7. External Dependencies

| Dependency | Used for | Fallback |
|---|---|---|
| Hosted LLM API | Generation + evaluation | Switch provider via config |
| BGE-M3 / bge-reranker (HuggingFace) | Embeddings, reranking | Hosted embedding API |
| PaddleOCR | Scanned PDFs | Tesseract |
| Cloud VM | Deployment | Local docker-compose for demo |
| Node.js (LTS) + Express | Public REST API, auth, CRUD, orchestration | None (core) |
| Python 3.11 + FastAPI | AI service | None (core) |
| Redis | Celery broker, cache | None (core) |

---

## 8. Assumptions and Constraints

- Team of 4; academic deadline (confirm date; roadmap assumes ~16 weeks).
- Limited budget: API credits and one small VM.
- No GPU: embeddings and reranker run on CPU or via hosted API.
- Policies are in English.
- **Tech stack (locked):** React + Vite (JavaScript) + Tailwind CSS frontend; Node.js + Express REST backend; Python (FastAPI + Celery) AI service; PostgreSQL + pgvector. No TypeScript, so contracts are enforced through OpenAPI and runtime validation.
- Team members are split across the Node/Python/frontend boundary; the contract between them must be frozen early (see roadmap Phase 0).

---

## 9. Acceptance Criteria (Definition of Done for MVP)

1. A User can upload a health policy PDF, wait for processing, and see a summary where every statement is cited.
2. A User can ask 5 different coverage questions and receive decisions with page/clause citations; at least 4 of 5 match the gold answers.
3. A question with no support in the document returns INSUFFICIENT_EVIDENCE, not a guess.
4. An Organization can add a policy; a User profile run returns ranked recommendations from it.
5. An Admin can log in and see role-restricted features; a User cannot access Organization/Admin endpoints (verified by tests).
6. The full stack starts with one `docker compose up`.
7. Gold-set metrics meet NFR-01 to NFR-03.

---

## 10. Out-of-Scope Ideas (Parking Lot)

Multilingual support, mobile app, premium calculator, insurer API integrations, claim-assistance flows, voice Q&A.
