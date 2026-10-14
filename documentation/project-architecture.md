# CoverageIQ — Project Architecture

> **Evidence-Grounded Insurance Policy Intelligence System**
> Source of truth for **flow and logic**: the architecture diagram (`CoverageIQ_Evidence-Grounded_System_Architecture.png`). Source of truth for **tech stack**: this document.
> The diagram still shows Next.js, a single REST API and Celery. Those labels are outdated (see Section 2A and Section 5). Re-export the diagram with the new stack labels. If the diagram and this file disagree on anything else, fix both in the same PR.

---

## 1. What CoverageIQ Is

Insurance policy documents are long, dense and full of exclusions. Users cannot easily tell *what is covered*, *what is not*, or *which policy suits them*. CoverageIQ does three things:

| Capability | Plain meaning |
|---|---|
| **Recommendation (6A)** | Given a user's profile and needs, filter and rank policies from the organization catalog. |
| **Summarization (6B)** | Given a policy PDF, produce a grounded summary of the clauses that matter. |
| **Query Retrieval / Q&A (6C)** | Given a question about a policy, answer with a decision (Covered / Not Covered / Conditional / Insufficient Evidence) and cite the page and clause. |

### Core Design Principle

> **NO EVIDENCE = NO CLAIM.**

Every statement the system outputs must be tied to a retrieved chunk (chunk-ID citation). An independent **Evaluator Agent** verifies this before anything reaches the user. If evidence is insufficient after retries, the system says so instead of guessing.

---

## 2. System Layers (Top to Bottom)

```
Actors (User / Organization / Admin)
        │
Authentication + RBAC Gateway
        │
Ingestion Pipeline ──► Knowledge Base (PostgreSQL + pgvector)
                              │
        ┌─────────────────────┼─────────────────────┐
   6A Recommendation     6B Summarization      6C Query Retrieval
        └─────────────────────┼─────────────────────┘
                              │
              Grounded LLM (JSON atomic claims + chunk IDs)
                              │
                 Evaluator Agent (independent verifier)
                    │ FAIL                    │ PASS
              Retry Loop (max 2)       Structured Output
                                              │
                                          Frontend
```

---

## 2A. Two-Service Architecture (Node + Python)

The system is split into two backend services plus a frontend. Everything below this section assumes this split.

| Service | Stack | Owns |
|---|---|---|
| **Frontend** | React + Vite, JavaScript, Tailwind CSS | UI, role-based dashboards, PDF viewer, evidence panels |
| **Node API (Backend)** | Node.js + Express, REST | Auth + RBAC, users/orgs, catalog CRUD, file upload + storage, job records, chat/session history, audit-log reads, orchestration of AI calls. **The only service the browser talks to.** |
| **Python AI Service** | FastAPI + Celery worker | PDF extraction + OCR, structuring, chunking, embeddings, hybrid retrieval + RRF, reranker, hard filtering + ML ranking (6A), summarization (6B), query retrieval (6C), Grounded LLM, Evaluator, retry loop |
| **PostgreSQL + pgvector** | one database | Shared by both services under the table-ownership rules below |

**Rule of thumb:** if it needs a model, an embedding, or an LLM call, it lives in Python. If it is identity, permissions, CRUD, files, or HTTP-facing orchestration, it lives in Node.

### Communication
```
Browser ──HTTPS──► Nginx ──► Node API ──internal HTTP──► Python AI Service
                              │                               │
                              └────────► PostgreSQL ◄─────────┘
                                         (pgvector)
                              Redis (Celery broker, cache)
```
- The browser never calls the Python service directly. Nginx does not expose it.
- Node → Python: internal REST over the docker network, with a shared service token (header `X-Service-Token`) and request ID propagation.
- Long tasks (ingestion): Node creates a job row and calls `POST /ai/ingest`. Python returns `202` and a Celery worker processes the PDF, updating the job row (`queued → processing → done | failed`). The frontend polls Node `GET /jobs/:id`.
- Short tasks (ask, summarize, recommend): synchronous call Node → Python with a timeout (default 45 s); Node returns the structured output to the browser.
- Auth: Node verifies the user JWT and passes a **trusted context** (`user_id`, `role`, `owner_scope`) to Python in the request body or headers. Python never parses user JWTs; it trusts only the service token plus that context. This keeps one place for RBAC.

### Database table ownership (single Postgres, one migration source)

| Owner | Tables (see `databaseDesign.md`) | The other service may |
|---|---|---|
| **Node** | users, organizations, roles, policies (metadata), policy_versions, jobs, chat_sessions, chat_messages, uploads | Python: read policies, update `jobs.status` |
| **Python** | sections, clauses, chunks (text, label, page, embedding, tsvector), policy_attributes, ranking_features, audit_claims, audit_verdicts | Node: read-only (for citations, PDF viewer page lookups, audit views) |

- **One migration tool for the whole schema** (Node side, e.g., `node-pg-migrate`, SQL files). Python never runs migrations; it only reads and writes its own tables via `psycopg` or SQLAlchemy Core.
- Any schema change touches both services, so it needs a PR reviewed by both owners.

### Internal AI Service API (contract v0, to be frozen in Phase 0)

| Endpoint | Purpose | Sync? |
|---|---|---|
| `POST /ai/ingest` `{policy_id, file_path, policy_type, owner_id, visibility}` | Start ingestion | async, `202` |
| `POST /ai/ask` `{policy_id, question, session_context, scope}` | Q&A (6C) → structured output | sync |
| `POST /ai/summarize` `{policy_id, scope}` | Summary (6B) | sync (cached) |
| `POST /ai/recommend` `{profile, scope}` | Recommendation (6A) | sync |
| `GET /ai/health` | Liveness + model readiness | sync |

All responses use the structured output shape in Section 3.10. The OpenAPI file for this contract lives in `/docs/contracts/ai-service.openapi.yaml` and is the single reference for both teams.

---

## 3. Layer-by-Layer Explanation

### 3.1 Actors and RBAC

Three roles, enforced at the gateway on **every** request.

| Role | Can do |
|---|---|
| **User** | Query policies, upload their own policy PDFs (private to them), get recommendations, summaries, Q&A. |
| **Organization** | Everything needed to manage the policy catalog: Create, Read, Update, Delete (CRUD) policies used for recommendations. |
| **Admin** | All User permissions + organization management and authorization (approve / suspend organizations, manage roles). |

**Rules**
- Implemented in the **Node API** (Express middleware).
- Authentication: JWT (access + refresh tokens), password hashing with bcrypt/argon2.
- Authorization: role claim in token, checked by `requireRole(...)` middleware on each route.
- **Data visibility:** organization-managed policies feed the shared recommendation catalog; user-uploaded PDFs stay private to that user (row-level `owner_id` filtering on every query, including vector search; Node passes the `owner_scope` to Python on every AI call).

### 3.2 Ingestion Pipeline

Turns a raw PDF into searchable, citable knowledge. Triggered by (a) a User upload or (b) Organization catalog CRUD.

| Step | What happens | Tooling |
|---|---|---|
| 1. Policy PDF Upload | Accept Car / Health / Life / Term policy PDFs. Validate type, size, virus scan hook. Store the file, create policy + job rows. | **Node**: Express + multer |
| 2. PDF Ingestion + OCR | Extract text with layout. Use OCR only for scanned pages. | **Python**: PyMuPDF (digital), PaddleOCR (scanned) |
| 3. Document Structuring | Detect sections, clauses, tables, and keep **page numbers** for each element. | **Python**: rule-based heading detection + layout blocks |
| 4. Semantic Chunking | Chunk by meaning, not fixed size. Tag each chunk as **Coverage / Exclusions / Conditions / Definitions**. Embed it. | **Python**: BGE-M3 embeddings |

**Output of ingestion:** rows in PostgreSQL (policy, sections, clauses, page references) and vectors + BM25 index (chunk embeddings and `tsvector`).

Ingestion is **asynchronous** — the Node API returns a job ID, the Python Celery worker processes the PDF and updates the job row, the frontend polls Node for status.

### 3.3 Knowledge Base

| Store | Contents | Purpose |
|---|---|---|
| **Relational DB (PostgreSQL)** | Policy metadata, sections, clauses, page references | Hard filtering, citations, CRUD, joins back to the exact page |
| **Vector DB (pgvector)** | Chunk embeddings + BM25 `tsvector` index | Hybrid retrieval (semantic + keyword) |

One database engine for both keeps deployment simple (one container, one backup).

### 3.4 Module 6A — Recommendation

Runs independently with its own entry point.

1. **Input:** user profile + requirements.
2. **Hard Filtering** — rules that remove policies which can never fit: insurance type, budget/premium range, age eligibility, applicant eligibility, individual/family coverage, sum insured, required benefits, pre-existing diseases, waiting periods, exclusions, deductible and co-payment, room-rent limits, hospital network, geographic availability, policy duration, renewal eligibility.
3. **ML Ranking (LightGBM / XGBoost)** — scores surviving policies → **Top N Policies**.
4. Top N goes to the Grounded LLM + Evaluator so the explanation of "why recommended" is itself evidence-backed.
5. Verified result appears in Structured Output as **Recommended Policies (verified)**. The user can then select a policy, which hands that policy to 6B / 6C (the "User selects policy" arrow on the diagram).

### 3.5 Module 6B — Summarization

Two entry points: a **user-uploaded PDF** (ingested on demand) or a **selected policy from 6A** (evaluator-verified recommendation).

1. **Relevant Clause Retrieval** — Vector + BM25 fused with Reciprocal Rank Fusion (RRF).
2. **Grounded LLM Summarization** — summary composed only from retrieved clauses.
3. Goes through the shared Grounded LLM → Evaluator path.

### 3.6 Module 6C — Query Retrieval (Q&A)

1. **Input:** user question + selected policy.
2. **Query Analysis** — detect intent (e.g., "is X covered?") and entities (treatment, condition, amount).
3. **Hybrid Retrieval** — Vector + BM25 over that policy's chunks only.
4. **Cross-Encoder Reranker (Top K Evidence)** — `bge-reranker` re-scores candidates, keeps best K.
5. Shared Grounded LLM → Evaluator path.

### 3.6a Why one policy at a time?
6B and 6C scope retrieval to a single policy ID. This prevents cross-policy contamination, which is the most common source of wrong coverage answers.

### 3.7 Grounded LLM

- Receives: query + retrieved evidence + policy rules.
- Must output **atomic claims in JSON**, each with `chunk_id` citation(s).
- Prompt contract: if no chunk supports a claim, the claim is not produced.
- API-hosted LLM endpoint (no local GPU required).

### 3.8 Evaluator Agent (Independent Verifier)

Separate prompt (ideally a separate model or at minimum a separate call with no shared scratchpad). Checks:

| Check | Question it answers |
|---|---|
| Evidence Support | Does each claim follow from its cited chunk? |
| Contradiction Detection | Do claims contradict each other or other retrieved chunks? |
| Citation Accuracy | Does the cited chunk ID/page actually contain this? |
| Completeness & Hallucination | Are important exclusions missing? Is anything invented? |
| Confidence Scoring | Numeric score used for the final decision and UI display. |

### 3.9 Retry Loop

On **FAIL**, max **2 retries**: re-retrieve (widen k, rewrite query) → re-rank → re-answer using the evaluator's feedback. If still failing: output **INSUFFICIENT EVIDENCE** (the "Retries exhausted" path in the diagram).

### 3.10 Structured Output

Every answer has the same shape:

```json
{
  "decision": "COVERED | NOT_COVERED | CONDITIONAL | INSUFFICIENT_EVIDENCE",
  "confidence": 0.0,
  "conditions": ["..."],
  "reason": "...",
  "citations": [{"chunk_id": "...", "page": 12, "clause": "4.2"}]
}
```

Recommendation responses add a ranked list of **Recommended Policies (verified)**, each with citations.

### 3.11 Frontend

React + Vite (JavaScript) with Tailwind CSS, built as a single-page app: Login, Upload, Recommendations, Summary, Q&A chat, **expandable evidence + PDF viewer jumping to the cited page**, and role-based dashboards (User / Organization / Admin).

---

## 4. Backend and Deployment

| Component | Role |
|---|---|
| **Nginx** | Reverse proxy, TLS, serves the built React app, forwards `/api` to Node, rate limiting. Python service is **not** exposed. |
| **Node API (Express)** | Auth, RBAC, CRUD, uploads, job tracking, orchestration, chat history |
| **Python AI Service (FastAPI)** | Retrieval, 6A/6B/6C pipelines, Grounded LLM, Evaluator |
| **Celery Worker (Python)** | Async ingestion, embedding, long runs (Redis broker) |
| **Redis** | Celery broker + short-lived cache (summaries, rate limits) |
| **PostgreSQL + pgvector** | Relational + vector storage (shared, table-ownership rules in 2A) |
| **API-hosted LLM & Evaluator endpoints** | Generation and verification |
| **Cross-Encoder Reranker (bge-reranker)** | Loaded inside the Python service (CPU) or hosted API |
| **Docker on Cloud VM (AWS/GCP/Azure)** | `docker compose`: nginx, frontend build, node-api, ai-service, ai-worker, redis, postgres |

Request flow: `Browser → Nginx → Node API → Python AI Service → PostgreSQL/pgvector`, with LLM/Evaluator calls going out from Python to hosted endpoints. Ingestion adds `Python AI Service → Celery Worker`.

---

## 5. Technology Decisions (Locked unless noted)

| Area | Decision | Note |
|---|---|---|
| Frontend | **React + Vite, JavaScript, Tailwind CSS** | No SSR needed. Routing: React Router. Server state: TanStack Query. |
| Backend API | **Node.js (LTS) + Express, REST** | Validation: Zod (or Joi). Auth: `jsonwebtoken` + bcrypt. Uploads: multer. Docs: OpenAPI (swagger). |
| AI service | **Python 3.11 + FastAPI** | PyMuPDF, PaddleOCR, BGE-M3, bge-reranker, LightGBM/XGBoost, Pydantic |
| Async jobs | **Celery + Redis** (Python side) | Node only creates job rows and calls Python |
| Database | **PostgreSQL 16 + pgvector** | One DB; table ownership in Section 2A |
| DB access | Node: Prisma or `pg` + query builder (**team to pick**). Python: SQLAlchemy Core / psycopg | One migration source (Node side, plain SQL migrations recommended because of pgvector/tsvector) |
| Embeddings | BGE-M3 | |
| Reranker | bge-reranker | |
| Ranking model | Weighted scorer first, LightGBM/XGBoost later | Needs labeled data; see R1 |
| Auth | JWT + RBAC (Node) | |
| Deployment | Docker Compose on one cloud VM | |

### Consequences of using JavaScript (not TypeScript) on both Node and React
- Nothing checks API shapes at compile time. Mitigation: OpenAPI contracts for both the Node API and the Python AI service, runtime validation (Zod) at Node boundaries, JSDoc type hints on shared shapes, and contract tests in CI.
- Node and Python will each have their own dependency, lint (ESLint / Ruff) and test setup (Jest or Vitest / pytest).

---

## 6. Cross-Cutting Concerns

- **Security:** RBAC at gateway, per-user data isolation in SQL and vector queries, signed URLs for PDFs, no PII in logs.
- **Observability:** log every query with retrieved chunk IDs, evaluator verdicts, retry count, latency, token cost.
- **Idempotency:** re-uploading the same PDF (hash match) must not create duplicate chunks.
- **Versioning:** every policy has a version; chunks reference the version so citations stay valid after updates.
- **Failure behavior:** the system prefers "Insufficient Evidence" over a confident wrong answer.

---

## 7. Known Risks and Open Decisions

| # | Risk / Decision | Why it matters | Proposed handling |
|---|---|---|---|
| R1 | **No training data for ML Ranking (6A)** | LightGBM/XGBoost need labeled examples of "good match". We have none. | Start with a transparent weighted scoring function; generate synthetic profile-to-policy labels; swap in LightGBM once labels exist. |
| R2 | **Evaluator doubles LLM cost and latency** | Each answer = generation + verification (+ up to 2 retries). | Cache, cap retries, use a smaller model for verification. |
| R3 | **Indian policy PDFs are inconsistent** (scanned, multilingual, tables) | Parsing quality drives everything downstream. | Build a gold set of ~10 real policies early; measure parsing before building on top. |
| R4 | **Hard-filter fields need structured data** | "Waiting period", "room-rent limit" etc. live in free text. | Extraction step at ingestion (LLM-assisted) into structured columns. |
| R5 | **Scope** | Three modules + RBAC + evaluator + dashboards is large. | MVP cut defined in `roadmap.md`. |
| D1 | ~~Backend language~~ | **Resolved:** Node/Express API + Python AI service. | Recorded in Section 2A and 5. |
| R6 | **Node↔Python contract drift** | Two teams, two languages, no shared types. A silent field rename breaks Q&A. | Freeze contract v0 in Phase 0, keep OpenAPI in `/docs/contracts`, run contract tests in CI, ship a mock AI service so Node and frontend are never blocked. |
| R7 | **Two services = more ops and latency** | More containers, one extra network hop, harder debugging. | Request ID in every log line, health endpoints, one `docker compose up`, keep the AI service stateless. |
| R8 | **Shared database, two writers** | Schema changes can break the other service. | Table ownership rules (2A), single migration source, both owners review schema PRs. |

---

## 8. Glossary

| Term | Meaning |
|---|---|
| **RAG** | Retrieval-Augmented Generation: LLM answers from retrieved text, not memory |
| **BM25** | Keyword relevance ranking |
| **RRF** | Reciprocal Rank Fusion: merges vector and BM25 rankings |
| **Cross-encoder** | Model that scores (query, chunk) pairs jointly; slower but more accurate |
| **Chunk ID** | Unique ID of a stored text chunk; the unit of citation |
| **Atomic claim** | One single verifiable statement |
| **Hard filtering** | Rule-based elimination before any ML |
