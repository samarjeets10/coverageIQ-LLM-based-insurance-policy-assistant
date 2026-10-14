# CoverageIQ — Modules and Features

> One section per module. Each module lists: **purpose, inputs, outputs, internal components, features, interfaces, dependencies, tests, and a "done when" checklist.**
> Use this file to split work. Requirement IDs (FR-xxx) map to `ProjectRequirements.md`.

---

## 0. Module Map

| ID | Module | Type | Owner (fill in) |
|---|---|---|---|
| M1 | Auth + RBAC Gateway | **Node** | |
| M2 | Ingestion Pipeline | **Node** (upload, jobs) + **Python** (processing) | |
| M3 | Knowledge Base (storage + retrieval layer) | **Python** (retrieval lib) + shared Postgres | |
| M4 | Catalog Management | **Node** + UI | |
| M5 | 6A Recommendation | **Python** (Node proxies) | |
| M6 | 6B Summarization | **Python** (Node proxies + caches) | |
| M7 | 6C Query Retrieval | **Python** (Node proxies + chat history) | |
| M8 | Grounded LLM + Evaluator + Retry Loop | **Python** (shared core) | |
| M9 | Structured Output Layer | **Python** (builds) + **Node** (validates, forwards) | |
| M10 | Frontend (React + Vite + JS + Tailwind) | UI | |
| M11 | Deployment and DevOps | Infra (both services) | |
| M12 | Evaluation and Testing Harness | Quality (Python-run, hits the AI service) | |

### Dependency Order

```
M11 (skeleton) ─► M1 ─► M2 ─► M3 ─► M7 ─► M8 ─► M9 ─► M10
                                 ├─► M6 ──┘
                          M4 ────┴─► M5 ──┘
M12 runs alongside everything (gold set first)
```

**Two codebases, one contract.** `backend-node/` (Express) and `ai-service/` (FastAPI) each have their own `common/` (config, DB, logging, errors; Python adds Pydantic schemas and the LLM client; Node adds Zod schemas). Services talk **only** through the internal REST contract in `ProjectArchitecture.md §2A` (OpenAPI in `/docs/contracts/`). Modules inside a service never import each other's internals.

**Service split:** Node owns identity, permissions, CRUD, files, job rows, chat history and orchestration. Python owns everything that needs a model, an embedding or an LLM call. Table ownership rules are in Architecture §2A.

---

## M1 — Authentication + RBAC Gateway

**Purpose:** Identify who is calling and what they may do.

**Features**
- Register, login, refresh token, logout (FR-AUTH-01..03)
- Role claim (`user`, `organization`, `admin`) in JWT
- Express middleware `requireRole(...)` on every route
- Builds the trusted scope (`user_id`, `role`, `owner_scope`) passed to the Python service; service token `X-Service-Token` on internal calls
- Organization approval flow (FR-AUTH-04)
- Row-level ownership helper: `owner_id` filter used by every data query

**Interfaces**
- `POST /auth/register`, `/auth/login`, `/auth/refresh`, `GET /auth/me`
- `PATCH /admin/organizations/{id}/approve`

**Stack:** Node.js, Express, `jsonwebtoken`, bcrypt/argon2, Zod. **Depends on:** PostgreSQL (users table)

**Tests:** forbidden-access matrix (each role × each endpoint), expired token, tampered token.

**Done when:** a User token cannot call any Organization/Admin route and vice versa, proven by automated tests.

---

## M2 — Ingestion Pipeline

**Purpose:** Convert a PDF into structured, chunked, embedded, citable knowledge.

**Inputs:** PDF file, policy type (Car/Health/Life/Term), owner (user or organization), visibility (private / catalog).
**Outputs:** policy row, sections, clauses, chunks with page numbers, embeddings, BM25 vectors, extracted attributes.

**Internal components**
| Component | Responsibility |
|---|---|
| `uploader` (**Node**, Express + multer) | Validate file, compute hash, store file, create policy + job rows, call `POST /ai/ingest` |
| `extractor` | PyMuPDF text + layout; detect scanned pages |
| `ocr` | PaddleOCR on scanned pages only |
| `structurer` | Sections, clauses, tables, page numbers |
| `chunker` | Semantic chunks tagged Coverage/Exclusions/Conditions/Definitions |
| `attribute_extractor` | Structured fields for hard filtering (premium, sum insured, waiting period, co-pay, room rent, etc.) |
| `embedder` | BGE-M3 vectors, `tsvector` |
| `worker` (**Python**, Celery) | Tasks orchestrating the steps below and updating `jobs.status` |

**Features:** FR-ING-01..12, job status endpoint, duplicate detection, re-ingestion on policy update.

**Interfaces**
- Public (Node): `POST /policies/upload` → `{job_id}`; `GET /jobs/{job_id}` → status + error
- Internal (Python): `POST /ai/ingest` → `202`; Celery task `ingest_policy(policy_id)`
- Everything in the component table except `uploader` runs in the Python service.

**Tests:** gold set parsing check (page-number correctness, table capture), idempotency, corrupt PDF handling.

**Done when:** the 10 gold policies ingest without error and ≥ 95% of sampled chunks map to the correct page.

> **Risk:** this module decides the quality ceiling of every later module. Spend extra time here.

---

## M3 — Knowledge Base and Retrieval Layer

**Purpose:** One clean API over storage so Python modules never write ad-hoc SQL. Lives in the Python service. Node reads chunks/pages read-only (citation lookup, PDF viewer page).

**Components**
- Relational tables (see `databaseDesign.md`, to be written)
- Vector index (pgvector, HNSW) + `tsvector` GIN index
- Retrieval library:
  - `vector_search(policy_id, query_embedding, k)`
  - `bm25_search(policy_id, query_text, k)`
  - `hybrid_search(policy_id, query, k)` → **RRF fusion**
  - `rerank(query, candidates, top_k)` → cross-encoder
  - `get_chunk(chunk_id)` with page and clause for citations

**Rules:** every function takes a scope (`policy_id`, `owner_id`); no unscoped searches.

**Tests:** recall@k on gold Q&A, isolation test (user A cannot retrieve user B's chunks).

**Done when:** hybrid + rerank beats vector-only on the gold set (measured and recorded).

---

## M4 — Catalog Management

**Purpose:** Let organizations manage the policies used for recommendations.

**Features:** FR-CAT-01..04; create (upload + metadata), list, update (re-ingest, new version), delete (soft delete, remove from search), admin deactivate.

**Interfaces (Node):** `POST/GET/PUT/DELETE /org/policies`, `GET /admin/policies`. Update triggers re-ingestion via the Python service.

**Done when:** an organization can only see and edit its own policies; deleted policies disappear from recommendations immediately.

---

## M5 — 6A Recommendation

**Purpose:** Turn a user profile into a verified Top-N list.

**Pipeline**
1. **Profile intake** — validated schema (type, budget, age, applicants, sum insured, required benefits, pre-existing conditions, location, duration).
2. **Hard filter** — SQL/rule engine over structured attributes: waiting periods, exclusions, deductible/co-pay, room-rent limits, hospital network, geographic availability, renewal eligibility, etc. Return elimination reasons.
3. **Ranking** — Phase A: transparent weighted score. Phase B: LightGBM/XGBoost once labels exist.
4. **Explanation** — retrieve supporting chunks for each Top-N policy; M8 generates and verifies "why recommended".
5. **Output** — Recommended Policies (verified).

**Features:** FR-REC-01..07

**Interfaces:** Public (Node, auth + proxy): `POST /recommendations`. Internal (Python): `POST /ai/recommend` → ranked list with citations and filter report. Hard filtering and ranking both run in Python so they share the policy attribute data.

**Tests:** filter correctness (unit tests per rule), ranking sanity (known profile → expected top policy), "no match" explanation.

**Done when:** for 10 hand-built profiles, the top-3 is judged reasonable by the team and each explanation passes the evaluator.

---

## M6 — 6B Summarization

**Purpose:** Evidence-backed policy summary.

**Pipeline:** policy (uploaded or selected) → targeted retrieval per summary section (coverage, exclusions, conditions, waiting periods, limits, claim process) using hybrid + RRF → M8 grounded summarization → verified summary.

**Features:** FR-SUM-01..04; section-wise summary; "Not found in document" handling; cached summary per policy version.

**Interfaces:** Public (Node): `POST /policies/:id/summary` (checks access, caches). Internal (Python): `POST /ai/summarize`.

**Tests:** every sentence has a citation; spot-check 20 summary statements against the PDF.

**Done when:** summaries for all gold policies pass evaluator with zero uncited statements.

---

## M7 — 6C Query Retrieval (Q&A)

**Purpose:** Answer a question about one policy with a decision and evidence.

**Pipeline:** question → **query analysis** (intent + entities) → hybrid retrieval → cross-encoder rerank (Top K) → M8.

**Features:** FR-QA-01..07; session chat history; follow-up resolution (P2).

**Interfaces:** Public (Node): `POST /policies/:id/ask` (checks access, stores chat message, forwards). Internal (Python): `POST /ai/ask` → structured answer (see M9). Node persists the answer in `chat_messages`.

**Tests:** 100-question gold set: decision accuracy, citation accuracy, INSUFFICIENT_EVIDENCE on unanswerable questions.

**Done when:** meets NFR-01..04 on the gold set.

---

## M8 — Grounded LLM + Evaluator Agent + Retry Loop (shared core)

**Purpose:** The trust engine. Every module routes through it.

**Components**
| Component | Detail |
|---|---|
| `LLMClient` | Provider-agnostic wrapper, timeouts, cost tracking |
| `GroundedGenerator` | Prompt = query + evidence + policy rules. Output: atomic claims JSON with `chunk_id`s. Rule: **NO EVIDENCE = NO CLAIM** |
| `Evaluator` | Independent prompt/model: evidence support, contradiction, citation accuracy, completeness/hallucination, confidence |
| `RetryController` | Max 2 retries: widen k → rewrite query → re-rank → re-answer with evaluator feedback; then INSUFFICIENT_EVIDENCE |
| `AuditLog` | Stores every claim, verdict, retries |

**Contract (generator output)**
```json
{"claims":[{"text":"...", "chunk_ids":["c_101"], "type":"coverage|exclusion|condition"}]}
```
**Contract (evaluator output)**
```json
{"verdict":"PASS|FAIL", "confidence":0.0, "issues":[{"claim_idx":0,"type":"unsupported|contradiction|bad_citation|missing_exclusion"}]}
```

**Tests:** adversarial set (questions with no answer, contradictory clauses, misleading phrasing), schema validation, retry-limit enforcement.

**Done when:** on the adversarial set, hallucinated claims reaching output ≤ 5%.

---

## M9 — Structured Output Layer

**Purpose:** Convert verified claims into the final response.

**Output states:** COVERED, NOT_COVERED, CONDITIONAL, INSUFFICIENT_EVIDENCE, plus Recommended Policies (verified).

**Fields:** decision, confidence, conditions, reason, citations (chunk_id, page, clause).

**Rules:** decision is derived from verified claims only; a claim set mixing coverage and exclusion yields CONDITIONAL or NOT_COVERED per defined precedence (document in code comments and tests).

**Done when:** schema is frozen as OpenAPI, Python emits it (Pydantic), Node validates it (Zod) before forwarding, and the frontend consumes the same shape.

---

## M10 — Frontend (React + Vite, JavaScript, Tailwind CSS)

**Stack:** React + Vite, JavaScript, Tailwind CSS, React Router, TanStack Query for server state, PDF viewer via `react-pdf`/pdf.js. Talks only to the Node API (base path `/api`).

**Pages**
| Page | Role | Features |
|---|---|---|
| Login / Register | all | FR-UI-01 |
| User Dashboard | User | uploads, history |
| Upload | User | progress, job status (FR-UI-02) |
| Recommendations | User | form, ranked results (FR-UI-03) |
| Summary | User | cited summary (FR-UI-04) |
| Q&A Chat | User | decision badge, expandable evidence (FR-UI-05/06) |
| PDF Viewer | User | jump to cited page, highlight (FR-UI-07) |
| Org Dashboard | Organization | catalog CRUD (FR-UI-08) |
| Admin Dashboard | Admin | approvals, users (FR-UI-09) |

**Frontend conventions**
- React + Vite single-page app (client-side routing with React Router; no SSR).
- Plain JavaScript. To avoid API drift without TypeScript: JSDoc typedefs for every API response, one `api/` folder holding all fetch calls, and a shared response shape copied from the OpenAPI schema.
- Tailwind CSS for styling; shared components (Button, Badge, Card, Modal) live in `components/ui/`.
- Auth: JWT stored in memory plus refresh token; protected routes by role.
- Suggested libraries: React Router, TanStack Query (server state), react-pdf (PDF viewer at cited page).

**Decision badge colors:** Covered = green, Not Covered = red, Conditional = amber, Insufficient Evidence = grey (matches diagram).

**Done when:** a new user completes upload → ask → view evidence without help.

---

## M11 — Deployment and DevOps

**Features:** Dockerfiles per service; `docker-compose.yml` (nginx, node-api, ai-service, ai-worker, redis, postgres+pgvector, frontend build); `.env.example`; GitHub Actions (lint + tests); one-command start; deployment to a cloud VM with TLS.

**Done when:** a fresh machine runs the whole stack via `docker compose up`, Python is not reachable from outside the docker network, and CI lints/tests both services (ESLint + Jest/Vitest, Ruff + pytest).

---

## M12 — Evaluation and Testing Harness

**Purpose:** Prove it works; guard against regressions.

**Contents:** gold policy set, 100 Q&A pairs, adversarial questions, scripts reporting retrieval recall@k, decision accuracy, citation accuracy, hallucination rate, latency and cost per query. Run on every major change; results saved in `/docs/eval-results/`.

**Done when:** one command prints the metrics table.

---

## Feature Priority Summary (MVP Cut)

| Priority | Included |
|---|---|
| **P0 (MVP)** | M1, M2 (digital PDFs), M3, M4 (basic CRUD), M6, M7, M8, M9, M10 core pages, M11, M12 |
| **P1** | OCR, query analysis, PDF highlight viewer, admin dashboard, duplicate detection |
| **P2** | Policy comparison, follow-up resolution, multilingual |
| **M5 note** | Ship with weighted scoring first; ML model only if labels are ready |

## Suggested Team Split (edit as needed)

Split by service so each person has a clear boundary. The Node↔Python contract is the handshake.

| Member | Primary | Secondary |
|---|---|---|
| A (Python, ingestion + retrieval) | M2 (Python side), M3 | M12 |
| B (Python, LLM core) | M8, M7, M9 (build) | M12 |
| C (Python + Node bridge) | M5, M6, M4 (Node) | Contract tests |
| D (Node + Frontend) | M1, M2 (Node side), M10 | M11 |

If one person owns Node entirely and two own Python, move M4 to the Node owner. Everyone co-owns M12 (gold set creation) in Phase 1. Every cross-service change is reviewed by one person from each side.

### Cross-service rules
- Contract first: change `/docs/contracts/*.openapi.yaml`, then both implementations, in one PR (or linked PRs).
- A **mock AI service** (returns fixed valid JSON for every `/ai/*` endpoint) is built in Phase 0 so Node and frontend work never waits on ML.
- Every internal call logs `request_id`; one request can be traced across Node and Python logs.
