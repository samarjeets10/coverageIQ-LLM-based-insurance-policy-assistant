# CoverageIQ — Complete Feature Connectivity & Real Data Flow Audit

- **Project:** CoverageIQ (Evidence-Grounded Insurance Policy Assistant)
- **Branch:** `prithviraj_features` (`HEAD`: `242d52899e44c84df70e1ab1d4b21fbb19e3de8e`)
- **Audit Standard:** `NO EVIDENCE = NO CLAIM` (verified directly against source code, running services, PostgreSQL 16 + pgvector, and live end-to-end execution; untested items are explicitly marked)

---

## A. Executive Summary — Why CoverageIQ Feels Disconnected

End-to-end inspection from the React UI down to PostgreSQL 16 and FastAPI proves a **clean vertical split** across the system:

1. **Authentication, Policy Creation, PDF Upload, Celery + BGE-M3 Ingestion, and Job Tracking work 100% end to end on real PostgreSQL 16 + pgvector + Redis 7.** Uploading a PDF through React/Express stores the binary on disk, creates persisted `policy_documents` and `ingestion_jobs` rows, processes pages via Celery (`PyMuPDFLoader`, `PaddleOCREngine`, `DocumentStructurer`, `SemanticChunker`), computes `1024`-dimensional `BAAI/bge-m3` embeddings on CUDA, writes `policy_chunks` + `chunk_embeddings` + `search_tsv` atomically, updates `policies.active_version_id`, and transitions the job to `COMPLETED` (`100%`).
2. **Every downstream user workflow beyond ingestion stops at the UI and/or Express Gateway boundary:**
   - **Coverage Q&A (M7/M8/M9)** is **fully implemented and operational in FastAPI (`POST /ai/ask`)** against live PostgreSQL 16 + pgvector + NVIDIA Nemotron (`nvidia/nemotron-3.5-lightning-30b-a3b`), as proven directly in Section D/I below. However, **Express exposes zero Q&A endpoints**, **PostgreSQL has no `chat_messages` table**, `client/src/lib/api.js` has no `qaApi`, and `UserDashboardPage.jsx` renders a disabled placeholder.
   - **Policy Recommendations (M5)** is **not implemented on `prithviraj_features`**. Teammate work on remote `origin/shravan/policy-recommendation` is unmerged because its schema (`003_recommendation_schema.sql`, `policy_versions`, `policies.owner_user_id`/`name`, SQLAlchemy) is incompatible with `node-pg-migrate`'s schema (`policy_documents`, `policies.owner_scope`/`title`), `ai_service/workers/pipeline.py` does not extract structured `policy_attributes` during PDF ingestion, Express has no `/api/v1/recommendations` route, and `UserDashboardPage.jsx` renders a static text card.
   - **Policy Summarization (M6)** is **not implemented** in React (`disabled` button), Express, FastAPI (`/ai/summarize` returns `HTTP 404`), or PostgreSQL.
   - **Policy Details / Versions (M4) & Org/Admin Management (M1)** are **partially backed in DB** (`policy_documents`, `organizations`, `organization_members`, `audit_logs` tables exist; `audit_logs` records uploads), **but lack Express read/update endpoints** (`GET /api/v1/policies/:policyId`, `GET /api/v1/policies/:policyId/documents`, `GET /api/v1/users`, `GET/POST /api/v1/organizations`, `GET /api/v1/admin/audit-logs`), forcing UI tabs in `OrganisationDashboardPage` and `AdminDashboardPage` to display placeholder notices.

Consequently, uploading a policy indexes its clauses in PostgreSQL, yet users in the browser see empty Q&A, Summarization, and Recommendation tabs because the **Express proxy routes + frontend interactive screens connecting PostgreSQL/FastAPI to those tabs have not been wired yet.**

---

## B. Feature Connectivity Matrix

Only features present in the repository (`client/`, `server/`, `ai_service/`) are listed:

| Feature | Frontend (`client/src/`) | API Client (`lib/api.js`) | Express Gateway (`server/src/`) | FastAPI / AI Service (`ai_service/`) | Database Source (`PostgreSQL 16`) | Real Data Verified? | Overall Status |
|---|---|---|---|---|---|---|---|
| **Authentication & Session Refresh** | `AuthContext.jsx`, `LoginPage.jsx`, `RegisterPage.jsx`, `ProtectedRoute.jsx`, `PublicOnlyRoute.jsx` | `authApi.*`, `tryRefreshAccessToken()` | `routes/auth.routes.js` (`/api/v1/users/*`), `user.controller.js`, `user.model.js` | N/A | `users`, `refresh_tokens` | **Yes** (4 users, 9 refresh tokens) | **Fully integrated** |
| **Policy Creation & Listing** | `PolicyUploadPanel.jsx`, `useWorkspaceData.js`, `UserDashboardPage`, `OrganisationDashboardPage`, `AdminDashboardPage` | `policiesApi.list()`, `policiesApi.create()` | `GET/POST /api/v1/policies` (`ingestion.routes.js`, `ingestionGateway.service.js`) | `PolicyKnowledgeRepository` (reads on `/ai/ask`) | `policies` | **Yes** (3 policies in live DB) | **Partially integrated** *(missing `GET /:id`, `PATCH/DELETE`, and UI `visibility='catalog'` toggle)* |
| **PDF Upload & Validation** | `PolicyUploadPanel.jsx`, `lib/pdfValidation.js` | `ingestionApi.uploadDocument()` (XHR FormData + progress) | `POST /api/v1/policies/:policyId/documents`, `pdfValidation.service.js`, `documentStorage.service.js` | `POST /internal/ingestion/enqueue`, `routes_ingestion.py` | `policy_documents`, `ingestion_jobs`, `audit_logs` | **Yes** (5 docs, disk PDFs verified) | **Fully integrated** |
| **Async Ingestion & Job Tracking** | `IngestionJobsPanel.jsx` (controlled polling, filters, pagination) | `ingestionApi.listJobs()`, `ingestionApi.getJob()` | `GET /api/v1/ingestion/jobs`, `GET /api/v1/ingestion/jobs/:jobId`, startup reconciler | Celery `ingest_policy_document`, `pipeline.py`, `PyMuPDFLoader`, `PaddleOCREngine`, `DocumentStructurer`, `SemanticChunker`, `BgeM3Embedder` | `ingestion_jobs`, `policy_documents`, `policy_chunks`, `chunk_embeddings`, `policies.active_version_id` | **Yes** (5 completed jobs, 318 chunks, 318 x 1024-dim vectors) | **Fully integrated** |
| **Policy Details & Version History** | `OrganisationDashboardPage.jsx` (`VersionsTab` shows only top-level `policies` list) | **None** (`GET /policies/:id` / `.../documents` absent) | **Missing** (`GET /policies/:id` returns `HTTP 404`) | `get_document()`, `set_active_document_version()` exist in `repositories.py` | `policy_documents` (stores all versions & SHA-256) | Backend persists versions; **no API/UI read path** | **Partially integrated** |
| **Coverage Q&A (Decision + Citations)** | `UserDashboardPage.jsx` (`QATab` disabled placeholder) | **None** | **Missing** (`POST /api/v1/policies/:id/ask` returns `HTTP 404`) | **Implemented & live:** `POST /ai/ask` (`routes_qa.py`, `ClosedRetryLoopOrchestrator`, `PersistentHybridRetriever`, `EvaluatorAgent`, `DeterministicDecisionGate`, `NvidiaNemotronProvider`) | Queries `policies`, `policy_documents`, `policy_chunks`, `chunk_embeddings` (no `chat_messages` table yet) | **Yes via FastAPI `/ai/ask`**; **No UI/Express route** | **Partially integrated (AI ready; Express & React disconnected)** |
| **Policy Recommendations** | `UserDashboardPage.jsx` (`RecommendationsTab` static card); `LandingPage.jsx` card | **None** | **Missing** (`POST /api/v1/recommendations` returns `HTTP 404`) | **Not on branch** (`POST /ai/recommend` -> `404`; unmerged on `origin/shravan/policy-recommendation` with schema mismatch; `attribute_extractor` not in `pipeline.py`) | `policy_attributes` / `ranking_features` **do not exist** in schema | **No** | **Frontend placeholder / Not integrated on branch** |
| **Policy Summarization** | `UserDashboardPage.jsx` (`SummariesTab` disabled button); `LandingPage.jsx` card | **None** | **Missing** (`POST /api/v1/policies/:id/summary` -> `HTTP 404`) | **Not implemented** (`POST /ai/summarize` -> `HTTP 404`) | No `policy_summaries` table | **No** | **Frontend placeholder / Not implemented** |
| **Workspace Analytics & Dashboard KPIs** | `UserDashboardPage`, `OrganisationDashboardPage`, `AdminDashboardPage` (`OverviewTab`, `KpiCard`, `DonutSummary`, `HorizontalBarChart`) | Reuses `policiesApi.list()` + `ingestionApi.listJobs({ limit: 20 })` | Uses existing `/policies` + `/ingestion/jobs` | N/A | `policies`, `ingestion_jobs` | **Yes for Policy & Job KPIs/charts**; Q&A/Summary KPIs hardcoded `'0'`; ignores pagination `total_items` when `> 20` | **Partially integrated** |
| **Organisation Teams & Catalog Publishing** | `OrganisationDashboardPage.jsx` (`MembersTab` static card; `CatalogTab` reuses `WorkspaceIngestionTab`) | **None** for orgs/members | **Missing** `organizations` CRUD endpoints (`createAuthorizedPolicy` checks `organization_members` if `organization_id` passed, but no route creates orgs) | Scope filter accepts `org:<id>` via `AuthorizedScope` | `organizations`, `organization_members` exist (`0` rows) | **No** | **Frontend placeholder / Backend CRUD missing** |
| **Admin Governance & Audit Logs** | `AdminDashboardPage.jsx` (`UsersTab`, `OrganisationsTab`, `AuditTab` static notices) | **None** | `PATCH /api/v1/users/:userId/role` exists only; **missing** `GET /users`, `GET /organizations`, `GET /audit-logs`; `user.controller.js` doesn't insert auth events into `audit_logs` | N/A | `users`, `audit_logs` (`5` rows written only by `uploadPolicyDocument`) | Partial (`audit_logs` populated on upload; unread) | **Frontend placeholder / Backend read APIs missing** |

---

## C. Placeholder Inventory

| File Path | Component / Function | Type | Current Behavior | Expected Real Behavior |
|---|---|---|---|---|
| [UserDashboardPage.jsx](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/client/src/pages/UserDashboardPage.jsx#L74-L115) | `OverviewTab` KPIs (`Q&A Queries`, `Summaries Generated`) | Hardcoded KPI strings (`'0'`) | Always displays `'0'` even if backend Q&A exists later | Fetch persisted Q&A count and summary count per user |
| [UserDashboardPage.jsx](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/client/src/pages/UserDashboardPage.jsx#L184-L209) | `QATab` | UI placeholder + `disabled` `<Button>` | Non-interactive divs mimicking policy selector & question input | Interactive policy selector (`active` ingested policies), question submit calling Express `POST /api/v1/policies/:id/ask`, displaying decision badge, conditions, and page/clause citations |
| [UserDashboardPage.jsx](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/client/src/pages/UserDashboardPage.jsx#L211-L231) | `SummariesTab` | UI placeholder + `disabled` `<Button>` | Disabled `"Generate Summary"` button, no API call | Select policy -> call `POST /api/v1/policies/:id/summary` -> render cited structured summary or honest `"Not implemented"` notice |
| [UserDashboardPage.jsx](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/client/src/pages/UserDashboardPage.jsx#L233-L253) | `RecommendationsTab` | Static informational card | Displays static description only; zero profile inputs, zero API call | Profile intake form -> `POST /api/v1/recommendations` -> ranked catalog policies with filter elimination report and verified chunk citations |
| [OrganisationDashboardPage.jsx](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/client/src/pages/OrganisationDashboardPage.jsx#L231-L241) | `MembersTab` | Static card | Displays signed-in user name/email and states invite APIs are planned | List `organization_members` (`GET /api/v1/organizations/:id/members`) and invite/manage members |
| [OrganisationDashboardPage.jsx](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/client/src/pages/OrganisationDashboardPage.jsx#L156-L229) | `VersionsTab` | Partial real table (policies, not versions) | Lists top-level `policies` rows (`pol.id`, `status`) rather than `policy_documents` version rows | Call `GET /api/v1/policies/:policyId/documents` to show `version_number`, `file_name`, `page_count`, `ingestion_status`, SHA-256, and activate/archive actions |
| [AdminDashboardPage.jsx](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/client/src/pages/AdminDashboardPage.jsx#L116-L189) | `PlaceholderGovernanceTab` (`users`, `organisations`, `audit` tabs) | Explicit placeholder cards | States directory/audit endpoints are not connected | Connected tables backed by `GET /api/v1/admin/users` (+ wiring `PATCH /:userId/role`), `GET /api/v1/organizations`, `GET /api/v1/admin/audit-logs` |
| [LandingPage.jsx](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/client/src/pages/LandingPage.jsx#L204-L320) | Feature bento cards (`Card 1..4`) | Static marketing illustrations | Shows fixed example (`"Is day-care surgery covered? COVERED"`, `"Shield Health Plus 94% Match"`) on public `/` landing page | Acceptable on marketing landing page; clearly label as illustrative product preview |
| [ai_service/database.py](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/ai_service/database.py#L9-L121) | `MockKnowledgeBase` | In-memory fixture corpus | Contains hardcoded `policy_health_01` clauses; **only active if `USE_MOCK_STORAGE=true`** (default in [config.py](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/ai_service/config.py#L103) and [ai_service/.env.example](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/ai_service/.env.example#L15)) | Fail closed in production/non-test runs unless `USE_MOCK_STORAGE=true` is explicitly intended for unit tests |
| [ai_service/llm/client.py](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/ai_service/llm/client.py#L168-L188) | `LLMClient.generate_json()` (`gemini` branch) | Dev heuristic fallback (`_offline_grounded_synthesis`) | In `gemini` mode only, errors log & fall back to `_offline_grounded_synthesis`. (`nvidia` mode used in live server does **not** fall back.) | Raise explicit provider error in `gemini` mode too instead of silently falling back |

---

## D. End-to-End Upload-to-Downstream Verification Trace (Phase B & Phase G)

A live end-to-end trace was executed against the running stack (`Express :8000`, `FastAPI :8001`, `Celery worker coverageiq-local@localhost`, `PostgreSQL 16 :5432`, `Redis 7 :6379`) using a disposable test user and disposable health policy (preserving all pre-existing database rows):

### 1. Upload, Ingestion & PostgreSQL Persistence Trace (Steps 1–8)
- **Step 1 — User Auth (`POST /api/v1/users/register`):** Returned `HTTP 201` with `HttpOnly` JWT cookies (`audit.trace.*@coverageiq.test`, role `user`).
- **Step 2 — Draft Policy Creation (`POST /api/v1/policies`):** Returned `HTTP 201` creating `policy_id = 1b741783-25bb-41e0-b9c4-02738ef0bb61` (`external_policy_key = pol_audit_trace_health_plan__c3f3c033`, `status = draft`, `visibility = private`).
- **Step 3 — Multipart PDF Upload (`POST /api/v1/policies/1b741783-25bb-41e0-b9c4-02738ef0bb61/documents`):**
  - Version 1 (`sha256_prefix = c44589e40e89`, `1739` bytes, 2 pages): Returned `HTTP 201` (`duplicate = false`, `enqueued = true`, `document_id = 4d261821-8599-4118-91f4-657c2d1c0cce`, `job_id = 45f76a4b-1be6-4183-8ca3-277ac06d9fd7`, `task_id = ingest-45f76a4b-1be6-4183-8ca3-277ac06d9fd7`, initial `status = queued`, `current_stage = QUEUED`).
  - Version 2 (`audit-v2-wrapped`, 2 pages, properly wrapped text): Uploaded via same endpoint -> `job_id = 5cdeb2e3-104c-4da2-88d5-f715744e7e60` -> `COMPLETED` (`3/3` chunks, `100%`).
- **Step 4 — Worker Stage Transitions Observed (`GET /api/v1/ingestion/jobs/:jobId`):**
  - `queued:QUEUED` -> `processing:VALIDATING` -> `processing:CHUNKING` -> `completed:COMPLETED` (`progress = { total_chunks: 3, processed_chunks: 3, percent: 100 }`).
- **Step 5 — Disk Storage & PostgreSQL Verification (Read-Only):**
  - Disk file confirmed present under `POLICY_UPLOAD_ROOT` (`1739` bytes for v1, SHA-256 matched `policy_documents.file_hash_sha256`).
  - `policies.status` transitioned from `'draft'` to `'active'`; `policies.active_version_id` updated atomically from v1 (`4d261821-...`) to v2 (`c401fe66-2bc9-4c62-b205-38fb08bab8ab`), with `doc_versions = 2`, `jobs = 2`.
  - Every chunk row in `policy_chunks` joined 1:1 with `chunk_embeddings` (`vector_dims = 1024`, `model_name = 'BAAI/bge-m3'`, non-null `search_tsv`, `page_number` 1–2, `clause_ref` `'Preamble'`, `'2'`, `'4'`).

### 2. Downstream Feature Verification on the Uploaded Policy
- **Missing Express & Recommendation/Summary Endpoints Confirmed (`HTTP 404`):**
  - `POST /api/v1/policies/:id/ask` -> `HTTP 404`
  - `POST /api/v1/policies/:id/summary` -> `HTTP 404`
  - `POST /api/v1/recommendations` -> `HTTP 404`
  - `GET /api/v1/policies/:id` -> `HTTP 404`
  - `POST http://127.0.0.1:8001/ai/recommend` -> `HTTP 404`
  - `POST http://127.0.0.1:8001/ai/summarize` -> `HTTP 404`
- **Live FastAPI Downstream Q&A Executed Directly (`POST http://127.0.0.1:8001/ai/ask`, authenticated via `X-Service-Token` + `AuthorizedScope`):**
  1. **Grounded Coverage + Limit Check (`v2` active version):**
     - Query: `"Is in-patient hospitalization for a minimum stay of 24 consecutive hours covered and what is the ICU limit?"`
     - Response (`HTTP 200`): `decision = "CONDITIONAL"`, `confidence_score = 1.0`, `evaluator_status = "VERIFIED_PASS"`, `storage_mode = "postgresql_pgvector"`, `retrieval_attempts[0].total_ms = 27.508`, `generation_llm.model = "nvidia/nemotron-3.5-lightning-30b-a3b"`, `citations = [{ chunk_id: "chk_pol_audit_trace__2_966f878ed7b3", page: 1, clause: "2", category: "Coverage", page_range: "1" }]`.
  2. **Grounded Exclusion Check (`v1`):**
     - Query: `"Is elective cosmetic surgery covered under this policy?"`
     - Response (`HTTP 200`): `decision = "NOT COVERED"`, `confidence_score = 1.0`, `evaluator_status = "VERIFIED_PASS"`, `citations = [{ chunk_id: "chk_pol_audit_trace__4_2d95968b35e2", page: 2, clause: "4", category: "Exclusions" }]`.
  3. **Grounded `NO EVIDENCE = NO CLAIM` Proof (`v1` clipped line + Unanswerable Question):**
     - When `v1`'s text had been clipped before the `"24 consecutive hours"` / `"ICU limit"` clauses, `/ai/ask` explicitly refused (`decision = "INSUFFICIENT EVIDENCE"`, `confidence_score = 0.95`) explaining the clauses were missing from retrieved text.
     - On `"Does this policy cover commercial spaceflight or lunar tourism?"`: returned `decision = "INSUFFICIENT EVIDENCE"`, `confidence_score = 0.95`, `citations = []`.
  4. **Cross-Tenant Isolation Check:**
     - Querying another tenant's existing `policy_id` with the disposable user's `ScopeContext` returned `HTTP 403` `{"detail":"Access denied for requested policy scope."}` prior to chunk retrieval.

---

## E. Missing Integration Points (Exact Gaps)

1. **Q&A Proxy & Persistence Bridge (`Express` + `PostgreSQL` + `React`):**
   - **Missing DB migration:** `chat_messages` (or `qa_sessions` / `qa_messages`) table to persist user questions, decisions, conditions, reasons, and cited chunk IDs per policy.
   - **Missing Express route/controller/service:** `POST /api/v1/policies/:policyId/ask` (must verify `verifyJWT`, `PERMISSIONS.CHAT_QUERY`, read authorization on `policyId`, verify active completed document exists, construct `ScopeContext` `{ user_id, role, owner_scope, allow_catalog_read }` strictly from DB auth context, forward with `X-Service-Token` to FastAPI `POST /ai/ask`, validate response shape, persist history, return JSON). Also `GET /api/v1/policies/:policyId/qa-history` for dashboard `Recent Queries` & KPI counts.
   - **Missing React UI wiring:** Add `qaApi` in `client/src/lib/api.js` and replace `UserDashboardPage.jsx` -> `QATab` placeholder with live policy selector (`status === 'active'`), question submission, decision badge (`COVERED`/`NOT COVERED`/`CONDITIONAL`/`INSUFFICIENT EVIDENCE`), conditions list, and expandable evidence/citation badges + history list.

2. **Recommendation Engine Schema Reconciliation & Gateway/Pipeline Wiring (M5 + M2 + M4):**
   - **Unmerged M5 code & schema mismatch:** `origin/shravan/policy-recommendation` expects `policy_versions` and `policies.(name, owner_user_id, organization_id, description, current_version_id)`, whereas live `node-pg-migrate` uses `policy_documents` and `policies.(title, owner_id, owner_scope, visibility, active_version_id)`.
   - **Missing attribute extraction in M2 (`pipeline.py`):** Uploaded PDFs generate `policy_chunks` and `chunk_embeddings`, **not** structured `policy_attributes` rows required by deterministic M5 filtering (`annual_premium_min/max`, `sum_insured_min/max`, `min/max_entry_age`, `ped_covered`, `ped_waiting_period_months`, `copay_pct`, etc.). Note also `PolicyUploadPanel.jsx` currently omits `visibility: 'catalog'` when `userRole === 'organisation'`.
   - **Missing Express & React M5 integration:** Express needs `POST /api/v1/recommendations` forwarding `TrustedScope` + validated profile payload to FastAPI; M5 `NullCitationProvider` needs wiring to M3/M8 (`PersistentHybridRetriever` + `ClosedRetryLoopOrchestrator`) for evidence-backed explanations; `UserDashboardPage` -> `RecommendationsTab` needs a real profile form and ranked results list with citations and elimination reasons.

3. **Policy Summarization Module (M6):**
   - **Missing AI service module:** `ai_service/` has no summarization service (`POST /ai/summarize`) retrieving section evidence via M3 and auditing statements via M8.
   - **Missing DB cache & Express proxy:** No `policy_summaries` table keyed by `(policy_id, document_id)` and no `POST /api/v1/policies/:policyId/summary` / `GET` routes.

4. **Policy Detail, Version Activation, & Citation Lookup APIs (M4):**
   - Express lacks `GET /api/v1/policies/:policyId`, `GET /api/v1/policies/:policyId/documents`, `PATCH /api/v1/policies/:policyId/active-version`, and chunk/evidence lookup `GET /api/v1/policies/:policyId/chunks/:chunkId` (needed for clickable PDF page/clause citation inspection).

5. **Organisation & Admin Governance APIs (M1):**
   - Express lacks `GET/POST /api/v1/organizations`, `GET/POST /api/v1/organizations/:id/members` (so `organisation` accounts currently have `0` rows in `organizations`), `GET /api/v1/admin/users`, and `GET /api/v1/admin/audit-logs`, and `user.controller.js` does not yet write `user.registered`, `user.logged_in`, `user.role_updated` rows into `audit_logs`.

6. **Dashboard Pagination & Telemetry Completeness:**
   - `useWorkspaceData.js` fetches `ingestionApi.listJobs({ page: 1, limit: 20 })` and computes KPI/chart counts from the returned slice `items`, ignoring `pagination.total_items` / server-side status aggregation if a user exceeds 20 jobs or 100 policies.

---

## F. Security Findings (Authorization, Isolation, Fallbacks & Leakage)

1. **Silent Mock Mode Default Hazard in AI Service (`High` if misconfigured):**
   - [ai_service/config.py](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/ai_service/config.py#L103) and [ai_service/.env.example](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/ai_service/.env.example#L15) default `USE_MOCK_STORAGE=True`, and `ai_service/.env` does not override it. While the running daemon explicitly sets `USE_MOCK_STORAGE=false` and `routes_ingestion.py` rejects `USE_MOCK_STORAGE=true` with `503`, starting Uvicorn purely from `ai_service/.env` would silently cause `/ai/ask` to query `MockKnowledgeBase` (`policy_health_01`) instead of PostgreSQL 16 + pgvector and skip `AuthorizedScope` token checks in `routes_qa.py:79-97`.
2. **Gemini-Mode Silent Fallback (`Medium`):**
   - [ai_service/llm/client.py](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/ai_service/llm/client.py#L168-L188): while `LLM_PROVIDER=nvidia` (current config) propagates failures cleanly without fallback, the `gemini` branch logs exceptions and silently runs `_offline_grounded_synthesis`.
3. **Catalog Visibility Omission on Organisation Upload (`Medium`):**
   - `PolicyUploadPanel.jsx` creates policies without sending `visibility: 'catalog'` or `organization_id` even when logged in as `organisation`, causing `createAuthorizedPolicy` ([ingestionGateway.service.js:968](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/server/src/services/ingestionGateway.service.js#L968)) to default all UI-created policies to `'private'` under `user:<id>`.
4. **M5 Unmerged Branch Insecure CORS & SQLAlchemy Pattern (`Medium` — avoided):**
   - `origin/shravan/policy-recommendation`'s `ai_service/main.py` re-enabled `CORSMiddleware(allow_origins=["*"], allow_credentials=True)` and omitted `/internal/ingestion` + CUDA pool cleanup; verified **not** merged.
5. **Verified Solid Security Controls (Current Running Stack):**
   - Cross-tenant Q&A isolation (`403 Access denied` before retrieval), upload ownership lock + atomic TX/file rollback, `X-Service-Token` constant-time verification (`403` on bad token), HttpOnly JWT + atomic `FOR UPDATE` refresh-token rotation + reuse revocation, strict Express & FastAPI CORS allowlists, Redis atomic sliding-window fail-closed limiter, and least-privilege `coverageiq_runtime` DB role (`pgmigrations` & DDL blocked) are all active and verified.

---

## G. Priority Classification

| Priority | Finding ID | Summary |
|---|---|---|
| **Critical** | `GAP-01` | **No Express proxy or React UI for Q&A (`POST /api/v1/policies/:id/ask` missing)** — core grounded answer & citation engine (`/ai/ask`) is invisible to end users. |
| **Critical** | `GAP-02` | **M5 Recommendations disconnected & schema-incompatible** — `003` SQL references legacy `policy_versions` / `policies.name`; M2 ingestion doesn't extract `policy_attributes`; Express & UI unbuilt. |
| **High** | `SEC-01` | **`USE_MOCK_STORAGE` defaults to `True` in `config.py` / `.env.example`** — risk of silent `MockKnowledgeBase` fallback on `/ai/ask` if FastAPI is launched without explicit `USE_MOCK_STORAGE=false`. |
| **High** | `GAP-03` | **Policy Summarization (M6) unimplemented** across React (`disabled`), Express, FastAPI, and PostgreSQL. |
| **High** | `GAP-04` | **No Policy Details / Version History / Citation Chunk Lookup Express APIs** (`GET /policies/:id`, `/documents`, `/chunks/:chunkId`). |
| **Medium** | `GAP-05` | **Organisation creation/membership (`organizations` = 0) and `visibility='catalog'` UI control missing**, preventing multi-user org collaboration and catalog publishing from UI. |
| **Medium** | `GAP-06` | **Admin Governance Read APIs missing** (`GET /users`, `GET /organizations`, `GET /audit-logs`; auth events not written to `audit_logs`). |
| **Medium** | `GAP-07` | **Dashboard KPI counts truncate at 20 jobs / 100 policies** because `useWorkspaceData` summarizes `listJobs({ limit: 20 }).items` client-side instead of dedicated aggregate stats. |
| **Low** | `SEC-02` | `LLMClient` `gemini` branch silently falls back to `_offline_grounded_synthesis` on exception (`nvidia` branch does not). |
| **Informational** | `INFO-01` | `LandingPage.jsx` cards render illustrative static examples (acceptable for public landing page, clearly non-account-specific). |

---

## H. Recommended Implementation Sequence (Awaiting User Approval)

1. **Step 1 — Safety Hardening (`config.py` & `.env.example` cleanup, small & zero-risk):**
   - Enforce fail-closed startup in non-test environments if `USE_MOCK_STORAGE=true`, remove `MockKnowledgeBase` silent default in `.env.example`, and remove silent fallback from `LLMClient`'s `gemini` branch (`ai_service/config.py`, `ai_service/llm/client.py`, `ai_service/.env.example`).
2. **Step 2 — Connect Coverage Q&A End to End (`M7/M8` -> `M1` -> `M10`):**
   - Migration `1764250200000_qa_history_and_summaries.js` (`server/migrations/`) creating `chat_messages` (scoped by `policy_id`, `document_id`, `user_id`, storing `decision`, `confidence_score`, `conditions`, `reason`, `citations`).
   - Express `server/src/controllers/qa.controller.js`, `server/src/routes/qa.routes.js` (mounted on `/api/v1/policies/:policyId/ask` and `/api/v1/policies/:policyId/qa-history`), forwarding `ScopeContext` + `X-Service-Token` to FastAPI `POST /ai/ask`.
   - Frontend `client/src/lib/api.js` (`qaApi`) + interactive `QATab` in `UserDashboardPage.jsx` (and reusable across workspaces) showing `COVERED`/`NOT COVERED`/`CONDITIONAL`/`INSUFFICIENT EVIDENCE`, conditions, and clickable page/clause citations.
3. **Step 3 — Policy Detail, Version History & Catalog Toggle (`M4`):**
   - Add `GET /api/v1/policies/:policyId`, `GET /api/v1/policies/:policyId/documents`, `PATCH /api/v1/policies/:policyId/active-version`, `GET /api/v1/policies/:policyId/chunks/:chunkId` in `server/src/controllers/ingestion.controller.js` + `ingestion.routes.js`.
   - Wire `PolicyUploadPanel.jsx` (`visibility: 'catalog'` toggle for `organisation`/`admin`) and `OrganisationDashboardPage.jsx` -> `VersionsTab` to real `policy_documents` rows.
4. **Step 4 — Reconcile & Integrate Policy Recommendation (`M5` + `M2` Attribute Extraction):**
   - Reconcile M5's `policy_attributes` + `ranking_features` schema into a `node-pg-migrate` migration referencing `policies(id)` and `policy_documents(id, policy_id)` (not legacy `policy_versions`) using existing Psycopg 3 repository/SQL patterns (without introducing an incompatible migration runner).
   - Add structured attribute extraction (`attribute_extractor`) during M2 Celery ingestion or organization catalog publication, inject M3/M8 `RecommendationCitationProvider`, expose `POST /api/v1/recommendations` in Express, and build the interactive form + cited results UI in `UserDashboardPage.jsx` -> `RecommendationsTab`.
5. **Step 5 — Policy Summarization (`M6`):**
   - Add `POST /ai/summarize` in `ai_service/` (targeted M3 retrieval per section + M8 verifier pass + `"Not found in document"` handling), cache in `policy_summaries`, expose `POST/GET /api/v1/policies/:policyId/summary` in Express, and wire `SummariesTab` in React.
6. **Step 6 — Organisation Team APIs, Admin Governance & Live Aggregate KPIs (`M1`):**
   - Add `GET/POST /api/v1/organizations`, `GET/POST /api/v1/organizations/:id/members`, `GET /api/v1/admin/users`, `GET /api/v1/admin/audit-logs`, `GET /api/v1/analytics/overview`, log auth/role changes in `audit_logs`, and wire `AdminDashboardPage.jsx` & `OrganisationDashboardPage.jsx` (`MembersTab`).

---

## I. Verification Evidence Summary

- **Runtime PostgreSQL Row Counts (Post-Audit Trace, Read-Only Query):**
  - `users`: `4` (`3` pre-existing + `1` disposable trace user)
  - `policies`: `3` (`2` pre-existing active ACKO term life policies + `1` disposable `Audit Trace Health Plan`)
  - `policy_documents`: `5` (`3` pre-existing completed + `2` disposable v1/v2 completed)
  - `ingestion_jobs`: `5` (`3` pre-existing + `2` disposable, all `status = 'completed'`, `current_stage = 'COMPLETED'`)
  - `policy_chunks`: `318` (`312` pre-existing + `6` from disposable v1/v2)
  - `chunk_embeddings`: `318` (`1024`-dim `vector`, matched 1:1 with `policy_chunks`)
  - `organizations`: `0`, `organization_members`: `0`, `refresh_tokens`: `9`, `audit_logs`: `5`
- **Service Health & Config Verified:**
  - Express (`8000`): `GET /api/health` -> `HTTP 200` (`database: coverageiq_db`, `redis: up`, `latencyMs: 0`)
  - FastAPI (`8001`): `GET /ai/health` -> `HTTP 200` (`database_mode: postgresql_pgvector`)
  - Celery worker (`coverageiq-local@localhost`): `inspect ping` -> `OK pong` (`USE_MOCK_STORAGE=false`, `EMBEDDING_PROVIDER=bge-m3`, `LLM_PROVIDER=nvidia`, `NVIDIA_API_KEY` set)
  - Frontend (`5173`): Vite dev server serving integrated React dashboards (`VITE_API_BASE_URL` defaults to `/api/v1` proxy)
- **Not Executed / Out-of-Scope Untested Paths:**
  - `origin/shravan/policy-recommendation` tests were **not** executed against this repo branch due to declared schema/dependency incompatibility.
  - Summarization (`POST /ai/summarize`) and Recommendations (`POST /api/v1/recommendations`) do not exist in the running endpoints (`HTTP 404` confirmed).
