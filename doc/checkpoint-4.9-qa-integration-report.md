# CoverageIQ Checkpoint 4.9 Completion Report
## End-to-End Policy Q&A Integration, Persistence & Hardening

**Branch:** `prithviraj_features` (uncommitted; 0 staged files, 0 commits, 0 pushes)  
**Core Principle:** `NO EVIDENCE = NO CLAIM`  
**Status:** Completed and verified across Python 3.11, Node.js/Express, PostgreSQL 16 + pgvector, Celery/Redis, React/Vitest, and live Google Chrome CDP + NVIDIA Nemotron (`nvidia/nemotron-3.5-lightning-30b-a3b`).

---

## 1. Scope Implemented

Checkpoint 4.9 connects the existing Python/FastAPI Q&A pipeline (`POST /ai/ask`) through the Node.js/Express gateway and PostgreSQL 16 (`chat_messages`) to the interactive React frontend using actual uploaded and indexed policy documents:

1. Mock-storage and offline LLM fallback fail-closed hardening.
2. Node-owned PostgreSQL Q&A persistence (`chat_messages` migration).
3. Express Q&A API proxy (`POST /api/v1/policies/:policyId/ask` and `GET /api/v1/policies/:policyId/qa-history`).
4. Interactive React Q&A workspace (`PolicyQAPanel.jsx`, `qaApi`, `useWorkspaceData.js`, `UserDashboardPage.jsx`) plus automatic policy status refresh on Celery job completion (`IngestionJobsPanel.jsx`).
5. Chronological Q&A conversation history and per-user isolation.
6. Unit, API, integration, UI, and live Chrome CDP + NVIDIA Nemotron end-to-end verification.

**Deferred outside Checkpoint 4.9 (unchanged):** policy recommendations, policy summarization, organisation member management, and admin directory endpoints.

---

## 2. Architecture & Request Data Flow

```text
React UI (PolicyQAPanel.jsx / qaApi)
  ├── POST /api/v1/policies/:policyId/ask  (cookie/Bearer JWT auth)
  ├── GET  /api/v1/policies/:policyId/qa-history
        │
        ▼
Express API Gateway (server/src/routes/qa.routes.js -> qa.controller.js -> qaGateway.service.js)
  ├── verifyJWT + authorizePermissions(PERMISSIONS.CHAT_QUERY)
  ├── authorizePolicyForQa (owner_scope / org memberships / active catalog; rejects client scope)
  ├── verifyActiveDocumentReadyForQa (requires completed active_version_id & policy_chunks)
  ├── mapPlatformRoleToAiRole ('organisation' -> 'organization')
  │
  ├── POST http://127.0.0.1:8001/ai/ask (X-Service-Token + trusted AuthorizedScope)
        │
        ▼
FastAPI AI Service (ai_service/api/routes_qa.py -> ClosedRetryLoopOrchestrator)
  ├── Fail-closed non-test validation (USE_MOCK_STORAGE=false, non-offline LLM & embeddings)
  ├── PG16 + pgvector PersistentHybridRetriever (BAAI/bge-m3 1024-dim + BM25 + RRF + MiniLM Reranker)
  ├── GroundedGenerator + Independent Verifier + DeterministicDecisionGate (retry loop)
        │
        ▼ (HTTP 200 validated answer ONLY)
PostgreSQL 16 (chat_messages inside withTransaction)
  ├── Insert 'user' row at clock_timestamp()
  └── Insert 'assistant' row at GREATEST(clock_timestamp(), userCreatedAt + 1 ms)
```

---

## 3. Files Created & Modified

### Python AI Service (`ai_service/`)
- [ai_service/config.py](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/ai_service/config.py): Added `APP_ENV` resolution (`_resolve_app_env()`), `is_non_test_env` (`APP_ENV in ("production", "staging", "prod", "stage")` or `DEBUG=false`), and fail-closed `Settings.validate_settings()` checks requiring explicit `USE_MOCK_STORAGE=false`, non-offline `LLM_PROVIDER`, and non-offline `EMBEDDING_PROVIDER` in non-test environments.
- [ai_service/.env.example](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/ai_service/.env.example): Documented `APP_ENV=development` and set default `USE_MOCK_STORAGE=false`.
- [ai_service/llm/client.py](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/ai_service/llm/client.py): Added `ExternalProviderConfigurationError` and `ExternalProviderError`; removed silent `_offline_grounded_synthesis` fallback on `gemini` errors or missing credentials; banned offline LLM execution in non-test environments; added `mock_mode: True` metadata marker in offline mode; fixed `_offline_grounded_synthesis` regex to parse multiline chunk content (`SemanticChunker` output) and page ranges (`1-2`).
- [ai_service/evaluator/retry_controller.py](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/ai_service/evaluator/retry_controller.py): Added `mock_mode` flag in `StructuredAnswerResponse.metadata`.
- [ai_service/api/routes_qa.py](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/ai_service/api/routes_qa.py): Expanded `_runtime_settings()` watched variables, guarded `get_orchestrator()` config validation cleanly with HTTP `503`, blocked `USE_MOCK_STORAGE=true` in non-test environments (`503`), and mapped `ExternalProviderConfigurationError` / `ExternalProviderError` cleanly.
- [ai_service/tests/test_phase4_9_qa_hardening.py](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/ai_service/tests/test_phase4_9_qa_hardening.py): Added unit/API regression tests for fail-closed non-test settings, Gemini non-fallback behavior, and `/ai/ask` error mapping.

### Node.js / Express & PostgreSQL Migrations (`server/`)
- [server/migrations/1764250200000_qa_chat_messages.js](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/server/migrations/1764250200000_qa_chat_messages.js): Node-owned migration adding `chat_messages` (`user` & `assistant` roles, `chk_chat_messages_assistant_fields`, JSONB `conditions` / `citations` / `provider_metadata`, 3 chronological indexes, and least-privilege DML grants for `coverageiq_runtime`).
- [server/src/services/qaGateway.service.js](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/server/src/services/qaGateway.service.js): Implemented `authorizePolicyForQa`, `verifyActiveDocumentReadyForQa`, `validateAndNormalizeAiAnswer`, `askPolicyQuestion`, and `listPolicyQaHistory` (`'organisation'` -> `'organization'` mapping, rejection of client-supplied `scope`/`owner_scope`/`user_id`, strict production `mock_mode` block, atomic transactional `user` + `assistant` persistence only after validated AI responses, and monotonic intra-transaction `clock_timestamp()` ordering).
- [server/src/services/ingestionGateway.service.js](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/server/src/services/ingestionGateway.service.js): Hardened `sanitizePublicErrorMessage()` to redact `nvapi-...`, `AIza...`, `sk-...`, and `Bearer ...` secrets in addition to database/Redis URLs and filesystem paths.
- [server/src/controllers/qa.controller.js](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/server/src/controllers/qa.controller.js), [server/src/routes/qa.routes.js](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/server/src/routes/qa.routes.js), [server/src/app.js](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/server/src/app.js), [server/.env.example](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/server/.env.example): Wired authenticated `/api/policies/:policyId/(ask|qa-history)` and `/api/v1/policies/:policyId/(ask|qa-history)` endpoints and documented `QA_HTTP_TIMEOUT_MS=90000`.
- [server/tests/qa.integration.test.js](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/server/tests/qa.integration.test.js): Added end-to-end integration tests for migration down/up redo, `coverageiq_runtime` privilege checks, RBAC/scope hardening, real Upload -> Celery -> PG16 -> `/ask` -> `chat_messages` persistence -> `/qa-history`, catalog per-user isolation, and fail-closed zero-persistence on AI errors/timeouts/secret leaks.

### React Frontend (`client/`)
- [client/src/lib/api.js](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/client/src/lib/api.js): Added `qaApi.ask()` and `qaApi.getHistory()`.
- [client/src/components/PolicyQAPanel.jsx](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/client/src/components/PolicyQAPanel.jsx): Built interactive Q&A panel filtering `status === 'active'` policies, rendering chronological thread history, New Thread toggling, Refresh, Retry on retryable HTTP failures, decision badges (`COVERED`, `NOT COVERED`, `CONDITIONAL`, `INSUFFICIENT EVIDENCE`), conditional numeric confidence badge, `evaluator_status`, conditions list, `NO EVIDENCE = NO CLAIM` banner on `INSUFFICIENT EVIDENCE`, and exact `citations` (page/range, clause, category, `chunk_id`, snippet).
- [client/src/hooks/useWorkspaceData.js](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/client/src/hooks/useWorkspaceData.js), [client/src/components/IngestionJobsPanel.jsx](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/client/src/components/IngestionJobsPanel.jsx), [client/src/components/dashboard/WorkspaceIngestionTab.jsx](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/client/src/components/dashboard/WorkspaceIngestionTab.jsx), [client/src/pages/UserDashboardPage.jsx](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/client/src/pages/UserDashboardPage.jsx), [client/src/pages/OrganisationDashboardPage.jsx](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/client/src/pages/OrganisationDashboardPage.jsx): Connected `PolicyQAPanel`, live `qaTotalQuestions` KPI metric, and `onJobCompleted` auto-refresh so newly completed uploads promote draft policies to `active` in the UI without manual page reload.
- [client/src/test/qaWorkflow.test.jsx](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/client/src/test/qaWorkflow.test.jsx): Added Vitest tests covering `qaApi` token-refresh retry, draft empty state navigation, history loading, New Thread behavior, `INSUFFICIENT EVIDENCE`, and transient 504 Retry recovery.

---

## 4. Verification Summary (Observed Tool Outputs Only)

| Suite / Check | Environment | Exact Command / Scope | Observed Result |
| --- | --- | --- | --- |
| Node.js / Express E2E | Ephemeral PG16 (`55432`) + Redis (`56379`), Python 3.11 FastAPI + Celery | `npm --prefix server test` (`hardening.security.test.js`, `ingestion.integration.test.js`, `qa.integration.test.js`) | **19 passed, 0 failed** (3 suites, 49.8s) |
| React Lint, Tests & Build | JS / Vitest / Vite 8 | `npm --prefix client run lint && npm --prefix client test && npm --prefix client run build` | ESLint clean (**0 errors**), Vitest **7 passed** (2 files), Vite build succeeded |
| Python Targeted Q&A / Hardening | Python 3.11 (`/home/prithviraj/ai-env-py311/bin/python`) | `test_phase4_9_qa_hardening.py`, `test_phase4_config.py`, `test_api_endpoints.py`, `test_retry_loop.py` | **29 passed, 0 failed** (23.75s) |
| Python Nvidia / Evaluator / Retrieval Unit | Python 3.11 | `test_nvidia_generation.py`, `test_nvidia_nemotron_provider.py`, `test_nvidia_verification.py`, `test_evaluator.py`, `test_hybrid_retrieval.py`, `test_query_analyzer.py`, `test_reranker.py` | **76 passed, 0 failed** (39.20s) |
| Full Python Regression Suite | Python 3.11 against ephemeral PG16 (`55432`) + Redis (`56379`) | `pytest ai_service/tests -q --tb=short` (run while live local daemons simultaneously held 2.44 GiB on RTX 3050 4GB) | **311 passed, 2 CUDA OOM failures** (`test_real_bge_m3_worker_ingestion_smoke_when_cached`, `test_bge_m3_real_cached_cuda_inference_batch_size_1`) |
| Contended CUDA BGE-M3 Checks (Isolated) | Python 3.11 after stopping resident daemons (2,998 MiB VRAM free) | Targeted re-run of `test_bge_m3_real_cached_cuda_inference_batch_size_1` & `test_real_bge_m3_worker_ingestion_smoke_when_cached` | **2/2 passed** (10.29s and 26.97s) |
| Live Chrome CDP + NVIDIA Nemotron E2E | Live `localhost:5173`, Express `8000`, FastAPI `8001`, Celery `ingestion,ocr`, PG16 `5432`, Redis `6379`, real `BAAI/bge-m3` + live `nvidia/nemotron-3.5-lightning-30b-a3b` | Headless Chrome CDP (`browser_e2e_qa_cdp.mjs`) registering disposable user -> creating policy -> uploading 2-page PDF -> Celery `COMPLETED` -> 3 live Q&A turns -> reload persistence -> Overview KPI check -> PG16 provenance check -> disposable cleanup | **Passed**: Q1 `CONDITIONAL` (Page 1), Q2 `NOT COVERED` (Page 2), Q3 `INSUFFICIENT EVIDENCE`, reload verified (`3 question(s)`), Overview KPI=`3`, PG16 audit `msg_count=6`, `valid_cited_chunks=2/2`, **0** browser page/console errors, baseline records intact (`users: 4, policies: 3, docs: 5, chunks: 318`) |

---

## 5. Observed Bugs Fixed During Integration Testing
1. **Stale active policy list after upload completion:** `UserDashboardPage` did not refresh `policies` when `IngestionJobsPanel` polled an in-flight job to `completed`, causing `Coverage Q&A` to show draft empty state after uploading until manual browser refresh. Fixed via `onJobCompleted` propagation plus tab-switch refresh to `qa` / `overview`.
2. **Thread reset race in `PolicyQAPanel.jsx`:** `conversationId` inside `useEffect` dependencies re-fetched history when clicking `New Thread` and immediately re-populated the previous thread ID. Fixed using `autoResumeThreadRef` and removing `conversationId` from the history `useEffect` dependency list.
3. **Intra-transaction `chat_messages` timestamp tie:** `DEFAULT NOW()` (`transaction_timestamp()`) assigned identical timestamps to `user` and `assistant` rows inside `withTransaction`, allowing random UUID tie-breaking in index scans. Fixed by inserting `user` with `clock_timestamp()` and `assistant` with `GREATEST(clock_timestamp(), userCreatedAt + interval '1 millisecond')`.
4. **Secret pattern leakage in upstream error sanitizer:** `sanitizePublicErrorMessage()` now redacts API key patterns (`nvapi-...`, `AIza...`, `sk-...`) and `Bearer` tokens.
5. **Multiline chunk parsing in `_offline_grounded_synthesis`:** Fixed regex stopping on the first newline of `SemanticChunker` chunks so offline PG integration tests parse full chunk content cleanly.
6. **Hardware VRAM note:** On an RTX 3050 4GB GPU, running a separate third CUDA BGE-M3 test process concurrently with both resident live daemons (FastAPI `~1.26 GiB` + Celery `~1.18 GiB`) exceeds `3.68 GiB` VRAM; live dual-daemon runtime (`FastAPI` + `Celery`) fits and runs cleanly together.
