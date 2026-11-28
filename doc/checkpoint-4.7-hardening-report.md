# CoverageIQ — Checkpoint 4.7 Hardening and Regression Review Report

- **Project:** CoverageIQ (Evidence-Grounded Insurance Policy Assistant)
- **Branch:** `prithviraj_features` (tracking `origin/prithviraj_features`, baseline commit `242d52899e44c84df70e1ab1d4b21fbb19e3de8e`)
- **Review Principle:** `NO EVIDENCE = NO CLAIM`
- **Git State:** Uncommitted implementation & verification only (`0` staged changes, no commits or pushes executed; Checkpoints 1–4.6 preserved)

---

## 1. Executive Summary & Production Readiness Assessment

Checkpoint 4.7 audited and hardened the end-to-end CoverageIQ stack across the React frontend (`client/`), Node.js/Express gateway & authentication layer (`server/`), PostgreSQL 16 + pgvector schema/migrations, Redis 7 + Celery asynchronous ingestion workers, BGE-M3 CUDA embedding lifecycle, PDF parsing/table extraction, and live headless Chromium browser workflows.

- **Overall Assessment:** **Conditionally Ready for Staging / Single-Instance Testing; Not Yet Ready for Multi-Replica Production Deployment.**
- **Resolved During Checkpoint 4.7:**
  - Two **High**-severity security issues were remediated and covered by automated tests: permissive credentialed CORS wildcard/reflection (`server/src/app.js`, `ai_service/main.py`) and non-atomic refresh-token rotation lacking reuse detection plus unvalidated admin `:userId` UUID parameters (`server/src/models/user.model.js`, `server/src/controllers/user.controller.js`).
  - One **Medium** GPU resource-lifecycle issue was fixed: added `BgeM3Embedder.close()` (`ai_service/embeddings/bge_m3.py`) and wired cleanup into `WorkerResourceManager.shutdown()` (`ai_service/workers/celery_app.py`) to offload CUDA tensors and empty PyTorch cache on shutdown/teardown.
  - One **Low** pre-existing PDF extraction regression (`test_table_with_empty_cells`) was fixed in `DocumentStructurer._extract_tables_and_mask_blocks` (`ai_service/ingestion/document_structurer.py`), stripping PyMuPDF narrow (`<= 6.0 pt`) phantom border divider columns while retaining wide blank data columns and column-aligned cell geometry.
- **Remaining Unverified / Production Prerequisites:**
  1. **Target Python 3.11 Runtime Unverified (`Medium`):** Host environment only had Python `3.14.4` (`python3.11` binary absent).
  2. **Live PaddleOCR Inference Unexecuted (`Medium`):** `paddleocr` / `paddlepaddle` are not installed in `~/ai-env` and not listed in `ai_service/requirements.txt`; only unavailable-state and mocked adapter handling ran.
  3. **In-Memory Upload Rate Limiter (`Medium`):** `server/src/middlewares/rateLimit.middleware.js` tracks request timestamps in a process-local `Map`, which requires migration to a Redis-backed store prior to horizontal scaling across multiple Node replicas.

---

## 2. Severity Findings Register & Remediation Status

| ID | Severity | Component | Finding Summary | Resolution & Verification |
| --- | --- | --- | --- | --- |
| SEC-01 | **High** | `server/src/app.js`, `ai_service/main.py` | Express defaulted `cors({ origin: process.env.CORS_ORIGIN \|\| true, credentials: true })`, reflecting any arbitrary `Origin` with credentials when `CORS_ORIGIN` was unset. FastAPI enabled `allow_origins=["*"]` with `allow_credentials=True`. | **Fixed.** Implemented `getAllowedCorsOrigins()` allowlist callback in Express (stripping `*`, restricted dev defaults only, fail-closed `Set()` in production if `CORS_ORIGIN` unset) + security headers (`nosniff`, `DENY`, `Referrer-Policy`, `x-powered-by` disabled). Restricted FastAPI CORS to `INTERNAL_CORS_ORIGINS` (default `[]`). Verified via Node security suite & `TestClient` tests. |
| SEC-02 | **High** | `server/src/models/user.model.js`, `server/src/controllers/user.controller.js` | `refreshAccessToken` read then later revoked refresh tokens outside a transaction (TOCTOU race allowing parallel chains), lacked reuse revocation on replayed tokens, omitted explicit cookie `path: '/'`, and passed unvalidated `:userId` into PostgreSQL `updateUserRole`. | **Fixed.** Added transactional `consumeAndRotateRefreshToken` with `SELECT ... FOR UPDATE` row locking and reuse detection (replaying a revoked token revokes all active tokens for that user), standardized `COOKIE_OPTIONS.path = '/'`, and enforced `UUID_RE` validation (`400`) on `updateUserPlatformRole`. Verified via concurrent race, replay, logout, and invalid-UUID tests. |
| REL-01 | **Medium** | `ai_service/embeddings/bge_m3.py`, `ai_service/workers/celery_app.py` | Back-to-back CUDA `BgeM3Embedder` lifecycles in one process exhausted 3.68 GiB RTX 3050 VRAM (`torch.OutOfMemoryError`) when prior instances retained references before `empty_cache()`. | **Fixed.** Added `BgeM3Embedder.close()` moving inner model weights to CPU, releasing references, running `gc.collect()`, and emptying CUDA cache; invoked in `WorkerResourceManager.shutdown()` and test teardowns. Verified via sequential cached CUDA tests in full `pytest` run. |
| ENV-01 | **Medium** | Host Python runtime | Architecture target is Python 3.11; host only provides Python `3.14.4` (`python3.11` missing). | **Open Blocker.** Explicitly flagged; requires Python 3.11 CI/venv validation. |
| ENV-02 | **Medium** | `ai_service/requirements.txt`, OCR runtime | `paddleocr` / `paddlepaddle` not installed on host and absent from `ai_service/requirements.txt`. | **Mitigated / Documented.** `PaddleOCREngine.is_available()` safely returns `False` and `process_page()` returns explicit `OCRResult(success=False)` without fabricating text; live OCR inference on scanned documents remains unexecuted. |
| OPS-01 | **Medium** | `server/src/middlewares/rateLimit.middleware.js` | Upload rate limiter uses process-local `Map` storage (`429` + `Retry-After` verified per single process). | **Documented Limitation.** Must move to shared Redis sliding-window store before running multiple Node instances behind a load balancer. |
| BUG-01 | **Low** | `ai_service/ingestion/document_structurer.py` | `test_table_with_empty_cells` failed because PyMuPDF `find_tables()` emitted `2.0 pt` empty border sliver columns, causing `len(raw_extract[0]) != len(headers)`, duplicating header row 0 into `data_rows[0]`, and misaligning `tbl.cells` bboxes. | **Fixed.** Filtered `<= 6.0 pt` all-empty phantom sliver columns cleanly (`kept_col_indices`), fixed header deduplication against filtered columns, mapped cell geometry via `orig_col`, and added 2 regression tests (`33/33` in `test_document_structurer.py`). |

---

## 3. Repository Audit & Cleanliness Check

- **Git Tracking Audit:** Confirmed `0` staged changes (`git diff --cached --name-only` empty), `0` commits or pushes performed, and `0` tracked `.env`, `.sqlite`/`.db`, `dist/`, `node_modules/`, `uploads/`, or scratch files in repository status.
- **Secrets Handling:** `.env` files were not modified or dumped; error responses and DB DSNs were verified redacted (`postgresql://<redacted>`).
- **Test Harness Concurrency Fix:** Updated `server/package.json` `"test"` script to `node --test --test-concurrency=1 tests/*.test.js` because integration test suites reset shared PostgreSQL 16 tables via `TRUNCATE ... CASCADE` in `beforeEach`.

---

## 4. Database, Migrations & Least-Privilege Verification

Verified against disposable ephemeral Docker `pgvector/pgvector:pg16` (`127.0.0.1:55432`) and `redis:7-alpine` (`127.0.0.1:56379`) containers (subsequently removed with `docker rm -f`):

- **Node-Owned `node-pg-migrate` Roundtrip:** Executed `migrate:up` -> `node-pg-migrate down 0` -> `migrate:up` on disposable database `coverageiq_mig_verify` across both migrations (`1764250000000_initial_pgvector_schema` and `1764250100000_auth_and_organizations`) with clean teardown and recreation (`pgmigrations` ledger, `vector` extension, `policies`, `policy_documents`, `policy_chunks`, `chunk_embeddings` with HNSW cosine index, `ingestion_jobs`, `users`, `organizations`, `organization_members`, `refresh_tokens`, `audit_logs`).
- **Role Separation Enforced:** `coverageiq_runtime` verified unable to execute DDL (`CREATE TABLE` and `ALTER TABLE` raise PostgreSQL `42501` permission denied), isolated from `coverageiq_migrator`.
- **pgvector Dimension Constraint:** Inserting a 768-dimensional vector into `chunk_embeddings` (`vector(1024)` + CHECK `vector_dims(embedding) = 1024`) rejected cleanly at DB constraint level.

---

## 5. Authentication, Authorization & Input Security

Verified across `server/tests/hardening.security.test.js`, `server/tests/ingestion.integration.test.js`, and `ai_service/tests/test_phase4_7_hardening.py`:

- **CORS & Headers:** Untrusted origins receive no `Access-Control-Allow-Origin` or `Access-Control-Allow-Credentials` header on Express (`development` & `production`) or FastAPI (`/ai/health`). Wildcard `*` entries in `CORS_ORIGIN` are stripped cleanly.
- **JWT & RBAC:** Expired tokens, wrong-secret tokens, self-registration with `role: "admin"` (`403`), non-admin `PATCH /api/v1/users/:userId/role` (`403`), malformed `userId` UUIDs (`400`), cross-tenant policy upload (`403`), and cross-tenant job reads (`404` non-enumerating) verified. Spoofed `owner_id`/`owner_scope` fields in `POST /api/policies` payloads are ignored (`policies` DB row inherits `user:<id>`).
- **Atomic Refresh Token Rotation & Theft Detection:** Parallel racing `/refresh-token` calls allow exactly one `200` winner and reject the concurrent caller (`401`); replaying a rotated/revoked token revokes all active refresh tokens for that user; `/logout` invalidates refresh tokens; auth cookies enforce `HttpOnly`, `SameSite=Lax`, `Path=/` (verified invisibility to `document.cookie` in Chrome).
- **Upload & Path Traversal Security:** Rejects `../`, `\0`, `/etc/passwd`, Windows-style `..\`, and out-of-root symlinks (`400` in Node, `PathTraversalError` in Python), plus non-PDF signatures, MIME spoofing, empty PDFs, corrupt trailers, and encrypted PDFs.

---

## 6. Concurrency, Celery Idempotency & Failure Recovery

- **Upload Versioning Concurrency:** 3 simultaneous PDF uploads (`Promise.all`) to the same `:policyId` serialized under PostgreSQL row lock (`FOR UPDATE`) and persisted strictly sequential unique `version_number` values `[1, 2, 3]` with zero SQL unique violations.
- **Upload Rollback Cleanup:** Simulated DB abort after disk write (`afterInsertBeforeCommit`) rolled back DB state and removed temporary/uploaded PDF files cleanly (`0` orphaned files).
- **Celery Atomic Claiming:** Concurrent `claim_ingestion_job_for_execution` calls from competing workers yield `['claimed', 'duplicate_in_progress']`; re-claiming completed jobs returns `'already_completed'`; checksum mismatches fail permanently (`SHA256_MISMATCH`) with redacted error strings.

---

## 7. Real Browser E2E Verification (`Google Chrome 158.0.8127.88` Headless CDP)

Note: Playwright and Puppeteer were not installed on the host, so a self-contained Node 22 WebSocket Chrome DevTools Protocol runner executed directly against native `/usr/bin/google-chrome --headless=new` driving the live Vite frontend proxied to Express, FastAPI, PostgreSQL 16 + pgvector, Redis 7, and a live Celery solo worker:

- **Verified Browser Flows (`8/8` passed, `0` uncaught JS/browser exceptions):**
  1. Unauthenticated `/dashboard` visit redirects cleanly to `/login`.
  2. Real DOM registration creates user + verifies `HttpOnly` cookie isolation (`document.cookie` free of JWTs).
  3. Draft policy creation via UI updates policy selector.
  4. Real file attachment (`DOM.setFileInputFiles`) + XHR multipart upload + live `<IngestionJobsPanel>` polling until Celery completes (`COMPLETED`, `100%`).
  5. Re-uploading identical PDF displays `"Duplicate PDF Detected"` (`HTTP 200`).
  6. Uploading corrupted `.pdf` displays `"Upload failed (HTTP 400)"`.
  7. Evicting only the `accessToken` cookie followed by panel refresh triggers transparent `/api/v1/users/refresh-token` rotation without logging out.
  8. Cross-tenant UUID target upload renders `"Upload failed (HTTP 403)"` without page crash.

---

## 8. PDF Extraction, BGE-M3 Embeddings & OCR Review

- **PDF Table Extraction Fix:** Updated `ai_service/ingestion/document_structurer.py` lines 370–473 (`_extract_tables_and_mask_blocks`) so PyMuPDF `<= 6.0 pt` empty border slivers are ignored cleanly without collapsing genuine empty columns or misaligning bounding boxes. Verified all `33/33` tests in `ai_service/tests/test_document_structurer.py`.
- **Cached BGE-M3 CUDA Inference:** Verified `BgeM3Embedder` loads locally cached `BAAI/bge-m3` on `NVIDIA GeForce RTX 3050 Laptop GPU` with `batch_size=1`, deterministic `.eval()` outputs, exact `1024` dimensions, finite `float32` values, unit L2 norms (`abs(norm - 1.0) < 1e-4`), OOM cache flush (`torch.cuda.empty_cache()`), strict failure with zero silent fallback to `OfflineHashEmbedder`, and full VRAM release via `.close()`.
- **OCR Boundary:** Verified `PageScannedDetector`, `MockOCREngine`, and `PaddleOCREngine` unavailable/empty/error-trapping behavior (`11/11` in `test_ocr_engine.py` + `test_paddle_ocr_engine_reports_unavailable_and_handles_empty_or_failing_output`). Live `paddleocr` execution could not run due to missing host dependency (`paddleocr`).

---

## 9. Categorized Test Results (Observed Post-Final-Edit)

All test commands below were executed cleanly **after the final code and test edits** against live ephemeral Docker PostgreSQL 16 (`pgvector/pgvector:pg16`) and Redis 7 (`redis:7-alpine`):

| Category | Command / Suite | Result | Details |
| --- | --- | --- | --- |
| **Frontend Lint & Build** | `npm --prefix client run lint && npm --prefix client run build` | **Passed** | ESLint `0` errors; Vite built `2181` modules (`202ms`). |
| **Frontend Unit / Component** | `npm --prefix client test` (`Vitest v5.1.2`) | **4 passed, 0 failed** | `src/test/ingestionWorkflow.test.jsx` (`1.23s`). |
| **Real Browser E2E** | Headless Chrome `158.0.8127.88` CDP against Vite + Express + FastAPI + PG16 + Redis 7 + Celery | **8/8 flows passed** | `0` uncaught browser exceptions. |
| **Node Security & Concurrency** | `npm --prefix server test` (`hardening.security.test.js`) | **5 passed, 0 failed** | CORS, headers, JWT/RBAC/UUID, atomic refresh rotation & replay, concurrent uploads, traversal/symlink, `coverageiq_runtime` DDL block. |
| **Node Upload & Job API E2E** | `npm --prefix server test` (`ingestion.integration.test.js`) | **8 passed, 0 failed** | Full PG16 + Redis + FastAPI + Celery upload/dedupe/versioning/reconciler/rate-limit/frontend-contract lifecycle (`13/13` total Node tests). |
| **Python Full Regression Suite** | `~/ai-env/bin/python -m pytest ai_service/tests/ -v --tb=short` (Python `3.14.4`) | **308 passed, 0 failed, 0 skipped** | `244.74s` (`4m04s`) across all 24 test modules (PG16+pgvector, Celery+Redis+PG, internal enqueue, persistent retrieval, cached BGE-M3 CUDA, PDF/table structurer, semantic chunker, OCR adapters, Nemotron/evaluator, config & hardening). See exact per-file counts below. |
| **Python Packaging Health** | `~/ai-env/bin/python -m pip check` | **Passed** | `No broken requirements found.` |
| **Python 3.11 Compatibility** | Check `/usr/bin/python3.11` | **Not Run (Blocked)** | Python 3.11 binary absent on host (`Python 3.14.4` only). |
| **Live PaddleOCR Inference** | `PaddleOCREngine` with real `paddleocr` runtime | **Not Run (Blocked)** | `paddleocr` / `paddlepaddle` absent from `~/ai-env` and `ai_service/requirements.txt`. |

### Exact Observed Python Per-File Counts (`pytest` Log Breakdown, Total = `308`)

- `test_api_endpoints.py`: `3`
- `test_bge_m3_embeddings.py`: `21`
- `test_celery_ingestion_unit.py`: `13`
- `test_celery_redis_pg_integration.py`: `4`
- `test_db_unit.py`: `4`
- `test_document_structurer.py`: `33`
- `test_evaluator.py`: `4`
- `test_hybrid_retrieval.py`: `1`
- `test_internal_ingestion_enqueue.py`: `4`
- `test_nvidia_generation.py`: `29`
- `test_nvidia_nemotron_provider.py`: `20`
- `test_nvidia_verification.py`: `20`
- `test_ocr_engine.py`: `11`
- `test_pdf_loader.py`: `15`
- `test_persistent_hybrid_retrieval.py`: `4`
- `test_phase3_3_hardening.py`: `25`
- `test_phase3_hardening.py`: `31`
- `test_phase4_7_hardening.py`: `8`
- `test_phase4_config.py`: `17`
- `test_postgres_pgvector_integration.py`: `8`
- `test_query_analyzer.py`: `1`
- `test_reranker.py`: `1`
- `test_retry_loop.py`: `4`
- `test_semantic_chunker.py`: `27`

---

## 10. Files Modified & Created During Checkpoint 4.7

### Hardening & Bug Fixes
- [server/src/app.js](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/server/src/app.js): Replaced permissive `CORS_ORIGIN || true` with allowlist validator `getAllowedCorsOrigins()`, added API security headers, disabled `x-powered-by`.
- [server/src/models/user.model.js](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/server/src/models/user.model.js): Added transactional `consumeAndRotateRefreshToken()` with `FOR UPDATE` locking and token-reuse session family revocation.
- [server/src/controllers/user.controller.js](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/server/src/controllers/user.controller.js): Updated `refreshAccessToken` to atomic rotation with cookie cleanup on failure, added `UUID_RE` check in `updateUserPlatformRole`, added `path: '/'` to `COOKIE_OPTIONS`.
- [server/package.json](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/server/package.json): Updated `test` script with `--test-concurrency=1` so multi-file DB integration suites run deterministically without `TRUNCATE` races.
- [ai_service/main.py](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/ai_service/main.py): Replaced `allow_origins=["*"]` + `allow_credentials=True` with explicit `INTERNAL_CORS_ORIGINS` allowlist (default `[]`).
- [ai_service/embeddings/bge_m3.py](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/ai_service/embeddings/bge_m3.py): Added `BgeM3Embedder.close()` to offload weights to CPU, release model references, GC, and flush CUDA cache cleanly.
- [ai_service/workers/celery_app.py](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/ai_service/workers/celery_app.py): Invoked `self._embedder.close()` inside `WorkerResourceManager.shutdown()`.
- [ai_service/ingestion/document_structurer.py](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/ai_service/ingestion/document_structurer.py): Fixed `_extract_tables_and_mask_blocks` phantom sliver column stripping and header deduplication / cell geometry mapping.

### Hardening Tests & Documentation
- [server/tests/hardening.security.test.js](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/server/tests/hardening.security.test.js) (new): Node security, CORS, refresh-token race/replay, concurrent upload versioning, rollback cleanup, traversal/symlink, rate limit, and `coverageiq_runtime` DDL block tests.
- [ai_service/tests/test_phase4_7_hardening.py](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/ai_service/tests/test_phase4_7_hardening.py) (new): FastAPI CORS & token auth, PDF/OCR failure isolation, BGE-M3 strictness + real cached CUDA inference + `.close()`, concurrent Celery claiming, 768-dim rejection, and SHA-256 tamper failure tests.
- [ai_service/tests/test_document_structurer.py](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/ai_service/tests/test_document_structurer.py): Added two table column/empty-header alignment regression tests.
- [ai_service/tests/test_celery_redis_pg_integration.py](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/ai_service/tests/test_celery_redis_pg_integration.py): Invoked `bge_emb.close()` in `test_real_bge_m3_worker_ingestion_smoke_when_cached` teardown.
- [doc/checkpoint-4.7-hardening-report.md](file:///home/prithviraj/Mega%20Project/coverageIQ-LLM-based-insurance-policy-assistant/doc/checkpoint-4.7-hardening-report.md) (new): This Checkpoint 4.7 report.

---

## 11. Next Steps Before Production Deployment

1. Validate full `ai_service/` dependencies and `pytest ai_service/tests/` under a clean **Python 3.11** environment matching production containers.
2. Decide scanned-document OCR deployment strategy (`paddleocr` / `paddlepaddle` installation in OCR worker image or dedicated OCR container) and execute live scanned-PDF OCR verification.
3. Upgrade `server/src/middlewares/rateLimit.middleware.js` to Redis-backed rate limiting if scaling Node gateway replicas horizontally.
4. Set explicit `CORS_ORIGIN` (frontend origin) and optional `INTERNAL_CORS_ORIGINS` (internal only, typically unset) in deployment secrets/environment configuration before setting `NODE_ENV=production`.
