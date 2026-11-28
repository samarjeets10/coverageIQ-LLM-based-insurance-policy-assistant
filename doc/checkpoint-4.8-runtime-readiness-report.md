# CoverageIQ — Checkpoint 4.8 Runtime Compatibility & Production Infrastructure Readiness Report

- **Project:** CoverageIQ (Evidence-Grounded Insurance Policy Assistant)
- **Branch:** `prithviraj_features`
- **Review Principle:** `NO EVIDENCE = NO CLAIM`
- **Git State:** Uncommitted implementation & verification only (`0` staged changes, no commits or pushes executed; Checkpoints 1–4.7 preserved)

---

## 1. Executive Summary & Readiness Status

Checkpoint 4.8 resolved the three outstanding runtime and infrastructure prerequisites identified in Checkpoint 4.7 (`ENV-01`, `ENV-02`, `OPS-01`):

1. **Python 3.11 Runtime Validated (`ENV-01` Resolved):** Provisioned an isolated Python `3.11.18` virtual environment at `/home/prithviraj/ai-env-py311` using `uv 0.13.10` while preserving the original Python `3.14.4` environment (`~/ai-env`) untouched. All 154 installed packages pass `uv pip check`, CUDA (`torch 2.15.0+cu132`) is active, and all **308/308** AI service tests pass under Python `3.11.18`.
2. **PaddleOCR Live Inference Validated (`ENV-02` Resolved on CPU):** Installed `paddlepaddle==3.3.1` (CPU) and `paddleocr==2.10.0` in `/home/prithviraj/ai-env-py311`, executed real 3-page scanned PDF OCR inference with high confidence (`0.9989` / `0.9987` on text pages, safe empty handling on blank page), added `PaddleOCREngine.close()`, and wired OCR cleanup into `WorkerResourceManager.shutdown()`.
3. **Redis-Backed Atomic Rate Limiter & Multi-Replica Readiness (`OPS-01` Resolved):** Replaced the process-local `Map` in `server/src/middlewares/rateLimit.middleware.js` with an atomic Redis Lua sliding-window limiter backed by `server/src/config/redis.js`, added Redis health checks and production startup/shutdown lifecycle checks in `server/server.js`, and verified multi-instance limit sharing, `X-Forwarded-For` spoof resistance, TTL expiry, and fail-closed `503`/`500` behavior.

- **Overall Assessment:** **Ready for Staging & Multi-Instance Node.js Gateway Deployment** (using Redis-backed rate limiting, PostgreSQL 16 + pgvector, Celery + Redis 7 workers, and Python 3.11.18 runtime).
- **Documented Hardware/Packaging Caveat:** `paddlepaddle-gpu==2.6.2` (latest GPU wheel on PyPI for Python 3.11) is built against older CUDA 11.x/12.x toolchains and failed probe import on the host's CUDA 13.2 / driver `595.99.02` environment without `setuptools`; CPU PaddleOCR (`paddlepaddle==3.3.1` + `paddleocr==2.10.0`) is the validated OCR runtime path, while BGE-M3 embeddings run on CUDA (`batch_size=1`).

---

## 2. Host Environment & Verified Dependency Matrix

| Component | Measured Version / Spec | Notes |
| --- | --- | --- |
| **OS / Kernel** | Ubuntu `26.04.1 LTS` (`Linux 7.0.0-44-generic x86_64`) | Host workstation |
| **System Memory / GPU** | `14 GiB` RAM, NVIDIA GeForce RTX 3050 Laptop GPU (`4096 MiB` VRAM, Driver `595.99.02`) | CUDA active in PyTorch |
| **Python 3.11 Venv** | Python `3.11.18` (`/home/prithviraj/ai-env-py311`) | `uv pip check`: 154 packages, all compatible |
| **Python 3.14 Venv (Preserved)** | Python `3.14.4` (`~/ai-env`) | Untouched comparison baseline |
| **PyTorch / CUDA** | `torch 2.15.0+cu132` (`cuda: True`) | BGE-M3 GPU inference (`batch_size=1`) |
| **AI Service Core** | `fastapi 0.165.0`, `celery 5.7.0`, `psycopg 3.3.9`, `pgvector 0.5.1`, `FlagEmbedding 1.4.2`, `transformers 5.24.1`, `pymupdf 2.2`, `redis 8.2.0` | All imported and tested in Python 3.11 |
| **OCR Stack (CPU)** | `paddlepaddle 3.3.1`, `paddleocr 2.10.0` | Real scanned PDF OCR verified |
| **Node.js / Express** | Node `v22.22.1`, `redis ^6.4.0`, `pg ^8.23.1`, `node-pg-migrate ^9.0.0` | 15/15 integration & security tests pass |
| **Database / Broker** | PostgreSQL 16 (`pgvector/pgvector:pg16`), Redis 7 (`redis:7-alpine`) | Ephemeral Docker test instances |

---

## 3. Python 3.11 Runtime Validation

- Provisioned `/home/prithviraj/ai-env-py311` with Python `3.11.18` via `uv 0.13.10` (`~/.local/bin/uv`), leaving system Python and `~/ai-env` (Python `3.14.4`) unmodified.
- Installed `ai_service/requirements.txt` and OCR dependencies (`paddlepaddle==3.3.1`, `paddleocr==2.10.0`).
- `uv pip check --python /home/prithviraj/ai-env-py311/bin/python` passed across all **154** packages with zero conflicts.
- Executed full `pytest ai_service/tests/` sequentially against live PostgreSQL 16 (`coverageiq_runtime` / `coverageiq_migrator`) and Redis 7 on both Python `3.11.18` and Python `3.14.4`:
  - **Python `3.11.18`:** **308 passed, 0 failed, 0 errors** in `256.79s` (2 warnings: PyMuPDF `fitz` deprecation notice, Paddle `ccache` notice).
  - **Python `3.14.4` (Comparison):** **308 passed, 0 failed, 0 errors** in `238.56s` (1 warning).
- Note on GPU concurrency: running both Python 3.11 and Python 3.14 test suites concurrently exhausted the `4096 MiB` RTX 3050 VRAM during `test_bge_m3_real_cached_cuda_inference_batch_size_1` (expected hardware limit when two processes allocate BGE-M3 weights simultaneously); running sequentially passes cleanly on both runtimes.

---

## 4. PaddleOCR Live Inference & Resource Lifecycle

Real OCR inference was executed using `PaddleOCREngine` + `PyMuPDFLoader` in `/home/prithviraj/ai-env-py311/bin/python` on a 3-page rasterized PDF (2 text pages + 1 blank page):

- **Initialization:** `is_available() = True`, `initialize() = True` in `2355.8 ms`.
- **3-Page OCR Execution:** `load_document()` completed in `1142.1 ms`:
  - **Page 1:** `method=OCR`, `ok=True`, `boxes=2`, `mean_confidence=0.9989`, text: `"SECTION 2.1 IN-PATIENT HOSPITALIZATION\nHospitalization expenses for minimum 24 hours are covered."`
  - **Page 2:** `method=OCR`, `ok=True`, `boxes=2`, `mean_confidence=0.9987`, text: `"SECTION 4.1 COSMETIC SURGERY EXCLUSION\nCosmetic treatment is excluded unless reconstructive."`
  - **Page 3 (Blank):** `method=OCR`, `ok=True`, `boxes=0`, `mean_confidence=None`, `text=''` (safe empty handling without errors).
- **Memory Footprint (RSS):** `82.99 MB` baseline -> `1122.5 MB` after PaddleOCR init -> `2061.66 MB` after 3-page OCR.
- **Lifecycle Cleanup:** Added `PaddleOCREngine.close()` in `ai_service/ingestion/ocr_engine.py` and invoked it during `WorkerResourceManager.shutdown()` in `ai_service/workers/celery_app.py`.
- **GPU Probe Finding:** Tested `paddlepaddle-gpu==2.6.2` in a disposable venv `/tmp/ciq-paddle-gpu-probe` (subsequently deleted); import failed (`ModuleNotFoundError: No module named 'setuptools'`) and `2.6.2` targets CUDA 11.x/12.x rather than CUDA 13.2. CPU PaddleOCR remains the recommended production OCR engine configuration on this stack.

---

## 5. Redis-Backed Rate Limiter & Server Production Hardening

Replaced process-local rate-limit state with shared Redis coordination (`redis@^6.4.0`):

- **`server/src/config/redis.js`:** Managed Redis client lifecycle (`getConnectedRedisClient()`, `checkRedisConnection()`, `closeRedisClients()`, `withTimeout()`, `sanitizeRedisUrl()`).
- **`server/src/middlewares/rateLimit.middleware.js`:**
  - Atomic Lua sliding-window script (`ZRANGEBYSCORE` / `ZADD` / `PEXPIRE`) using server-side `redis.call('TIME')` and unique request member IDs (`Date.now():crypto.randomUUID()`).
  - Sets `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `Retry-After` headers.
  - **Fail-Closed Guarantees:** Returns `503 RATE_LIMIT_UNAVAILABLE` if Redis is unreachable/times out, and `500 RATE_LIMIT_CONFIG_INVALID` on invalid configuration (e.g., `max <= 0`).
  - **Identity & Spoof Resistance:** Uses `req.user.id` (JWT-authenticated) or proxy-aware `req.ip`, ignoring raw `X-Forwarded-For` headers when `TRUST_PROXY` is configured safely (`resolveTrustProxySetting()` in `server/src/app.js` rejects insecure `TRUST_PROXY=true`).
  - Exposes `createRateLimiter()` and `reset()` (`SCAN` + `DEL`).
- **`server/src/routes/health.routes.js` & `server/server.js`:**
  - `/api/health` checks both PostgreSQL and Redis (`services.database`, `services.redis`).
  - Production startup validates required secrets/configuration (`ACCESS_TOKEN_SECRET`, `REFRESH_TOKEN_SECRET`, `INTERNAL_SERVICE_TOKEN`, `CORS_ORIGIN`, `DATABASE_URL`, `REDIS_URL`, `POLICY_UPLOAD_ROOT`), enforces distinct JWT secrets, checks PG + Redis connectivity at boot, and closes Redis + PG pools on graceful shutdown (with 10s timeout guard).
- **Automated Verification (`server/tests/hardening.security.test.js`):**
  - Verified two independent Express app instances sharing one Redis prefix allow exactly `3/6` concurrent requests, set `pTTL`, resist `X-Forwarded-For` spoofing, return `503` on Redis outage, and return `500` on invalid config.

---

## 6. Shared Storage Cross-Runtime Verification

- Added cross-runtime storage test in `server/tests/hardening.security.test.js` (`verifies shared storage permissions and cross-runtime readability between Node gateway and Python worker storage`).
- Confirmed Node.js gateway upload to `POLICY_UPLOAD_ROOT` is immediately readable by Python `LocalFilesystemDocumentStorage.resolve_for_reading()` with identical SHA-256 digest and byte size.
- **Deployment Requirement:** In multi-node deployments where Node.js gateway and Celery workers run on separate hosts, `POLICY_UPLOAD_ROOT` must be backed by a shared POSIX filesystem (e.g., NFS/EFS) with matching UID/GID permissions, or migrated to object storage in a future checkpoint.

---

## 7. Measured Performance & Resource Benchmarks

All measurements collected directly on the test host (`Python 3.11.18`, `RTX 3050 4GB`, Docker `pgvector/pgvector:pg16`, Docker `redis:7-alpine`):

| Operation | Measured Result | Resource / Notes |
| --- | --- | --- |
| **BGE-M3 Model Init + Warmup (`cuda`, `batch_size=1`)** | `6462.0 ms` | VRAM: `0.0 MB` -> `1085.04 MB` allocated (`1090.0 MB` reserved) |
| **BGE-M3 Embed 2 Texts (`cuda`, `batch_size=1`)** | `21.4 ms` | `1024` dimensions; VRAM `1085.04 MB` |
| **BGE-M3 `close()` + `empty_cache()`** | — | VRAM freed back to `0.0 MB` allocated (`0.0 MB` reserved) |
| **PaddleOCR Init (CPU)** | `2355.8 ms` | Process RSS: `82.99 MB` -> `1122.5 MB` |
| **PaddleOCR 3-Page Scanned PDF Load + OCR** | `1142.1 ms` | Process RSS: `1122.5 MB` -> `2061.66 MB` |
| **Digital PDF Parse + Structure + Chunk (1 page)** | `2771.0 ms` | Includes first-call BGE-M3 tokenizer load; `1` chunk |
| **Redis `PING` Latency** | `2.10 ms` first, `0.17 ms` avg (10 calls) | Local Docker `redis:7-alpine` |
| **PostgreSQL Pool Open / `SELECT 1`** | `15.42 ms` open / `0.35 ms` acquire+query | `PostgresPoolManager` (`coverageiq_runtime`) |

---

## 8. End-to-End Regression Summary (Post-Final-Edit)

| Suite | Command / Target | Result |
| --- | --- | --- |
| **Python 3.11 AI Service** | `/home/prithviraj/ai-env-py311/bin/python -m pytest ai_service/tests/` | **308 passed**, 0 failed (`256.79s`) |
| **Python 3.14 AI Service (Comparison)** | `~/ai-env/bin/python -m pytest ai_service/tests/` | **308 passed**, 0 failed (`238.56s`) |
| **Node.js Security & Integration** | `npm --prefix server test` (`AI_PYTHON_BIN=/home/prithviraj/ai-env-py311/bin/python`) | **15 passed**, 0 failed (`24.73s`: 7/7 hardening + 8/8 ingestion) |
| **Frontend Lint / Unit / Build** | `npm --prefix client run lint && npm --prefix client test && npm --prefix client run build` | **0 lint errors**, **4/4 Vitest passed** (`1.74s`), Vite build **275ms** (`2181` modules) |
| **Headless Chrome CDP E2E** | `browser_e2e_cdp.mjs` (Python 3.11 FastAPI + Celery + Redis 7 + PG16 + Express + Vite) | **8/8 steps passed**, `0` uncaught browser exceptions |

---

## 9. Modified & Added Files (Checkpoint 4.8)

- **New:** `server/src/config/redis.js`
- **New:** `doc/checkpoint-4.8-runtime-readiness-report.md`
- **Modified:**
  - `server/src/middlewares/rateLimit.middleware.js`
  - `server/src/routes/health.routes.js`
  - `server/server.js`
  - `server/src/app.js`
  - `server/package.json`, `server/package-lock.json`
  - `server/.env.example`, `ai_service/.env.example`
  - `server/tests/hardening.security.test.js`
  - `server/tests/ingestion.integration.test.js`
  - `ai_service/ingestion/ocr_engine.py`
  - `ai_service/workers/celery_app.py`
