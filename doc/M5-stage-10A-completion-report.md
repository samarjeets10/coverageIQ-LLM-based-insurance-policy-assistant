# CoverageIQ — Module M5 (Policy Recommendation) Stage 10A Completion & Verification Report

**Project:** CoverageIQ — LLM-Based Insurance Policy Assistant  
**Module:** M5 — Policy Recommendation (`6A`)  
**Developer:** Shravan Sambhaji Kapare (Roll No. 203)  
**Git Branch:** `shravan/policy-recommendation` (base `origin/main@261896e`, latest commit `57f1676`)  
**Stage Scope:** Read-only repository/integration/DB/cross-service verification + safe M5-owned completion & hardening (**no commits, no push/merge/PR, no migration 003 execution, no edits to shared Node/M2/M3/M4/M8 code**)

---

## 1. Executive Summary

Stage 10A audited commit `57f1676`, `M5-INTEGRATION-REPORT.md`, prior Stage 8A–9B-1 reports, live read-only PostgreSQL `18.6`, and all peer module branches/codebases (M2 on `origin/prithviraj_features`, M3/M7/M8 in `ai_service/`, and Node `server/`).

**Key Conclusions:**
- **M5 Core Recommendation Engine:** All M5-owned pipeline responsibilities—deterministic hard eligibility filtering across all 4 categories (`health`, `car`, `life`, `term`), frozen 59-column feature engineering (`FEATURE_SCHEMA_VERSION = "1.0.0"`), synthetic LightGBM LambdaRank training/inference (`lightgbm_ranker.joblib` + warm-up), explicit opt-in transparent weighted fallback (`ranking_model="weighted"`), fail-closed non-dev `X-Service-Token` auth, structured JSON PII-scrubbed logging, version-aware catalog SQL (`policies.current_version_id`), and frozen OpenAPI contract (`docs/contracts/ai-service.openapi.yaml`)—are **VERIFIED** / **IMPLEMENTED**.
- **Audit Catches & Fixes Implemented in Stage 10A (M5-owned only):**
  1. **Ruff False-Pass Configuration & Formatting Drift Fixed:** Running `ruff` from the repository root matched `0` files due to relative `include` globs in `ai_service/pyproject.toml`; inside `ai_service/`, `3` test files needed reformatting and `12` `I001` import-sorting errors fired because `known-first-party = ["ai_service"]` was absent. Fixed `ai_service/pyproject.toml`, formatted all files, and documented the `cd ai_service` Ruff invocation in `ai_service/README.md`.
  2. **Real-Catalog Hardening in `filters.py` & `features.py`:** Fixed non-numeric/non-finite (`"NaN"`, `"Inf"`, `"unknown"`) and `bool` attribute parsing so a single malformed catalog policy fails closed cleanly (`MISSING_MANDATORY_POLICY_DATA` / `INVALID_POLICY_METADATA` / category rules) instead of bypassing filters or raising `HTTP 500` across valid catalog rows; resolved `TENURE_COMPATIBILITY` conflict against `003_recommendation_schema.sql` DDL defaults (`1..3`) when Life/Term `type_specific_metadata` supplies `policy_term_years_min/max`; fixed falsy `0.0 or ...` checks for `life_cover_min/max` and `policy_term_years_max`; enforced `DuplicateCatalogPolicyError` inside pure `filter_policies()`; aligned `filter_policies_from_db()` to push `policy_type` to SQL.
  3. **API Lineage, Citation Validation & Traceability (`schemas.py`, `citations.py`, `service.py`, `router.py`, `common/logging.py`):** Added additive optional `policy_version_id` on `RecommendedPolicy`, typed `CitationItem`/`dict` validation (`chunk_id`, `page >= 1`, `clause`) + `OSError`/`TimeoutError` isolation in citation retrieval, normalized `TrustedScope.role` case while still rejecting invalid roles (`422`), echoed `X-Request-ID` on `401`/`500`/`501`/`503` errors and `auth.rejected`/`auth.misconfigured` events, and added `"scope"`, `"monthly_budget_inr"`, and `"monthly_budget"` to log PII scrubbing.
- **External / Unblocked Production Gaps (`BLOCKED` / `NOT IMPLEMENTED` outside M5):**
  - Live PostgreSQL (`coverageiq_db`) still has only `001` & `002` applied (`0` catalog rows, `policy_attributes = False`, `ranking_features = False`).
  - M4 Catalog CRUD, M2 `attribute_extractor` for `policy_attributes`, M3 PostgreSQL/pgvector persistence & retrieval for citations, M8 recommendation explanation synthesis, and Node↔Python authenticated proxy routes do **not** exist in the repository yet.
  - Model weights remain trained exclusively on synthetic rule-generated data (`80` policies, `60` profiles); real-world accuracy is **not validated**.

---

## 2. Verified Completed Responsibilities

Every item below carries an explicit verification classification:

### Phase A — Repository & Integration Commit Verification
- **Branch & Commit Identity (`VERIFIED`):** Repository root `I:\Mega Project\coverageIQ-LLM-based-insurance-policy-assistant` is on branch `shravan/policy-recommendation`, 1 commit ahead of `origin/main@261896e`, at commit `57f1676` (`feat(m5): integrate policy recommendation engine into unified ai_service`, 54 files changed, `+20,406 / -184` lines).
- **Namespace Refactoring Discrepancy Note (`VERIFIED`):** Prompt-referenced `ai_service/config/database.py` and `ai_service/config/settings.py` were refactored into [`ai_service/recommendation/db.py`](../ai_service/recommendation/db.py), [`ai_service/recommendation/settings.py`](../ai_service/recommendation/settings.py), and [`ai_service/recommendation/openapi_export.py`](../ai_service/recommendation/openapi_export.py) in commit `57f1676` to avoid colliding with upstream [`ai_service/config.py`](../ai_service/config.py).
- **Upstream Preservation (`VERIFIED`):** Upstream `POST /ai/ask`, M3 retrieval/reranker/evaluator/retry modules, `ai_service/config.py`, `ai_service/database.py`, `ai_service/__init__.py`, and all 14 upstream Q&A tests pass unmodified.

### Phase B — M5 Implementation Audit

#### A. Hard Filtering (`ai_service/recommendation/filters.py`)
- **All 4 Insurance Categories (`VERIFIED`):** Supports `'health'`, `'car'`, `'life'`, `'term'`.
- **Mandatory & Domain Eligibility Rules (`VERIFIED`):** Covers all 16 Architecture §3.4 dimensions plus catalog safety gates (`MISSING_MANDATORY_POLICY_DATA`, `INVALID_POLICY_METADATA`, `POLICY_ACTIVE_STATUS`, `CATALOG_ELIGIBLE`, `POLICY_TYPE_MATCH`, `AGE_MIN/MAX`, `RENEWAL_AGE_MAX`, `LIFELONG_RENEWAL`, `APPLICANT_TYPE`, `BUDGET_MIN/MAX`, `SUM_INSURED_MIN/MAX`, `COVERAGE_MODE`, `FAMILY_SIZE_MAX`, `GEOGRAPHIC_AVAILABILITY`, `TENURE_COMPATIBILITY`, `REQUIRED_BENEFITS`, `REQUIRED_BENEFITS_EXCLUDED`, `COPAY_MAX`, `DEDUCTIBLE_MAX`, `CASHLESS_REQUIRED`, `HOSPITAL_NETWORK_MIN`, `INITIAL_WAITING_PERIOD`, `PED_COVERAGE`, `PED_WAITING_PERIOD`, `ROOM_RENT_LIMIT`, `CAR_*`, `LIFE_*`, `TERM_*`).
- **Missing & Malformed Metadata Fail-Closed Behavior (`VERIFIED` + `IMPLEMENTED`):** Missing mandatory fields, unverified capability flags (`ped_covered=False`, `cashless_available=False`, empty `vehicle_types`, unflagged `smoking_status_eligibility`), non-numeric strings, and `"NaN"`/`"Inf"` values fail closed per policy without crashing valid candidates.
- **Elimination Reasons & Zero-Match Behavior (`VERIFIED`):** Produces `FilterReport` with per-policy `PolicyFilterEvaluation` and `FilterEliminationReason`; verified zero-match profiles (`PROF-HEA-015`, `PROF-CAR-015`, `PROF-LIF-015`, `PROF-TER-015`) return `NO_MATCHING_POLICIES` and skip Stage 5 feature construction and Stage 6/weighted scoring.
- **Current Policy-Version Filtering (`VERIFIED`):** Parameterized `build_catalog_policy_attributes_query()` enforces `organization_id IS NOT NULL`, `owner_user_id IS NULL`, `is_catalog_eligible IS TRUE`, and `(policy_attributes.policy_version_id == policies.current_version_id OR (current_version_id IS NULL AND policy_version_id IS NULL))`.
- **Duplicate Policy Handling (`VERIFIED` + `IMPLEMENTED`):** Both `load_catalog_policies_from_db()` and direct `filter_policies()` now raise `DuplicateCatalogPolicyError` (`CatalogIntegrityError` -> `HTTP 500`) on duplicate `policy_id` entries.

#### B. Feature Engineering (`ai_service/recommendation/features.py` & `FEATURES.md`)
- **All 59 Features & Schema Version (`VERIFIED`):** `len(FEATURE_COLUMNS) == 59`, `FEATURE_SCHEMA_VERSION == "1.0.0"`, `float64` finite matrix (`36` policy-attribute + `7` user-profile + `16` interaction signals).
- **Training/Inference Parity & Leakage Prevention (`VERIFIED`):** Both offline training (`build_training_feature_dataset`) and online inference (`build_feature_matrix`) call shared `_normalize_profile_fields`, `_normalize_policy_fields`, and `_extract_from_normalized`; `FORBIDDEN_FEATURE_COLUMNS` strictly excludes targets (`suitability_score`, `relevance_label`), filter artifacts, and IDs.
- **Missing Values & Category Isolation (`VERIFIED`):** Optional missingness uses `0.0` + explicit indicators (`has_annual_income`, `has_claim_settlement_ratio`, `has_vehicle_age`); Health (`8`), Car (`8`), and Life/Term (`10`) features zero cleanly outside their categories.

#### C. Machine Learning (`ai_service/recommendation/train_lightgbm.py` & `artifacts/models/`)
- **LightGBM LambdaRank & Group-Aware Splitting (`VERIFIED`):** `GroupShuffleSplit` and 5-fold `GroupKFold` group strictly by `profile_id` with zero profile overlap across Train (`461` rows, `38` groups), Validation (`135` rows, `9` groups), and Test (`124` rows, `9` groups); `ranking_model="auto"` / `"lightgbm"` uses the single loaded `LGBMRanker` (`best_iteration = 24`).
- **Model Artifact Compatibility, Loading & Warm-Up (`VERIFIED`):** `load_model_state()` validates `feature_schema_version == "1.0.0"`, exact 59-column ordering, artifact integrity (missing/corrupt/unfitted), and executes `_warm_up_lightgbm_model()` once at FastAPI startup lifespan.
- **Evaluation Metrics & Baselines — Synthetic Data Only (`VERIFIED`):** Test NDCG@3 = `1.0000`, NDCG@5 = `0.9917` vs Random (`0.7285` / `0.7248`), Cheapest (`0.7233` / `0.7389`), 5-fold CV mean NDCG@3 = `0.9946 ± 0.0053`. **Important limitation:** All relevance grades `{1, 2, 3}` come from a synthetic suitability heuristic; these metrics verify ranking mechanics only and **do not** demonstrate real-world insurance recommendation accuracy.
- **XGBoost Status (`VERIFIED` — Intentionally Not Implemented):** Checked `doc/project-architecture.md` (§3.4, §5), `doc/modules-features.md` (M5), and `doc/roadmap.md` (Phase 5 & Scope Control). Every specification uses `"LightGBM / XGBoost"` as a slash alternative (with weighted scorer mandatory first and ML optional/stretch), never requiring XGBoost alongside LightGBM. Consistently, `ranking_model="xgboost"` returns controlled `HTTP 501`.

#### D. Recommendation API (`router.py`, `service.py`, `schemas.py`, `main.py`, `citations.py`)
- **Dual Routes & Frozen Contract (`VERIFIED`):** `POST /api/v1/recommend`, `POST /ai/recommend`, `GET /health`, `GET /ai/health`, plus upstream `POST /ai/ask` are synchronized with [`docs/contracts/ai-service.openapi.yaml`](../docs/contracts/ai-service.openapi.yaml) (`verify_openapi_contract()` = PASS).
- **Auth, Validation, Limits & Errors (`VERIFIED` + `IMPLEMENTED`):** Non-dev fail-closed `X-Service-Token` (`401`), constant-time `hmac.compare_digest`, `RequestBodyLimitMiddleware` (`413`), NaN-safe `RequestValidationError` (`422`), `UnsupportedRankingModelError` (`501`), `CatalogIntegrityError` (`500`), missing table/outage/model unavailability (`503`), `X-Request-ID` propagation, 6-stage `latency_ms` logging, and `NullCitationProvider` (`citations=[]`) with validated M3/M8 interface seam.

#### E. Weighted Scoring Fallback (`ai_service/recommendation/weighted.py`)
- **Transparent Deterministic Phase-A Scorer (`VERIFIED`):** `WeightedDimensionWeights` validates finite non-negative weights summing to `1.0` (`low_premium`, `balanced`, `comprehensive_coverage`); operates strictly over the 59-column Stage 5 feature matrix with zero weight on `annual_income`; never silently substitutes for failed LightGBM (`"auto"` / `"lightgbm"` fail `503`; only explicit `ranking_model="weighted"` runs weighted scoring).

---

## 3. Remaining Incomplete Responsibilities

| Responsibility | Classification | Explanation |
|---|---|---|
| Apply database migration `003_recommendation_schema.sql` (`policy_attributes`, `ranking_features`) | **BLOCKED** | Migration SQL exists at `server/migrations/003_recommendation_schema.sql`; cannot be applied without Node/DB owner review (`Risk R8`). |
| Real catalog data seeding across Health, Car, Life, Term | **BLOCKED / NOT IMPLEMENTED** | `coverageiq_db` `policies` table has `0` rows; M4 catalog CRUD routes and M2 attribute extraction do not exist yet. |
| M2 structured `attribute_extractor` populating `policy_attributes` | **NOT IMPLEMENTED** (on M2 branch) | Checked `origin/prithviraj_features`: M2 implements PDF/OCR/structuring/chunking only; zero attribute extraction or DB writers exist. |
| M5 -> M3 -> M8 grounded explanation & chunk citations | **BLOCKED** | Interface seam (`RecommendationCitationProvider`) + Pydantic/dict validation are ready, but M3 has no PostgreSQL/pgvector chunk storage (`chunks` table absent) and M8 has no recommendation explainer. |
| Node.js gateway proxy (`POST /recommendations` -> `POST /ai/recommend`) + M1 RBAC/JWT + `X-Service-Token` | **NOT IMPLEMENTED** | Node `server/` currently only has health check, migrations runner (`001`/`002`), and incomplete broken `auth.routes.js`/`user.controller.js`. |
| M12 real-world gold-set evaluation (10 real policies, real relevance labels, calibration of CSR/PED missingness) | **BLOCKED** | Requires real extracted policy records and human relevance judgments. |

---

## 4. Implementation Changes Made in Stage 10A

All changes stay strictly within M5-owned Python files, configuration, tests, OpenAPI contract, and M5 documentation:

1. **`ai_service/recommendation/filters.py`:**
   - Hardened `_is_missing`, `_to_optional_int`, and `_to_optional_float` to reject `bool`, handle non-numeric strings cleanly, and reject non-finite values (`"NaN"`, `"Inf"`).
   - Hardened `_check_mandatory_policy_fields` to parse required numeric attributes cleanly and reject negative/inverted entry ages, premiums, or non-positive sum insured (`MISSING_MANDATORY_POLICY_DATA` / `INVALID_POLICY_METADATA`) fail-closed per candidate policy.
   - Aligned `TENURE_COMPATIBILITY` for `life`/`term` policies to check `type_specific_metadata.policy_term_years_min/max` before falling back to `003` DDL default `min_tenure_years`/`max_tenure_years`.
   - Replaced truthy `or` with explicit `is None` checks for `life_cover_min` / `life_cover_max`.
   - Enforced `DuplicateCatalogPolicyError` inside pure `filter_policies()`.
   - Passed `policy_type=normalized_profile.policy_type` pushdown in `filter_policies_from_db()`.
2. **`ai_service/recommendation/features.py`:**
   - Replaced truthy `or` with explicit `is not None` checks for `life_cover_max`, `policy_term_years_max`, and `premium_payment_term_years`.
3. **`ai_service/recommendation/schemas.py`:**
   - Added additive optional `policy_version_id: str | None = None` to `RecommendedPolicy`.
   - Added `_normalize_scope_role` on `TrustedScope` to normalize whitespace/case before `Literal` validation.
4. **`ai_service/recommendation/citations.py` & `ai_service/recommendation/service.py`:**
   - Added `normalize_and_validate_citation()` supporting `dict` and Pydantic models (`CitationItem`-compatible) while enforcing `chunk_id`, `page >= 1`, and `clause`.
   - Populated `RecommendedPolicy.policy_version_id` from retrieved catalog rows.
   - Caught `DuplicateCatalogPolicyError` explicitly around `filter_policies()` -> `CatalogIntegrityError`, and added `OSError`/`TimeoutError` handling around `citation_provider.get_citations()`.
5. **`ai_service/recommendation/router.py` & `ai_service/common/logging.py`:**
   - Propagated `X-Request-ID` onto `401`, `500`, `501`, and `503` `HTTPException` headers and `auth.rejected` / `auth.misconfigured` structured log events (without logging tokens).
   - Added `"scope"`, `"monthly_budget_inr"`, and `"monthly_budget"` to `_PII_KEY_NAMES` in `common/logging.py`.
6. **`ai_service/pyproject.toml`, `ai_service/README.md`, `docs/contracts/ai-service.openapi.yaml`, & formatting:**
   - Added `[tool.ruff.lint.isort] known-first-party = ["ai_service"]` and registered `tests/test_stage10a_m5_completion.py`.
   - Formatted existing M5 test files cleanly via Ruff and regenerated `docs/contracts/ai-service.openapi.yaml`.
7. **`ai_service/tests/test_stage10a_m5_completion.py`:**
   - Added `6` integration, resilience, citation, security, and 10-profile batch sanity tests.

---

## 5. Files Created and Modified (Uncommitted Working Tree per Stage 10A Rules)

| File | Action | Owner Scope |
|---|---|---|
| `ai_service/recommendation/filters.py` | Modified | M5 |
| `ai_service/recommendation/features.py` | Modified | M5 |
| `ai_service/recommendation/schemas.py` | Modified | M5 |
| `ai_service/recommendation/citations.py` | Modified | M5 |
| `ai_service/recommendation/service.py` | Modified | M5 |
| `ai_service/recommendation/router.py` | Modified | M5 |
| `ai_service/common/logging.py` | Modified | M5 / shared logging |
| `ai_service/pyproject.toml` | Modified | M5 build/lint config |
| `ai_service/README.md` | Modified | Docs |
| `docs/contracts/ai-service.openapi.yaml` | Regenerated | M5/Node contract |
| `ai_service/tests/test_inference.py` | Formatted | M5 tests |
| `ai_service/tests/test_phase9b1_hardening.py` | Formatted | M5 tests |
| `ai_service/tests/test_stage8b_versioning_and_schema.py` | Formatted | M5 tests |
| `ai_service/tests/test_stage10a_m5_completion.py` | Created | Stage 10A M5 tests |
| `doc/M5-stage-10A-completion-report.md` | Created | Stage 10A deliverable |

---

## 6. Database Integration Status

Verified via direct read-only inspection against live `coverageiq_db` (`PostgreSQL 18.6`):

| Object / Check | Status | Details |
|---|---|---|
| PostgreSQL Engine & Extensions | **VERIFIED** | PostgreSQL `18.6`; `plpgsql` `1.0` and `vector` `0.8.7` installed (`vector` remains out-of-band, absent from `001`/`002`). |
| Applied Migrations (`schema_migrations`) | **VERIFIED** | `001_initial_schema.sql` and `002_add_platform_role.sql` applied; `003_recommendation_schema.sql` **UNAPPLIED**. |
| `policy_attributes` & `ranking_features` in Live DB | **NOT IMPLEMENTED (unmigrated)** | `tables_present["policy_attributes"] == False`, `tables_present["ranking_features"] == False`, `recommendation_catalog_ready == False`. |
| Migration File & SQLAlchemy Alignment (`common/tables.py` vs `003`) | **VERIFIED** | Columns, defaults, constraints (`UNIQUE NULLS NOT DISTINCT`, composite FK to `policy_versions(id, policy_id)`), and PG DDL compilation align cleanly. |
| Version-Aware Catalog Query (`filters.py`) | **VERIFIED** | SQL compiles cleanly and joins `policy_version_id == policies.current_version_id` (plus `NULL` fallback); tested behaviorally on multi-version SQLite fixtures. |
| Live Catalog Records & Fail-Closed DB Path | **VERIFIED (0 rows / safe 503)** | All `12` existing PostgreSQL tables hold `0` rows (`policies = 0`); live `load_catalog_policies_from_db()` fails closed cleanly with `PolicyAttributesTableUnavailableError` -> `HTTP 503`. |
| Readiness for Real Catalog Records | **IMPLEMENTED (Schema Ready / Pending 003 Execution)** | Python catalog loader, Decimal/UUID handling, fail-closed bounds checking, and version matching are ready once `003` is applied through DB governance. |

---

## 7. M2 / M3 / M4 / M8 Cross-Service Dependencies

| Module | Inspection Result | Status vs M5 |
|---|---|---|
| **M2 (Ingestion Pipeline)** | Inspected `remotes/origin/prithviraj_features` (`7d5bc47`, `517a7dd`, unmerged to `main`): contains PDF validator/loader, OCR, document structurer, semantic chunker (`PolicyChunk`). Missing: Celery worker, `POST /ai/ingest`, DB persistence, and `attribute_extractor` for `policy_attributes` (`policies.policy_type` normalization & `type_specific_metadata` population). | **BLOCKED / NOT IMPLEMENTED** |
| **M3 (Knowledge Base & Retrieval)** | Current branch has in-memory `MockKnowledgeBase` (only 1 hardcoded policy `policy_health_01`), BM25, 768-dim dense retriever (vs Architecture §5 BGE-M3 1024-dim spec — **unresolved**), RRF, and CrossEncoder reranker. Missing: PostgreSQL/pgvector `sections`/`clauses`/`chunks` tables, `hybrid_search(policy_id, ...)` over DB, and `RecommendationCitationProvider` adapter. | **BLOCKED / NOT IMPLEMENTED** |
| **M4 (Catalog Management)** | Zero `/org/policies` or `/admin/policies` routes/controllers exist in `server/`; `policies` table has `0` rows. | **NOT IMPLEMENTED / BLOCKED** |
| **M8 (Grounded LLM & Evaluator)** | Current branch handles Q&A (`POST /ai/ask`) via `ClosedRetryLoopOrchestrator` + `EvaluatorAgent`. Missing: M5 "why recommended" explanation & claim-verification hook wired to `RecommendationCitationProvider`. | **BLOCKED** |

---

## 8. Node Integration Status

Inspected [`server/`](../server/):
- **Implemented (`VERIFIED`):** PostgreSQL pool (`server/src/config/db.js`), sequential migration runner (`server/scripts/migrate.js`), migrations `001` & `002`, Express app scaffold (`server/src/app.js`), and `GET /api/health`.
- **Broken Upstream Code (Left Untouched per Rules):** `server/src/controllers/user.controller.js` line 10 contains an incomplete statement (`const user = await UserModel.`), and `server/src/routes/auth.routes.js` line 17 misassigns `router.route = ('./logout', userLogout)`.
- **M5 Proxy Integration (`NOT IMPLEMENTED / BLOCKED`):** No `/api/v1/recommendations` proxy route, no M1 JWT verification / RBAC `requireRole` middleware constructing `TrustedScope`, and no `X-Service-Token` / `X-Request-ID` HTTP client calling Python `/api/v1/recommend` or `/ai/recommend`.
- **Contract Readiness (`VERIFIED`):** Frozen OpenAPI contract at [`docs/contracts/ai-service.openapi.yaml`](../docs/contracts/ai-service.openapi.yaml) provides exact request/response/auth/error schemas (`ServiceTokenAuth`, `TrustedScope`, `RecommendRequest`, `RecommendResponse`, `AIHealthResponse`, `ErrorDetailResponse`) for Node integration.

---

## 9. Security & Reliability Findings

1. **Non-Dev Internal Authentication (`VERIFIED`):** Outside `{development, dev, local, test, testing}`, startup rejects missing/blank `AI_SERVICE_TOKEN` or `ai_service_require_auth=False`, and `verify_service_token()` enforces constant-time `hmac.compare_digest` (`HTTP 401`) on both M5 routes.
2. **Upstream `/ai/ask` Authentication Gap (`VERIFIED` — Cross-Team Finding):** Upstream `ai_service/api/routes_qa.py` does **not** attach `verify_service_token` (left untouched per Stage 10A rule not to modify M3/M7/M8 code; requires M7/M8 owner review before public production exposure behind nginx/Node).
3. **Payload & DoS Bounds (`VERIFIED`):** `RequestBodyLimitMiddleware` caps body size (`64 KiB` default -> `HTTP 413`), tag lists capped at `50` items (`<= 100` chars) prior to deduplication, `allow_inf_nan=False`, and NaN-safe `422` validation handler.
4. **Traceability & Privacy (`IMPLEMENTED` / `VERIFIED`):** `X-Request-ID` propagated on responses and operational error responses (`401`/`500`/`501`/`503`); structured JSON logs scrub passwords, DSN credentials, `token`, `profile`, `scope`, `user_id`, `owner_scope`, income, budgets, age, medical conditions, location, and smoking status (`[REDACTED]`).
5. **DB & Catalog Resilience (`IMPLEMENTED` / `VERIFIED`):** Parameterized SQLAlchemy Core queries with `connect_timeout`, `pool_timeout`, and libpq `statement_timeout`; per-policy malformed attribute isolation prevents single broken rows from crashing valid catalog results; duplicate IDs and DB outages map distinctly (`500` vs `503`).

---

## 10. Test Results (Directly Observed After Final Edit)

| Check | Exact Command | Exit Code | Result | Notes / Skipped / External Service |
|---|---|---|---|---|
| Ruff Format Check | `cd ai_service ; <clean_venv>\python.exe -m ruff format --check .` | `0` | **PASS** (`31 files already formatted`) | Covered all M5 source + test files in `pyproject.toml` |
| Ruff Lint Check | `cd ai_service ; <clean_venv>\python.exe -m ruff check .` | `0` | **PASS** (`All checks passed!`) | Uses `[tool.ruff.lint.isort] known-first-party = ["ai_service"]` |
| OpenAPI Contract Check | `<clean_venv>\python.exe -c "from ai_service.recommendation.openapi_export import verify_openapi_contract; verify_openapi_contract(); print('CONTRACT_VERIFY: PASS')"` | `0` | **PASS** (`CONTRACT_VERIFY: PASS`) | Live `create_app().openapi()` matches `docs/contracts/ai-service.openapi.yaml` strictly |
| Pytest Run 1 (No DB Env Vars) | `<clean_venv>\python.exe -m pytest ai_service/tests/ -v` | `0` | **150 passed, 3 skipped, 0 failed** (`74.29s`) | Skipped the 3 live PostgreSQL tests (`DB_PASSWORD` unset in default shell env) |
| Pytest Run 2 (Live Read-Only PG via `DB_*` from `server/.env`) | `<clean_venv>\python.exe -m pytest ai_service/tests/ -v` *(with `DB_*` env loaded in process)* | `0` | **153 passed, 0 skipped, 0 failed** (`34.69s`) | Executed all unit + integration + all **3** live read-only PostgreSQL 18.6 tests (`statement_timeout`, table absence -> `503`, `001`/`002` catalog query) |

*(Live PG tests connect to local PostgreSQL `18.6` `coverageiq_db` read-only using `DB_*` credentials from `server/.env`; in environments without PostgreSQL running, those 3 tests skip cleanly via `pytest.skip`.)*

---

## 11. Performance Evaluation Limitations

1. **In-Memory Mocked DB Timing Only:** Both Stage 9A's `500`-candidate micro-benchmark (`~208.6 ms` LightGBM / `~189.5 ms` weighted median) and Stage 10A's `10`-profile sanity check (`< 5.0 s` over `50` synthetic dicts) mock DB retrieval (`load_catalog_policies_from_db`) and exclude network/PostgreSQL I/O and FastAPI lifespan startup.
2. **Synthetic Data Circularity:** Near-perfect NDCG (`1.0000` Test NDCG@3, `0.9917` Test NDCG@5) reflects synthetic training/test data generated by `dataset.py`'s deterministic `compute_suitability_score()` oracle over `56` eligible profile groups (`80` policies). This **does not** prove real-world ranking precision, user conversion, or generalization across real insurer brochures.
3. **Missingness Distribution Skew:** Synthetic training profiles had `has_annual_income == 1.0` for `720/720` rows; when `annual_income_inr` is omitted in LightGBM inference, explicit `ProfileCompleteness` (`income_distribution_mismatch_risk=True`) and warnings surface honestly without fabricating synthetic income.

---

## 12. Outstanding Blockers

1. **Dual-Owner Migration PR Review (`BLOCKED`):** Approval needed from Node/DB owner + M5 owner before running `npm run migrate` for `server/migrations/003_recommendation_schema.sql`.
2. **M2 Branch Merge & `attribute_extractor` + Catalog Seeding (`BLOCKED`):** `origin/prithviraj_features` is not merged into `main`, has no `policy_attributes` extractor, and `policies` contains `0` rows.
3. **M3 PostgreSQL/pgvector Schema & Embedding Dimension Decision (`BLOCKED`):** Resolve `768` (`dense_retriever.py`) vs `1024` (BGE-M3 Architecture §5) conflict and formalize `CREATE EXTENSION vector` + `sections`/`clauses`/`chunks` tables.
4. **Node Gateway Auth, Catalog CRUD (M4), & M5 Proxy (`BLOCKED`):** Implement/fix Node M1 auth, M4 catalog CRUD, and `/api/v1/recommendations` proxy carrying `X-Service-Token`, `X-Request-ID`, and verified `TrustedScope`.
5. **M3/M8 Grounded Recommendation Citations (`BLOCKED`):** Implement `RecommendationCitationProvider` backed by M3 pgvector retrieval and M8 claim verifier.

---

## 13. Final M5 Completion Assessment

**Module M5 (Policy Recommendation Engine — Core Python Service, Filtering, Features, ML/Weighted Ranking, Contract & Read-Only DB Integration) is technically COMPLETE (`IMPLEMENTED` / `VERIFIED`) within M5's architectural boundary.**

Every M5-owned responsibility that does not depend on unapplied shared migrations (`003`), Node M1/M4/proxy routes, M2 attribute extraction, or M3/M8 citation storage is implemented, hardened against malformed real-catalog inputs, lint-clean (`ruff` 0 errors), contract-frozen (`docs/contracts/ai-service.openapi.yaml`), and passing **153 / 153** tests (including all 3 live read-only PostgreSQL 18.6 checks). End-to-end production catalog serving remains **BLOCKED** solely on the cross-team dependencies documented in Section 12.
