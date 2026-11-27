# Phase 9B-1 — Production Safety Hardening & API Contract Freeze Completion Report

**Project:** CoverageIQ — LLM-Based Insurance Policy Assistant (Module M5)  
**Developer:** Shravan Sambhaji Kapare (Roll No. 203, Group B2)  
**Branch:** `shravan-feature/recommendation-module` (Base commit `7953ea24`, uncommitted working tree per Phase 9B-1 rules)  
**Status:** Phase 9B-1 Complete (`003_recommendation_schema.sql` remains UNAPPLIED; M2/M3/M4/DB/LightGBM artifacts untouched).

---

## 1. Implementation Summary

Phase 9B-1 hardened the `ai_service` security, validation, error-handling, query-efficiency, and API contract layers surfaced during the Stage 9B readiness audit without changing Stage 5 `FEATURE_SCHEMA_VERSION = "1.0.0"`, Stage 6 LightGBM training/weights, `ranking_features`, or shared `server/` modules:

1. **Fail-Closed Non-Dev Authentication:** Enforced mandatory non-empty `AI_SERVICE_TOKEN` in all non-development environments (`Settings._validate_security_configuration` + runtime defense-in-depth inside `verify_service_token()`).
2. **Input Bounds & Payload Resource Protection:** Capped `required_benefits` and `pre_existing_conditions` at `50` elements (`100` chars per tag), added `allow_inf_nan=False`, length limits on `TrustedScope`, ASGI `RequestBodyLimitMiddleware` (`max_request_body_bytes = 65536` -> `HTTP 413`), and a NaN-safe `RequestValidationError` handler returning `HTTP 422`.
3. **Database Reliability & Error Isolation:** Mapped duplicate catalog records to `DuplicateCatalogPolicyError` -> `CatalogIntegrityError` (`HTTP 500`, distinct from DB outage `HTTP 503`), eliminated redundant per-request `information_schema` round-trips on real SQLAlchemy connections via SQLSTATE `42P01` / missing-table detection, and guarded unexpected `ValueError` escapes at both service and router boundaries.
4. **Query Efficiency:** Pushed `policy_type=request.profile.policy_type` cleanly into `build_catalog_policy_attributes_query` via `load_catalog_policies_from_db()` while preserving active/inactive filter semantics, organization catalog isolation, and version-aware matching.
5. **TrustedScope Boundary & OpenAPI Contract Freeze:** Forwarded validated `TrustedScope` to `PolicyCitationRequest.scope` (without trusting unauthenticated identity or leaking `user_id`/`owner_scope` into logs) and froze [`docs/contracts/ai-service.openapi.yaml`](../docs/contracts/ai-service.openapi.yaml) with automated drift verification.

---

## 2. Exact Modified & Created Files

| File | Action | Purpose |
|---|---|---|
| [`ai_service/config/settings.py`](../ai_service/config/settings.py) | **Modified** | Added `ai_service_require_auth`, `max_request_body_bytes`, `is_development_env`, `auth_enforced`, empty env string handling, and fail-closed `_validate_security_configuration`. |
| [`ai_service/config/openapi_export.py`](../ai_service/config/openapi_export.py) | **Created** | Deterministic OpenAPI YAML exporter (`export_openapi_contract`) and schema drift validator (`verify_openapi_contract`). |
| [`ai_service/main.py`](../ai_service/main.py) | **Modified** | Added `RequestBodyLimitMiddleware` (`HTTP 413`), `_sanitize_validation_errors` + `RequestValidationError` (`HTTP 422`), `HealthResponse` OpenAPI metadata, and startup security validation. |
| [`ai_service/recommendation/schemas.py`](../ai_service/recommendation/schemas.py) | **Modified** | Enforced `MAX_TAG_LIST_ITEMS = 50` (including raw duplicate count), `MAX_TAG_ITEM_LENGTH = 100`, `allow_inf_nan=False`, `TrustedScope` ID bounds/docstrings, `ErrorDetailResponse`, and `HealthResponse`. |
| [`ai_service/recommendation/citations.py`](../ai_service/recommendation/citations.py) | **Modified** | Added optional `scope: TrustedScope | None = None` to `PolicyCitationRequest` for downstream M3/M8 tenant-aware citation lookup. |
| [`ai_service/recommendation/filters.py`](../ai_service/recommendation/filters.py) | **Modified** | Added `DuplicateCatalogPolicyError(ValueError)`, `_is_missing_policy_attributes_error()` (`42P01` / missing table detection), and direct single-query execution on real SQLAlchemy connections. |
| [`ai_service/recommendation/service.py`](../ai_service/recommendation/service.py) | **Modified** | Added `CatalogIntegrityError`, `_load_catalog_candidates` (`policy_type` SQL pushdown), guarded filtering/DB/citation exceptions, `scope` forwarding to citations, and safe non-PII `scope_present`/`scope_role` telemetry. |
| [`ai_service/recommendation/router.py`](../ai_service/recommendation/router.py) | **Modified** | Replaced optional `Header` token check with OpenAPI `APIKeyHeader` (`ServiceTokenAuth`), added fail-closed runtime token enforcement, documented `413` + `ErrorDetailResponse` contracts, and added `ValueError` router guard. |
| [`ai_service/requirements.txt`](../ai_service/requirements.txt) | **Modified** | Explicitly declared `pyyaml>=6.0.0,<7.0.0` for contract generation and verification. |
| [`docs/contracts/ai-service.openapi.yaml`](../docs/contracts/ai-service.openapi.yaml) | **Created** | Frozen OpenAPI 3.1.0 contract exported strictly from FastAPI `create_app().openapi()` (uncommitted in workspace per branch/commit restriction). |
| [`ai_service/tests/test_phase9b1_hardening.py`](../ai_service/tests/test_phase9b1_hardening.py) | **Created** | `13` tests covering fail-closed auth, list/body limits, `NaN` 422 safety, duplicate 500 mapping, single-query missing-table checks, DB outage 503s, `policy_type` SQL pushdown, `TrustedScope` forwarding, and OpenAPI drift. |
| [`ai_service/README.md`](../ai_service/README.md) | **Modified** | Documented Phase 9B-1 hardening deliverables and OpenAPI contract reference. |
| [`doc/M5-phase-9B-1-completion-report.md`](M5-phase-9B-1-completion-report.md) | **Created** | This Phase 9B-1 completion report. |

**Verified Untouched:** `ai_service/artifacts/models/*`, `recommendation/train_lightgbm.py`, `recommendation/features.py` (`FEATURE_SCHEMA_VERSION = "1.0.0"`), `recommendation/weighted.py`, `common/tables.py`, `server/`, `server/migrations/003_recommendation_schema.sql`, and all M2/M3/M4 scopes.

---

## 3. Security Fixes

1. **Non-Development Fail-Closed Auth:** Any environment other than `{"development", "dev", "local", "test", "testing"}` now fails startup (`ValidationError`) if `AI_SERVICE_TOKEN` is missing, blank, or `ai_service_require_auth=False`. `verify_service_token()` re-validates this at request time (`HTTP 401`, constant-time `hmac.compare_digest`, zero token logging).
2. **DoS & Payload Hardening:** Bounded raw HTTP body bytes (`64 KiB` default -> `HTTP 413`), max `50` items (`<= 100` chars) on `required_benefits`/`pre_existing_conditions` before deduplication (`HTTP 422`), `allow_inf_nan=False`, and fixed FastAPI's `RequestValidationError` `NaN` serialization `500` crash to return safe `HTTP 422`.
3. **Identity Isolation (`TrustedScope`):** Validated `TrustedScope` lengths (`128` chars) and documented M1 gateway attestation responsibility; SQL catalog filters retain `organization_id IS NOT NULL AND owner_user_id IS NULL`; `user_id` and `owner_scope` are scrubbed from logs.

---

## 4. API Contract Details

- **Frozen Contract:** [`docs/contracts/ai-service.openapi.yaml`](../docs/contracts/ai-service.openapi.yaml) (`OpenAPI 3.1.0`).
- **Included Paths:** `/api/v1/recommend` (`POST`), `/ai/recommend` (`POST`), `/health` (`GET`), `/ai/health` (`GET`).
- **Security & Errors Documented:** `ServiceTokenAuth` (`X-Service-Token` `apiKey` header), `X-Request-ID` header, `RecommendRequest`, `RecommendResponse`, `ProfileCompleteness`, `TrustedScope`, `HealthResponse`, `ErrorDetailResponse`, and HTTP `200 / 401 / 413 / 422 / 500 / 501 / 503` responses.
- **Drift Check:** Verified programmatically via `config.openapi_export.verify_openapi_contract()` and automated pytest assertion.

---

## 5. Database Reliability Improvements

- **Single-Query Catalog Retrieval:** `load_catalog_policies_from_db()` skips the extra `information_schema` round-trip on real SQLAlchemy `Connection`s (reducing DB queries per recommendation request from `2` to `1`) while explicitly converting PostgreSQL SQLSTATE `42P01` (`UndefinedTable`) or SQLite missing-table errors on `policy_attributes` into `PolicyAttributesTableUnavailableError` -> sanitized `HTTP 503`.
- **Distinct Integrity Error Mapping:** Duplicate current-version `policy_attributes` rows raise `DuplicateCatalogPolicyError` -> `CatalogIntegrityError` -> `HTTP 500` rather than falsely claiming PostgreSQL is unavailable (`HTTP 503`).
- **Preserved Safely:** Version-aware `current_version_id` joins, parameterized SQLAlchemy Core statements, `pool_pre_ping`, and configurable `connect_timeout` / `pool_timeout` / `statement_timeout`.

---

## 6. Performance Improvements

- `_load_catalog_candidates()` passes `policy_type=request.profile.policy_type` into `load_catalog_policies_from_db()` -> `build_catalog_policy_attributes_query()`, pushing category filtering directly into PostgreSQL `WHERE policies.policy_type = :policy_type_1`.
- Eliminates fetching unrelated `car` / `life` / `term` rows during `health` requests (and vice versa), reducing network payload size and Python filter iterations while preserving active/inactive (`active_only=False`), organization ownership, version matching, and final recommendation lists identically (proven via multi-category SQLite catalog test).
- Eliminated 1 redundant `information_schema` round-trip per recommendation request on real connections.

---

## 7. Tests Executed & Results

- `ruff format --check .` & `ruff check .` (`ai_service/`): **`33 files already formatted`**, **`All checks passed!`** (`0` errors).
- `python -m config.openapi_export` / `verify_openapi_contract()`: **`CONTRACT_VERIFY: PASS`**.
- **Run 1 (`pytest -v` without shell `DB_*` env vars):** **`130 passed, 2 skipped`** (existing suites + `13` new `test_phase9b1_hardening.py` tests; `test_stage9a_robustness.py::test_live_postgresql_statement_timeout_and_read_only_readiness` ran live via `../server/.env`).
- **Run 2 (`pytest -v` with `server/.env` loaded into process env):** **`132 passed, 0 skipped`** against live read-only PostgreSQL 18.6 (`schema_migrations = ['001_initial_schema.sql', '002_add_platform_role.sql']`).

---

## 8. Skipped Tests & Reasons

- **Run 1 only (`2 skipped`):**
  1. `tests/test_database_integration.py::test_postgresql_read_only_integration` — skipped (`DB_PASSWORD not configured; skipping live PostgreSQL read-only test`).
  2. `tests/test_inference.py::test_live_postgresql_read_only_recommendation_readiness` — skipped (`DB_PASSWORD not configured; skipping live PostgreSQL Stage 7 check`).
- **Run 2 (`server/.env` exported):** **`0` skipped** (all `132` tests passed).

---

## 9. Remaining Known Risks (Preserved Intentionally in Phase 9B-1)

1. **Missing `claim_settlement_ratio` & `ped_waiting_period_months` on Real Policies:** Preserved existing Stage 5 (`0.0` CSR / `0.0` PED waiting days when NULL) and Stage 6 LightGBM scoring behavior untouched per Phase 9B-1 instructions; uncalibrated risk remains until real catalog gold validation in Phase 9B-3.
2. **Unapplied Schema & Empty DB (`0` rows):** `003_recommendation_schema.sql` remains unapplied; live DB catalog has `0` policies.
3. **Upstream Modules Unimplemented:** No Node.js proxy/M1 RBAC, M2 `attribute_extractor`, M3 retrieval chunks, M4 catalog CRUD, or M8 grounded explainer exists in the repo yet (`citations=[]`).
4. **Canonical Vocabulary Drift (`filters.py` vs. M2):** Deferred canonical taxonomy coordination with M2/M4 (Phase 9B-3).

---

## 10. Backup ZIP & Manifest Paths

- **Pre-Phase-9B-1 Backup ZIP:** `I:\Mega Project\coverageiq-m5-backups\coverageiq-m5-backup-20261127-120324.zip` (`223,393` bytes, `51` files, CRC verified)
- **Pre-Phase-9B-1 Manifest:** `I:\Mega Project\coverageiq-m5-backups\coverageiq-m5-backup-20261127-120324.manifest.txt` (`2,838` bytes)
- **Earlier Stage 9A Verified Backup (untouched):** `I:\Mega Project\coverageiq-m5-backups\coverageiq-m5-backup-20261127-111838.zip`

---

## 11. Git Branch & Commit Status

- **Current Branch:** `shravan-feature/recommendation-module` (HEAD `7953ea24 feat: establish PostgreSQL database foundation`).
- **Commit Status:** Uncommitted working tree only (`git diff --stat` empty; zero tracked files altered; all M5 & contract files present as untracked workspace deliverables per Strict Restriction #13: *"Do not commit, push, merge, or switch branches."*). Note `docs/contracts/ai-service.openapi.yaml` was generated and verified cleanly in the working tree awaiting your team commit approval.

---

## 12. Updated M5 Readiness Assessment

| Dimension | Phase 9B-1 Status | Notes |
|---|---|---|
| **Architecture Ready** | **YES** | Fail-closed auth, payload limits, `TrustedScope` boundary documentation, single-query SQL pushdown, and frozen OpenAPI contract (`docs/contracts/ai-service.openapi.yaml`) complete. |
| **Integration Ready (Python API Side)** | **YES (Frozen Contract & Guardrails) / Blocked Upstream** | `ai_service` contract is frozen and hardened for Node proxy integration; end-to-end live integration remains blocked on Node M1/M4 proxy and `003_recommendation_schema.sql` promotion (Phase 9B-2). |
| **Real Catalog Ready** | **NO** | Awaits Phase 9B-2 (`003` migration + transactional PG tests) and Phase 9B-3 (real brochure gold evaluation & M2/M4 vocabulary/missingness alignment). |
| **Production Ready** | **NO** | Blocked by real catalog seeding/validation, Node API/Auth integration, M3/M8 evidence citations, and deployment setup (`M11`). |
