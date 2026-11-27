# Stage 9A — Recommendation Engine Robustness & Production Hardening Report

**Project:** CoverageIQ — LLM-Based Insurance Policy Assistant (Module M5)  
**Developer:** Shravan Sambhaji Kapare (Roll No. 203, Group B2)  
**Branch:** `shravan-feature/recommendation-module`  
**Status:** Stage 9A Complete (`003_recommendation_schema.sql` remains UNAPPLIED; live database and shared Node `server/` left unmodified).

---

## 1. Deliverables Summary (Files Created & Modified)

| File | Action | Purpose |
|---|---|---|
| [`ai_service/recommendation/weighted.py`](../ai_service/recommendation/weighted.py) | **Created** | Transparent, deterministic, configurable weighted scoring backend (`WEIGHTED_SCORER_NAME = "coverageiq_weighted_scorer"`, v`1.0.0`, Phase A fallback) strictly over the 59-column Stage 5 feature matrix (`annual_income` weight = `0.0`). |
| [`ai_service/recommendation/citations.py`](../ai_service/recommendation/citations.py) | **Created** | Interface-only citation seam (`PolicyCitationRequest`, `RecommendationCitationProvider` Protocol) and `NullCitationProvider` returning `[]` without M3/M8 coupling or fabricated citations. |
| [`ai_service/common/logging.py`](../ai_service/common/logging.py) | **Created** | `StructuredJsonFormatter`, `sanitize_log_value`, `emit_structured_event`, and `configure_logging` enforcing structured JSON telemetry while scrubbing passwords, tokens, DB URL credentials, and customer PII (`NFR-11`). |
| [`ai_service/config/settings.py`](../ai_service/config/settings.py) | **Modified** | Added `db_connect_timeout_seconds`, `db_statement_timeout_ms`, `db_pool_timeout_seconds`, `log_level`, and `log_json`. |
| [`ai_service/config/database.py`](../ai_service/config/database.py) | **Modified** | Passed `pool_timeout`, `connect_timeout`, and libpq `options="-c statement_timeout=..."` to SQLAlchemy `create_db_engine()`. |
| [`ai_service/recommendation/schemas.py`](../ai_service/recommendation/schemas.py) | **Modified** | Added additive `ProfileCompleteness` model + optional `profile_completeness` on `RecommendResponse`, plus OpenAPI `examples` on `UserProfileRequest`, `RecommendRequest`, and `RecommendResponse`. |
| [`ai_service/recommendation/service.py`](../ai_service/recommendation/service.py) | **Modified** | Added startup warm-up (`_warm_up_lightgbm_model`), corrupt-artifact catching (`EOFError`, `zlib.error`, `pickle.UnpicklingError`, `json.JSONDecodeError`), explicit `"weighted"` opt-in routing without silent `"auto"` fallback, stage latency telemetry, `ProfileCompleteness` generation, and citation provider invocation. |
| [`ai_service/recommendation/router.py`](../ai_service/recommendation/router.py) | **Modified** | Added `get_citation_provider` DI dependency and explicit OpenAPI HTTP `401 / 422 / 500 / 501 / 503` response documentation on `/api/v1/recommend` and `/ai/recommend`. |
| [`ai_service/main.py`](../ai_service/main.py) | **Modified** | Configured structured logging on app startup, initialized `NullCitationProvider`, and exposed `model_warmed_up` alongside `model_ready` on `/health` and `/ai/health`. |
| [`ai_service/tests/test_inference.py`](../ai_service/tests/test_inference.py) | **Modified** | Updated the single Stage 7 unit assertion (`line 729`) that previously expected `"weighted"` to raise `UnsupportedRankingModelError` to assert `"xgboost"` raises `UnsupportedRankingModelError`. |
| [`ai_service/tests/test_stage9a_robustness.py`](../ai_service/tests/test_stage9a_robustness.py) | **Created** | Added `19` robustness, artifact corruption, startup warm-up, weighted fallback, incomplete-profile matrix, JSON logging/PII redaction, citation interface, DB timeout (with live PG check), 500-policy determinism/latency, and OpenAPI verification tests. |
| [`ai_service/README.md`](../ai_service/README.md) | **Modified** | Updated module status, active Stage 9A capabilities, and endpoint behavior notes. |
| [`doc/M5-stage-9A-implementation-report.md`](M5-stage-9A-implementation-report.md) | **Created** | This technical report. |

**Preserved Untouched:** `ai_service/artifacts/models/{lightgbm_ranker.joblib, lightgbm_metadata.json, training_metrics.json}`, `recommendation/train_lightgbm.py`, `recommendation/features.py` (`FEATURE_SCHEMA_VERSION = "1.0.0"`), `recommendation/dataset.py`, `recommendation/filters.py`, `common/tables.py`, `server/`, `server/migrations/003_recommendation_schema.sql`, and all Stage 8A/8A.1/8B SQL & architecture docs. No `train_xgboost.py` or M3 retrieval implementation was introduced.

---

## 2. Preserving LightGBM & Hardening Artifact Loading

1. **Zero Retraining or Modification:** Existing Stage 6 artifact `lightgbm_ranker.joblib` and training logic in `train_lightgbm.py` remain byte-for-byte intact.
2. **Corrupt & Unfitted Artifact Handling:**
   - `load_model_state()` now catches `(json.JSONDecodeError, EOFError, zlib.error, pickle.UnpicklingError, AttributeError, ModuleNotFoundError, KeyError, TypeError, ValueError)` as `error_kind="corrupt"` alongside `"missing"` and `"incompatible"`.
   - `_warm_up_lightgbm_model(model)` executes a 1-row finite forward pass via `model.predict(warm_df)` across the exact `59` ordered `FEATURE_COLUMNS` at startup so unfitted (`NotFittedError`) or structurally broken `LGBMRanker` boosters are rejected during FastAPI lifespan (`model_ready=False`, `model_warmed_up=False`).
   - `/health` and `/ai/health` remain operational (HTTP `200`) while `/api/v1/recommend` and `/ai/recommend` with `ranking_model="auto"` or `"lightgbm"` return controlled HTTP `503` without exposing host filesystem paths.

---

## 3. Explicit Weighted Fallback Backend (`ranking_model="weighted"`)

1. **Architecture Alignment & Explicit Opt-In:**
   - Implements CoverageIQ's `Phase A` transparent weighted scorer using five `[0, 1]` dimensions computed strictly from Stage 5 features: `affordability`, `coverage`, `benefits`, `cost_waiting`, and `insurer_network`.
   - Category-aware adjustments handle `is_health == 0.0` cleanly (neutral default CSR `90.0%` only when `has_claim_settlement_ratio == 0.0`; `pan_india_coverage_indicator` fallback when hospital network count is zeroed outside health).
   - Risk-preference weights (`low_premium`, `balanced`, `comprehensive_coverage`) and scaling bounds live in `WeightedScoringConfig` + `WeightedDimensionWeights`, validated to be finite, non-negative, and sum to `1.0`.
   - **Non-Silent Fallback Guarantee:** When `ranking_model` is `"auto"` (default) or `"lightgbm"`, missing/corrupt LightGBM artifacts ALWAYS fail with HTTP `503` — `weighted` is used ONLY when explicitly requested (`ranking_model="weighted"`). `ranking_model="xgboost"` remains unimplemented (HTTP `501`).

2. **Empirical Comparison vs. LightGBM (Held-Out Synthetic Test Split):**

| Backend | Test `NDCG@1` | Test `NDCG@3` | Test `NDCG@5` | Test `NDCG@10` | Scoring Time (`124` rows, `9` groups) |
|---|---|---|---|---|---|
| LightGBM (`lightgbm_ranker.joblib`) | `1.0000` | `1.0000` | `0.9917` | `0.9927` | `3.32 ms` |
| Weighted (`score_policies_weighted`) | `1.0000` | `1.0000` | `1.0000` | `0.9996` | `2.90 ms` |

> [!IMPORTANT]
> **Evaluation Caveat:** The `1.0000` synthetic `NDCG@5` of the weighted backend must **not** be interpreted as superiority over LightGBM. Stage 3 synthetic ground-truth relevance grades (`{1, 2, 3}` in `dataset.py`) were deterministically generated by `compute_suitability_score()`, which uses the same underlying five-dimension weighting scheme. This synthetic-split comparison therefore overlaps circularly with the synthetic label oracle (`Suitability BL (ceil)`). Neither backend has yet been evaluated against real-world user outcomes or curated gold relevance labels.

---

## 4. Incomplete Customer Profile Investigation & Hardening

### 4.1 Observed Training vs. Inference Income Distribution Mismatch

Inspecting the 720-row Stage 5/6 synthetic training dataset and trained `lightgbm_ranker.joblib` gain importances:
- Training `annual_income` values: `min = 565,000.0`, `median = 1,915,000.0`, `mean = 2,061,498.92`, `max = 4,177,000.0`.
- `has_annual_income == 1.0` for **100%** (`720/720`) of synthetic training pairs -> LightGBM split gain on `has_annual_income` is `0.0000` (and `0.0000` on `risk_preference_encoded`), while `annual_income` itself carries gain importance `15.4805` (#7 overall feature).
- On the 56-group synthetic evaluation benchmark (`Test NDCG@3 = 1.0000, @5 = 0.9917` baseline), masking `annual_income_inr` yielded:
  - `annual_income = 0.0` (current Stage 5 missingness encoding): `Test NDCG@3 = 1.0000`, `NDCG@5 = 0.9917`, `max_score_delta = 0.5752`, `2 / 56` groups changed Top-1.
  - `np.nan`: identical (`max_score_delta = 0.5752`, `2 / 56` groups changed Top-1) because LightGBM sends unseen `NaN`s down the default child identical to `0.0`.
  - Substituting `train_median` (`1,915,000.0`): `Test NDCG@3 = 1.0000`, `NDCG@5 = 0.9917`, `max_score_delta = 0.4219`, `3 / 56` groups changed Top-1 (worse Top-1 stability while also fabricating unverified customer income).

### 4.2 Design Resolution
1. Rejected fake median imputation (`train_median`) — never fabricate customer financial data.
2. Preserved `annual_income_inr > 0.0` validation (`HTTP 422` on `0` or negative values) so missing income (`None`) cannot be confused with genuine invalid input, while preserving Stage 5 `annual_income=0.0, has_annual_income=0.0` schema stability.
3. Added explicit `ProfileCompleteness` response metadata (`is_complete`, `missing_fields`, `applied_assumptions`, `income_distribution_mismatch_risk`) alongside explicit `warnings`. Note that the explicit `weighted` backend places zero weight on `annual_income`, making weighted recommendations completely invariant when `annual_income_inr` is omitted (`income_distribution_mismatch_risk=False`).

---

## 5. Operational Hardening, Citations Seam & End-to-End Scale Benchmark

- **Structured JSON Logging (`common/logging.py`):** Every request emits compact JSON log events (`recommendation.completed`, `recommendation.no_match`, `recommendation.model_unavailable`, `recommendation.db_unavailable`, `recommendation.unsupported_model`, etc.) containing `request_id`, `model_used`, candidate counts (`total_candidates`, `eligible_count`, `rejected_count`, `returned_count`), profile completeness field names (never values), and 6-stage `latency_ms` breakdowns (`db`, `filter`, `features`, `scoring`, `citations`, `total`). Sensitive keys and PostgreSQL URLs are scrubbed automatically (`[REDACTED]`).
- **Database Timeouts (`config/settings.py`, `config/database.py`):** Added configurable `db_connect_timeout_seconds=5`, `db_statement_timeout_ms=10000`, and `db_pool_timeout_seconds=10.0`, verified live against read-only PostgreSQL 18.6 (`SELECT setting FROM pg_settings WHERE name = 'statement_timeout'` confirmed `"7500"` ms under test config).
- **Citation Interface Seam (`recommendation/citations.py`):** `NullCitationProvider` attaches `citations=[]` by default without M3 retrieval or ungrounded claims; DI wiring via `get_citation_provider` was verified end-to-end with an injected stub provider and invalid-payload guard (`HTTP 500`).
- **500-Candidate Micro-Benchmark (`top_n=10`, 5 sequential `TestClient` requests per backend):**
  - **Methodology Caveat:** Uses `500` synthetic in-memory `_sample_policy()` dictionaries with **mocked database access** (`get_db_connection` + `load_catalog_policies_from_db` monkeypatched, so DB time is `0.001–0.005 ms`), **excluding** startup `lifespan` model load and warm-up. Represents a `5`-request in-process compute sanity check only, not a statistically significant production latency benchmark.
  - **LightGBM:** `[207.83, 209.82, 208.62, 230.15, 192.19]` ms — **median `208.62 ms`** (in-process stage breakdown: filter `~125–167 ms`, features `~32–52 ms`, LightGBM scoring `~2.9–7.7 ms`).
  - **Weighted:** `[177.91, 189.51, 173.38, 199.66, 295.83]` ms — **median `189.51 ms`** (observed weighted scoring `~3.3–7.4 ms`).

---

## 6. Final Verification Summary

- `ruff format --check .` & `ruff check .` (from `ai_service/`): **`31 files already formatted`**, **`All checks passed!`** (`0` errors).
- **Run 1 (`pytest -v` without `DB_*` env vars, `ai_service/.env` absent):** **`117 passed, 2 skipped`** (`18.89s`).
  - `test_database_integration.py::test_postgresql_read_only_integration` and `test_inference.py::test_live_postgresql_read_only_recommendation_readiness` call bare `Settings()` and skipped cleanly because `DB_PASSWORD` was unset.
  - `test_stage9a_robustness.py::test_live_postgresql_statement_timeout_and_read_only_readiness` explicitly loads `Settings(_env_file="../server/.env")` when present, so it executed against live PostgreSQL in Run 1 as well.
- **Run 2 (`pytest -v` with `server/.env` variables exported to process env):** **`119 passed, 0 skipped`** (`10.59s`), executing **all three** live PostgreSQL read-only integration tests.
- **PostgreSQL `schema_migrations` verified read-only:** `['001_initial_schema.sql', '002_add_platform_role.sql']` (`003_recommendation_schema.sql` UNAPPLIED; `policies` row count = `0`; `policy_attributes = False`, `ranking_features = False`).
- **Stage 9A Completion:** Marked Stage 9A complete on `shravan-feature/recommendation-module` with zero commits, zero DB writes/migrations, and unchanged LightGBM training code and artifacts.

---

## 7. Explicit Deferrals (Outside Stage 9A)

1. **`ranking_model="xgboost"`** (`train_xgboost.py` / XGBoost backend) remains deferred (`HTTP 501`).
2. **M3 Evidence Retrieval & M8 Grounded Explanations/Verification** remain deferred behind `RecommendationCitationProvider` (`NullCitationProvider` returns `[]`).
3. **Database Migration Review & Catalog Seeding** (`server/migrations/003_recommendation_schema.sql` & M2 attribute ingestion) remain pending cross-team PR review before live recommendation data is served.
