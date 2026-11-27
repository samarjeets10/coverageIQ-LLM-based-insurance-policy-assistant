# coverageIQ-LLM-based-insurance-policy-assistant
AI-powered insurance policy analysis, retrieval and recommendation system.

---

## M5 — Policy Recommendation: Implementation Status

- **Branch:** `shravan/policy-recommendation`
- **Detailed Handoff:** [doc/M5-completion-and-integration-handoff.md](doc/M5-completion-and-integration-handoff.md)
- **Overall Status:** **M5 IMPLEMENTATION COMPLETE WITH EXTERNAL INTEGRATION BLOCKERS**

### A. Implemented Functionality

Module M5 delivers the Python FastAPI Policy Recommendation service under [`ai_service/recommendation/`](ai_service/recommendation/) and [`ai_service/main.py`](ai_service/main.py):

- **Multi-Category Hard Filtering ([`filters.py`](ai_service/recommendation/filters.py)):** Evaluates deterministic eligibility gates and category-specific constraints across `health`, `car`, `life`, and `term` policies prior to feature engineering or scoring, returning `NO_MATCHING_POLICIES` cleanly when zero candidates match.
- **Metadata Validation & Missing-Value Handling ([`coercion.py`](ai_service/recommendation/coercion.py), [`filters.py`](ai_service/recommendation/filters.py), [`features.py`](ai_service/recommendation/features.py), [`service.py`](ai_service/recommendation/service.py)):** Strictly validates numeric, integer, and domain bounds across filtering and direct feature extraction; rejects booleans, `"NaN"`, `Decimal("NaN")`, `float("inf")`, and out-of-range attributes; normalizes supported booleans (`bool`, `0`/`1`, `"true"`/`"false"`, `"yes"`/`"no"`) without coercing ambiguous strings to `True`; isolates malformed catalog candidates as `INVALID_POLICY_METADATA`; guards factual highlight extraction against invalid inputs; applies a conservative `36`-month (`1080`-day) fallback when covered health policies omit `ped_waiting_period_months` without fabricating `"36-month"` highlights; and grants zero PED wait credit when `ped_covered=False`.
- **59-Feature Engineering ([`features.py`](ai_service/recommendation/features.py), [`FEATURES.md`](ai_service/recommendation/FEATURES.md)):** Builds a finite `float64` matrix under frozen feature schema version `1.0.0` (`59` ordered columns) with category isolation, availability indicators, and target/identifier leakage exclusion.
- **LightGBM LambdaRank Ranking ([`train_lightgbm.py`](ai_service/recommendation/train_lightgbm.py), [`artifacts/models/`](ai_service/artifacts/models/)):** Loads and warms up the committed `lightgbm_ranker.joblib` artifact once at startup, enforcing `1.0.0` schema and 59-column compatibility (`503` on missing or incompatible artifacts).
- **Explicitly Enabled Weighted Ranking Fallback ([`weighted.py`](ai_service/recommendation/weighted.py)):** Supports deterministic transparent scoring when callers explicitly set `ranking_model="weighted"` (never silently substituted for failed LightGBM; `ranking_model="xgboost"` returns controlled HTTP `501`).
- **Deterministic Ranking & Tie-Breaking ([`service.py`](ai_service/recommendation/service.py)):** Sorts eligible candidates deterministically by `(-score, str(policy_id))`.
- **Version-Aware Catalog Loading ([`filters.py`](ai_service/recommendation/filters.py), [`tables.py`](ai_service/common/tables.py), [`db.py`](ai_service/recommendation/db.py)):** Uses parameterized SQLAlchemy Core joins requiring `organization_id IS NOT NULL`, `owner_user_id IS NULL`, `is_catalog_eligible IS TRUE`, optional `policy_type` pushdown, and `policy_attributes.policy_version_id == policies.current_version_id` (or both `NULL`), with duplicate detection (`500`) and missing-table handling (`503`).
- **FastAPI Recommendation Endpoints & Contract ([`main.py`](ai_service/main.py), [`router.py`](ai_service/recommendation/router.py), [`schemas.py`](ai_service/recommendation/schemas.py)):** Exposes `POST /api/v1/recommend`, `POST /ai/recommend`, `GET /health`, and `GET /ai/health`, synchronized with [`docs/contracts/ai-service.openapi.yaml`](docs/contracts/ai-service.openapi.yaml).
- **Service-Token Auth & `TrustedScope` ([`router.py`](ai_service/recommendation/router.py), [`settings.py`](ai_service/recommendation/settings.py)):** Enforces non-development `AI_SERVICE_TOKEN` startup configuration, constant-time `X-Service-Token` checks (`401`), `RequestBodyLimitMiddleware` (`413`), and normalized `TrustedScope` propagation.
- **Structured Logging, Request-ID Propagation & Error Handling ([`logging.py`](ai_service/common/logging.py), [`main.py`](ai_service/main.py), [`router.py`](ai_service/recommendation/router.py)):** Scrubs secrets, database credentials, `scope`, and customer PII from structured JSON logs while preserving `X-Request-ID` across `200`, `401`, `413`, `422`, `500`, `501`, and `503` responses.
- **Citation-Provider Interface ([`citations.py`](ai_service/recommendation/citations.py), [`service.py`](ai_service/recommendation/service.py)):** Provides the `RecommendationCitationProvider` boundary (`NullCitationProvider` default) and `normalize_and_validate_citation`, returning empty citation lists safely when citation evidence is unavailable rather than fabricating citations or policy facts.
- **Unit, Regression & End-to-End Integration Tests ([`ai_service/tests/`](ai_service/tests/)):** Includes deterministic unit tests, missingness/sensitivity coverage (`M5.I1`), boolean/metadata hardening regressions (`M5.I2`), and isolated `003`-shaped SQLite end-to-end route tests ([`test_m5_final_integration.py`](ai_service/tests/test_m5_final_integration.py)).

> **Model & Evaluation Limitation:** The committed LightGBM ranker was trained exclusively on synthetic policy and profile data. Passing unit, regression, and integration tests confirm service mechanics and contract compliance; they do **not** establish real-world recommendation accuracy.

---

### B. Remaining Implementation and Integration Work

| Remaining work | Owner/dependency | Why it is not completed | What is needed to finish |
|---|---|---|---|
| **1. Shared database migration (`003_recommendation_schema.sql`) and live catalog readiness** | Shared database / backend owners ([`003_recommendation_schema.sql`](server/migrations/003_recommendation_schema.sql), [`migrate.js`](server/scripts/migrate.js)) | Requires formal review and approval prior to execution through the Node migration runner; M5 does not independently apply shared migrations. | Approved migration review, authorized execution via Node migrations, and passing read-only PostgreSQL readiness and catalog checks. |
| **2. M1 Node.js API gateway integration** | M1 / backend team | Node gateway proxy routing and authentication/authorization forwarding to the Python recommendation endpoints are not yet wired on this branch. | Authenticated Node-to-Python recommendation proxy routes, verified `TrustedScope` and `X-Service-Token` propagation, and Node integration tests. |
| **3. M2/M4 real policy ingestion and catalog publication** | M2 ingestion and M4 catalog teams | M5 depends on real, version-aligned `policy_attributes` records and cannot invent policy metadata or bypass M2/M4 publication ownership. | Published policy versions with matching structured `policy_attributes` rows for supported categories, followed by M5 catalog readiness and recommendation tests. |
| **4. M3/M8 citation grounding** | M3 retrieval and M8 evidence/explanation teams | M5 provides the `RecommendationCitationProvider` boundary and safe empty-citation behavior (`NullCitationProvider`), pending integration of a concrete verified evidence provider. | Retrieval-backed `RecommendationCitationProvider` returning verified clause/page citations with source references, plus tests covering missing or partial evidence. |
| **5. Real-world recommendation evaluation and model calibration** | M5 ML work in coordination with data and domain-review teams | Committed LightGBM artifacts were trained on synthetic data; representative production policy/profile outcomes and independent reviewer relevance judgments are not yet available. | Curated real policy data, independent user profiles, domain-reviewed relevance labels/outcomes, reproducible benchmark evaluation, and approved model/schema updates where justified. |
| **6. Feature schema 2.0 and PED missingness** | M5, dependent on real data and model evaluation | Active schema `1.0.0` (`59` features) does not include an explicit PED waiting-period missingness feature; covered health policies missing PED wait periods use the conservative `36`-month interim fallback (`2.0` is **not** implemented yet). | Approved feature-schema `2.0` specification, model retraining, backwards-compatibility validation, and evaluation on real policy data. |
| **7. Request-ID header hardening** | M5 / Python AI-service maintenance ([`main.py`](ai_service/main.py), [`router.py`](ai_service/recommendation/router.py)) | Final audit observed that `_extract_request_id_headers` and router header handling strip whitespace and cap length (`128`) without removing or rejecting embedded CR/LF control characters. | Strip or reject CR/LF and control characters consistently across header helpers with regression tests across all response paths. |

---

### C. Current Verification Evidence

Latest local verification results recorded in [doc/M5-completion-and-integration-handoff.md](doc/M5-completion-and-integration-handoff.md) for the M5 code state (`fb4b952`) on branch `shravan/policy-recommendation` (with all declared packages from [`ai_service/requirements.txt`](ai_service/requirements.txt) installed; live PostgreSQL integration and remote/production deployment were **not** verified):

- **Full AI-service test suite (`pytest tests/ -ra --tb=short`):** `219 passed, 2 skipped` (`38.92s`).
- **Four new M5 end-to-end integration tests (`pytest tests/test_m5_final_integration.py -v`):** `4 passed` (`4.13s`).
- **Ruff formatting & lint (`ruff format --check .`, `ruff check .` in `ai_service/`):** `35 files already formatted`, `All checks passed!`.
- **OpenAPI contract verification (`verify_openapi_contract()`):** Passed; live FastAPI schema matches [`docs/contracts/ai-service.openapi.yaml`](docs/contracts/ai-service.openapi.yaml).
- **Feature schema & model artifact:** `FEATURE_SCHEMA_VERSION = "1.0.0"`, `59` columns in exact order; committed [`lightgbm_ranker.joblib`](ai_service/artifacts/models/lightgbm_ranker.joblib) preserved unchanged (`18428` bytes, SHA-256 `0b2124916d2940c8767960f66d37f2d885b3b9bdf151f3b8486111c7343c3ddb`).
- **Skipped tests note:** Live PostgreSQL read-only checks ([`test_database_integration.py`](ai_service/tests/test_database_integration.py) and [`test_inference.py:759`](ai_service/tests/test_inference.py#L755-L760)) were skipped because `DB_PASSWORD` and live PostgreSQL access were unavailable; these skipped checks are **not** counted as passing.

---

### D. Branch and Implementation Status

- **Feature branch:** `shravan/policy-recommendation`
- **M5 commits on branch ahead of `origin/main`:**
  - `57f1676` — `feat(m5): integrate policy recommendation engine into unified ai_service`
  - `348da80` — `fix(m5): harden catalog metadata filtering, version lineage, and traceability`
  - `08bfeb8` — `test(m5): add missingness and ranking sensitivity coverage`
  - `d2a564b` — `fix(m5): harden metadata validation and missing-value handling`
  - `fb4b952` — `test(m5): verify end-to-end recommendation integration`
  - `18178aa` — `docs(m5): add completion and integration handoff`
- **Detailed handoff document:** [doc/M5-completion-and-integration-handoff.md](doc/M5-completion-and-integration-handoff.md)
- **Overall status:** **M5 IMPLEMENTATION COMPLETE WITH EXTERNAL INTEGRATION BLOCKERS**

M5's Python recommendation engine and local verification suite are complete, while full CoverageIQ production integration remains incomplete until the cross-module database, gateway, catalog publication, citation grounding, and real-data evaluation dependencies in Section B are closed.

---

### E. Next Steps and Recommended Order

Several tracks below can proceed in parallel where team ownership and service boundaries allow:

1. Obtain shared database-owner approval for [`003_recommendation_schema.sql`](server/migrations/003_recommendation_schema.sql) and apply it through the agreed Node migration workflow.
2. Coordinate M2 ingestion and M4 catalog publication to populate and verify real, version-aligned `policy_attributes` data.
3. Complete M1 Node.js API gateway authentication, `TrustedScope` extraction, and `X-Service-Token` forwarding to `/api/v1/recommend` or `/ai/recommend`.
4. Integrate M3/M8 verified citation grounding behind `RecommendationCitationProvider`.
5. Run authorized live PostgreSQL readiness and integration tests against the migrated, populated database (`DB_PASSWORD` configured).
6. Conduct real-world recommendation evaluation and model calibration (including assessing schema `v2.0.0` for explicit PED missingness).
7. Complete the minor request-ID CR/LF control-character hardening.
