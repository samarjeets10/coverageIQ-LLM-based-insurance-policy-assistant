# Stage 8B — Architecture-Aligned Database Integration Report

**Project:** CoverageIQ — LLM-Based Insurance Policy Assistant (Module M5)  
**Developer:** Shravan Sambhaji Kapare (Roll No. 203, Group B2)  
**Branch:** `shravan-feature/recommendation-module`  
**Architecture Baseline:** `doc/project-architecture.md` & `feature/coverageiq-database-foundation` (`7953ea24172b3803dcf6c1dfcbb732b95f066112`)  
**Status:** Architecture-aligned migration, Python sync, tests, and documentation ready for cross-owner PR review (`003_recommendation_schema.sql` NOT executed against any database).

---

## 1. Files Created & Modified

| File | Action | Purpose |
|---|---|---|
| [`server/migrations/003_recommendation_schema.sql`](../server/migrations/003_recommendation_schema.sql) | **Created** (for review, unapplied) | Node-runner compatible DDL creating Python-owned `policy_attributes` (with all validated v3 constraints & version-aware `UNIQUE NULLS NOT DISTINCT`) and architecture-mandated `ranking_features` (`DEFAULT '1.0.0'`). |
| [`ai_service/common/tables.py`](../ai_service/common/tables.py) | **Modified** | Synchronized `policy_attributes` fail-closed defaults, cross-column `CHECK` constraints, and `UniqueConstraint(..., postgresql_nulls_not_distinct=True)` with `003`; updated `ranking_features` schema default `'1.0.0'`, score bounds `[0, 1]`, and versioned uniqueness. |
| [`ai_service/config/database.py`](../ai_service/config/database.py) | **Modified** | Preserved all `EXPECTED_TABLES` (`policies`, `policy_versions`, `organizations`, `policy_attributes`, `ranking_features`); added non-breaking `recommendation_catalog_ready` indicator in `inspect_database_readiness()`. |
| [`ai_service/recommendation/filters.py`](../ai_service/recommendation/filters.py) | **Modified** | Updated `build_catalog_policy_attributes_query()` join to enforce `policies.current_version_id` `NULL`-safe matching; added duplicate `policy_id` runtime guard in `load_catalog_policies_from_db()`. |
| [`ai_service/tests/test_schema_draft.py`](../ai_service/tests/test_schema_draft.py) | **Modified** | Verified `server/migrations/003_recommendation_schema.sql` column/constraint parity with `common/tables.py`, `CREATE TABLE` PostgreSQL compilation, exclusion of M2/M3 vector tables, and preservation of v1–v3 SQL drafts. |
| [`ai_service/tests/test_stage8b_versioning_and_schema.py`](../ai_service/tests/test_stage8b_versioning_and_schema.py) | **Created** | `6` tests covering current-version matching, exclusion of previous/pending/stray-NULL/private rows, duplicate rejection, fail-closed missing/incomplete attributes, `ranking_features` alignment, and readiness checks. |
| [`doc/M5-recommendation-database-design.md`](M5-recommendation-database-design.md) | **Modified** | Added Stage 8B Architecture Alignment Addendum & M2 Policy Attribute Input Contract; updated verified `36 / 7 / 16 = 59` feature breakdown. |
| [`ai_service/README.md`](../ai_service/README.md) | **Modified** | Recorded Stage 8A, Stage 8A.1, and Stage 8B completion/review state. |
| [`doc/M5-stage-8B-implementation-report.md`](M5-stage-8B-implementation-report.md) | **Created** | This Stage 8B implementation summary. |

**Preserved Untouched:** `001_initial_schema.sql`, `002_add_platform_role.sql`, `server/scripts/migrate.js`, `proposed_003_recommendation_schema.sql` (v1), `proposed_003_recommendation_schema_v2.sql` (v2), `proposed_003_recommendation_schema_v3.sql` (v3), `doc/M5-stage-8A-final-validation.md`, `features.py` (`59` features, `FEATURE_SCHEMA_VERSION = "1.0.0"`), `train_lightgbm.py`, and `service.py` API contract.

---

## 2. Architecture Ownership & `ranking_features` Compatibility Decision

1. **Migration Ownership Respected:** `server/migrations/003_recommendation_schema.sql` conforms strictly to Node [`server/scripts/migrate.js`](../server/scripts/migrate.js) conventions (sequential numbering, idempotent `IF NOT EXISTS`, no inner `BEGIN`/`COMMIT`) and creates **only** Python-owned M5 tables (`policy_attributes`, `ranking_features`). M2/M3 vector infrastructure (`sections`, `clauses`, `chunks`, `CREATE EXTENSION vector`) was intentionally excluded.
2. **`ranking_features` Decision:** Retained in both [`003_recommendation_schema.sql`](../server/migrations/003_recommendation_schema.sql) and [`ai_service/common/tables.py`](../ai_service/common/tables.py) per [`doc/project-architecture.md §2A`](project-architecture.md#L83), upgraded to:
   - `feature_schema_version VARCHAR(32) NOT NULL DEFAULT '1.0.0'` (aligned with Stage 5 `FEATURE_SCHEMA_VERSION = "1.0.0"` instead of `'v1'`).
   - Normalized quality score bounds `[0, 1]` on the 5 optional subscores (`value_for_money_score`, `coverage_breadth_score`, `waiting_period_friendliness`, `network_strength_score`, `claim_reliability_score`).
   - Version-scoped uniqueness: `UNIQUE NULLS NOT DISTINCT (policy_id, policy_version_id, feature_schema_version)`.
   - **Dynamic vs. Precomputed Assessment:** M5 Stage 5 ([`features.py`](../ai_service/recommendation/features.py)) continues computing all `59` features dynamically at inference time (`36` policy-attribute signals + `7` profile signals + `16` interaction signals; `23` depend on runtime user profile inputs and cannot be stored statically per policy). `ranking_features` serves strictly as an optional policy-static subscore/feature cache table preserving `§2A` compatibility.

---

## 3. Version-Aware Join & Fail-Closed Retrieval Behavior

In [`ai_service/recommendation/filters.py`](../ai_service/recommendation/filters.py#L1031-L1079), `build_catalog_policy_attributes_query()` now joins `policy_attributes` to `policies` via:

```python
policies.join(
    policy_attributes,
    and_(
        policies.c.id == policy_attributes.c.policy_id,
        or_(
            policy_attributes.c.policy_version_id == policies.c.current_version_id,
            and_(
                policies.c.current_version_id.is_(None),
                policy_attributes.c.policy_version_id.is_(None),
            ),
        ),
    ),
)
```

Guaranteed runtime properties:
- **Current-Version Isolation:** Active `v1` attributes stay live during async `v2` ingestion; switching `policies.current_version_id` cuts over atomically to `v2` while preserving historical `v1`.
- **No Duplicate or Stale Rows:** `policy_attributes_policy_version_uq` prevents duplicate DB rows per version, and `load_catalog_policies_from_db()` adds a defensive runtime duplicate `policy_id` check raising `ValueError`.
- **Fail-Closed Stale Cutover Protection:** If `policies.current_version_id = V3` before `V3` attributes are written, the join returns `0` rows rather than serving outdated `V1` attributes.
- **Read-Only Inference Guarantee:** Database queries remain strictly `SELECT`-only (`Connection.execute` reads catalog attributes; missing `policy_attributes` raises `PolicyAttributesTableUnavailableError` → `HTTP 503`).

---

## 4. M2 Ingestion Contract Summary (Coordinated Boundary)

Full specification added in [`M5-recommendation-database-design.md §8.1`](M5-recommendation-database-design.md#L211-L234):
- **Categories:** Normalized lowercase `'health' | 'car' | 'life' | 'term'` on `policies.policy_type`.
- **Activation Gate:** A policy becomes recommendation-eligible **only** when ALL hold simultaneously:
  1. `policies.organization_id IS NOT NULL` and `policies.owner_user_id IS NULL` (private user uploads excluded per `FR-ING-12`).
  2. `policies.status = 'active'` and `policies.current_version_id = policy_attributes.policy_version_id`.
  3. `policy_attributes.is_catalog_eligible = TRUE`.
  4. All mandatory numeric bounds (`min_entry_age`, `max_entry_age`, `annual_premium_min/max`, `sum_insured_min/max`) and category-specific `type_specific_metadata` keys (`vehicle_types` / `vehicle_age_limit_years` for `car`; `smoking_status_eligibility` for `term`; etc.) are populated.
- **Fail-Closed Behavior:** Unset DDL defaults (`is_catalog_eligible=FALSE`, `coverage_mode='individual'`, `ped_covered=FALSE`, `cashless_available=FALSE`, `pan_india_coverage=FALSE`) and Stage 4 `MANDATORY_POLICY_DATA` prevent incomplete drafts from entering Stage 5/6 scoring.
- **Requires Coordination with M2 / M4:** Lowercase `policy_type` enforcement on Node catalog writes, M2 `attribute_extractor` JSON Schema validation for `type_specific_metadata`, and atomic cutover protocol when updating `policies.current_version_id`.

---

## 5. Verification Results (Directly Observed)

| Check | Exact Command | Observed Output |
|---|---|---|
| Ruff Check | `.venv\Scripts\ruff.exe check .` | **`All checks passed!`** (`0` errors) |
| Pytest (no DB credentials) | `.venv\Scripts\pytest.exe -v` | **`98 passed, 2 skipped`** in `5.28s` (`test_database_integration.py` & `test_inference.py` live DB tests skipped cleanly without `DB_PASSWORD`) |
| Pytest (read-only live PostgreSQL via `server/.env`) | `python -m pytest -v` (with `server/.env` env) | **`100 passed`** in `5.49s` (`test_database_integration.py` + `test_schema_draft.py` (3) + `test_stage8b_versioning_and_schema.py` (6) + Stages 2–7 suites all green) |

---

## 6. Remaining Cross-Owner Review & Execution Requirements

1. **Dual-Owner Migration PR Review (`§2A` / Risk R8):** Review and approve [`server/migrations/003_recommendation_schema.sql`](../server/migrations/003_recommendation_schema.sql) across Node/DB owner and M5 owner before running `npm run migrate`.
2. **Pre-Migration Backup & Execution:** Create a `pg_dump` backup of PostgreSQL before executing `node server/scripts/migrate.js` in `server/` (not executed during Stage 8B).
3. **Shared `pgvector` Extension Migration (M2 / M3 / Shared DB):** Formalize `CREATE EXTENSION IF NOT EXISTS vector;` and vector tables (`sections`, `clauses`, `chunks`) in a separate M2/M3 PR (`vector 0.8.7` is currently only installed out-of-band on the dev DB).
4. **M2/M4 Ingestion Contract Alignment:** Enforce `policies.policy_type` domain validation, `type_specific_metadata` completeness, and catalog publishing before live data seeding (`policies` currently has `0` rows).

## 7. Known Limitations

- **Unapplied Migration:** `policy_attributes` and `ranking_features` do not exist in live PostgreSQL until `003_recommendation_schema.sql` is reviewed and executed; live `/api/v1/recommend` continues returning controlled `HTTP 503` while `/health` remains `200 OK`.
- **Empty Catalog & Unpopulated RAG Citations:** Live DB holds `0` catalog policies, `ranking_features` remains unpopulated (Stage 5 computes all `59` features dynamically), and `RecommendedPolicy.citations` remains `[]` until M3/M8 integration.
- **Git Safety:** Branch remains `shravan-feature/recommendation-module`; no commits, pushes, merges, PRs, or DB writes performed.
