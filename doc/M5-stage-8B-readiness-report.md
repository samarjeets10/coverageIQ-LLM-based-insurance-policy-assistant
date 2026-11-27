# Stage 8B — Pre-Implementation Readiness & Approval Verification Report

**Project:** CoverageIQ — LLM-Based Insurance Policy Assistant  
**Module:** M5 — Policy Recommendation (Architecture §3.4 / §2A)  
**Developer:** Shravan Sambhaji Kapare (Roll No. 203, Group B2)  
**Branch:** `shravan-feature/recommendation-module`  
**Status:** Read-Only Readiness Verification (No database writes, no migrations applied, no Stages 2–7 or `server/` modifications)

---

## 1. Verified File & Branch Inventory (No Assumed Paths)

Direct repository and remote-branch inspection confirmed actual file locations vs. referenced names:

| Referenced Name | Actual Verified Status / Location |
|---|---|
| `ai_service/doc/M5-recommendation-database-design.md` | **Path does not exist under `ai_service/`** → Actual file: [`doc/M5-recommendation-database-design.md`](M5-recommendation-database-design.md) |
| `M5-stage-8A-final-validation.md` | Actual file: [`doc/M5-stage-8A-final-validation.md`](M5-stage-8A-final-validation.md) |
| `ai_service/README.md` | Present: [`ai_service/README.md`](../ai_service/README.md) |
| `ai_service/recommendation/tables.py` | **Does not exist** → Actual file: [`ai_service/common/tables.py`](../ai_service/common/tables.py) |
| `ai_service/recommendation/database.py` | **Does not exist** → Actual file: [`ai_service/config/database.py`](../ai_service/config/database.py) |
| `proposed_003_recommendation_schema_v3.sql` | Present: [`ai_service/sql/proposed_003_recommendation_schema_v3.sql`](../ai_service/sql/proposed_003_recommendation_schema_v3.sql) (v1 & v2 preserved alongside) |
| `/docs/contracts/ai-service.openapi.yaml`, `decisionLog.md`, `databaseDesign.md` | **Do not exist anywhere** across local or remote branches (`main`, `feature/coverageiq-database-foundation`, `prithviraj-feature`, `samarjeet-feat`, `yash-feature`) |
| `server/migrations/` | Contains only `001_initial_schema.sql` and `002_add_platform_role.sql` |
| `server/scripts/migrate.js` | Present; Node `pg` sequential migration runner matching `/^\d+_[a-z0-9_-]+\.sql$/i` |

---

## 2. Team Approval Matrix (Decisions D1–D5)

Every remote branch (`origin/main`, `origin/feature/coverageiq-database-foundation`, `origin/prithviraj-feature`, `origin/samarjeet-feat`, `origin/yash-feature`) and all docs were inspected directly. None of **D1–D5** has recorded team sign-off in repository documentation or migrations.

| ID | Decision | Verification Status | Evidence & Conflicting Documentation |
|---|---|---|---|
| **D1** | Promote `proposed_003_recommendation_schema_v3.sql` into `server/migrations/003_recommendation_schema.sql` | **Not confirmed — Requires human / team approval** | [`doc/project-architecture.md`](project-architecture.md#L78-L87) mandates a single Node migration stream with dual-owner PR review (`Risk R8`). `server/migrations/` has only `001` & `002`. **Tooling conflict:** [`doc/roadmap.md`](roadmap.md#L51) mentions `Alembic`, whereas [`project-architecture.md`](project-architecture.md#L85-L245) & [`migrate.js`](../server/scripts/migrate.js) use Node plain SQL migrations. |
| **D2** | Defer/remove unused `ranking_features` table and code references | **Not confirmed — Conflicting documentation; requires human / team approval** | [`project-architecture.md §2A`](project-architecture.md#L83) explicitly lists `ranking_features` as a Python-owned table, and Stage 2 [`common/tables.py`](../ai_service/common/tables.py), [`config/database.py`](../ai_service/config/database.py), `v1` SQL, & [`test_schema_draft.py`](../ai_service/tests/test_schema_draft.py) define it. Stage 8A/8A.1 (`v3`) proved all `59` Stage 5 features compute dynamically (`36` static + `7` profile + `16` interaction) with zero `ranking_features` consumers. |
| **D3** | Version-aware uniqueness (`UNIQUE NULLS NOT DISTINCT (policy_id, policy_version_id)`) and `policies.current_version_id` matching | **Not confirmed — Compatible with `001`, requires team approval** | [`001_initial_schema.sql`](../server/migrations/001_initial_schema.sql) supports `policies.current_version_id` + `policy_versions(id, policy_id)`, and [`FR-CAT-02`](project-requirement.md#L69) requires versioned updates. However, Stage 2 `common/tables.py` used `UNIQUE (policy_id)` and Stage 4 [`filters.py`](../ai_service/recommendation/filters.py#L1032) currently joins only on `policy_id`. Coordinated Stage 8B join alignment is required. |
| **D4** | `pgvector` extension & migration ownership between shared DB owners and M2/M3 | **Not confirmed — Conflicting / incomplete documentation; requires human / team approval** | M5 role summary listed `"PostgreSQL + pgvector schema"`, while [`modules-features.md`](modules-features.md#L64-L150) & [`project-architecture.md`](project-architecture.md#L78-L87) assign vector storage/retrieval (`sections`, `clauses`, `chunks`) to **M2/M3**, relational `policy_attributes` to **M5**, and migrations to **Node**. Local PG `18.6` has `vector 0.8.7` installed **out-of-band** (absent from `001`/`002`), risking missing extension on clean CI/Docker DBs. |
| **D5** | M2 ingestion contract for populating `policy_attributes` and setting `is_catalog_eligible` | **Not confirmed — Missing contract & implementation; requires human / team approval** | [`FR-ING-11/12`](project-requirement.md#L62-L63) & [`modules-features.md`](modules-features.md#L79) assign `attribute_extractor` to M2, but no M2 Python code or `/docs/contracts/ai-service.openapi.yaml` exists on any branch. Furthermore, `001` leaves `policies.policy_type` nullable without category CHECKs, and `policy_attributes` single-table CHECKs cannot validate `type_specific_metadata` keys per category—requiring an M2/M4 publish gate. |

---

## 3. Outstanding Ownership Decisions & Unresolved Blockers

1. **Migration Governance Owner Sign-Off (Node DB Owner + M5):** Official approval to add `003_recommendation_schema.sql` into `server/migrations/` and reconcile whether `ranking_features` is formally dropped from `project-architecture.md §2A`.
2. **pgvector Formalization Migration (Shared DB / M2 / M3):** Confirm whether `CREATE EXTENSION IF NOT EXISTS vector;` belongs in an independent infrastructure migration (before M2/M3 `chunks`) or inside M2/M3's upcoming migration.
3. **Catalog Writer Guarantee (M2 / M4):** Confirm who populates normalized lowercase `policies.policy_type` (`'health' | 'car' | 'life' | 'term'`), composite `(policy_id, policy_version_id)`, valid JSONB `type_specific_metadata`, and atomic promotion (`policies.current_version_id`, `policies.status = 'active'`, `policy_attributes.is_catalog_eligible = TRUE`).
4. **Live Catalog Content (`0` rows):** Even after `003` creates `policy_attributes`, `coverageiq_db` holds **zero `organizations`, zero `policies`, and zero `policy_versions`**, so live `/api/v1/recommend` will transition from `503 Service Unavailable` to `200 NO_MATCHING_POLICIES` until real catalog policies are seeded via M4/M2.

---

## 4. Exact Files Stage 8B Will Modify (Post-Authorization Only)

No existing files were modified in Stage 8B. Once explicitly authorized, Stage 8B requires changes across the following exact targets:

| File / Target | Scope of Stage 8B Change | Requirement Satisfied |
|---|---|---|
| `server/migrations/003_recommendation_schema.sql` *(promoted copy of `v3`)* | Promote [`proposed_003_recommendation_schema_v3.sql`](../ai_service/sql/proposed_003_recommendation_schema_v3.sql) into shared Node migration directory after D1/D2/D3 sign-off. | Schema promotion via existing `migrate.js` runner |
| [`ai_service/common/tables.py`](../ai_service/common/tables.py) | Update `policy_attributes` SQLAlchemy Core metadata to match `v3` (`coverage_mode` default `'individual'`, fail-closed defaults, `UniqueConstraint("policy_id", "policy_version_id", postgresql_nulls_not_distinct=True)`, `v3` CHECK constraints); remove `ranking_features` table object (pending D2). | SQLAlchemy parity + D2 deferral |
| [`ai_service/config/database.py`](../ai_service/config/database.py) | Remove `"ranking_features"` from `EXPECTED_TABLES` (pending D2) so readiness checks track `("policies", "policy_versions", "organizations", "policy_attributes")`. | D2 removal + clean `/health` readiness |
| [`ai_service/recommendation/filters.py`](../ai_service/recommendation/filters.py#L960-L1036) | Update `build_catalog_policy_attributes_query` join from `policies.c.id == policy_attributes.c.policy_id` to include `or_(policy_attributes.c.policy_version_id == policies.c.current_version_id, and_(policies.c.current_version_id.is_(None), policy_attributes.c.policy_version_id.is_(None)))`. Keep fail-closed `MANDATORY_POLICY_DATA` & `CATALOG_ELIGIBLE` rules intact. | Preserve history; prevent duplicate/stale rows; fail closed |
| [`ai_service/recommendation/service.py`](../ai_service/recommendation/service.py) & [`schemas.py`](../ai_service/recommendation/schemas.py) | **No breaking API or feature schema edits needed.** Existing `FEATURE_SCHEMA_VERSION = "1.0.0"` (`59` features), error handlers, and `RecommendResponse` contract remain intact. Verify empty-catalog DB returns `200 NO_MATCHING_POLICIES` cleanly once `policy_attributes` exists. | Preserve existing API & feature schema |
| [`ai_service/tests/test_schema_draft.py`](../ai_service/tests/test_schema_draft.py) | Update draft path / assertions to validate promoted `003_recommendation_schema.sql` (or `v3`) vs `common/tables.py` without expecting `ranking_features` (pending D2). | Schema-SQL parity test |
| [`ai_service/tests/test_filters.py`](../ai_service/tests/test_filters.py#L420-L446) | Add compiled SQL assertions verifying version-safe `current_version_id` join predicates and zero regression across existing 10 filter tests. | Query safety verification |
| [`ai_service/tests/test_health_and_config.py`](../ai_service/tests/test_health_and_config.py#L54-L66) | Update mocked readiness fixture keys if `ranking_features` is removed from `EXPECTED_TABLES`. | Health test consistency |
| [`ai_service/tests/test_database_integration.py`](../ai_service/tests/test_database_integration.py) & [`tests/test_inference.py`](../ai_service/tests/test_inference.py#L753-L779) | Assert `policy_attributes` present in live DB once migrated; verify rolled-back transactional integration checks (no persistent DB writes) for `v1`/`v2` cutoff and fail-closed incomplete rows. | Live read-only + transactional verification |
| [`doc/project-architecture.md`](project-architecture.md#L83) *(shared PR)* | Update `§2A` table ownership list (`ranking_features` deferred/removed) alongside migration PR review. | Eliminate doc contradiction |

---

## 5. Proposed Migration Sequence (Post-Sign-Off)

1. **Governance Sign-Off:** Obtain explicit Node/DB + M2/M3/M5 approval on D1–D5 (plus decision on `CREATE EXTENSION IF NOT EXISTS vector;` placement).
2. **Backup Dev/Target PostgreSQL:** Run pre-migration snapshot (`pg_dump`) before executing DDL.
3. **Promote Migration SQL:** Place approved SQL into `server/migrations/003_recommendation_schema.sql` (verified syntax clean, no inner `BEGIN`/`COMMIT`, `IF NOT EXISTS`).
4. **Run Existing Node Migration Runner:** Execute `npm run migrate` (`node server/scripts/migrate.js`) from `server/`.
5. **Verify Live Catalog State:** Inspect `schema_migrations`, `policy_attributes` columns/defaults/CHECK constraints, composite FK `policy_attributes_version_fk`, unique index `policy_attributes_policy_version_uq`, and partial/GIN indexes read-only.
6. **Apply Stage 8B Python Sync (`ai_service/`):** Update `common/tables.py`, `config/database.py`, `filters.py` version join, and unit/integration tests; run Ruff + full pytest.

---

## 6. Stage 8B Execution Checklist (Do Not Execute Yet)

- [ ] **1. Pre-migration database backup:** Take clean `pg_dump` schema + data backup (`pg_dump --format=custom --file=backup_pre_003.dump ...`) and record `SELECT * FROM schema_migrations;` baseline.
- [ ] **2. Migration ownership & peer review confirmation:** Dual sign-off from Node/DB owner and M5 developer confirming `003_recommendation_schema.sql` matches [`proposed_003_recommendation_schema_v3.sql`](../ai_service/sql/proposed_003_recommendation_schema_v3.sql), `CREATE EXTENSION IF NOT EXISTS vector;` ownership is settled, and `ranking_features` deferral (D2) is approved.
- [ ] **3. Migration execution via Node runner:** Promote SQL to `server/migrations/003_recommendation_schema.sql` and execute `npm run migrate` (`node scripts/migrate.js`); confirm `Applied: 003_recommendation_schema.sql` and entry recorded in `schema_migrations`.
- [ ] **4. Schema & default verification:** Confirm `policy_attributes` exists with fail-closed defaults (`is_catalog_eligible=false`, `coverage_mode='individual'`, `ped_covered=false`, `cashless_available=false`, `pan_india_coverage=false`) and all 17+ domain `CHECK` constraints active.
- [ ] **5. Foreign-key verification:** Confirm `policy_attributes_policy_id_fkey` (`ON DELETE CASCADE`) and composite `policy_attributes_version_fk` (`(policy_version_id, policy_id) -> policy_versions(id, policy_id) ON DELETE CASCADE`) in `pg_constraint`.
- [ ] **6. Index verification:** Confirm unique B-tree `policy_attributes_policy_version_uq` (`indnullsnotdistinct = true`), 3 partial B-tree indexes (`WHERE is_catalog_eligible = TRUE`), and 5 `jsonb_path_ops` GIN indexes in `pg_index`.
- [ ] **7. Current-version query verification:** Verify updated `filters.build_catalog_policy_attributes_query()` compiles clean parameterized SQL matching `policies.current_version_id` and returns zero duplicates/stale versions across rolled-back `v1`/`v2`/unextracted/`NULL` test fixtures.
- [ ] **8. Database rollback plan readiness:** Confirm `migrate.js` rolls back failed runs automatically; validate manual rollback runbook for reverting a committed `003` run cleanly (`BEGIN; DROP TABLE IF EXISTS policy_attributes CASCADE; DELETE FROM schema_migrations WHERE filename = '003_recommendation_schema.sql'; COMMIT;` or restore `backup_pre_003.dump`).
- [ ] **9. Existing & updated test suite verification:** Run `.venv\Scripts\ruff.exe check .` and full `pytest -v` (verifying Stages 2–7 pass with zero regressions and `FEATURE_SCHEMA_VERSION == "1.0.0"` preserved).
- [ ] **10. Read-only live database integration tests:** Execute `test_database_integration.py` and `test_inference.py::test_live_postgresql_read_only_recommendation_readiness` against live PostgreSQL verifying `/health` reports `policy_attributes: True` and `/api/v1/recommend` stops returning missing-table `503` (returning `200 NO_MATCHING_POLICIES` on empty catalog or `RECOMMENDATIONS_READY` if test rows exist).

---

## 7. Database Backup and Rollback Strategy

- **Pre-Migration Backup:** Mandatory `pg_dump` custom-format archive (`--format=custom`) prior to invoking `npm run migrate`. Verify restore readability (`pg_restore --list`) before applying DDL.
- **Automatic Failure Protection (`migrate.js`):** [`server/scripts/migrate.js`](../server/scripts/migrate.js#L29-L63) wraps `003_recommendation_schema.sql` in a single PostgreSQL transaction (`BEGIN ... COMMIT`); any syntax, permission, or FK failure executes `ROLLBACK` before writing `schema_migrations`, leaving the live schema untouched.
- **Post-Commit Rollback (Manual Only — `migrate.js` has no `down` command):**
  1. Stop `ai_service` / pause catalog ingestion jobs.
  2. Verify no active external FKs depend on `policy_attributes` (it is a leaf table in `v3`).
  3. Execute inside a controlled transaction:
     ```sql
     BEGIN;
     DROP TABLE IF EXISTS policy_attributes CASCADE;
     DELETE FROM schema_migrations WHERE filename = '003_recommendation_schema.sql';
     COMMIT;
     ```
  4. Revert Stage 8B Python changes or restore pre-migration `pg_dump` archive if data loss recovery is needed.

---

## 8. Stage 8B Verification & Test Plan

1. **Ruff Compliance:** `ruff format --check .` and `ruff check .` across `ai_service/` (`0` errors).
2. **Unit & Mocked Tests (Zero DB Dependency):**
   - Updated `test_schema_draft.py` confirming `common/tables.py` aligns column-for-column and constraint-for-constraint with `003_recommendation_schema.sql`.
   - Updated `test_filters.py` asserting parameterized version-aligned `JOIN policy_attributes` SQL and bound parameters (`no raw interpolation`), plus fail-closed `MANDATORY_POLICY_DATA` handling for partial rows.
   - All existing `test_dataset.py`, `test_features.py`, `test_train_lightgbm.py`, `test_schemas.py`, and `test_inference.py` (`>= 93` total tests).
3. **Rolled-Back PostgreSQL Versioning Test (Local Dev DB Only, Inside `conn.begin()` -> `rollback()`):**
   - Prove `v1` active → `v2` processing → `v1` still served → `current_version_id = v2` → `v2` served (`1` row, no duplicates) → `v2` unextracted → fail-closed `0` rows → default `INSERT` succeeds with `coverage_mode='individual'` -> rollback cleans 100%.
4. **Live Read-Only Integration Verification:**
   - `GET /health` & `GET /ai/health` return `200` with `tables_present.policy_attributes == True`.
   - `POST /api/v1/recommend` returns controlled `200 NO_MATCHING_POLICIES` on empty catalog (`0` policies) rather than `503`.

---

## 9. Remaining Blockers & Go / No-Go Recommendation

### Remaining Blockers
1. **Unapproved Team Decisions D1–D5** (no PR/sign-off on `003_recommendation_schema.sql`, `ranking_features` removal from `project-architecture.md §2A`, or `pgvector` formal migration).
2. **No M2 Ingestion / M4 Catalog Seeding Implementation:** `policies` currently has `0` rows across all categories (`health`, `car`, `life`, `term`), and no M2/M4 contract test exists in `/docs/contracts/`.

### Explicit Go / No-Go Recommendation

> **NO-GO for executing Stage 8B code or database changes right now.**  
> **GO for team review sign-off.**

- **Why technical schema readiness is GO:** `proposed_003_recommendation_schema_v3.sql` is validated against PostgreSQL `18.6`, composite FK `policy_versions(id, policy_id)`, `migrate.js`, all `59` Stage 5 features, and all `80` synthetic policies.
- **Why implementation is NO-GO until sign-off:** Modifying `server/migrations/`, applying DDL to shared PostgreSQL, removing `ranking_features` specified in [`project-architecture.md §2A`](project-architecture.md#L83), or altering Stage 4–7 code requires explicit resolution of **D1–D5** by the shared DB/Node owner and team plus your authorization to start Stage 8B.
