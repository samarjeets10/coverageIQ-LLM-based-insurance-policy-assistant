# Stage 8A — M5 Recommendation PostgreSQL Schema Audit & Database Design

**Project:** CoverageIQ — LLM-Based Insurance Policy Assistant  
**Module:** M5 — Policy Recommendation (Architecture §3.4 / §2A)  
**Developer:** Shravan Sambhaji Kapare (Roll No. 203, Group B2)  
**Git Branch:** `shravan-feature/recommendation-module`  
**Stage Scope:** Read-Only Schema Audit & Final Database Design (No migrations applied; no Python/Node/DB mutations)

---

## 1. Current PostgreSQL Audit (Verified Read-Only)

Direct read-only inspection of local PostgreSQL (`coverageiq_db`) confirmed:

- **PostgreSQL Engine:** `PostgreSQL 18.6` (`x86_64-windows`)
- **Installed Extensions:**
  - `plpgsql` (`1.0`, schema `pg_catalog`)
  - `vector` (`0.8.7`, schema `public`) — **installed out-of-band** (neither [`001_initial_schema.sql`](../server/migrations/001_initial_schema.sql) nor [`002_add_platform_role.sql`](../server/migrations/002_add_platform_role.sql) contains `CREATE EXTENSION IF NOT EXISTS vector`)
- **Vector Columns / HNSW Indexes Currently Present:** **None (`0`)**
- **Applied Migrations in `schema_migrations`:**
  1. `001_initial_schema.sql`
  2. `002_add_platform_role.sql`

### 1.1 Existing Table Inventory (12 Public Tables)

| Table | Owner | Row Count | Recommendation Role | Missing Elements |
|---|---:|---:|---|---|
| `users` | Node | `0` | Owner reference for private uploads (`owner_user_id`, `created_by`) | `platform_role` added by `002` |
| `organizations` | Node | `0` | Catalog publisher (`organization_id`) | Present; metadata only |
| `organization_members` | Node | `0` | RBAC / catalog publish rights (Node gateway) | Present |
| `refresh_tokens` | Node | `0` | Auth only | None for M5 |
| `policies` | Node | `0` | **Core parent catalog metadata** (`id`, `name`, `insurer_name`, `policy_type`, `status`, `current_version_id`, ownership scope) | Lacks `CHECK` / `NOT NULL` on `policy_type`; contains `0` rows |
| `policy_versions` | Node | `0` | Versioned PDF lineage (`id`, `policy_id`, `version_number`, `UNIQUE (id, policy_id)`) | Present; no attributes child table exists |
| `uploads` | Node | `0` | Raw uploaded file metadata | None for M5 |
| `jobs` | Node | `0` | Async M2 ingestion job tracking | Upstream producer for M5 attributes |
| `chat_sessions` | Node | `0` | M7 Q&A sessions | None for M5 |
| `chat_messages` | Node | `0` | M7 Q&A answers + citations | None for M5 |
| `audit_logs` | Node | `0` | Cross-service action audit | Present |
| `schema_migrations` | Node | `2` | Migration runner ledger (`filename`, `applied_at`) | Only `001` & `002` applied |

### 1.2 Missing Planned Tables Identified

- **`policy_attributes`** — **Missing** (causes `/api/v1/recommend` and `/ai/recommend` to return controlled `HTTP 503` in Stage 7).
- **`ranking_features`** — **Missing** (audited in Section 5; recommended **not required**).
- **M2/M3 knowledge-base tables (`sections`, `clauses`, `chunks`) & M8 audit tables (`audit_claims`, `audit_verdicts`)** — **Missing** (owned by M2/M3/M8, outside M5 relational filtering).

### 1.3 Verified Constraints & Indexes on `policies` and `policy_versions`

- `policies_check`: mutual exclusivity between private uploads (`owner_user_id IS NOT NULL AND organization_id IS NULL`) and catalog policies (`owner_user_id IS NULL AND organization_id IS NOT NULL`).
- `policies_status_check`: `'draft' | 'processing' | 'active' | 'failed' | 'archived'`.
- `policies_current_version_fk`: `FOREIGN KEY (current_version_id, id) REFERENCES policy_versions (id, policy_id) DEFERRABLE INITIALLY DEFERRED`.
- `policies` partial indexes: `policies_organization_idx` (`WHERE organization_id IS NOT NULL`) and `policies_owner_user_idx` (`WHERE owner_user_id IS NOT NULL`).
- `policy_versions`: composite `UNIQUE (id, policy_id)` enabling version-safe composite foreign keys from child tables (`policy_attributes`).

---

## 2. Recommendation Data Contract (`policies` + `policy_attributes`)

[`filters.py`](../ai_service/recommendation/filters.py) (Stage 4), [`features.py`](../ai_service/recommendation/features.py) (Stage 5), and [`service.py`](../ai_service/recommendation/service.py) (Stage 7) consume policy attributes via [`build_catalog_policy_attributes_query`](../ai_service/recommendation/filters.py#L960-L1036).

### 2.1 Parent Catalog Fields (`policies` Table — Node-Owned)

| Column | Type | Used By | Purpose & Behavioral Note |
|---|---|---|---|
| `id` (`policy_id`) | `UUID` PK | `filters`, `service` | Unique policy identifier, tie-breaker in ranking sort, API `RecommendedPolicy.policy_id` |
| `organization_id` | `UUID` FK | `filters` SQL query | Must be `NOT NULL` for catalog inclusion |
| `owner_user_id` | `UUID` FK | `filters` SQL query | Must be `NULL` (excludes private uploads per `FR-ING-12`) |
| `name` (`policy_name`) | `VARCHAR(250)` | `filters`, `service` | Display title in `FilterReport` and `RecommendedPolicy` |
| `insurer_name` | `VARCHAR(200)` NULL | `service` | Display insurer in `RecommendedPolicy.insurer_name` |
| `policy_type` | `VARCHAR(50)` NULL | `filters`, `features`, `service` | Must hold `'health' | 'car' | 'life' | 'term'` for catalog entries (unconstrained in `001` — see Mismatch #1) |
| `description` | `TEXT` NULL | `filters` select list | Preserved metadata for catalog inspection |
| `status` (`policy_status`) | `VARCHAR(20)` | `filters` (`POLICY_ACTIVE_STATUS`) | Must equal `'active'` to pass hard filter |
| `current_version_id` | `UUID` NULL | `filters` select list | Points to active `policy_versions.id` (version join gap — see Section 6.2) |

### 2.2 Relational Policy Attributes (`policy_attributes` Table — M5-Owned)

#### A. Common Core Attributes (All 4 Categories)

| Column | SQL Type | Required | Consumed By (`filters.py` / `features.py` / `service.py`) | Purpose |
|---|---|---|---|---|
| `id` | `UUID` PK | Yes | Internal PK | Surrogate key (`gen_random_uuid()`) |
| `policy_id` | `UUID` FK → `policies(id)` | Yes | Join key + cascade delete | Links attributes to parent catalog policy |
| `policy_version_id` | `UUID` FK (composite) | Recommended `NULL`-tolerant FK | Version provenance (`policy_versions(id, policy_id)`) | Isolates live `v1` attributes during `v2` ingestion |
| `is_catalog_eligible` | `BOOLEAN` | Yes | `filters` (`CATALOG_ELIGIBLE`) + SQL `WHERE` | Explicit catalog opt-in independent of upload status |
| `min_entry_age`, `max_entry_age` | `INTEGER` | Yes | `filters` (`AGE_MIN`, `AGE_MAX`, mandatory check); `features` (`age_eligibility_margin`, `remaining_eligible_years`) | Entry age bounds `[0, 120]` |
| `max_renewal_age`, `lifelong_renewal` | `INTEGER` NULL, `BOOLEAN` | `lifelong` Yes | `filters` (`RENEWAL_AGE_MAX`, `LIFELONG_RENEWAL`); `features` (`remaining_eligible_years`); `service` highlight | Renewal horizon evaluation |
| `coverage_mode`, `max_family_members` | `VARCHAR(30)`, `INTEGER` | Yes | `filters` (`COVERAGE_MODE`, `FAMILY_SIZE_MAX`); `features` (`family_floater_indicator`) | `'individual' | 'family_floater' | 'both'` |
| `eligible_applicant_types` | `JSONB` (array) | Yes | `filters` (`APPLICANT_TYPE`) | Array of e.g. `'salaried'`, `'self_employed'`, `'student'`, `'senior_citizen'` |
| `annual_premium_min`, `annual_premium_max` | `NUMERIC(12,2)` | Yes | `filters` (`BUDGET_MAX`, `BUDGET_MIN`); `features` (`annual_premium`, `monthly_premium`, `premium_to_budget_ratio`, `affordability_margin`, `affordability_score`); `service` | Premium bounds (`Decimal` → `float`) |
| `sum_insured_min`, `sum_insured_max` | `NUMERIC(14,2)` | Yes | `filters` (`SUM_INSURED_MIN`, `SUM_INSURED_MAX`); `features` (`sum_insured`, `coverage_to_requirement_ratio`, `coverage_headroom`); `service` | Base cover bounds (`Decimal` → `float`) |
| `deductible_amount`, `copay_pct` | `NUMERIC(12,2)`, `NUMERIC(5,2)` | Yes | `filters` (`DEDUCTIBLE_MAX`, `COPAY_MAX`); `features` (`deductible_amount`, `co_payment_percentage`); `service` highlight | Cost-sharing parameters |
| `network_hospitals_count`, `cashless_available` | `INTEGER`, `BOOLEAN` | Yes | `filters` (`HOSPITAL_NETWORK_MIN`, `CASHLESS_REQUIRED`); `features` (`hospital_network_count`, `cashless_availability_indicator` gated on `health`); `service` highlight | Cashless network strength (hospital/garage count) |
| `pan_india_coverage`, `available_regions` | `BOOLEAN`, `JSONB` (array) | Yes | `filters` (`GEOGRAPHIC_AVAILABILITY`); `features` (`pan_india_coverage_indicator`) | Geographic eligibility |
| `min_tenure_years`, `max_tenure_years` | `INTEGER` | Yes | `filters` (`TENURE_COMPATIBILITY`, life/term fallback); `features` (`policy_term_years` fallback) | Policy duration bounds |
| `covered_benefits`, `exclusions` | `JSONB` (arrays) | Yes | `filters` (`REQUIRED_BENEFITS`, `REQUIRED_BENEFITS_EXCLUDED`, `PED_COVERAGE`); `features` (`total_benefit_count`, `matched_required_benefits`, `required_benefit_match_ratio`, `exclusion_conflict_count`) | Standardized lowercase tag matching |
| `claim_settlement_ratio` | `NUMERIC(5,2)` NULL | Nullable | `features` (`claim_settlement_ratio`, `has_claim_settlement_ratio`); `service` highlight | Insurer reliability (ranked #2 gain in Stage 6) |
| `no_claim_bonus_pct`, `restoration_benefit` | `NUMERIC(5,2)`, `BOOLEAN` | Yes | `features` (`no_claim_bonus_pct`, `restoration_benefit_indicator`); `service` highlight | Value-add benefits |
| `extraction_source`, `extraction_confidence` | `VARCHAR(30)`, `NUMERIC(4,3)` NULL | Source Yes | Provenance auditing (`'manual' | 'synthetic' | 'ingestion_llm' | 'verified'`) | Separates verified catalog rows from unreviewed extractions |

#### B. Health-Specific Relational Columns (`policy_attributes`)

| Column | Type | Consumed By | Purpose |
|---|---|---|---|
| `room_rent_limit_type` | `VARCHAR(30)` | `filters` (`ROOM_RENT_LIMIT`), `features` (`room_rent_restriction_indicator`) | `'no_limit' | 'single_private' | 'shared' | 'fixed_per_day' | 'pct_sum_insured'` |
| `room_rent_limit_value` | `NUMERIC(12,2)` NULL | Audit/display metadata | Required if limit type is `'fixed_per_day'` or `'pct_sum_insured'` |
| `initial_waiting_period_days` | `INTEGER` | `filters` (`INITIAL_WAITING_PERIOD`), `features` (`initial_waiting_period_days` health-gated) | Initial waiting days |
| `ped_covered` | `BOOLEAN` | `filters` (`PED_COVERAGE`), `features` (`ped_coverage_indicator`), `service` | Pre-existing disease coverage flag |
| `ped_waiting_period_months` | `INTEGER` NULL | `filters` (`PED_WAITING_PERIOD`), `features` (`ped_waiting_period_days` = months × 30), `service` | Nullable when `ped_covered = FALSE` or category ≠ `'health'` |
| `specific_disease_waiting_months`, `maternity_waiting_months` | `INTEGER`, `INTEGER` NULL | Catalog metadata (reserved; not consumed in Stage 4/5 rules yet) | Health disclosure fields from M2 extraction |

#### C. Category-Specific JSONB (`type_specific_metadata`) for Car, Life, and Term

Stored as typed JSONB (`object`) rather than sparse nullable columns on `policy_attributes`:

| Category | `type_specific_metadata` Keys | Consumed By | Missing / Fallback Behavior |
|---|---|---|---|
| **Car** | `vehicle_types` (`list[str]`) | `filters` (`CAR_VEHICLE_TYPE`) | Rejects buyer specifying `vehicle_type` if absent/empty |
| **Car** | `vehicle_age_limit_years` (`int`) | `filters` (`CAR_VEHICLE_AGE_MAX`), `features` (`vehicle_age_limit`, `vehicle_age_eligibility_margin`) | Rejects buyer specifying `vehicle_age_years` if `None`; features → `0.0` |
| **Car** | `own_damage_coverage`, `third_party_coverage`, `zero_depreciation`, `roadside_assistance`, `engine_protection` (`bool`) | `filters` (`CAR_*`), `features` (`*_indicator`) | `filters` falls back to `covered_benefits` tags when `None`; `features` → `0.0` |
| **Life & Term** | `life_cover_min`, `life_cover_max` (`number`) | `filters` (`*_COVER_MIN/MAX`), `features` (`life_cover`, `life_cover_ratio`) | Cleanly falls back to `sum_insured_min` / `sum_insured_max` |
| **Life & Term** | `policy_term_years_min`, `policy_term_years_max` (`int`) | `filters` (`*_POLICY_TERM`), `features` (`policy_term_years`) | Cleanly falls back to `min_tenure_years` / `max_tenure_years` |
| **Life & Term** | `premium_payment_term_years` (`int`) | `filters` (`*_PREMIUM_PAYMENT_TERM`), `features` (`premium_payment_term_years`) | `filters` rejects if user caps PPT and value is `None`; `features` → `0.0` |
| **Life** | `maturity_benefit`, `surrender_value_available` (`bool`) | `filters` (`LIFE_MATURITY_BENEFIT`, `LIFE_SURRENDER_VALUE`), `features` | Fallback on `maturity_benefit` tag (`covered_benefits`); `0.0` if missing |
| **Term** | `smoking_status_eligibility` (`'all' \| 'smoker_surcharge' \| 'non_smoker_only'`) | `filters` (`TERM_SMOKING_ELIGIBILITY`), `features` (`smoking_eligibility_compatibility`) | `filters` fail-closes smokers unless `'all' \| 'smoker_surcharge'`; `features` defaults to `'all'` (`1.0`) if unflagged |
| **Term** | `critical_illness_rider`, `accidental_death_rider` (`bool`) | `filters` (`TERM_*_RIDER`), `features` (`*_indicator`) | `filters` checks tag fallback; `features` → `0.0` |

> **Design Note:** `policy_code` (present in synthetic CSVs and `features.TrainingFeatureDataset.metadata`) is **not** a column in `001_initial_schema.sql`'s `policies` table (`features.py` safely uses `.get("policy_code", "")`, and `service.py` does not depend on it).

---

## 3. Complete Feature Matrix Mapping vs. `ranking_features` Assessment

### 3.1 Static vs. Pairwise Split of the 59 Stage 5 Features

Stage 5 [`FEATURE_COLUMNS`](../ai_service/recommendation/features.py#L53-L122) contains `59` features:

- **`32` Purely Policy-Static Signals (100% present in `policies` + `policy_attributes`):**
  - Category flags (`policy_type_encoded`, `is_health`, `is_car`, `is_life`, `is_term`, `has_claim_settlement_ratio`)
  - Premiums & cover (`annual_premium`, `monthly_premium`, `sum_insured`)
  - Quality & benefit counts (`total_benefit_count`, `claim_settlement_ratio`, `deductible_amount`, `no_claim_bonus_pct`, `restoration_benefit_indicator`, `pan_india_coverage_indicator`)
  - Category-gated policy specs (`ped_coverage_indicator`, `ped_waiting_period_days`, `initial_waiting_period_days`, `room_rent_restriction_indicator`, `cashless_availability_indicator`, `hospital_network_count`, `co_payment_percentage`, `family_floater_indicator`, `vehicle_age_limit`, `own_damage_indicator`, `third_party_indicator`, `zero_depreciation_indicator`, `roadside_assistance_indicator`, `engine_protection_indicator`, `life_cover`, `policy_term_years`, `premium_payment_term_years`, `maturity_benefit_indicator`, `surrender_value_indicator`, `critical_illness_rider_indicator`, `accidental_death_rider_indicator`)
- **`27` Query-Dependent / User-Policy Interaction Signals (cannot be precomputed per policy):**
  - User profile inputs (`has_annual_income`, `has_vehicle_age`, `applicant_age`, `annual_income`, `monthly_budget`, `family_size`, `risk_preference_encoded`, `requested_sum_insured`, `vehicle_age`, `requested_life_cover`)
  - Pairwise interactions (`premium_to_budget_ratio`, `affordability_margin`, `affordability_score`, `coverage_to_requirement_ratio`, `coverage_headroom`, `age_eligibility_margin`, `remaining_eligible_years`, `matched_required_benefits`, `required_benefit_match_ratio`, `exclusion_conflict_count`, `vehicle_age_eligibility_margin`, `life_cover_ratio`, `smoking_eligibility_compatibility`)

### 3.2 Verdict: Is `ranking_features` Necessary?

**No — `ranking_features` is unnecessary and should be deferred/removed from the `003` migration.**

1. **Zero Usage:** Neither `filters.py`, `features.py`, `train_lightgbm.py`, nor `service.py` queries `ranking_features` (its 5 summary subscores—`value_for_money_score`, `coverage_breadth_score`, `waiting_period_friendliness`, `network_strength_score`, `claim_reliability_score`—are unused anywhere in Stages 4–7).
2. **Schema Drift Already Observed:** Original [`proposed_003_recommendation_schema.sql`](../ai_service/sql/proposed_003_recommendation_schema.sql#L137) defaulted `ranking_features.feature_schema_version` to `'v1'`, conflicting with Stage 5/6/7 `FEATURE_SCHEMA_VERSION = "1.0.0"`.
3. **Dynamic Runtime Efficiency:** All ~32 policy attributes already reside canonically in `policy_attributes`, while the ~27 pairwise interaction features depend strictly on the incoming user request. Computing all 59 floats dynamically in pandas/NumPy executes in milliseconds per request with guaranteed train/serve parity.

> Until deferred persistence is needed at massive scale (`>10k` catalog policies) or `ranking_features` is deprecated in [`common/tables.py`](../ai_service/common/tables.py) and [`config/database.py`](../ai_service/config/database.py), [`config/database.py`](../ai_service/config/database.py) will simply report `tables_present["ranking_features"] = False` alongside healthy DB readiness.

---

## 4. pgvector Ownership & Cross-Module Architecture

Per [`project-architecture.md`](project-architecture.md#L78-L87), [`modules-features.md`](modules-features.md#L64-L150), and team responsibility assignments:

| Layer / Object | Owning Module | Service / Writer | M5 (Recommendation) Relationship |
|---|---|---|---|
| Shared Migration Runner (`server/migrations/*.sql`) | Node / DB Governance | Reviewed jointly before merge | Only place DDL runs (`migrate.js`) |
| `CREATE EXTENSION IF NOT EXISTS vector;` | Shared DB Infra (or M3) | Currently missing from `001`/`002` (installed out-of-band on dev DB) | Must be formalized in `server/migrations/` before M2/M3 |
| `users`, `organizations`, `policies`, `policy_versions`, `jobs`, `chat_*` | Node API (`M1`, `M4`) | Node writes; Python reads `policies` / updates `jobs.status` | Read-only parent join for `policy_attributes` |
| `policy_attributes` | **M5** (schema) & **M2/M4** (populators) | M2 ingestion extracts + M4 org curation activates; M5 reads | Primary relational store for hard filtering + ML features |
| `sections`, `clauses`, `chunks (embedding vector, tsvector)` | **M2** (write) & **M3** (retrieval API) | Python M2/M3 only; Node read-only for PDF citations | **No direct SQL dependency from M5.** Future explanation grounding calls M3 retrieval (`hybrid_search(policy_id, ...)`) on Top-N outputs only, then passes evidence to M8 to populate `citations`. |

**Why this separation is strict:** Hard filtering and tabular ranking require exact numeric/boolean/set equality constraints (`<= budget_max`, `>= min_sum_insured`, `ped_covered = TRUE`), which pgvector approximate nearest-neighbor search cannot guarantee. pgvector belongs strictly to semantic clause retrieval (`M3` / `M6B` / `M6C` and post-ranking `M5 -> M3 -> M8` citation grounding).

---

## 5. Audit of `proposed_003_recommendation_schema.sql` & Compatibility Analysis

Auditing [`ai_service/sql/proposed_003_recommendation_schema.sql`](../ai_service/sql/proposed_003_recommendation_schema.sql) against `001_initial_schema.sql`, `server/scripts/migrate.js`, `filters.py`, `features.py`, and `synthetic_policies.csv` identified **7 concrete issues**:

1. **Missing Version-Level Scoping (`UNIQUE (policy_id)` vs. Multi-Version `policies`):**
   - `policies.current_version_id` supports policy re-ingestion (`FR-CAT-02`), creating new `policy_versions` rows (`v1`, `v2`). `UNIQUE (policy_id)` prevents storing `v2` extracted attributes safely while `v1` remains live and active.
   - **Fix in v2:** `CONSTRAINT policy_attributes_policy_version_uq UNIQUE NULLS NOT DISTINCT (policy_id, policy_version_id)` (native in PostgreSQL 15+, verified running PG 18.6).
   - **Code compatibility mismatch:** Stage 4 [`build_catalog_policy_attributes_query`](../ai_service/recommendation/filters.py#L1031-L1033) joins solely on `policies.c.id == policy_attributes.c.policy_id`. Once `policy_attributes` holds multiple versions per `policy_id`, **Stage 8B must update that join** to filter `(policy_attributes.policy_version_id == policies.current_version_id)` (or `NULL` fallback when unversioned) to avoid duplicate rows.
2. **Non-Idempotent DDL:** Original `003` uses bare `CREATE TABLE` and `CREATE INDEX`. Revised [`proposed_003_recommendation_schema_v2.sql`](../ai_service/sql/proposed_003_recommendation_schema_v2.sql) adds `IF NOT EXISTS` throughout.
3. **Permissive Fail-Open Defaults:** Original `003` defaults `is_catalog_eligible DEFAULT TRUE` and `ped_covered DEFAULT TRUE`, risking unverified policies passing `CATALOG_ELIGIBLE` or `PED_COVERAGE`. Revised `v2` switches both to fail-closed `DEFAULT FALSE`.
4. **Missing Cross-Column Domain Checks:** Original `003` allows `coverage_mode = 'family_floater'` with `max_family_members = 1`, and `room_rent_limit_type IN ('fixed_per_day', 'pct_sum_insured')` with `room_rent_limit_value IS NULL`. Revised `v2` adds both `CHECK` constraints (verified against all `80` synthetic policies: `0` conflicts).
5. **Suboptimal & Incomplete Indexing:** Original `003` puts 4 inequality columns `(annual_premium_min, annual_premium_max, sum_insured_min, sum_insured_max)` in one B-tree index and omits GIN indexes on `eligible_applicant_types` and `type_specific_metadata`. Revised `v2` replaces it with `(policy_id, policy_version_id)`, `(annual_premium_min, sum_insured_max)`, plus GIN (`jsonb_path_ops`) on all JSONB filter columns.
6. **Parent `policies.policy_type` Left Unconstrained in `001`:** `001_initial_schema.sql` leaves `policies.policy_type` nullable `VARCHAR(50)` without lowercase/enum checks, whereas `filters.py` and `features.py` fail closed if `policy_type` is missing or invalid (`'health' | 'car' | 'life' | 'term'`). Single-table `CHECK` constraints on `policy_attributes` also cannot validate category-specific keys in `type_specific_metadata` across tables. **Requires M4/M2 ingestion validation gate prior to publishing `is_catalog_eligible = TRUE`.**
7. **Superfluous `ranking_features` Table & Version Drift:** Removed from [`proposed_003_recommendation_schema_v2.sql`](../ai_service/sql/proposed_003_recommendation_schema_v2.sql) (`'v1'` vs. `'1.0.0'` drift and zero code consumers). Note that [`test_schema_draft.py`](../ai_service/tests/test_schema_draft.py) and [`common/tables.py`](../ai_service/common/tables.py) still reference `ranking_features` from Stage 2 and must remain untouched until Stage 8B sign-off updates `common/tables.py` together with the approved migration.

---

## 6. Unresolved Team Dependencies & Required Approvals

| # | Decision / Dependency | Affected Teams / Owners | Impact if Unresolved |
|---|---|---|---|
| **D1** | **Approve & promote revised migration to `server/migrations/`** (`proposed_003_recommendation_schema_v2.sql` → `003_recommendation_schema.sql`) | Node/DB Owner (Prithviraj Patil / Samarjeet) + M5 (Shravan) | Live `/api/v1/recommend` remains `HTTP 503` |
| **D2** | **Approve removing `ranking_features`** from `003` (and cleaning up `common/tables.py` + `database.py` + `test_schema_draft.py` in Stage 8B) | Node/DB Owner + M5 | Keeping an empty unused table with `'v1'` vs `'1.0.0'` drift |
| **D3** | **Confirm version-scoped `policy_attributes` strategy** (`UNIQUE NULLS NOT DISTINCT (policy_id, policy_version_id)` + updating `filters.py` Stage 8B join to `policies.current_version_id`) vs. in-place `UNIQUE (policy_id)` upsert | M2 + M4 + M5 | Re-ingesting policy `v2` either risks clobbering active `v1` or returning duplicate candidates |
| **D4** | **Formalize `CREATE EXTENSION IF NOT EXISTS vector;`** and M2/M3 `chunks`/`sections`/`clauses` migration | Shared DB / M2 / M3 | Fresh CI / Docker PostgreSQL databases will lack pgvector despite dev DB having `0.8.7` |
| **D5** | **Catalog ingestion & seeding pipeline (`FR-ING-11`, `M4`)**: enforce lowercase `policies.policy_type IN ('health','car','life','term')` and populated `type_specific_metadata` before setting `is_catalog_eligible = TRUE`; seed real catalog data via Node/M4 (never unflagged synthetic rows in production) | M2 + M4 + M5 | Live catalog has `0` policies even after schema creation |

---

## 7. Recommended Implementation Sequence (Post-Approval, Stage 8B+)

1. **Team Review & Sign-Off:** Resolve Decisions **D1–D4** cleanly across M2, M3, M4/Node, and M5 owners.
2. **Official Migration PR (Node + DB):** Copy approved `proposed_003_recommendation_schema_v2.sql` into `server/migrations/003_recommendation_schema.sql` (and coordinate `CREATE EXTENSION IF NOT EXISTS vector;` + `policies.policy_type` validation rule). Execute via `node server/scripts/migrate.js`.
3. **Stage 8B Python Sync (`ai_service/`):** Update `common/tables.py`, `filters.build_catalog_policy_attributes_query` (version-scoped `current_version_id` join), `config/database.py`, and `test_schema_draft.py` to match the exact approved migration.
4. **Catalog Population & End-to-End Live Verification:** Populate approved organization catalog policies (`policies` + `policy_versions` + `policy_attributes` via M4/M2 or controlled dev catalog fixture), verify live `POST /api/v1/recommend` and `POST /ai/recommend` end-to-end against PostgreSQL, and proceed to next model/integration stages.

---

## 8. Stage 8B Architecture Alignment Addendum & M2 Ingestion Contract

> **Superseded Notes from Earlier Stage 8A Draft:**
> - **Feature Count Precision (Stage 8A.1):** Verified individually across all 59 rows of `FEATURE_COLUMNS`: **`36` policy-attribute signals + `7` customer-profile signals + `16` computed interaction signals = `59`** (`23` depend on runtime profile inputs; `36` depend solely on policy attributes).
> - **`ranking_features` Retained for `project-architecture.md §2A` Compliance (Stage 8B):** While M5 continues computing all 59 dynamic features at runtime (`FEATURE_SCHEMA_VERSION = "1.0.0"`), [`server/migrations/003_recommendation_schema.sql`](../server/migrations/003_recommendation_schema.sql) and [`ai_service/common/tables.py`](../ai_service/common/tables.py) **preserve `ranking_features`** per [`project-architecture.md §2A`](project-architecture.md#L83), upgraded with version-scoped uniqueness (`UNIQUE NULLS NOT DISTINCT (policy_id, policy_version_id, feature_schema_version)`), normalized subscore bounds `[0, 1]`, and `DEFAULT '1.0.0'`.

### 8.1 M5 Policy Attribute Input Contract (for M2 Ingestion & M4 Catalog Curation)

1. **Supported Categories:** Lowercase `'health'`, `'car'`, `'life'`, `'term'` stored in `policies.policy_type`.
2. **Required Common Columns (`policy_attributes` + parent `policies`):**
   - Parent `policies`: non-null `organization_id`, `owner_user_id IS NULL`, `status = 'active'`, `current_version_id = policy_versions.id`, lowercase `policy_type`.
   - `policy_attributes`: `policy_id`, `policy_version_id`, `is_catalog_eligible`, `min_entry_age`, `max_entry_age`, `coverage_mode` (`max_family_members >= 2` when `'family_floater'`/`'both'`), `eligible_applicant_types` (non-empty lowercase JSON array), `annual_premium_min <= annual_premium_max`, `sum_insured_min <= sum_insured_max`, `min_tenure_years <= max_tenure_years`, `covered_benefits`, `exclusions`, `network_hospitals_count`, `deductible_amount`, `copay_pct`, `initial_waiting_period_days`, `no_claim_bonus_pct`, `restoration_benefit`.
3. **Category-Specific JSONB (`type_specific_metadata`):**
   - **Health:** relational `room_rent_limit_type` (`room_rent_limit_value` required if `'fixed_per_day'`/`'pct_sum_insured'`), `ped_covered`, `ped_waiting_period_months`; optional metadata keys (`family_floater_supported`, `icu_limit_type`).
   - **Car:** `vehicle_types` (`list[str]`), `vehicle_age_limit_years` (`int`), boolean coverage flags (`own_damage_coverage`, `third_party_coverage`, `zero_depreciation`, `roadside_assistance`, `engine_protection`).
   - **Life:** `life_cover_min`, `life_cover_max`, `policy_term_years_min`, `policy_term_years_max`, `premium_payment_term_years`, `maturity_benefit`, `surrender_value_available`.
   - **Term:** `life_cover_min`, `life_cover_max`, `policy_term_years_min`, `policy_term_years_max`, `premium_payment_term_years`, `smoking_status_eligibility` (`'all' | 'smoker_surcharge' | 'non_smoker_only'`), `critical_illness_rider`, `accidental_death_rider`.
4. **Versioning & Activation Rules (`policy_version_id` ↔ `policies.current_version_id`):**
   - M2 writes new extracted rows against `(policy_id, policy_versions.id)` with `is_catalog_eligible = FALSE` until validated.
   - Only after validation should `policies.current_version_id`, `policies.status = 'active'`, and `policy_attributes.is_catalog_eligible = TRUE` be committed atomically (never exposing private user uploads where `owner_user_id IS NOT NULL`).
5. **Fail-Closed Guarantees:**
   - Unset defaults in `003_recommendation_schema.sql` (`is_catalog_eligible = FALSE`, `coverage_mode = 'individual'`, `ped_covered = FALSE`, `cashless_available = FALSE`, `pan_india_coverage = FALSE`) prevent partial records from passing M5 filters.
   - Missing mandatory numeric/category fields fail closed in Stage 4 (`MANDATORY_POLICY_DATA`, `CAR_VEHICLE_TYPE`, `TERM_SMOKING_ELIGIBILITY`, etc.), and missing `v2` attribute rows after `current_version_id` cutover return `0` rows from Stage 8B SQL instead of stale `v1` data.
6. **Items Requiring M2 / M4 Coordination Before Production Seeding:** *(a)* Lowercase `policy_type` enforcement at Node/M4 publish; *(b)* JSON Schema / Pydantic validation inside M2 `attribute_extractor` for `type_specific_metadata` before setting `is_catalog_eligible = TRUE`; *(c)* OpenAPI/contract publication under `/docs/contracts/`.
