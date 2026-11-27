# Stage 8A.1 — Final SQL Design Validation (`proposed_003_recommendation_schema_v2.sql` → `v3`)

**Project:** CoverageIQ — LLM-Based Insurance Policy Assistant  
**Module:** M5 — Policy Recommendation  
**Developer:** Shravan Sambhaji Kapare (Roll No. 203, Group B2)  
**Git Branch:** `shravan-feature/recommendation-module`  
**Validation Mode:** Read-only PostgreSQL + static SQL/Python validation (zero DB writes, zero applied migrations, Stages 2–7 untouched)

---

## Executive Validation Result

Validation of [`ai_service/sql/proposed_003_recommendation_schema_v2.sql`](../ai_service/sql/proposed_003_recommendation_schema_v2.sql) against live PostgreSQL `18.6` (`pg_constraint` / `pg_index`), [`001_initial_schema.sql`](../server/migrations/001_initial_schema.sql), [`migrate.js`](../server/scripts/migrate.js), [`filters.py`](../ai_service/recommendation/filters.py), [`features.py`](../ai_service/recommendation/features.py), and all `80` synthetic policies passed Checks 1–3 cleanly **except for 1 blocking default-vs-CHECK defect and 2 safety/consistency improvements identified in Check 4**:

- **Blocking Defect Found in `v2`:** `coverage_mode` defaulted to `'both'` (line 54) while `max_family_members` defaulted to `1` (line 56), causing **every default `INSERT` omitting both columns to fail** `CONSTRAINT policy_attributes_family_members_check CHECK (coverage_mode = 'individual' OR max_family_members >= 2)`.
- **Additional Corrections Required:**
  1. Unify fail-closed boolean defaults (`cashless_available DEFAULT FALSE`, `pan_india_coverage DEFAULT FALSE`, matching `is_catalog_eligible DEFAULT FALSE` and `ped_covered DEFAULT FALSE`).
  2. Enforce `CONSTRAINT policy_attributes_renewal_consistency_check CHECK (NOT lifelong_renewal OR max_renewal_age IS NULL)` matching [`features.py`](../ai_service/recommendation/features.py#L550-L555) horizon precedence (`0/80` conflicts in synthetic catalog).

Per Stage 8A.1 instructions, both [`proposed_003_recommendation_schema.sql`](../ai_service/sql/proposed_003_recommendation_schema.sql) (v1) and [`proposed_003_recommendation_schema_v2.sql`](../ai_service/sql/proposed_003_recommendation_schema_v2.sql) (v2) were **preserved untouched**, and the corrected DDL was written to a separate proposal:

> **[`ai_service/sql/proposed_003_recommendation_schema_v3.sql`](../ai_service/sql/proposed_003_recommendation_schema_v3.sql)** *(not applied to PostgreSQL)*

---

## 1. Composite Foreign Key Validation

Direct inspection of `pg_constraint` and `pg_index` on `coverageiq_db` (`PostgreSQL 18.6`) confirmed:

| Catalog View / Object | Constraint / Index Name | Observed Definition | Validation Verdict |
|---|---|---|---|
| `pg_constraint` (`policy_versions`) | `policy_versions_id_policy_id_key` | `contype = 'u'`, `UNIQUE (id, policy_id)` | **PASS** — Exact column order `(id, policy_id)` matches `FOREIGN KEY (policy_version_id, policy_id) REFERENCES policy_versions (id, policy_id)` |
| `pg_index` (`policy_versions`) | `policy_versions_id_policy_id_key` | `CREATE UNIQUE INDEX policy_versions_id_policy_id_key ON public.policy_versions USING btree (id, policy_id)` (`indisunique=True`, `indnullsnotdistinct=False`) | **PASS** — Backing unique B-tree index exists |
| `pg_constraint` (`policy_versions` columns) | `policy_versions_id_not_null`, `policy_versions_policy_id_not_null` | Both `id` and `policy_id` are `NOT NULL` | **PASS** |
| `pg_constraint` (`policies`) | `policies_current_version_fk` | `FOREIGN KEY (current_version_id, id) REFERENCES policy_versions(id, policy_id) DEFERRABLE INITIALLY DEFERRED`, `confmatchtype = 's'` (`MATCH SIMPLE`) | **Precedent Confirmed** — Existing `001` schema already uses this exact composite FK pattern |

### `MATCH SIMPLE` & Cascade Delete Edge Cases Verified
- Under PostgreSQL default `MATCH SIMPLE`, if `policy_attributes.policy_version_id IS NULL` (manual catalog row without a PDF version), the composite FK `policy_attributes_version_fk` skips validation while the separate single-column FK `policy_id REFERENCES policies(id) ON DELETE CASCADE` still enforces valid parent existence and cascade deletion when the parent `policies` row is removed.
- If `policy_version_id` is non-null, PostgreSQL guarantees BOTH `(policy_version_id)` belongs to `policy_versions` AND that version belongs to the exact same `policy_id`, preventing cross-policy version hijacking.
- Deleting a `policy_versions` row (`ON DELETE CASCADE`) deletes only `policy_attributes` rows linked to that version (`policy_version_id`), leaving `NULL`-version rows intact unless the parent `policies` row itself is deleted. Note an existing `001` quirk: `policies.policies_current_version_fk` uses `ON DELETE NO ACTION`, so deleting a `policy_versions` row that is still referenced by `policies.current_version_id` is rejected unless `policies.current_version_id` is updated/cleared first (soft deletion via `policies.status = 'archived'` + `policy_attributes.is_catalog_eligible = FALSE` avoids this cleanly).

---

## 2. Policy Versioning Lifecycle & Stage 8B Query Validation

### 2.1 Four-Column Versioning Matrix

Relationship across the four keys:
- `policies.id` (`PK`, `NOT NULL`) ↔ `policy_attributes.policy_id` (`FK -> policies.id`, `NOT NULL`)
- `policies.current_version_id` (`FK -> policy_versions(id, policy_id)`, nullable)
- `policy_versions.id` (`PK`, `NOT NULL`) ↔ `policy_attributes.policy_version_id` (nullable composite `FK`)
- `policy_attributes` identity: `CONSTRAINT policy_attributes_policy_version_uq UNIQUE NULLS NOT DISTINCT (policy_id, policy_version_id)`

| Scenario | `policies.current_version_id` | `policy_attributes` Rows (`policy_version_id`) | Version-Aligned Join Outcome (`Stage 8B`) |
|---|---|---|---|
| **Manual catalog policy (no PDF uploaded yet)** | `NULL` | 1 row `(P1, NULL)` (`UNIQUE NULLS NOT DISTINCT` prevents a second `NULL` row) | Matches **1 row** `(P1, NULL)`. Duplicate manual rows rejected at DB level. |
| **Newly PDF-ingested v1 before activation** | `NULL` (still draft) | 1 row `(P1, V1)` (`is_catalog_eligible = FALSE`, `policies.status = 'processing'`) | `NULL != V1` → **0 rows** returned until `policies.current_version_id = V1` AND `status = 'active'` AND `is_catalog_eligible = TRUE`. |
| **Active v1 published** | `V1` | `(P1, V1)` + optional historical `(P1, NULL)` | Exact match on `pa.policy_version_id = p.current_version_id` returns **only `(P1, V1)`** (1 row). |
| **Concurrent v2 re-ingestion (`FR-CAT-02`)** | `V1` (live) | `(P1, V1)` + in-progress `(P1, V2)` (`is_catalog_eligible = FALSE`) | Returns **only `(P1, V1)`** cleanly with zero downtime and zero overwrites. |
| **Atomic v2 cutover (normal)** | `V2` | `(P1, V1)` + completed `(P1, V2)` (`is_catalog_eligible = TRUE`) | Returns **only `(P1, V2)`** (1 row); `(P1, V1)` retained for audit/rollback. |
| **Stale state (`current_version_id = V2`, no `V2` attribute row yet)** | `V2` | Only `(P1, V1)` exists | Returns **0 rows** (**fail-closed**: refuses to serve stale `v1` attributes under `v2`). |
| **Rollback to `V1`** | Reset to `V1` | Both `(P1, V1)` and `(P1, V2)` exist | Immediately returns `(P1, V1)` cleanly without re-extracting PDF. |

### 2.2 Required Stage 8B Query Change in `ai_service/recommendation/filters.py`

Currently Stage 4 [`build_catalog_policy_attributes_query`](../ai_service/recommendation/filters.py#L1031-L1033) (left untouched in Stage 8A/8A.1) joins only on `policies.c.id == policy_attributes.c.policy_id`. Before applying the multi-version `v3` table to PostgreSQL, Stage 8B must add the index-friendly `NULL`-safe version predicate (equivalent to PostgreSQL `IS NOT DISTINCT FROM`, avoiding `OR` planner penalties by pairing exact equality with `NULL` guard):

```python
from sqlalchemy import and_, or_

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

Because `policy_attributes_policy_version_uq` guarantees uniqueness across `(policy_id, policy_version_id)` with `NULLS NOT DISTINCT`, this join condition **mathematically cannot return duplicate rows or stale versions** for any `policies` record.

---

## 3. Complete 59-Row Feature Source Mapping (`FEATURE_COLUMNS`)

Inspected directly from [`ai_service/recommendation/features.py`](../ai_service/recommendation/features.py#L53-L709) (`FEATURE_SCHEMA_VERSION = "1.0.0"`).  
Exact verified source breakdown (correcting Stage 8A's earlier `32/27` estimate):
- **`36`** pure `policy attribute` features
- **`7`** pure `customer profile` features
- **`16`** `computed interaction` features (including 3 category-gated profile inputs: `has_vehicle_age`, `vehicle_age`, `requested_life_cover`)
- **Total = `59` features** (`23` depend on customer profile inputs; `36` depend solely on policy catalog attributes)

| # | Feature Name | Source Category | Exact DB Column / Profile Field | Computation Location | Missing-Value Behavior | Mandatory Input? |
|---|---|---|---|---|---|---|
| 1 | `policy_type_encoded` | `policy attribute` | `policies.policy_type` | `features.py` (`POLICY_TYPE_ENCODING`) | Raises `MissingMandatoryFeatureInputError` | **Yes** |
| 2 | `is_health` | `policy attribute` | `policies.policy_type` | `features.py` (`_extract_from_normalized`) | Raises if `policy_type` missing/unsupported | **Yes** |
| 3 | `is_car` | `policy attribute` | `policies.policy_type` | `features.py` (`_extract_from_normalized`) | Raises if `policy_type` missing/unsupported | **Yes** |
| 4 | `is_life` | `policy attribute` | `policies.policy_type` | `features.py` (`_extract_from_normalized`) | Raises if `policy_type` missing/unsupported | **Yes** |
| 5 | `is_term` | `policy attribute` | `policies.policy_type` | `features.py` (`_extract_from_normalized`) | Raises if `policy_type` missing/unsupported | **Yes** |
| 6 | `has_claim_settlement_ratio` | `policy attribute` | `policy_attributes.claim_settlement_ratio` | `features.py` (`_normalize_policy_fields`) | `0.0` when `claim_settlement_ratio IS NULL` | No |
| 7 | `has_annual_income` | `customer profile` | `UserProfileRequest.annual_income_inr` | `features.py` (`_normalize_profile_fields`) | `0.0` when omitted or `<= 0` (+ Stage 7 warning) | No |
| 8 | `has_vehicle_age` | `computed interaction` | `policies.policy_type` + `profile.vehicle_age_years` | `features.py` (`_extract_from_normalized`) | `0.0` unless `car` and `vehicle_age_years` set | No |
| 9 | `applicant_age` | `customer profile` | `UserProfileRequest.age` | `features.py` (`_normalize_profile_fields`) | Raises `MissingMandatoryFeatureInputError` / 422 | **Yes** |
| 10 | `annual_income` | `customer profile` | `UserProfileRequest.annual_income_inr` | `features.py` (`_normalize_profile_fields`) | `0.0` sentinel (+ `has_annual_income=0.0`) | No |
| 11 | `monthly_budget` | `customer profile` | `profile.monthly_budget_inr` or `budget_max` | `features.py` (`_normalize_profile_fields`) | Derived as `budget_max / 12.0` (`budget_max` req.) | Indirect (`budget_max`) |
| 12 | `family_size` | `customer profile` | `UserProfileRequest.family_size` | `features.py` (`_normalize_profile_fields`) | Defaults to `1.0` | No (default `1`) |
| 13 | `risk_preference_encoded` | `customer profile` | `UserProfileRequest.risk_preference` | `features.py` (`RISK_PREFERENCE_ENCODING`) | Defaults to `1.0` (`balanced`) + Stage 7 warning | No |
| 14 | `annual_premium` | `policy attribute` | `policy_attributes.annual_premium_min`, `annual_premium_max` | `features.py` (`0.5 * (min + max)`) | Raises `MissingMandatoryFeatureInputError` | **Yes** |
| 15 | `monthly_premium` | `policy attribute` | `policy_attributes.annual_premium_min`, `annual_premium_max` | `features.py` (`annual_premium / 12.0`) | Raises `MissingMandatoryFeatureInputError` | **Yes** |
| 16 | `premium_to_budget_ratio` | `computed interaction` | `annual_premium_min/max`, `profile.budget_max` | `features.py` (`_safe_divide`) | Raises on missing mandatory inputs; `_safe_divide` → `0.0` | **Yes** |
| 17 | `affordability_margin` | `computed interaction` | `profile.budget_max`, `annual_premium_min/max` | `features.py` (`budget_max - annual_premium`) | Raises on missing mandatory inputs | **Yes** |
| 18 | `affordability_score` | `computed interaction` | `profile.budget_max`, `policy_attributes.annual_premium_min` | `features.py` (`clip((budget_max - min) / budget_max, 0, 1)`) | Raises on missing mandatory inputs | **Yes** |
| 19 | `sum_insured` | `policy attribute` | `policy_attributes.sum_insured_max` | `features.py` (`_normalize_policy_fields`) | Raises `MissingMandatoryFeatureInputError` | **Yes** |
| 20 | `requested_sum_insured` | `customer profile` | `UserProfileRequest.min_sum_insured` | `features.py` (`_normalize_profile_fields`) | Raises `MissingMandatoryFeatureInputError` / 422 | **Yes** |
| 21 | `coverage_to_requirement_ratio` | `computed interaction` | `policy_attributes.sum_insured_max`, `profile.min_sum_insured` | `features.py` (`_safe_divide`) | Raises on missing mandatory inputs | **Yes** |
| 22 | `coverage_headroom` | `computed interaction` | `policy_attributes.sum_insured_max`, `profile.min_sum_insured` | `features.py` (`sum_insured - req_sum`) | Raises on missing mandatory inputs | **Yes** |
| 23 | `age_eligibility_margin` | `computed interaction` | `policy_attributes.max_entry_age`, `profile.age` | `features.py` (`max_entry_age - age`) | Raises on missing mandatory inputs | **Yes** |
| 24 | `remaining_eligible_years` | `computed interaction` | `policy_attributes.max_renewal_age`, `lifelong_renewal`, `max_entry_age`, `profile.age` | `features.py` (`max(0, horizon - age)`) | `lifelong_renewal=True` → `100.0`; else `max_entry_age` | Indirect (`max_entry_age`, `age`) |
| 25 | `total_benefit_count` | `policy attribute` | `policy_attributes.covered_benefits` | `features.py` (`len(covered_benefits)`) | Empty JSON / missing → `0.0` | No |
| 26 | `matched_required_benefits` | `computed interaction` | `profile.required_benefits`, `policy_attributes.covered_benefits` | `features.py` (`len(req & covered)`) | Empty lists → `0.0` | No |
| 27 | `required_benefit_match_ratio` | `computed interaction` | `profile.required_benefits`, `policy_attributes.covered_benefits` | `features.py` (`_extract_from_normalized`) | `1.0` when `required_benefits` is empty | No |
| 28 | `exclusion_conflict_count` | `computed interaction` | `profile.required_benefits`, `pre_existing_conditions`, `policy_attributes.exclusions` | `features.py` (`len((req \| ped) & exclusions)`) | Empty lists → `0.0` | No |
| 29 | `claim_settlement_ratio` | `policy attribute` | `policy_attributes.claim_settlement_ratio` | `features.py` (`_normalize_policy_fields`) | `0.0` sentinel (+ `has_claim_settlement_ratio=0.0`) | No (nullable) |
| 30 | `deductible_amount` | `policy attribute` | `policy_attributes.deductible_amount` | `features.py` (`_normalize_policy_fields`) | `0.0` when missing/`NULL` | No |
| 31 | `no_claim_bonus_pct` | `policy attribute` | `policy_attributes.no_claim_bonus_pct` | `features.py` (`_normalize_policy_fields`) | `0.0` when missing/`NULL` | No |
| 32 | `restoration_benefit_indicator` | `policy attribute` | `policy_attributes.restoration_benefit` | `features.py` (`_extract_from_normalized`) | `0.0` (`False`) when missing | No |
| 33 | `pan_india_coverage_indicator` | `policy attribute` | `policy_attributes.pan_india_coverage` | `features.py` (`_extract_from_normalized`) | `0.0` (`False`) when missing | No |
| 34 | `ped_coverage_indicator` | `policy attribute` | `policy_attributes.ped_covered` + `policies.policy_type` | `features.py` (health-gated) | `0.0` if non-health or missing | No |
| 35 | `ped_waiting_period_days` | `policy attribute` | `policy_attributes.ped_waiting_period_months` + `policies.policy_type` | `features.py` (`months * 30.0`, health-gated) | `0.0` if `NULL` or non-health | No (nullable) |
| 36 | `initial_waiting_period_days` | `policy attribute` | `policy_attributes.initial_waiting_period_days` + `policies.policy_type` | `features.py` (health-gated) | `0.0` if missing or non-health | No |
| 37 | `room_rent_restriction_indicator` | `policy attribute` | `policy_attributes.room_rent_limit_type` + `policies.policy_type` | `features.py` (`type != 'no_limit'`, health-gated) | Defaults `'no_limit'` → `0.0`; `0.0` non-health | No |
| 38 | `cashless_availability_indicator` | `policy attribute` | `policy_attributes.cashless_available` + `policies.policy_type` | `features.py` (health-gated) | `0.0` if missing or non-health | No |
| 39 | `hospital_network_count` | `policy attribute` | `policy_attributes.network_hospitals_count` + `policies.policy_type` | `features.py` (health-gated) | `0.0` if missing or non-health | No |
| 40 | `co_payment_percentage` | `policy attribute` | `policy_attributes.copay_pct` + `policies.policy_type` | `features.py` (health-gated) | `0.0` if missing or non-health | No |
| 41 | `family_floater_indicator` | `policy attribute` | `policy_attributes.coverage_mode` + `policies.policy_type` | `features.py` (`in ('family_floater','both')`, health-gated) | Defaults `'individual'` → `0.0` | No |
| 42 | `vehicle_age` | `computed interaction` | `profile.vehicle_age_years` + `policies.policy_type` | `features.py` (car-gated) | `0.0` if non-car or `vehicle_age_years` omitted | No |
| 43 | `vehicle_age_limit` | `policy attribute` | `policy_attributes.type_specific_metadata->>'vehicle_age_limit_years'` + `policy_type` | `features.py` (car-gated) | `0.0` if missing or non-car | No (JSONB) |
| 44 | `vehicle_age_eligibility_margin` | `computed interaction` | `meta->>'vehicle_age_limit_years'`, `profile.vehicle_age_years`, `policy_type` | `features.py` (`limit - age`, car-gated) | `0.0` if either input missing or non-car | No |
| 45 | `own_damage_indicator` | `policy attribute` | `type_specific_metadata->>'own_damage_coverage'` + `policy_type` | `features.py` (car-gated) | `0.0` if missing or non-car | No (JSONB) |
| 46 | `third_party_indicator` | `policy attribute` | `type_specific_metadata->>'third_party_coverage'` + `policy_type` | `features.py` (car-gated) | `0.0` if missing or non-car | No (JSONB) |
| 47 | `zero_depreciation_indicator` | `policy attribute` | `type_specific_metadata->>'zero_depreciation'` + `policy_type` | `features.py` (car-gated) | `0.0` if missing or non-car | No (JSONB) |
| 48 | `roadside_assistance_indicator` | `policy attribute` | `type_specific_metadata->>'roadside_assistance'` + `policy_type` | `features.py` (car-gated) | `0.0` if missing or non-car | No (JSONB) |
| 49 | `engine_protection_indicator` | `policy attribute` | `type_specific_metadata->>'engine_protection'` + `policy_type` | `features.py` (car-gated) | `0.0` if missing or non-car | No (JSONB) |
| 50 | `life_cover` | `policy attribute` | `type_specific_metadata->>'life_cover_max'` fallback `sum_insured_max` + `policy_type` | `features.py` (life/term-gated) | Falls back to `sum_insured_max`; `0.0` if health/car | Indirect (`sum_insured_max`) |
| 51 | `requested_life_cover` | `computed interaction` | `profile.min_sum_insured` + `policies.policy_type` | `features.py` (life/term-gated) | `0.0` for health/car | Indirect (`min_sum_insured`) |
| 52 | `life_cover_ratio` | `computed interaction` | `life_cover`, `profile.min_sum_insured`, `policies.policy_type` | `features.py` (`_safe_divide`, life/term-gated) | `0.0` for health/car | Indirect (`sum_insured_max`, `min_sum_insured`) |
| 53 | `policy_term_years` | `policy attribute` | `type_specific_metadata->>'policy_term_years_max'` fallback `max_tenure_years` + `policy_type` | `features.py` (life/term-gated) | Falls back to `max_tenure_years` (default `1.0`); `0.0` health/car | No |
| 54 | `premium_payment_term_years` | `policy attribute` | `type_specific_metadata->>'premium_payment_term_years'` + `policy_type` | `features.py` (life/term-gated) | `0.0` if missing or health/car | No (JSONB) |
| 55 | `maturity_benefit_indicator` | `policy attribute` | `type_specific_metadata->>'maturity_benefit'` + `policy_type` | `features.py` (life-gated) | `0.0` if missing or non-life | No (JSONB) |
| 56 | `surrender_value_indicator` | `policy attribute` | `type_specific_metadata->>'surrender_value_available'` + `policy_type` | `features.py` (life-gated) | `0.0` if missing or non-life | No (JSONB) |
| 57 | `smoking_eligibility_compatibility` | `computed interaction` | `profile.smoking_status`, `type_specific_metadata->>'smoking_status_eligibility'`, `policy_type` | `features.py` (`_compute_smoking_compatibility`, term-gated) | Defaults missing metadata to `'all'` (`1.0`); `0.0` non-term | No |
| 58 | `critical_illness_rider_indicator` | `policy attribute` | `type_specific_metadata->>'critical_illness_rider'` + `policy_type` | `features.py` (term-gated) | `0.0` if missing or non-term | No (JSONB) |
| 59 | `accidental_death_rider_indicator` | `policy attribute` | `type_specific_metadata->>'accidental_death_rider'` + `policy_type` | `features.py` (term-gated) | `0.0` if missing or non-term | No (JSONB) |

> **Verification Result:** Every single one of the `59` features traces unambiguously to verified `policies`, `policy_attributes` (relational or `type_specific_metadata` JSONB), or `UserProfileRequest` inputs, with zero reliance on unstored or leaking targets (`FORBIDDEN_FEATURE_COLUMNS`).

---

## 4. Migration Safety Audit (`v2` vs. `v3`)

| Safety Dimension | Audit Finding on `v2` | Resolution in `v3` / Governance Note |
|---|---|---|
| **Foreign-Key Dependency Order** | `policy_attributes` references only `policies(id)` and `policy_versions(id, policy_id)` (both created in `001_initial_schema.sql`, already applied). `policy_attributes` creates no circular FKs. | **Valid** — clean leaf child table in migration sequence `003+`. |
| **Constraint vs. Default Conflict (CRITICAL)** | **`v2` Defect:** `coverage_mode DEFAULT 'both'` + `max_family_members DEFAULT 1` violates `CHECK (coverage_mode = 'individual' OR max_family_members >= 2)`. | **Fixed in `v3`:** `coverage_mode DEFAULT 'individual'` + `max_family_members DEFAULT 1`. |
| **Defense-in-Depth Constraints & Defaults** | `v2` had permissive `TRUE` defaults on `cashless_available` & `pan_india_coverage`, and allowed contradictory `lifelong_renewal = TRUE` alongside non-null `max_renewal_age`. | **Fixed in `v3`:** All capability booleans default to `FALSE` (fail-closed) + added `policy_attributes_renewal_consistency_check`. All 80 rows of `synthetic_policies.csv` pass 100% of `v3` constraints programmatically. |
| **Index Definitions** | Partial B-tree indexes (`WHERE is_catalog_eligible = TRUE`) + GIN indexes with `jsonb_path_ops`. Note: `policy_attributes_policy_idx` (`policy_id, policy_version_id`) mirrors `policy_attributes_policy_version_uq` restricted to eligible catalog rows. | **Valid** — explicitly documented that `jsonb_path_ops` accelerates `@>` / `@?` / `@@` (not `?` / `?\|` operators). Queries should use `@>`. |
| **Idempotency Limitations** | `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS` skip gracefully **only** when schema already matches; they do **not** alter pre-existing mismatched columns/constraints without `ALTER TABLE`. | **Documented limitation** — safe when orchestrated by `schema_migrations`, but manual schema drift must be resolved via explicit `ALTER TABLE` migrations, not re-running `003`. |
| **Migration Runner Compatibility (`server/scripts/migrate.js`)** | `migrate.js` matches filenames `/^\d+_[a-z0-9_-]+\.sql$/i`, records applied filename in `schema_migrations`, and executes the entire SQL file inside an explicit `BEGIN ... COMMIT` / `ROLLBACK` transaction. | **Compatible** — `v3` omits manual `BEGIN`/`COMMIT` statements and uses standard sequential naming `003_recommendation_schema.sql` once promoted to `server/migrations/`. |
| **Rollback Strategy** | `migrate.js` implements forward-only migrations (no automated `down` script). Failed runs automatically `ROLLBACK` before recording `schema_migrations`. | **Documented manual rollback:** `DROP TABLE IF EXISTS policy_attributes CASCADE;` (plus delete row from `schema_migrations` if reverting a committed run cleanly). |
| **PostgreSQL 18 Compatibility** | Live server verified as `PostgreSQL 18.6`. `gen_random_uuid()` (built-in PG 13+) and `UNIQUE NULLS NOT DISTINCT` (PG 15+) are natively supported. | **100% PG 18.6 Compatible** (minimum engine requirement: PostgreSQL 15+ due to `NULLS NOT DISTINCT`). |

---

## 5. Pre-Stage-8B Sign-Off Checklist

Before proceeding to **Stage 8B** (updating Python `common/tables.py`, `filters.py` version-safe join, `config/database.py`, `test_schema_draft.py`, and promoting migration SQL under team governance):

1. Promote [`ai_service/sql/proposed_003_recommendation_schema_v3.sql`](../ai_service/sql/proposed_003_recommendation_schema_v3.sql) (preserving `v1` and `v2` proposals unchanged until cutover PR).
2. Confirm with Node/DB owners & M2/M3: formalizing `CREATE EXTENSION IF NOT EXISTS vector;` in shared migrations alongside M2/M3 vector tables (`sections`, `clauses`, `chunks`), and enforcing lowercase `policies.policy_type IN ('health','car','life','term')` + `type_specific_metadata` completeness before publishing `is_catalog_eligible = TRUE`.
3. Authorize Stage 8B Python alignment (`filters.py` `current_version_id` join, `common/tables.py` `policy_attributes` sync, `ranking_features` removal/deprecation cleanup).
