-- =============================================================================
-- REVISED DRAFT SCHEMA PROPOSAL v3 ONLY (NOT APPLIED TO LIVE DATABASE)
-- =============================================================================
-- Proposed File Name (when approved for server/migrations/):
--     003_recommendation_schema.sql  (or next sequential number via migrate.js)
-- Supersedes Drafts:
--     ai_service/sql/proposed_003_recommendation_schema.sql    (v1)
--     ai_service/sql/proposed_003_recommendation_schema_v2.sql (v2)
-- Author: Shravan Sambhaji Kapare (Roll 203, Group B2 — Module M5)
-- Status: Pending team ownership review (Node/DB Owner + M2 + M3 + M5)
-- Reference Design Docs:
--     doc/M5-recommendation-database-design.md
--     doc/M5-stage-8A-final-validation.md
--
-- Corrections applied vs. v2 (identified during Stage 8A.1 final SQL validation):
--   1. FIXED BLOCKING BUG: Changed coverage_mode DEFAULT from 'both' to
--      'individual' to match max_family_members DEFAULT 1; previously in v2,
--      any INSERT omitting coverage_mode & max_family_members failed
--      CONSTRAINT policy_attributes_family_members_check immediately.
--   2. UNIFIED FAIL-CLOSED DEFAULTS: Changed cashless_available and
--      pan_india_coverage defaults from TRUE to FALSE so unverified/unextracted
--      records never falsely pass CASHLESS_REQUIRED or GEOGRAPHIC_AVAILABILITY.
--   3. ADDED RENEWAL CONSISTENCY CHECK: Enforces
--      (NOT lifelong_renewal OR max_renewal_age IS NULL) matching features.py
--      horizon precedence and verified across all 80 synthetic_policies.csv rows.
--   4. DOCUMENTED EXECUTION/INDEX/ROLLBACK SEMANTICS: Verified PG 18.6 DDL,
--      server/scripts/migrate.js single-transaction wrapper (no BEGIN/COMMIT),
--      jsonb_path_ops @> operator scope, and explicit manual rollback DDL.
--
-- Ownership & Compatibility Notes:
--   - Target referenced unique constraint verified in PostgreSQL 18.6:
--     policy_versions_id_policy_id_key -> UNIQUE (id, policy_id)
--   - Requires Stage 8B update in ai_service/recommendation/filters.py
--     (build_catalog_policy_attributes_query) to align
--     policy_attributes.policy_version_id with policies.current_version_id
--     using index-friendly NULL-safe equality:
--       (pa.policy_version_id = p.current_version_id
--        OR (p.current_version_id IS NULL AND pa.policy_version_id IS NULL))
--   - ranking_features remains deferred/omitted (Stage 5 computes all 59
--     features dynamically; 23 depend on live profile context).
--   - pgvector (0.8.7) is installed out-of-band on the dev DB; shared infra/M3
--     should formalize `CREATE EXTENSION IF NOT EXISTS vector;` in migrations.
--
-- Manual Rollback Reference (if ever reverted manually outside migrate.js):
--   DROP TABLE IF EXISTS policy_attributes CASCADE;
-- =============================================================================

CREATE TABLE IF NOT EXISTS policy_attributes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_id UUID NOT NULL REFERENCES policies(id) ON DELETE CASCADE,
    policy_version_id UUID,

    -- Explicit opt-in for shared catalog recommendations (private uploads stay FALSE)
    is_catalog_eligible BOOLEAN NOT NULL DEFAULT FALSE,

    -- Age & applicant eligibility constraints
    min_entry_age INTEGER NOT NULL DEFAULT 18
        CHECK (min_entry_age >= 0 AND min_entry_age <= 120),
    max_entry_age INTEGER NOT NULL DEFAULT 65
        CHECK (max_entry_age >= min_entry_age AND max_entry_age <= 120),
    max_renewal_age INTEGER
        CHECK (max_renewal_age IS NULL OR max_renewal_age >= max_entry_age),
    lifelong_renewal BOOLEAN NOT NULL DEFAULT FALSE,
    coverage_mode VARCHAR(30) NOT NULL DEFAULT 'individual'
        CHECK (coverage_mode IN ('individual', 'family_floater', 'both')),
    max_family_members INTEGER NOT NULL DEFAULT 1
        CHECK (max_family_members >= 1),
    eligible_applicant_types JSONB NOT NULL
        DEFAULT '["salaried","self_employed","student","senior_citizen"]'::JSONB
        CHECK (jsonb_typeof(eligible_applicant_types) = 'array'),

    -- Financial & coverage bounds
    annual_premium_min NUMERIC(12, 2) NOT NULL
        CHECK (annual_premium_min >= 0),
    annual_premium_max NUMERIC(12, 2) NOT NULL
        CHECK (annual_premium_max >= annual_premium_min),
    sum_insured_min NUMERIC(14, 2) NOT NULL
        CHECK (sum_insured_min > 0),
    sum_insured_max NUMERIC(14, 2) NOT NULL
        CHECK (sum_insured_max >= sum_insured_min),
    deductible_amount NUMERIC(12, 2) NOT NULL DEFAULT 0
        CHECK (deductible_amount >= 0),
    copay_pct NUMERIC(5, 2) NOT NULL DEFAULT 0
        CHECK (copay_pct >= 0 AND copay_pct <= 100),
    room_rent_limit_type VARCHAR(30) NOT NULL DEFAULT 'no_limit'
        CHECK (room_rent_limit_type IN (
            'no_limit', 'single_private', 'shared', 'fixed_per_day', 'pct_sum_insured'
        )),
    room_rent_limit_value NUMERIC(12, 2)
        CHECK (room_rent_limit_value IS NULL OR room_rent_limit_value >= 0),

    -- Waiting periods & pre-existing disease (PED) rules
    initial_waiting_period_days INTEGER NOT NULL DEFAULT 30
        CHECK (initial_waiting_period_days >= 0),
    ped_covered BOOLEAN NOT NULL DEFAULT FALSE,
    ped_waiting_period_months INTEGER
        CHECK (ped_waiting_period_months IS NULL OR ped_waiting_period_months >= 0),
    specific_disease_waiting_months INTEGER NOT NULL DEFAULT 24
        CHECK (specific_disease_waiting_months >= 0),
    maternity_waiting_months INTEGER
        CHECK (maternity_waiting_months IS NULL OR maternity_waiting_months >= 0),

    -- Hospital/garage network, geographic coverage & duration
    network_hospitals_count INTEGER NOT NULL DEFAULT 0
        CHECK (network_hospitals_count >= 0),
    cashless_available BOOLEAN NOT NULL DEFAULT FALSE,
    pan_india_coverage BOOLEAN NOT NULL DEFAULT FALSE,
    available_regions JSONB NOT NULL DEFAULT '[]'::JSONB
        CHECK (jsonb_typeof(available_regions) = 'array'),
    min_tenure_years INTEGER NOT NULL DEFAULT 1
        CHECK (min_tenure_years >= 1),
    max_tenure_years INTEGER NOT NULL DEFAULT 3
        CHECK (max_tenure_years >= min_tenure_years),

    -- Benefits, exclusions & ranking signals
    covered_benefits JSONB NOT NULL DEFAULT '[]'::JSONB
        CHECK (jsonb_typeof(covered_benefits) = 'array'),
    exclusions JSONB NOT NULL DEFAULT '[]'::JSONB
        CHECK (jsonb_typeof(exclusions) = 'array'),
    claim_settlement_ratio NUMERIC(5, 2)
        CHECK (
            claim_settlement_ratio IS NULL
            OR (claim_settlement_ratio >= 0 AND claim_settlement_ratio <= 100)
        ),
    no_claim_bonus_pct NUMERIC(5, 2) NOT NULL DEFAULT 0
        CHECK (no_claim_bonus_pct >= 0 AND no_claim_bonus_pct <= 200),
    restoration_benefit BOOLEAN NOT NULL DEFAULT FALSE,

    -- Category-specific attributes for car / life / term / health extensions
    type_specific_metadata JSONB NOT NULL DEFAULT '{}'::JSONB
        CHECK (jsonb_typeof(type_specific_metadata) = 'object'),

    -- Extraction provenance
    extraction_source VARCHAR(30) NOT NULL DEFAULT 'manual'
        CHECK (extraction_source IN ('manual', 'synthetic', 'ingestion_llm', 'verified')),
    extraction_confidence NUMERIC(4, 3)
        CHECK (extraction_confidence IS NULL OR extraction_confidence BETWEEN 0 AND 1),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Cross-column domain invariants
    CONSTRAINT policy_attributes_family_members_check
        CHECK (coverage_mode = 'individual' OR max_family_members >= 2),
    CONSTRAINT policy_attributes_room_limit_value_check
        CHECK (
            room_rent_limit_type NOT IN ('fixed_per_day', 'pct_sum_insured')
            OR room_rent_limit_value IS NOT NULL
        ),
    CONSTRAINT policy_attributes_renewal_consistency_check
        CHECK (NOT lifelong_renewal OR max_renewal_age IS NULL),

    -- Version-aware identity & composite FK against policy_versions(id, policy_id)
    -- PG 15+ NULLS NOT DISTINCT guarantees at most one unversioned (NULL) row
    -- and one row per distinct (policy_id, policy_version_id).
    CONSTRAINT policy_attributes_policy_version_uq
        UNIQUE NULLS NOT DISTINCT (policy_id, policy_version_id),
    CONSTRAINT policy_attributes_version_fk
        FOREIGN KEY (policy_version_id, policy_id)
        REFERENCES policy_versions (id, policy_id)
        ON DELETE CASCADE
);

-- Lookup & filtering indexes for eligible catalog queries
-- Note: jsonb_path_ops indexes optimize @> containment queries (and @? / @@).
CREATE INDEX IF NOT EXISTS policy_attributes_policy_idx
    ON policy_attributes (policy_id, policy_version_id)
    WHERE is_catalog_eligible = TRUE;

CREATE INDEX IF NOT EXISTS policy_attributes_premium_sum_idx
    ON policy_attributes (annual_premium_min, sum_insured_max)
    WHERE is_catalog_eligible = TRUE;

CREATE INDEX IF NOT EXISTS policy_attributes_age_mode_idx
    ON policy_attributes (min_entry_age, max_entry_age, coverage_mode)
    WHERE is_catalog_eligible = TRUE;

CREATE INDEX IF NOT EXISTS policy_attributes_applicants_gin
    ON policy_attributes USING GIN (eligible_applicant_types jsonb_path_ops);

CREATE INDEX IF NOT EXISTS policy_attributes_benefits_gin
    ON policy_attributes USING GIN (covered_benefits jsonb_path_ops);

CREATE INDEX IF NOT EXISTS policy_attributes_exclusions_gin
    ON policy_attributes USING GIN (exclusions jsonb_path_ops);

CREATE INDEX IF NOT EXISTS policy_attributes_regions_gin
    ON policy_attributes USING GIN (available_regions jsonb_path_ops);

CREATE INDEX IF NOT EXISTS policy_attributes_type_meta_gin
    ON policy_attributes USING GIN (type_specific_metadata jsonb_path_ops);
