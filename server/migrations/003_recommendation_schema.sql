-- =============================================================================
-- Migration 003: Policy Recommendation Relational Schema (Module M5)
-- =============================================================================
-- Architecture Reference:
--   - doc/project-architecture.md (§2A Database table ownership, §3.4, §5)
--   - Base commit: feature/coverageiq-database-foundation (7953ea24)
--   - Prepared from validated schema proposal:
--       ai_service/sql/proposed_003_recommendation_schema_v3.sql
--
-- Ownership & Cross-Service Review Governance:
--   - Migration execution owner: Node API runner (server/scripts/migrate.js)
--   - Table / data owners (Python AI Service — M5):
--       1. policy_attributes (required by Stage 4 hard filters & Stage 5 features)
--       2. ranking_features  (preserved per project-architecture.md §2A contract)
--   - Requires dual-owner PR review (Node/DB owner + Python M5 owner) prior to
--     execution via `npm run migrate` inside server/.
--   - Vector tables (sections, clauses, chunks) and `CREATE EXTENSION vector`
--     belong to shared DB infra / M2 & M3 and are intentionally omitted here.
--
-- Schema Dependencies:
--   - Requires 001_initial_schema.sql (policies, policy_versions, organizations)
--   - Requires 002_add_platform_role.sql
--   - Requires PostgreSQL 15+ (gen_random_uuid(), UNIQUE NULLS NOT DISTINCT)
--
-- Notes on ranking_features Architectural Role:
--   - project-architecture.md §2A lists `ranking_features` as Python-owned.
--   - M5 Stage 5 (ai_service/recommendation/features.py) computes all 59
--     ranking features dynamically at inference time (36 policy-attribute signals
--     + 7 profile signals + 16 profile-policy interaction signals; 23 depend on
--     the runtime customer profile and cannot be stored statically per policy).
--   - `ranking_features` is retained as a version-aware, optional persistence
--     target for future policy-level quality subscores or cached static vectors,
--     aligned to `FEATURE_SCHEMA_VERSION = '1.0.0'`.
--
-- Rollback Reference (migrate.js rolls back failures automatically via BEGIN/
-- ROLLBACK; reverting an already-committed run requires manual execution):
--   BEGIN;
--   DROP TABLE IF EXISTS ranking_features CASCADE;
--   DROP TABLE IF EXISTS policy_attributes CASCADE;
--   DELETE FROM schema_migrations WHERE filename = '003_recommendation_schema.sql';
--   COMMIT;
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


-- =============================================================================
-- Python-Owned Precomputed Policy Subscores / Vector Cache (Architecture §2A)
-- =============================================================================
CREATE TABLE IF NOT EXISTS ranking_features (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_id UUID NOT NULL REFERENCES policies(id) ON DELETE CASCADE,
    policy_version_id UUID,
    feature_schema_version VARCHAR(32) NOT NULL DEFAULT '1.0.0',

    -- Optional normalized policy-only quality subscores [0, 1]
    value_for_money_score NUMERIC(6, 4)
        CHECK (value_for_money_score IS NULL OR (value_for_money_score >= 0 AND value_for_money_score <= 1)),
    coverage_breadth_score NUMERIC(6, 4)
        CHECK (coverage_breadth_score IS NULL OR (coverage_breadth_score >= 0 AND coverage_breadth_score <= 1)),
    waiting_period_friendliness NUMERIC(6, 4)
        CHECK (
            waiting_period_friendliness IS NULL
            OR (waiting_period_friendliness >= 0 AND waiting_period_friendliness <= 1)
        ),
    network_strength_score NUMERIC(6, 4)
        CHECK (network_strength_score IS NULL OR (network_strength_score >= 0 AND network_strength_score <= 1)),
    claim_reliability_score NUMERIC(6, 4)
        CHECK (claim_reliability_score IS NULL OR (claim_reliability_score >= 0 AND claim_reliability_score <= 1)),

    feature_vector JSONB NOT NULL DEFAULT '{}'::JSONB
        CHECK (jsonb_typeof(feature_vector) = 'object'),
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT ranking_features_policy_version_schema_uq
        UNIQUE NULLS NOT DISTINCT (policy_id, policy_version_id, feature_schema_version),
    CONSTRAINT ranking_features_version_fk
        FOREIGN KEY (policy_version_id, policy_id)
        REFERENCES policy_versions (id, policy_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ranking_features_policy_idx
    ON ranking_features (policy_id, policy_version_id, feature_schema_version);
