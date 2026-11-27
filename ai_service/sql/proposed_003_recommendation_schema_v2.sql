-- =============================================================================
-- REVISED DRAFT SCHEMA PROPOSAL ONLY (NOT APPLIED TO LIVE DATABASE)
-- =============================================================================
-- Proposed File Name (when approved for server/migrations/):
--     003_recommendation_schema.sql  (or next sequential number via migrate.js)
-- Supersedes Draft: ai_service/sql/proposed_003_recommendation_schema.sql (v1)
-- Author: Shravan Sambhaji Kapare (Roll 203, Group B2 — Module M5)
-- Status: Pending team ownership review (Node/DB Owner + M2 + M3 + M5)
-- Reference Design Doc: doc/M5-recommendation-database-design.md
--
-- Revisions vs. original proposed_003_recommendation_schema.sql:
--   1. Made DDL idempotent (CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT EXISTS)
--      compatible with server/scripts/migrate.js transaction runner.
--   2. Switched policy_attributes uniqueness from UNIQUE(policy_id) to version-scoped
--      CONSTRAINT policy_attributes_policy_version_uq
--      UNIQUE NULLS NOT DISTINCT (policy_id, policy_version_id) (PostgreSQL 15+)
--      so async v2 PDF ingestion (FR-CAT-02) does not clobber live v1 attributes.
--      (Note: requires Stage 8B update to filters.py join to align with
--      policies.current_version_id.)
--   3. Tightened fail-closed defaults (is_catalog_eligible DEFAULT FALSE,
--      ped_covered DEFAULT FALSE) and added integrity checks on family floater
--      member capacity and fixed/percentage room-rent limit values.
--   4. Replaced 4-column B-tree range index with targeted partial B-tree indexes
--      and added missing GIN indexes on eligible_applicant_types and
--      type_specific_metadata.
--   5. Deferred/omitted ranking_features table: Stage 5 computes 59 features dynamically
--      (~27 depend on the live user profile and cannot be stored per policy), and
--      all policy-level attributes already reside cleanly in policy_attributes.
--
-- Ownership Note on pgvector:
--   Live DB currently has extension 'vector' (v0.8.7) installed outside
--   server/migrations/. Shared DB infra / M3 should formalize
--   `CREATE EXTENSION IF NOT EXISTS vector;` before creating M2/M3 vector tables
--   (sections, clauses, chunks). M5 hard filtering and ML ranking do NOT require
--   pgvector directly.
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
    coverage_mode VARCHAR(30) NOT NULL DEFAULT 'both'
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
    cashless_available BOOLEAN NOT NULL DEFAULT TRUE,
    pan_india_coverage BOOLEAN NOT NULL DEFAULT TRUE,
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

    -- Version-aware identity & FK integrity against policy_versions(id, policy_id)
    CONSTRAINT policy_attributes_policy_version_uq
        UNIQUE NULLS NOT DISTINCT (policy_id, policy_version_id),
    CONSTRAINT policy_attributes_version_fk
        FOREIGN KEY (policy_version_id, policy_id)
        REFERENCES policy_versions (id, policy_id)
        ON DELETE CASCADE
);

-- Lookup & filtering indexes for eligible catalog queries
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
