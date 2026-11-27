-- =============================================================================
-- DRAFT SCHEMA PROPOSAL ONLY (NOT APPLIED TO LIVE DATABASE)
-- =============================================================================
-- Proposed File Name (when approved): 003_recommendation_schema.sql
-- Author: Shravan Sambhaji Kapare (Roll 203, Group B2)
-- Status: Pending team ownership review with Prithviraj Patil (Roll 220)
--
-- Purpose:
--   Adds structured relational policy attribute tables required by Module 6A
--   (Policy Recommendation: Deterministic Hard Filtering + ML Ranking).
--   Aligns with existing UUID/composite-FK conventions in
--   server/migrations/001_initial_schema.sql.
-- =============================================================================

CREATE TABLE policy_attributes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_id UUID NOT NULL REFERENCES policies(id) ON DELETE CASCADE,
    policy_version_id UUID,

    -- Catalog recommendation eligibility flag (distinct from private user uploads)
    is_catalog_eligible BOOLEAN NOT NULL DEFAULT TRUE,

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
    eligible_applicant_types JSONB NOT NULL DEFAULT '["salaried","self_employed","student","senior_citizen"]'::JSONB
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
    ped_covered BOOLEAN NOT NULL DEFAULT TRUE,
    ped_waiting_period_months INTEGER
        CHECK (ped_waiting_period_months IS NULL OR ped_waiting_period_months >= 0),
    specific_disease_waiting_months INTEGER NOT NULL DEFAULT 24
        CHECK (specific_disease_waiting_months >= 0),
    maternity_waiting_months INTEGER
        CHECK (maternity_waiting_months IS NULL OR maternity_waiting_months >= 0),

    -- Hospital network, geographic coverage & duration
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
        CHECK (claim_settlement_ratio IS NULL OR (claim_settlement_ratio >= 0 AND claim_settlement_ratio <= 100)),
    no_claim_bonus_pct NUMERIC(5, 2) NOT NULL DEFAULT 0
        CHECK (no_claim_bonus_pct >= 0 AND no_claim_bonus_pct <= 200),
    restoration_benefit BOOLEAN NOT NULL DEFAULT FALSE,
    type_specific_metadata JSONB NOT NULL DEFAULT '{}'::JSONB
        CHECK (jsonb_typeof(type_specific_metadata) = 'object'),

    -- Provenance tracking (manual/synthetic seeding vs automated PDF attribute extraction)
    extraction_source VARCHAR(30) NOT NULL DEFAULT 'manual'
        CHECK (extraction_source IN ('manual', 'synthetic', 'ingestion_llm', 'verified')),
    extraction_confidence NUMERIC(4, 3)
        CHECK (extraction_confidence IS NULL OR extraction_confidence BETWEEN 0 AND 1),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (policy_id),
    CONSTRAINT policy_attributes_version_fk
        FOREIGN KEY (policy_version_id, policy_id)
        REFERENCES policy_versions (id, policy_id)
        ON DELETE CASCADE
);

CREATE INDEX policy_attributes_budget_sum_idx
    ON policy_attributes (
        annual_premium_min,
        annual_premium_max,
        sum_insured_min,
        sum_insured_max
    )
    WHERE is_catalog_eligible = TRUE;

CREATE INDEX policy_attributes_age_mode_idx
    ON policy_attributes (min_entry_age, max_entry_age, coverage_mode)
    WHERE is_catalog_eligible = TRUE;

CREATE INDEX policy_attributes_benefits_gin
    ON policy_attributes USING GIN (covered_benefits jsonb_path_ops);

CREATE INDEX policy_attributes_exclusions_gin
    ON policy_attributes USING GIN (exclusions jsonb_path_ops);

CREATE INDEX policy_attributes_regions_gin
    ON policy_attributes USING GIN (available_regions jsonb_path_ops);


CREATE TABLE ranking_features (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_id UUID NOT NULL REFERENCES policies(id) ON DELETE CASCADE,
    policy_version_id UUID,
    feature_schema_version VARCHAR(32) NOT NULL DEFAULT 'v1',

    -- Precomputed policy-level quality indicators
    value_for_money_score NUMERIC(6, 4),
    coverage_breadth_score NUMERIC(6, 4),
    waiting_period_friendliness NUMERIC(6, 4),
    network_strength_score NUMERIC(6, 4),
    claim_reliability_score NUMERIC(6, 4),

    feature_vector JSONB NOT NULL DEFAULT '{}'::JSONB
        CHECK (jsonb_typeof(feature_vector) = 'object'),
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (policy_id, feature_schema_version),
    CONSTRAINT ranking_features_version_fk
        FOREIGN KEY (policy_version_id, policy_id)
        REFERENCES policy_versions (id, policy_id)
        ON DELETE CASCADE
);

CREATE INDEX ranking_features_policy_idx
    ON ranking_features (policy_id, feature_schema_version);
