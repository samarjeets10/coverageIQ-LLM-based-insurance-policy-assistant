/**
 * CoverageIQ Checkpoint 5.0 Policy Attributes & Ranking Features Migration (M5)
 *
 * Reconciles M5 recommendation storage cleanly onto CoverageIQ's authoritative
 * PostgreSQL 16 schema (`policies` + `policy_documents` + `policy_chunks`),
 * owned by Node.js through node-pg-migrate without duplicating `policy_versions`.
 *
 * Strictly preserves explicit missingness (`NULL` instead of fabricated defaults)
 * and stores chunk/page/clause provenance for every extracted attribute.
 */

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql(`
    CREATE TABLE IF NOT EXISTS policy_attributes (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        policy_id UUID NOT NULL REFERENCES policies(id) ON DELETE CASCADE,
        document_id UUID NOT NULL,

        -- Explicit opt-in catalog recommendation visibility
        is_catalog_eligible BOOLEAN NOT NULL DEFAULT FALSE,

        -- Extraction lifecycle & provenance metadata (nullable attributes remain NULL when unstated)
        extraction_status VARCHAR(32) NOT NULL DEFAULT 'pending'
            CHECK (extraction_status IN ('pending', 'completed', 'partial', 'insufficient_data', 'failed')),
        extraction_source VARCHAR(32) NOT NULL DEFAULT 'ingestion_llm'
            CHECK (extraction_source IN ('manual', 'synthetic', 'ingestion_llm', 'ingestion_deterministic', 'verified')),
        extraction_confidence NUMERIC(4, 3) NULL
            CHECK (extraction_confidence IS NULL OR (extraction_confidence >= 0 AND extraction_confidence <= 1)),
        extraction_error TEXT NULL,
        source_chunk_ids JSONB NOT NULL DEFAULT '[]'::jsonb
            CHECK (jsonb_typeof(source_chunk_ids) = 'array'),
        field_provenance JSONB NOT NULL DEFAULT '{}'::jsonb
            CHECK (jsonb_typeof(field_provenance) = 'object'),

        -- Age & applicant eligibility constraints (nullable when omitted from PDF)
        min_entry_age INTEGER NULL
            CHECK (min_entry_age IS NULL OR (min_entry_age >= 0 AND min_entry_age <= 120)),
        max_entry_age INTEGER NULL
            CHECK (
                max_entry_age IS NULL
                OR (
                    max_entry_age >= 0
                    AND max_entry_age <= 120
                    AND (min_entry_age IS NULL OR max_entry_age >= min_entry_age)
                )
            ),
        max_renewal_age INTEGER NULL
            CHECK (
                max_renewal_age IS NULL
                OR (
                    max_renewal_age >= 0
                    AND max_renewal_age <= 120
                    AND (max_entry_age IS NULL OR max_renewal_age >= max_entry_age)
                )
            ),
        lifelong_renewal BOOLEAN NULL,
        coverage_mode VARCHAR(30) NULL
            CHECK (coverage_mode IS NULL OR coverage_mode IN ('individual', 'family_floater', 'both')),
        max_family_members INTEGER NULL
            CHECK (max_family_members IS NULL OR max_family_members >= 1),
        eligible_applicant_types JSONB NOT NULL DEFAULT '[]'::jsonb
            CHECK (jsonb_typeof(eligible_applicant_types) = 'array'),

        -- Financial & coverage bounds
        annual_premium_min NUMERIC(12, 2) NULL
            CHECK (annual_premium_min IS NULL OR annual_premium_min >= 0),
        annual_premium_max NUMERIC(12, 2) NULL
            CHECK (
                annual_premium_max IS NULL
                OR (
                    annual_premium_max >= 0
                    AND (annual_premium_min IS NULL OR annual_premium_max >= annual_premium_min)
                )
            ),
        sum_insured_min NUMERIC(14, 2) NULL
            CHECK (sum_insured_min IS NULL OR sum_insured_min > 0),
        sum_insured_max NUMERIC(14, 2) NULL
            CHECK (
                sum_insured_max IS NULL
                OR (
                    sum_insured_max > 0
                    AND (sum_insured_min IS NULL OR sum_insured_max >= sum_insured_min)
                )
            ),
        deductible_amount NUMERIC(12, 2) NULL
            CHECK (deductible_amount IS NULL OR deductible_amount >= 0),
        copay_pct NUMERIC(5, 2) NULL
            CHECK (copay_pct IS NULL OR (copay_pct >= 0 AND copay_pct <= 100)),
        room_rent_limit_type VARCHAR(30) NULL
            CHECK (
                room_rent_limit_type IS NULL
                OR room_rent_limit_type IN ('no_limit', 'single_private', 'shared', 'fixed_per_day', 'pct_sum_insured')
            ),
        room_rent_limit_value NUMERIC(12, 2) NULL
            CHECK (room_rent_limit_value IS NULL OR room_rent_limit_value >= 0),

        -- Waiting periods & pre-existing disease (PED) rules
        initial_waiting_period_days INTEGER NULL
            CHECK (initial_waiting_period_days IS NULL OR initial_waiting_period_days >= 0),
        ped_covered BOOLEAN NULL,
        ped_waiting_period_months INTEGER NULL
            CHECK (ped_waiting_period_months IS NULL OR ped_waiting_period_months >= 0),
        specific_disease_waiting_months INTEGER NULL
            CHECK (specific_disease_waiting_months IS NULL OR specific_disease_waiting_months >= 0),
        maternity_waiting_months INTEGER NULL
            CHECK (maternity_waiting_months IS NULL OR maternity_waiting_months >= 0),

        -- Network, geography, tenure & product metadata
        network_hospitals_count INTEGER NULL
            CHECK (network_hospitals_count IS NULL OR network_hospitals_count >= 0),
        cashless_available BOOLEAN NULL,
        pan_india_coverage BOOLEAN NULL,
        available_regions JSONB NOT NULL DEFAULT '[]'::jsonb
            CHECK (jsonb_typeof(available_regions) = 'array'),
        min_tenure_years INTEGER NULL
            CHECK (min_tenure_years IS NULL OR min_tenure_years >= 1),
        max_tenure_years INTEGER NULL
            CHECK (
                max_tenure_years IS NULL
                OR (
                    max_tenure_years >= 1
                    AND (min_tenure_years IS NULL OR max_tenure_years >= min_tenure_years)
                )
            ),
        covered_benefits JSONB NOT NULL DEFAULT '[]'::jsonb
            CHECK (jsonb_typeof(covered_benefits) = 'array'),
        exclusions JSONB NOT NULL DEFAULT '[]'::jsonb
            CHECK (jsonb_typeof(exclusions) = 'array'),
        claim_settlement_ratio NUMERIC(5, 2) NULL
            CHECK (claim_settlement_ratio IS NULL OR (claim_settlement_ratio >= 0 AND claim_settlement_ratio <= 100)),
        no_claim_bonus_pct NUMERIC(5, 2) NULL
            CHECK (no_claim_bonus_pct IS NULL OR (no_claim_bonus_pct >= 0 AND no_claim_bonus_pct <= 200)),
        restoration_benefit BOOLEAN NULL,
        type_specific_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
            CHECK (jsonb_typeof(type_specific_metadata) = 'object'),

        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

        -- Cross-column domain invariants
        CONSTRAINT chk_policy_attributes_family_members
            CHECK (
                coverage_mode IS NULL
                OR coverage_mode = 'individual'
                OR max_family_members IS NULL
                OR max_family_members >= 2
            ),
        CONSTRAINT chk_policy_attributes_room_limit_value
            CHECK (
                room_rent_limit_type IS NULL
                OR room_rent_limit_type NOT IN ('fixed_per_day', 'pct_sum_insured')
                OR room_rent_limit_value IS NOT NULL
            ),
        CONSTRAINT chk_policy_attributes_renewal_consistency
            CHECK (NOT COALESCE(lifelong_renewal, FALSE) OR max_renewal_age IS NULL),

        CONSTRAINT uq_policy_attributes_policy_document UNIQUE (policy_id, document_id),
        CONSTRAINT fk_policy_attributes_document_policy
            FOREIGN KEY (policy_id, document_id)
            REFERENCES policy_documents (policy_id, id)
            ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_policy_attributes_catalog_doc
        ON policy_attributes (policy_id, document_id)
        WHERE is_catalog_eligible = TRUE;

    CREATE INDEX IF NOT EXISTS idx_policy_attributes_premium_sum
        ON policy_attributes (annual_premium_min, sum_insured_max)
        WHERE is_catalog_eligible = TRUE;

    CREATE INDEX IF NOT EXISTS idx_policy_attributes_applicants_gin
        ON policy_attributes USING GIN (eligible_applicant_types jsonb_path_ops);

    CREATE INDEX IF NOT EXISTS idx_policy_attributes_benefits_gin
        ON policy_attributes USING GIN (covered_benefits jsonb_path_ops);

    CREATE INDEX IF NOT EXISTS idx_policy_attributes_exclusions_gin
        ON policy_attributes USING GIN (exclusions jsonb_path_ops);

    CREATE INDEX IF NOT EXISTS idx_policy_attributes_regions_gin
        ON policy_attributes USING GIN (available_regions jsonb_path_ops);

    CREATE INDEX IF NOT EXISTS idx_policy_attributes_type_meta_gin
        ON policy_attributes USING GIN (type_specific_metadata jsonb_path_ops);

    CREATE TRIGGER trg_policy_attributes_updated_at
        BEFORE UPDATE ON policy_attributes
        FOR EACH ROW EXECUTE FUNCTION set_row_updated_at();
  `);

  pgm.sql(`
    CREATE TABLE IF NOT EXISTS ranking_features (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        policy_id UUID NOT NULL REFERENCES policies(id) ON DELETE CASCADE,
        document_id UUID NOT NULL,
        feature_schema_version VARCHAR(32) NOT NULL DEFAULT '1.0.0',

        value_for_money_score NUMERIC(6, 4) NULL
            CHECK (value_for_money_score IS NULL OR (value_for_money_score >= 0 AND value_for_money_score <= 1)),
        coverage_breadth_score NUMERIC(6, 4) NULL
            CHECK (coverage_breadth_score IS NULL OR (coverage_breadth_score >= 0 AND coverage_breadth_score <= 1)),
        waiting_period_friendliness NUMERIC(6, 4) NULL
            CHECK (
                waiting_period_friendliness IS NULL
                OR (waiting_period_friendliness >= 0 AND waiting_period_friendliness <= 1)
            ),
        network_strength_score NUMERIC(6, 4) NULL
            CHECK (network_strength_score IS NULL OR (network_strength_score >= 0 AND network_strength_score <= 1)),
        claim_reliability_score NUMERIC(6, 4) NULL
            CHECK (claim_reliability_score IS NULL OR (claim_reliability_score >= 0 AND claim_reliability_score <= 1)),

        feature_vector JSONB NOT NULL DEFAULT '{}'::jsonb
            CHECK (jsonb_typeof(feature_vector) = 'object'),
        computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

        CONSTRAINT uq_ranking_features_doc_schema
            UNIQUE (policy_id, document_id, feature_schema_version),
        CONSTRAINT fk_ranking_features_document_policy
            FOREIGN KEY (policy_id, document_id)
            REFERENCES policy_documents (policy_id, id)
            ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_ranking_features_doc_schema
        ON ranking_features (policy_id, document_id, feature_schema_version);
  `);

  pgm.sql(`
    DO $$
    BEGIN
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'coverageiq_runtime') THEN
            GRANT USAGE ON SCHEMA public TO coverageiq_runtime;
            GRANT SELECT, INSERT, UPDATE, DELETE
                ON TABLE policy_attributes, ranking_features
                TO coverageiq_runtime;
        END IF;
    END
    $$;
  `);
};

exports.down = (pgm) => {
  pgm.sql(`
    DROP TRIGGER IF EXISTS trg_policy_attributes_updated_at ON policy_attributes;
    DROP TABLE IF EXISTS ranking_features CASCADE;
    DROP TABLE IF EXISTS policy_attributes CASCADE;
  `);
};
