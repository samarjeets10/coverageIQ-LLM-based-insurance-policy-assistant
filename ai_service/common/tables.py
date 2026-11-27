"""SQLAlchemy Core metadata synchronized with 001 and 003 PostgreSQL migrations."""

from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Connection,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint,
    and_,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

metadata = MetaData()

# ---------------------------------------------------------------------------
# Existing Node-owned tables (read-only reflection for Python AI Service)
# Matches server/migrations/001_initial_schema.sql
# ---------------------------------------------------------------------------

organizations = Table(
    "organizations",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("name", String(200), nullable=False),
    Column("slug", String(100), nullable=False, unique=True),
    Column("created_by", UUID(as_uuid=True), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

policies = Table(
    "policies",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("owner_user_id", UUID(as_uuid=True), nullable=True),
    Column(
        "organization_id",
        UUID(as_uuid=True),
        ForeignKey("organizations.id"),
        nullable=True,
    ),
    Column("created_by", UUID(as_uuid=True), nullable=False),
    Column("name", String(250), nullable=False),
    Column("insurer_name", String(200), nullable=True),
    Column("policy_type", String(50), nullable=True),
    Column("description", Text, nullable=True),
    Column("status", String(20), nullable=False, server_default="draft"),
    Column("current_version_id", UUID(as_uuid=True), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

policy_versions = Table(
    "policy_versions",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "policy_id",
        UUID(as_uuid=True),
        ForeignKey("policies.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("version_number", Integer, nullable=False),
    Column("original_filename", Text, nullable=False),
    Column("mime_type", String(100), nullable=False),
    Column("file_size_bytes", BigInteger, nullable=False),
    Column("sha256", String(64), nullable=True),
    Column("storage_key", Text, nullable=False),
    Column("uploaded_by", UUID(as_uuid=True), nullable=False),
    Column("processing_status", String(20), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("policy_id", "version_number"),
    UniqueConstraint("id", "policy_id"),
)

# ---------------------------------------------------------------------------
# Python-owned recommendation tables (Architecture §2A)
# Synchronized with server/migrations/003_recommendation_schema.sql
# ---------------------------------------------------------------------------

policy_attributes = Table(
    "policy_attributes",
    metadata,
    Column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    ),
    Column(
        "policy_id",
        UUID(as_uuid=True),
        ForeignKey("policies.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("policy_version_id", UUID(as_uuid=True), nullable=True),
    Column("is_catalog_eligible", Boolean, nullable=False, server_default=text("FALSE")),
    Column("min_entry_age", Integer, nullable=False, server_default=text("18")),
    Column("max_entry_age", Integer, nullable=False, server_default=text("65")),
    Column("max_renewal_age", Integer, nullable=True),
    Column("lifelong_renewal", Boolean, nullable=False, server_default=text("FALSE")),
    Column("coverage_mode", String(30), nullable=False, server_default=text("'individual'")),
    Column("max_family_members", Integer, nullable=False, server_default=text("1")),
    Column(
        "eligible_applicant_types",
        JSONB,
        nullable=False,
        server_default=text('\'["salaried","self_employed","student","senior_citizen"]\'::JSONB'),
    ),
    Column("annual_premium_min", Numeric(12, 2), nullable=False),
    Column("annual_premium_max", Numeric(12, 2), nullable=False),
    Column("sum_insured_min", Numeric(14, 2), nullable=False),
    Column("sum_insured_max", Numeric(14, 2), nullable=False),
    Column("deductible_amount", Numeric(12, 2), nullable=False, server_default=text("0")),
    Column("copay_pct", Numeric(5, 2), nullable=False, server_default=text("0")),
    Column(
        "room_rent_limit_type",
        String(30),
        nullable=False,
        server_default=text("'no_limit'"),
    ),
    Column("room_rent_limit_value", Numeric(12, 2), nullable=True),
    Column("initial_waiting_period_days", Integer, nullable=False, server_default=text("30")),
    Column("ped_covered", Boolean, nullable=False, server_default=text("FALSE")),
    Column("ped_waiting_period_months", Integer, nullable=True),
    Column(
        "specific_disease_waiting_months",
        Integer,
        nullable=False,
        server_default=text("24"),
    ),
    Column("maternity_waiting_months", Integer, nullable=True),
    Column("network_hospitals_count", Integer, nullable=False, server_default=text("0")),
    Column("cashless_available", Boolean, nullable=False, server_default=text("FALSE")),
    Column("pan_india_coverage", Boolean, nullable=False, server_default=text("FALSE")),
    Column("available_regions", JSONB, nullable=False, server_default=text("'[]'::JSONB")),
    Column("min_tenure_years", Integer, nullable=False, server_default=text("1")),
    Column("max_tenure_years", Integer, nullable=False, server_default=text("3")),
    Column("covered_benefits", JSONB, nullable=False, server_default=text("'[]'::JSONB")),
    Column("exclusions", JSONB, nullable=False, server_default=text("'[]'::JSONB")),
    Column("claim_settlement_ratio", Numeric(5, 2), nullable=True),
    Column("no_claim_bonus_pct", Numeric(5, 2), nullable=False, server_default=text("0")),
    Column("restoration_benefit", Boolean, nullable=False, server_default=text("FALSE")),
    Column("type_specific_metadata", JSONB, nullable=False, server_default=text("'{}'::JSONB")),
    Column("extraction_source", String(30), nullable=False, server_default=text("'manual'")),
    Column("extraction_confidence", Numeric(4, 3), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("NOW()")),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=text("NOW()")),
    UniqueConstraint(
        "policy_id",
        "policy_version_id",
        name="policy_attributes_policy_version_uq",
        postgresql_nulls_not_distinct=True,
    ),
    ForeignKeyConstraint(
        ["policy_version_id", "policy_id"],
        ["policy_versions.id", "policy_versions.policy_id"],
        ondelete="CASCADE",
        name="policy_attributes_version_fk",
    ),
    CheckConstraint("min_entry_age >= 0 AND min_entry_age <= 120"),
    CheckConstraint("max_entry_age >= min_entry_age AND max_entry_age <= 120"),
    CheckConstraint("max_renewal_age IS NULL OR max_renewal_age >= max_entry_age"),
    CheckConstraint("coverage_mode IN ('individual', 'family_floater', 'both')"),
    CheckConstraint("max_family_members >= 1"),
    CheckConstraint("annual_premium_min >= 0 AND annual_premium_max >= annual_premium_min"),
    CheckConstraint("sum_insured_min > 0 AND sum_insured_max >= sum_insured_min"),
    CheckConstraint("deductible_amount >= 0"),
    CheckConstraint("copay_pct >= 0 AND copay_pct <= 100"),
    CheckConstraint(
        "coverage_mode = 'individual' OR max_family_members >= 2",
        name="policy_attributes_family_members_check",
    ),
    CheckConstraint(
        "room_rent_limit_type NOT IN ('fixed_per_day', 'pct_sum_insured') "
        "OR room_rent_limit_value IS NOT NULL",
        name="policy_attributes_room_limit_value_check",
    ),
    CheckConstraint(
        "NOT lifelong_renewal OR max_renewal_age IS NULL",
        name="policy_attributes_renewal_consistency_check",
    ),
)

ranking_features = Table(
    "ranking_features",
    metadata,
    Column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    ),
    Column(
        "policy_id",
        UUID(as_uuid=True),
        ForeignKey("policies.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("policy_version_id", UUID(as_uuid=True), nullable=True),
    Column(
        "feature_schema_version",
        String(32),
        nullable=False,
        server_default=text("'1.0.0'"),
    ),
    Column("value_for_money_score", Numeric(6, 4), nullable=True),
    Column("coverage_breadth_score", Numeric(6, 4), nullable=True),
    Column("waiting_period_friendliness", Numeric(6, 4), nullable=True),
    Column("network_strength_score", Numeric(6, 4), nullable=True),
    Column("claim_reliability_score", Numeric(6, 4), nullable=True),
    Column("feature_vector", JSONB, nullable=False, server_default=text("'{}'::JSONB")),
    Column("computed_at", DateTime(timezone=True), nullable=False, server_default=text("NOW()")),
    UniqueConstraint(
        "policy_id",
        "policy_version_id",
        "feature_schema_version",
        name="ranking_features_policy_version_schema_uq",
        postgresql_nulls_not_distinct=True,
    ),
    ForeignKeyConstraint(
        ["policy_version_id", "policy_id"],
        ["policy_versions.id", "policy_versions.policy_id"],
        ondelete="CASCADE",
        name="ranking_features_version_fk",
    ),
    CheckConstraint(
        "value_for_money_score IS NULL "
        "OR (value_for_money_score >= 0 AND value_for_money_score <= 1)"
    ),
    CheckConstraint(
        "coverage_breadth_score IS NULL "
        "OR (coverage_breadth_score >= 0 AND coverage_breadth_score <= 1)"
    ),
    CheckConstraint(
        "waiting_period_friendliness IS NULL "
        "OR (waiting_period_friendliness >= 0 AND waiting_period_friendliness <= 1)"
    ),
    CheckConstraint(
        "network_strength_score IS NULL "
        "OR (network_strength_score >= 0 AND network_strength_score <= 1)"
    ),
    CheckConstraint(
        "claim_reliability_score IS NULL "
        "OR (claim_reliability_score >= 0 AND claim_reliability_score <= 1)"
    ),
)


def fetch_catalog_policies_metadata(
    conn: Connection,
    policy_type: str | None = None,
    active_only: bool = True,
) -> list[dict[str, Any]]:
    """Query organization-owned catalog policies from PostgreSQL (excludes private uploads)."""
    conditions = [
        policies.c.organization_id.is_not(None),
        policies.c.owner_user_id.is_(None),
    ]
    if active_only:
        conditions.append(policies.c.status == "active")
    if policy_type is not None:
        conditions.append(policies.c.policy_type == policy_type.strip().lower())

    stmt = (
        select(
            policies.c.id,
            policies.c.organization_id,
            policies.c.name,
            policies.c.insurer_name,
            policies.c.policy_type,
            policies.c.description,
            policies.c.status,
            policies.c.current_version_id,
        )
        .where(and_(*conditions))
        .order_by(policies.c.created_at.desc())
    )
    rows = conn.execute(stmt).mappings().all()
    return [dict(row) for row in rows]
