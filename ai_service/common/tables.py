"""SQLAlchemy Core metadata synchronized with authoritative CoverageIQ PostgreSQL migrations."""

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
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

metadata = MetaData()

# ---------------------------------------------------------------------------
# Node-owned tables (read-only reflection aligned with node-pg-migrate)
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
    Column("external_policy_key", Text, nullable=False),
    Column("title", Text, nullable=False),
    Column("insurer_name", Text, nullable=True),
    Column("policy_type", Text, nullable=False, server_default="general"),
    Column("visibility", Text, nullable=False, server_default="private"),
    Column("owner_id", Text, nullable=False),
    Column("owner_scope", Text, nullable=False),
    Column("status", Text, nullable=False, server_default="draft"),
    Column("active_version_id", UUID(as_uuid=True), nullable=True),
    Column("archived_at", DateTime(timezone=True), nullable=True),
    Column("metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

policy_documents = Table(
    "policy_documents",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "policy_id",
        UUID(as_uuid=True),
        ForeignKey("policies.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("version_number", Integer, nullable=False),
    Column("version_label", Text, nullable=True),
    Column("file_name", Text, nullable=False),
    Column("storage_path", Text, nullable=False),
    Column("mime_type", Text, nullable=False),
    Column("file_size_bytes", BigInteger, nullable=False),
    Column("file_hash_sha256", String(64), nullable=False),
    Column("page_count", Integer, nullable=False, server_default="0"),
    Column("is_encrypted", Boolean, nullable=False, server_default=text("FALSE")),
    Column("ingestion_status", Text, nullable=False, server_default="pending"),
    Column("is_archived", Boolean, nullable=False, server_default=text("FALSE")),
    Column("archived_at", DateTime(timezone=True), nullable=True),
    Column("document_metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("policy_id", "id", name="uq_policy_documents_policy_id_id"),
    UniqueConstraint("policy_id", "version_number", name="uq_policy_documents_policy_version"),
)

# ---------------------------------------------------------------------------
# Checkpoint 5.0 Policy Recommendation Tables (`1764250300000_recommendation_policy_attributes.js`)
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
    Column("document_id", UUID(as_uuid=True), nullable=False),
    Column("is_catalog_eligible", Boolean, nullable=False, server_default=text("FALSE")),
    Column("extraction_status", String(32), nullable=False, server_default=text("'pending'")),
    Column("extraction_source", String(32), nullable=False, server_default=text("'ingestion_llm'")),
    Column("extraction_confidence", Numeric(4, 3), nullable=True),
    Column("extraction_error", Text, nullable=True),
    Column("source_chunk_ids", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("field_provenance", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Column("min_entry_age", Integer, nullable=True),
    Column("max_entry_age", Integer, nullable=True),
    Column("max_renewal_age", Integer, nullable=True),
    Column("lifelong_renewal", Boolean, nullable=True),
    Column("coverage_mode", String(30), nullable=True),
    Column("max_family_members", Integer, nullable=True),
    Column("eligible_applicant_types", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("annual_premium_min", Numeric(12, 2), nullable=True),
    Column("annual_premium_max", Numeric(12, 2), nullable=True),
    Column("sum_insured_min", Numeric(14, 2), nullable=True),
    Column("sum_insured_max", Numeric(14, 2), nullable=True),
    Column("deductible_amount", Numeric(12, 2), nullable=True),
    Column("copay_pct", Numeric(5, 2), nullable=True),
    Column("room_rent_limit_type", String(30), nullable=True),
    Column("room_rent_limit_value", Numeric(12, 2), nullable=True),
    Column("initial_waiting_period_days", Integer, nullable=True),
    Column("ped_covered", Boolean, nullable=True),
    Column("ped_waiting_period_months", Integer, nullable=True),
    Column("specific_disease_waiting_months", Integer, nullable=True),
    Column("maternity_waiting_months", Integer, nullable=True),
    Column("network_hospitals_count", Integer, nullable=True),
    Column("cashless_available", Boolean, nullable=True),
    Column("pan_india_coverage", Boolean, nullable=True),
    Column("available_regions", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("min_tenure_years", Integer, nullable=True),
    Column("max_tenure_years", Integer, nullable=True),
    Column("covered_benefits", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("exclusions", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("claim_settlement_ratio", Numeric(5, 2), nullable=True),
    Column("no_claim_bonus_pct", Numeric(5, 2), nullable=True),
    Column("restoration_benefit", Boolean, nullable=True),
    Column("type_specific_metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("NOW()")),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=text("NOW()")),
    UniqueConstraint("policy_id", "document_id", name="uq_policy_attributes_policy_document"),
    ForeignKeyConstraint(
        ["policy_id", "document_id"],
        ["policy_documents.policy_id", "policy_documents.id"],
        ondelete="CASCADE",
        name="fk_policy_attributes_document_policy",
    ),
    CheckConstraint("min_entry_age IS NULL OR (min_entry_age >= 0 AND min_entry_age <= 120)"),
    CheckConstraint("max_entry_age IS NULL OR (max_entry_age >= 0 AND max_entry_age <= 120)"),
    CheckConstraint("annual_premium_min IS NULL OR annual_premium_min >= 0"),
    CheckConstraint("sum_insured_max IS NULL OR sum_insured_max > 0"),
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
    Column("document_id", UUID(as_uuid=True), nullable=False),
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
    Column("feature_vector", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Column("computed_at", DateTime(timezone=True), nullable=False, server_default=text("NOW()")),
    UniqueConstraint(
        "policy_id",
        "document_id",
        "feature_schema_version",
        name="uq_ranking_features_doc_schema",
    ),
    ForeignKeyConstraint(
        ["policy_id", "document_id"],
        ["policy_documents.policy_id", "policy_documents.id"],
        ondelete="CASCADE",
        name="fk_ranking_features_document_policy",
    ),
)


def fetch_catalog_policies_metadata(
    conn: Connection,
    *,
    policy_type: str | None = None,
    active_only: bool = True,
) -> list[dict[str, Any]]:
    """Return read-only catalog policy records using CoverageIQ visibility & status."""
    stmt = select(
        policies.c.id.label("policy_id"),
        policies.c.external_policy_key,
        policies.c.owner_id,
        policies.c.owner_scope,
        policies.c.visibility,
        policies.c.title.label("policy_name"),
        policies.c.insurer_name,
        policies.c.policy_type,
        policies.c.status,
        policies.c.active_version_id,
        policies.c.created_at,
        policies.c.updated_at,
    ).where(policies.c.visibility == "catalog")

    if active_only:
        stmt = stmt.where(policies.c.status == "active")
    if policy_type is not None:
        stmt = stmt.where(policies.c.policy_type == policy_type.strip().lower())

    stmt = stmt.order_by(policies.c.created_at.desc(), policies.c.id.asc())
    return [dict(row) for row in conn.execute(stmt).mappings().all()]
