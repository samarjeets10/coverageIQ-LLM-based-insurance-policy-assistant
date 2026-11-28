/**
 * CoverageIQ Checkpoint 4.5 Additive Auth, Organization & Audit Migration
 *
 * Adopts teammate authentication/RBAC user & organization schema into the single
 * Node-owned node-pg-migrate migration timeline without replacing or altering the
 * pgvector policies/documents/chunks/embeddings/ingestion_jobs schema.
 */

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql(`
    CREATE TABLE IF NOT EXISTS users (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        email VARCHAR(320) NOT NULL,
        password_hash TEXT NOT NULL,
        full_name VARCHAR(150) NOT NULL,
        platform_role VARCHAR(20) NOT NULL DEFAULT 'user'
            CHECK (platform_role IN ('user', 'admin', 'organisation')),
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        email_verified_at TIMESTAMPTZ NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE UNIQUE INDEX IF NOT EXISTS users_email_lower_uq
        ON users (LOWER(email));
  `);

  pgm.sql(`
    CREATE TABLE IF NOT EXISTS organizations (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name VARCHAR(200) NOT NULL,
        slug VARCHAR(100) NOT NULL UNIQUE,
        created_by UUID NOT NULL REFERENCES users(id),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS organization_members (
        organization_id UUID NOT NULL
            REFERENCES organizations(id) ON DELETE CASCADE,
        user_id UUID NOT NULL
            REFERENCES users(id) ON DELETE CASCADE,
        role VARCHAR(20) NOT NULL DEFAULT 'member'
            CHECK (role IN ('owner', 'admin', 'member')),
        joined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (organization_id, user_id)
    );

    CREATE INDEX IF NOT EXISTS organization_members_user_idx
        ON organization_members (user_id);
  `);

  pgm.sql(`
    CREATE TABLE IF NOT EXISTS refresh_tokens (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        token_hash TEXT NOT NULL UNIQUE,
        expires_at TIMESTAMPTZ NOT NULL,
        revoked_at TIMESTAMPTZ NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        replaced_by UUID NULL REFERENCES refresh_tokens(id),
        CHECK (expires_at > created_at)
    );

    CREATE INDEX IF NOT EXISTS refresh_tokens_user_idx
        ON refresh_tokens (user_id);

    CREATE INDEX IF NOT EXISTS refresh_tokens_active_expiry_idx
        ON refresh_tokens (expires_at)
        WHERE revoked_at IS NULL;
  `);

  pgm.sql(`
    CREATE TABLE IF NOT EXISTS audit_logs (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        actor_user_id UUID NULL REFERENCES users(id) ON DELETE SET NULL,
        organization_id UUID NULL REFERENCES organizations(id) ON DELETE SET NULL,
        action VARCHAR(100) NOT NULL,
        resource_type VARCHAR(100) NOT NULL,
        resource_id UUID NULL,
        request_id VARCHAR(100) NULL,
        details JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS audit_logs_created_idx
        ON audit_logs (created_at DESC);

    CREATE INDEX IF NOT EXISTS audit_logs_actor_idx
        ON audit_logs (actor_user_id, created_at DESC);
  `);

  pgm.sql(`
    DROP TRIGGER IF EXISTS trg_users_updated_at ON users;
    CREATE TRIGGER trg_users_updated_at
        BEFORE UPDATE ON users
        FOR EACH ROW EXECUTE FUNCTION set_row_updated_at();

    DROP TRIGGER IF EXISTS trg_organizations_updated_at ON organizations;
    CREATE TRIGGER trg_organizations_updated_at
        BEFORE UPDATE ON organizations
        FOR EACH ROW EXECUTE FUNCTION set_row_updated_at();
  `);

  pgm.sql(`
    DO $$
    BEGIN
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'coverageiq_runtime') THEN
            REVOKE CREATE ON SCHEMA public FROM PUBLIC;
            REVOKE CREATE ON SCHEMA public FROM coverageiq_runtime;
            GRANT USAGE ON SCHEMA public TO coverageiq_runtime;
            GRANT SELECT, INSERT, UPDATE, DELETE
                ON TABLE users, organizations, organization_members, refresh_tokens, audit_logs
                TO coverageiq_runtime;
        END IF;
    END
    $$;
  `);
};

exports.down = (pgm) => {
  pgm.sql(`
    DROP TRIGGER IF EXISTS trg_organizations_updated_at ON organizations;
    DROP TRIGGER IF EXISTS trg_users_updated_at ON users;

    DROP TABLE IF EXISTS audit_logs CASCADE;
    DROP TABLE IF EXISTS refresh_tokens CASCADE;
    DROP TABLE IF EXISTS organization_members CASCADE;
    DROP TABLE IF EXISTS organizations CASCADE;
    DROP TABLE IF EXISTS users CASCADE;
  `);
};
