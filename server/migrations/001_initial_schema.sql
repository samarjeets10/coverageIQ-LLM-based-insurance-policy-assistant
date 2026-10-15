


CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(320) NOT NULL,
    password_hash TEXT NOT NULL,
    full_name VARCHAR(150) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    email_verified_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX users_email_lower_uq
    ON users (LOWER(email));

CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL,
    slug VARCHAR(100) NOT NULL UNIQUE,
    created_by UUID NOT NULL REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE organization_members (
    organization_id UUID NOT NULL
        REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL
        REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL DEFAULT 'member'
        CHECK (role IN ('owner', 'admin', 'member')),
    joined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (organization_id, user_id)
);

CREATE INDEX organization_members_user_idx
    ON organization_members (user_id);

CREATE TABLE refresh_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    replaced_by UUID REFERENCES refresh_tokens(id),
    CHECK (expires_at > created_at)
);

CREATE INDEX refresh_tokens_user_idx
    ON refresh_tokens (user_id);

CREATE INDEX refresh_tokens_active_expiry_idx
    ON refresh_tokens (expires_at)
    WHERE revoked_at IS NULL;

CREATE TABLE policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    owner_user_id UUID REFERENCES users(id),
    organization_id UUID REFERENCES organizations(id),

    created_by UUID NOT NULL REFERENCES users(id),
    name VARCHAR(250) NOT NULL,
    insurer_name VARCHAR(200),
    policy_type VARCHAR(50),
    description TEXT,

    status VARCHAR(20) NOT NULL DEFAULT 'draft'
        CHECK (status IN (
            'draft', 'processing', 'active',
            'failed', 'archived'
        )),

    current_version_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CHECK (
        (owner_user_id IS NOT NULL AND organization_id IS NULL)
        OR
        (owner_user_id IS NULL AND organization_id IS NOT NULL)
    )
);

CREATE INDEX policies_owner_user_idx
    ON policies (owner_user_id)
    WHERE owner_user_id IS NOT NULL;

CREATE INDEX policies_organization_idx
    ON policies (organization_id)
    WHERE organization_id IS NOT NULL;

CREATE INDEX policies_created_by_idx
    ON policies (created_by);

CREATE TABLE policy_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_id UUID NOT NULL
        REFERENCES policies(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL CHECK (version_number > 0),
    original_filename TEXT NOT NULL,
    mime_type VARCHAR(100) NOT NULL DEFAULT 'application/pdf',
    file_size_bytes BIGINT NOT NULL CHECK (file_size_bytes > 0),
    sha256 CHAR(64),
    storage_key TEXT NOT NULL,
    uploaded_by UUID NOT NULL REFERENCES users(id),
    processing_status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (processing_status IN (
            'pending', 'processing', 'completed', 'failed'
        )),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (policy_id, version_number),
    UNIQUE (id, policy_id)
);

ALTER TABLE policies
    ADD CONSTRAINT policies_current_version_fk
    FOREIGN KEY (current_version_id, id)
    REFERENCES policy_versions (id, policy_id)
    DEFERRABLE INITIALLY DEFERRED;

CREATE INDEX policy_versions_policy_idx
    ON policy_versions (policy_id, version_number DESC);

CREATE TABLE uploads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_version_id UUID NOT NULL
        REFERENCES policy_versions(id) ON DELETE CASCADE,
    uploaded_by UUID NOT NULL REFERENCES users(id),
    storage_key TEXT NOT NULL UNIQUE,
    original_filename TEXT NOT NULL,
    mime_type VARCHAR(100) NOT NULL,
    file_size_bytes BIGINT NOT NULL CHECK (file_size_bytes > 0),
    sha256 CHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_version_id UUID NOT NULL
        REFERENCES policy_versions(id) ON DELETE CASCADE,
    requested_by UUID NOT NULL REFERENCES users(id),
    job_type VARCHAR(30) NOT NULL DEFAULT 'ingestion'
        CHECK (job_type IN ('ingestion', 'reprocessing')),
    status VARCHAR(20) NOT NULL DEFAULT 'queued'
        CHECK (status IN (
            'queued', 'processing', 'completed', 'failed', 'cancelled'
        )),
    attempt_count INTEGER NOT NULL DEFAULT 0
        CHECK (attempt_count >= 0),
    error_code VARCHAR(100),
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX jobs_status_created_idx
    ON jobs (status, created_at);

CREATE INDEX jobs_version_idx
    ON jobs (policy_version_id, created_at DESC);

CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    policy_id UUID NOT NULL REFERENCES policies(id) ON DELETE CASCADE,
    title VARCHAR(250),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX chat_sessions_user_updated_idx
    ON chat_sessions (user_id, updated_at DESC);

CREATE INDEX chat_sessions_policy_idx
    ON chat_sessions (policy_id);

CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL
        REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL
        CHECK (role IN ('system', 'user', 'assistant')),
    content TEXT NOT NULL,
    decision VARCHAR(30)
        CHECK (decision IS NULL OR decision IN (
            'COVERED', 'NOT_COVERED',
            'CONDITIONAL', 'INSUFFICIENT_EVIDENCE'
        )),
    confidence NUMERIC(4,3)
        CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    citations JSONB NOT NULL DEFAULT '[]'::JSONB
        CHECK (jsonb_typeof(citations) = 'array'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX chat_messages_session_created_idx
    ON chat_messages (session_id, created_at);

CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    organization_id UUID REFERENCES organizations(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id UUID,
    request_id VARCHAR(100),
    details JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX audit_logs_created_idx
    ON audit_logs (created_at DESC);

CREATE INDEX audit_logs_actor_idx
    ON audit_logs (actor_user_id, created_at DESC);
