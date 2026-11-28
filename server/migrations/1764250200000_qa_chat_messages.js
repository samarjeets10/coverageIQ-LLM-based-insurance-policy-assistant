/**
 * CoverageIQ Checkpoint 4.9 Q&A Conversation Persistence Migration
 *
 * Adds PostgreSQL-backed `chat_messages` table owned by Node.js (node-pg-migrate)
 * for tenant-isolated Q&A questions and verified AI responses with citations.
 */

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql(`
    CREATE TABLE IF NOT EXISTS chat_messages (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        organization_id UUID NULL REFERENCES organizations(id) ON DELETE SET NULL,
        policy_id UUID NOT NULL REFERENCES policies(id) ON DELETE CASCADE,
        document_id UUID NULL REFERENCES policy_documents(id) ON DELETE SET NULL,
        owner_scope TEXT NOT NULL CHECK (length(btrim(owner_scope)) > 0),
        conversation_id UUID NOT NULL,
        role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
        content TEXT NOT NULL CHECK (length(btrim(content)) > 0),
        decision VARCHAR(32) NULL
            CHECK (decision IS NULL OR decision IN ('COVERED', 'NOT COVERED', 'CONDITIONAL', 'INSUFFICIENT EVIDENCE')),
        confidence_score NUMERIC(5, 4) NULL
            CHECK (confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)),
        conditions JSONB NOT NULL DEFAULT '[]'::jsonb
            CHECK (jsonb_typeof(conditions) = 'array'),
        citations JSONB NOT NULL DEFAULT '[]'::jsonb
            CHECK (jsonb_typeof(citations) = 'array'),
        evaluator_status VARCHAR(64) NULL,
        retry_count INTEGER NULL CHECK (retry_count IS NULL OR retry_count >= 0),
        provider_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
            CHECK (jsonb_typeof(provider_metadata) = 'object'),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT chk_chat_messages_assistant_fields CHECK (
            role = 'user'
            OR (decision IS NOT NULL AND evaluator_status IS NOT NULL)
        )
    );

    CREATE INDEX IF NOT EXISTS idx_chat_messages_policy_user_chrono
        ON chat_messages (policy_id, user_id, created_at ASC, id ASC);

    CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_chrono
        ON chat_messages (conversation_id, created_at ASC, id ASC);

    CREATE INDEX IF NOT EXISTS idx_chat_messages_user_recent
        ON chat_messages (user_id, created_at DESC);
  `);

  pgm.sql(`
    DO $$
    BEGIN
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'coverageiq_runtime') THEN
            GRANT USAGE ON SCHEMA public TO coverageiq_runtime;
            GRANT SELECT, INSERT, UPDATE, DELETE
                ON TABLE chat_messages
                TO coverageiq_runtime;
        END IF;
    END
    $$;
  `);
};

exports.down = (pgm) => {
  pgm.sql(`
    DROP TABLE IF EXISTS chat_messages CASCADE;
  `);
};
