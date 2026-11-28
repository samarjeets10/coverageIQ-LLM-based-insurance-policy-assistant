/**
 * CoverageIQ Checkpoint 4.2 Initial PostgreSQL 16 + pgvector Schema Migration
 *
 * Migration ownership: Node.js (`node-pg-migrate`) using the dedicated migration role.
 * Note on sparse search: `policy_chunks.search_tsv` + GIN index uses PostgreSQL's
 * full-text search (`tsvector`/`ts_rank_cd`). PostgreSQL FTS is NOT true Okapi BM25
 * (it lacks corpus-level IDF saturation `k1` and average-document-length normalization `b`);
 * exact `BM25Okapi` scoring remains in Python (`ai_service/retrieval/bm25_retriever.py`).
 */

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql(`
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
            CREATE EXTENSION vector;
        END IF;
    END
    $$;
  `);

  pgm.sql(`
    CREATE OR REPLACE FUNCTION set_row_updated_at()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = NOW();
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
  `);

  pgm.sql(`
    CREATE TABLE policies (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        external_policy_key TEXT NOT NULL CHECK (length(btrim(external_policy_key)) > 0),
        title TEXT NOT NULL CHECK (length(btrim(title)) > 0),
        insurer_name TEXT NULL,
        policy_type TEXT NOT NULL DEFAULT 'general'
            CHECK (policy_type IN ('health', 'car', 'life', 'term', 'travel', 'motor', 'pension', 'general')),
        visibility TEXT NOT NULL DEFAULT 'private'
            CHECK (visibility IN ('private', 'catalog')),
        owner_id TEXT NOT NULL CHECK (length(btrim(owner_id)) > 0),
        owner_scope TEXT NOT NULL CHECK (length(btrim(owner_scope)) > 0),
        status TEXT NOT NULL DEFAULT 'draft'
            CHECK (status IN ('draft', 'active', 'inactive', 'archived')),
        active_version_id UUID NULL,
        archived_at TIMESTAMPTZ NULL,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT uq_policies_external_key UNIQUE (external_policy_key)
    );
  `);

  pgm.sql(`
    CREATE TABLE policy_documents (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        policy_id UUID NOT NULL REFERENCES policies(id) ON DELETE CASCADE,
        version_number INTEGER NOT NULL CHECK (version_number >= 1),
        version_label TEXT NULL,
        file_name TEXT NOT NULL CHECK (length(btrim(file_name)) > 0),
        storage_path TEXT NOT NULL CHECK (length(btrim(storage_path)) > 0),
        mime_type TEXT NOT NULL DEFAULT 'application/pdf' CHECK (length(btrim(mime_type)) > 0),
        file_size_bytes BIGINT NOT NULL CHECK (file_size_bytes >= 0),
        file_hash_sha256 CHAR(64) NOT NULL CHECK (file_hash_sha256 ~ '^[0-9a-f]{64}$'),
        page_count INTEGER NOT NULL DEFAULT 0 CHECK (page_count >= 0),
        is_encrypted BOOLEAN NOT NULL DEFAULT FALSE,
        ingestion_status TEXT NOT NULL DEFAULT 'pending'
            CHECK (ingestion_status IN ('pending', 'queued', 'processing', 'completed', 'failed', 'archived')),
        is_archived BOOLEAN NOT NULL DEFAULT FALSE,
        archived_at TIMESTAMPTZ NULL,
        document_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT uq_policy_documents_policy_id_id UNIQUE (policy_id, id),
        CONSTRAINT uq_policy_documents_policy_version UNIQUE (policy_id, version_number)
    );
  `);

  // Enforce at the schema level that policies.active_version_id can only reference a
  // document belonging to that exact same policy row. On document deletion, only
  // active_version_id is cleared (PG 15+ column-scoped ON DELETE SET NULL).
  pgm.sql(`
    ALTER TABLE policies
        ADD CONSTRAINT fk_policies_active_version_same_policy
        FOREIGN KEY (id, active_version_id)
        REFERENCES policy_documents(policy_id, id)
        ON DELETE SET NULL (active_version_id)
        DEFERRABLE INITIALLY DEFERRED;
  `);

  pgm.sql(`
    CREATE TABLE policy_chunks (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        policy_id UUID NOT NULL,
        document_id UUID NOT NULL,
        chunk_key TEXT NOT NULL CHECK (length(btrim(chunk_key)) > 0),
        chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
        section_id TEXT NULL,
        clause_ref TEXT NOT NULL CHECK (length(btrim(clause_ref)) > 0),
        category TEXT NOT NULL
            CHECK (category IN ('Coverage', 'Exclusions', 'Conditions', 'Definitions', 'General')),
        page_number INTEGER NOT NULL CHECK (page_number >= 1),
        page_start INTEGER NOT NULL CHECK (page_start >= 1),
        page_end INTEGER NOT NULL CHECK (page_end >= page_start),
        content_text TEXT NOT NULL CHECK (length(btrim(content_text)) > 0),
        source_block_ids TEXT[] NOT NULL DEFAULT '{}'::text[],
        table_id TEXT NULL,
        extraction_method TEXT NOT NULL DEFAULT 'DIGITAL'
            CHECK (extraction_method IN ('DIGITAL', 'OCR', 'HYBRID')),
        token_count INTEGER NULL CHECK (token_count IS NULL OR token_count >= 0),
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        search_tsv tsvector GENERATED ALWAYS AS (
            setweight(to_tsvector('english', coalesce(clause_ref, '')), 'A') ||
            setweight(to_tsvector('english', coalesce(content_text, '')), 'B')
        ) STORED,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT fk_policy_chunks_document_policy
            FOREIGN KEY (document_id, policy_id)
            REFERENCES policy_documents(id, policy_id)
            ON DELETE CASCADE,
        CONSTRAINT uq_policy_chunks_doc_chunk_key UNIQUE (document_id, chunk_key),
        CONSTRAINT uq_policy_chunks_doc_policy_id_id UNIQUE (document_id, policy_id, id)
    );
  `);

  pgm.sql(`
    CREATE TABLE chunk_embeddings (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        chunk_id UUID NOT NULL,
        policy_id UUID NOT NULL,
        document_id UUID NOT NULL,
        model_name TEXT NOT NULL CHECK (length(btrim(model_name)) > 0),
        model_revision TEXT NOT NULL DEFAULT 'default' CHECK (length(btrim(model_revision)) > 0),
        embedding_dim INTEGER NOT NULL DEFAULT 1024 CHECK (embedding_dim = 1024),
        is_normalized BOOLEAN NOT NULL DEFAULT TRUE,
        embedding vector(1024) NOT NULL CHECK (vector_dims(embedding) = 1024),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT fk_chunk_embeddings_chunk_doc_policy
            FOREIGN KEY (document_id, policy_id, chunk_id)
            REFERENCES policy_chunks(document_id, policy_id, id)
            ON DELETE CASCADE,
        CONSTRAINT uq_chunk_embeddings_chunk_model UNIQUE (chunk_id, model_name, model_revision)
    );
  `);

  // Ingestion job history uses ON DELETE SET NULL so job audit history survives
  // document/policy archival or deletion.
  pgm.sql(`
    CREATE TABLE ingestion_jobs (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        policy_id UUID NULL REFERENCES policies(id) ON DELETE SET NULL,
        document_id UUID NULL REFERENCES policy_documents(id) ON DELETE SET NULL,
        external_policy_key TEXT NOT NULL CHECK (length(btrim(external_policy_key)) > 0),
        document_version_snapshot INTEGER NULL CHECK (document_version_snapshot IS NULL OR document_version_snapshot >= 1),
        file_hash_sha256 CHAR(64) NULL CHECK (file_hash_sha256 IS NULL OR file_hash_sha256 ~ '^[0-9a-f]{64}$'),
        owner_scope TEXT NOT NULL CHECK (length(btrim(owner_scope)) > 0),
        celery_task_id TEXT NULL,
        status TEXT NOT NULL DEFAULT 'queued'
            CHECK (status IN ('queued', 'processing', 'completed', 'failed', 'cancelled')),
        current_stage TEXT NULL,
        total_chunks INTEGER NOT NULL DEFAULT 0 CHECK (total_chunks >= 0),
        processed_chunks INTEGER NOT NULL DEFAULT 0 CHECK (processed_chunks >= 0),
        error_code TEXT NULL,
        error_message TEXT NULL,
        attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
        started_at TIMESTAMPTZ NULL,
        completed_at TIMESTAMPTZ NULL,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT uq_ingestion_jobs_celery_task_id UNIQUE (celery_task_id)
    );
  `);

  // Dense semantic search (pgvector HNSW cosine distance) & sparse lexical search (FTS GIN)
  pgm.sql(`
    CREATE INDEX idx_chunk_embeddings_hnsw_cosine
        ON chunk_embeddings USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);

    CREATE INDEX idx_policy_chunks_search_tsv_gin
        ON policy_chunks USING gin (search_tsv);
  `);

  // Composite filtering indexes
  pgm.sql(`
    CREATE INDEX idx_policies_owner_scope_status
        ON policies (owner_scope, status, visibility);

    CREATE INDEX idx_policies_active_version
        ON policies (active_version_id)
        WHERE active_version_id IS NOT NULL;

    CREATE INDEX idx_policy_documents_policy_status
        ON policy_documents (policy_id, is_archived, ingestion_status, version_number DESC);

    CREATE INDEX idx_policy_documents_hash
        ON policy_documents (policy_id, file_hash_sha256);

    CREATE INDEX idx_policy_chunks_policy_doc_cat_page
        ON policy_chunks (policy_id, document_id, category, page_number);

    CREATE INDEX idx_policy_chunks_policy_doc_order
        ON policy_chunks (policy_id, document_id, chunk_index);

    CREATE INDEX idx_chunk_embeddings_policy_doc_model
        ON chunk_embeddings (policy_id, document_id, model_name, model_revision);

    CREATE INDEX idx_ingestion_jobs_policy_created
        ON ingestion_jobs (policy_id, created_at DESC);

    CREATE INDEX idx_ingestion_jobs_doc_created
        ON ingestion_jobs (document_id, created_at DESC);

    CREATE INDEX idx_ingestion_jobs_scope_status
        ON ingestion_jobs (owner_scope, status, created_at DESC);
  `);

  // Automatic updated_at triggers
  pgm.sql(`
    CREATE TRIGGER trg_policies_updated_at
        BEFORE UPDATE ON policies
        FOR EACH ROW EXECUTE FUNCTION set_row_updated_at();

    CREATE TRIGGER trg_policy_documents_updated_at
        BEFORE UPDATE ON policy_documents
        FOR EACH ROW EXECUTE FUNCTION set_row_updated_at();

    CREATE TRIGGER trg_policy_chunks_updated_at
        BEFORE UPDATE ON policy_chunks
        FOR EACH ROW EXECUTE FUNCTION set_row_updated_at();

    CREATE TRIGGER trg_chunk_embeddings_updated_at
        BEFORE UPDATE ON chunk_embeddings
        FOR EACH ROW EXECUTE FUNCTION set_row_updated_at();

    CREATE TRIGGER trg_ingestion_jobs_updated_at
        BEFORE UPDATE ON ingestion_jobs
        FOR EACH ROW EXECUTE FUNCTION set_row_updated_at();
  `);

  // Least-privilege grants when dedicated runtime role exists
  pgm.sql(`
    DO $$
    BEGIN
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'coverageiq_runtime') THEN
            REVOKE CREATE ON SCHEMA public FROM PUBLIC;
            REVOKE CREATE ON SCHEMA public FROM coverageiq_runtime;
            GRANT USAGE ON SCHEMA public TO coverageiq_runtime;
            GRANT SELECT, INSERT, UPDATE, DELETE
                ON TABLE policies, policy_documents, policy_chunks, chunk_embeddings, ingestion_jobs
                TO coverageiq_runtime;
        END IF;
    END
    $$;
  `);
};

exports.down = (pgm) => {
  pgm.sql(`
    ALTER TABLE policies DROP CONSTRAINT IF EXISTS fk_policies_active_version_same_policy;

    DROP TRIGGER IF EXISTS trg_ingestion_jobs_updated_at ON ingestion_jobs;
    DROP TRIGGER IF EXISTS trg_chunk_embeddings_updated_at ON chunk_embeddings;
    DROP TRIGGER IF EXISTS trg_policy_chunks_updated_at ON policy_chunks;
    DROP TRIGGER IF EXISTS trg_policy_documents_updated_at ON policy_documents;
    DROP TRIGGER IF EXISTS trg_policies_updated_at ON policies;

    DROP TABLE IF EXISTS ingestion_jobs CASCADE;
    DROP TABLE IF EXISTS chunk_embeddings CASCADE;
    DROP TABLE IF EXISTS policy_chunks CASCADE;
    DROP TABLE IF EXISTS policy_documents CASCADE;
    DROP TABLE IF EXISTS policies CASCADE;

    DROP FUNCTION IF EXISTS set_row_updated_at();
  `);
};
