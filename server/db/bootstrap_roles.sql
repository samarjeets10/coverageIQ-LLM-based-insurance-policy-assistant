-- Bootstrap script for CoverageIQ PostgreSQL 16 + pgvector role separation.
-- Run as PostgreSQL cluster superuser BEFORE executing node-pg-migrate.
-- Replace placeholder passwords in production and inject via environment secrets only.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'coverageiq_migrator') THEN
        CREATE ROLE coverageiq_migrator LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT
            PASSWORD 'change_me_migrator';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'coverageiq_runtime') THEN
        CREATE ROLE coverageiq_runtime LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT
            PASSWORD 'change_me_runtime';
    END IF;
END
$$;

-- Create target database owned by coverageiq_migrator if missing:
-- CREATE DATABASE coverageiq_db OWNER coverageiq_migrator;

-- Then inside coverageiq_db (connected as superuser once):
-- CREATE EXTENSION IF NOT EXISTS vector;
-- ALTER SCHEMA public OWNER TO coverageiq_migrator;
-- REVOKE CREATE ON SCHEMA public FROM PUBLIC;
-- GRANT USAGE ON SCHEMA public TO coverageiq_runtime;
