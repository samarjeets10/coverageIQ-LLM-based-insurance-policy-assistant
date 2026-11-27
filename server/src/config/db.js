require('dotenv').config();

const fs = require('node:fs/promises');
const path = require('node:path');
const { Pool } = require('pg');

const pool = new Pool({
    host: process.env.DB_HOST || 'localhost',
    port: Number(process.env.DB_PORT || 5432),
    database: process.env.DB_NAME || 'coverageiq_db',
    user: process.env.DB_USER || 'postgres',
    password: process.env.DB_PASSWORD,
    ssl: process.env.DB_SSL === 'true'
        ? { rejectUnauthorized: true }
        : false,
    connectionTimeoutMillis: 5000,
    idleTimeoutMillis: 30000,
    max: 10,
});

pool.on('error', (error) => {
    console.error('Unexpected idle PostgreSQL client error:', error.message);
});

async function query(text, params = []) {
    return pool.query(text, params);
}

async function checkDatabaseConnection() {
    const result = await pool.query(`
        SELECT
            current_database() AS database,
            current_user AS username,
            current_timestamp AS checked_at
    `);

    return result.rows[0];
}

async function runMigrations({ closePool = false } = {}) {
    const client = await pool.connect();

    try {
        await client.query('BEGIN');

        await client.query(`
            CREATE TABLE IF NOT EXISTS schema_migrations (
                filename TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        `);

        await client.query('COMMIT');

        const migrationsDir = path.resolve(__dirname, '../../migrations');
        const files = (await fs.readdir(migrationsDir))
            .filter((file) => /^\d+_[a-z0-9_-]+\.sql$/i.test(file))
            .sort();

        for (const filename of files) {
            await client.query('BEGIN');

            try {
                const existing = await client.query(
                    'SELECT 1 FROM schema_migrations WHERE filename = $1',
                    [filename]
                );

                if (existing.rowCount > 0) {
                    await client.query('COMMIT');
                    console.log(`Already applied: ${filename}`);
                    continue;
                }

                const sql = await fs.readFile(
                    path.join(migrationsDir, filename),
                    'utf8'
                );

                await client.query(sql);

                await client.query(
                    'INSERT INTO schema_migrations (filename) VALUES ($1)',
                    [filename]
                );

                await client.query('COMMIT');
                console.log(`Applied: ${filename}`);
            } catch (error) {
                await client.query('ROLLBACK');
                throw new Error(
                    `Migration ${filename} failed: ${error.message}`,
                    { cause: error }
                );
            }
        }
    } finally {
        client.release();
        if (closePool) {
            await pool.end();
        }
    }
}

if (require.main === module) {
    runMigrations({ closePool: true }).catch((error) => {
        console.error(error.message);
        process.exitCode = 1;
    });
}

module.exports = {
    pool,
    query,
    checkDatabaseConnection,
    runMigrations,
};
