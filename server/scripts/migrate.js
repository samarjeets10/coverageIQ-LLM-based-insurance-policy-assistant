
require('dotenv').config();

const fs = require('node:fs/promises');
const path = require('node:path');
const { pool } = require('../src/config/db');

async function migrate() {
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

        const migrationsDir = path.join(__dirname, '..', 'migrations');
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
        await pool.end();
    }
}

migrate().catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
});
