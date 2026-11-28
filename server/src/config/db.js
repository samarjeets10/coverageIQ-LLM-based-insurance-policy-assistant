const { Pool } = require('pg');

function buildPoolConfig() {
    const ssl = process.env.DB_SSL === 'true' ? { rejectUnauthorized: true } : false;
    const base = {
        ssl,
        connectionTimeoutMillis: Number(process.env.DB_CONNECTION_TIMEOUT_MS || 5000),
        idleTimeoutMillis: Number(process.env.DB_IDLE_TIMEOUT_MS || 30000),
        max: Number(process.env.DB_POOL_MAX || 10),
    };

    if (process.env.DATABASE_URL && process.env.DATABASE_URL.trim() !== '') {
        return {
            ...base,
            connectionString: process.env.DATABASE_URL.trim(),
        };
    }

    return {
        ...base,
        host: process.env.DB_HOST || 'localhost',
        port: Number(process.env.DB_PORT || 5432),
        database: process.env.DB_NAME || 'coverageiq_db',
        user: process.env.DB_USER || 'coverageiq_runtime',
        password: process.env.DB_PASSWORD,
    };
}

const pool = new Pool(buildPoolConfig());

pool.on('error', (error) => {
    console.error('Unexpected idle PostgreSQL client error:', error.message);
});

async function query(text, params = []) {
    return pool.query(text, params);
}

async function withTransaction(callback) {
    const client = await pool.connect();
    try {
        await client.query('BEGIN');
        const result = await callback(client);
        await client.query('COMMIT');
        return result;
    } catch (error) {
        try {
            await client.query('ROLLBACK');
        } catch (_rollbackError) {
            // Preserve original transaction failure
        }
        throw error;
    } finally {
        client.release();
    }
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

module.exports = {
    pool,
    query,
    withTransaction,
    checkDatabaseConnection,
};
