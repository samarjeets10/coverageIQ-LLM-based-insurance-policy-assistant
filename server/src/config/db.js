
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
    checkDatabaseConnection,
};
