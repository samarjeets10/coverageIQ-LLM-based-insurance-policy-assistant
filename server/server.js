
require('dotenv').config();

const app = require('./src/app');
const {
    pool,
    checkDatabaseConnection,
} = require('./src/config/db');

const PORT = Number(process.env.SERVER_PORT || 8000);

async function startServer() {
    try {
        const database = await checkDatabaseConnection();

        console.log(
            `PostgreSQL connected: ${database.database}`
        );

        const server = app.listen(PORT, () => {
            console.log(`Server is running on port ${PORT}.`);
        });

        server.on('error', async (error) => {
            console.error('HTTP server error:', error.message);
            await pool.end();
            process.exitCode = 1;
        });

        const shutdown = (signal) => {
            console.log(`${signal} received. Shutting down...`);

            server.close(async () => {
                try {
                    await pool.end();
                    process.exit(0);
                } catch (error) {
                    console.error(
                        'Database shutdown error:',
                        error.message
                    );
                    process.exit(1);
                }
            });
        };

        process.once('SIGINT', () => shutdown('SIGINT'));
        process.once('SIGTERM', () => shutdown('SIGTERM'));
    } catch (error) {
        console.error(
            'Server startup failed:',
            error.message
        );

        await pool.end();
        process.exitCode = 1;
    }
}

startServer();
