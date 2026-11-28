require('dotenv').config();

const fs = require('node:fs/promises');
const path = require('node:path');
const app = require('./src/app');
const {
    pool,
    checkDatabaseConnection,
} = require('./src/config/db');
const {
    checkRedisConnection,
    closeRedisClients,
} = require('./src/config/redis');
const { startIngestionReconciler } = require('./src/services/ingestionGateway.service');

const PORT = Number(process.env.SERVER_PORT || 8000);

function validateOperationalConfig() {
    const isProd = process.env.NODE_ENV === 'production';
    const uploadRoot = path.resolve(String(process.env.POLICY_UPLOAD_ROOT || './uploads/policies').trim());

    if (isProd) {
        const requiredVars = [
            'ACCESS_TOKEN_SECRET',
            'REFRESH_TOKEN_SECRET',
            'INTERNAL_SERVICE_TOKEN',
            'CORS_ORIGIN',
            'DATABASE_URL',
            'REDIS_URL',
            'POLICY_UPLOAD_ROOT',
        ];
        const missing = requiredVars.filter((k) => !process.env[k] || String(process.env[k]).trim() === '');
        if (missing.length > 0) {
            throw new Error(`Missing required production environment variable(s): ${missing.join(', ')}`);
        }
        if (process.env.ACCESS_TOKEN_SECRET === process.env.REFRESH_TOKEN_SECRET) {
            throw new Error('ACCESS_TOKEN_SECRET and REFRESH_TOKEN_SECRET must be distinct in production.');
        }
    }

    return { uploadRoot };
}

async function startServer() {
    let reconciler = null;
    try {
        const { uploadRoot } = validateOperationalConfig();
        await fs.mkdir(uploadRoot, { recursive: true });
        await fs.access(uploadRoot, fs.constants.R_OK | fs.constants.W_OK);

        const [database, redisStatus] = await Promise.all([
            checkDatabaseConnection(),
            checkRedisConnection(),
        ]);

        console.log(
            `PostgreSQL connected: ${database.database}; Redis connected (${redisStatus.latencyMs}ms)`
        );

        reconciler = startIngestionReconciler();

        const server = app.listen(PORT, () => {
            console.log(`Server is running on port ${PORT}.`);
        });

        const cleanupResources = async () => {
            if (reconciler) {
                reconciler.stop();
                reconciler = null;
            }
            await Promise.allSettled([
                pool.end(),
                closeRedisClients(),
            ]);
        };

        server.on('error', async (error) => {
            console.error('HTTP server error:', error.message);
            await cleanupResources();
            process.exitCode = 1;
        });

        const shutdown = (signal) => {
            console.log(`${signal} received. Shutting down...`);
            if (reconciler) {
                reconciler.stop();
                reconciler = null;
            }

            const forceTimer = setTimeout(() => {
                console.error('Graceful shutdown timed out; forcing exit.');
                process.exit(1);
            }, 10000);
            if (typeof forceTimer.unref === 'function') {
                forceTimer.unref();
            }

            server.close(async () => {
                try {
                    await cleanupResources();
                    clearTimeout(forceTimer);
                    process.exit(0);
                } catch (error) {
                    console.error(
                        'Resource shutdown error:',
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

        if (reconciler) {
            reconciler.stop();
        }
        await Promise.allSettled([
            pool.end(),
            closeRedisClients(),
        ]);
        process.exitCode = 1;
    }
}

startServer();
