const express = require('express');
const { checkDatabaseConnection } = require('../config/db');
const { checkRedisConnection } = require('../config/redis');

const router = express.Router();

router.get('/', async (_req, res) => {
    const [dbResult, redisResult] = await Promise.allSettled([
        checkDatabaseConnection(),
        checkRedisConnection(),
    ]);

    const dbOk = dbResult.status === 'fulfilled';
    const redisOk = redisResult.status === 'fulfilled';

    if (!dbOk) {
        console.error('Health check PostgreSQL failure:', dbResult.reason?.message || 'error');
    }
    if (!redisOk) {
        console.error('Health check Redis failure:', redisResult.reason?.message || 'error');
    }

    if (dbOk && redisOk) {
        return res.status(200).json({
            success: true,
            status: 'healthy',
            services: {
                api: 'up',
                database: 'up',
                redis: 'up',
            },
            database: dbResult.value.database,
            redis: {
                latencyMs: redisResult.value.latencyMs,
            },
        });
    }

    return res.status(503).json({
        success: false,
        status: 'unhealthy',
        services: {
            api: 'up',
            database: dbOk ? 'up' : 'down',
            redis: redisOk ? 'up' : 'down',
        },
    });
});

module.exports = router;
