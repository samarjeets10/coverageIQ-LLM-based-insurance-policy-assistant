
const express = require('express');
const { checkDatabaseConnection } = require('../config/db');

const router = express.Router();

router.get('/', async (req, res) => {
    try {
        const database = await checkDatabaseConnection();

        return res.status(200).json({
            success: true,
            status: 'healthy',
            services: {
                api: 'up',
                database: 'up',
            },
            database: database.database,
        });
    } catch (error) {
        console.error('Health check failed:', error.message);

        return res.status(503).json({
            success: false,
            status: 'unhealthy',
            services: {
                api: 'up',
                database: 'down',
            },
        });
    }
});

module.exports = router;
