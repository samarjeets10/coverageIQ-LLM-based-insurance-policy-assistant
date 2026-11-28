const express = require('express');
const cors = require('cors');
const cookieParser = require('cookie-parser');

const healthRoutes = require('./routes/health.routes');
const authRoutes = require('./routes/auth.routes');
const ingestionRoutes = require('./routes/ingestion.routes');
const qaRoutes = require('./routes/qa.routes');
const recommendationRoutes = require('./routes/recommendation.routes');
const errorHandler = require('./middlewares/error.middleware');

const app = express();

function resolveTrustProxySetting() {
    const raw = process.env.TRUST_PROXY;
    if (raw === undefined || raw === null || String(raw).trim() === '') {
        return false;
    }
    const clean = String(raw).trim();
    const lower = clean.toLowerCase();
    if (lower === 'false' || lower === '0') {
        return false;
    }
    if (lower === 'true') {
        throw new Error('TRUST_PROXY=true is insecure; specify a numeric hop count (e.g. 1) or explicit subnet/proxy list.');
    }
    if (/^\d+$/.test(clean)) {
        return Number(clean);
    }
    return clean.split(',').map((item) => item.trim()).filter(Boolean);
}

app.set('trust proxy', resolveTrustProxySetting());
app.disable('x-powered-by');

const DEV_DEFAULT_CORS_ORIGINS = [
    'http://localhost:5173',
    'http://127.0.0.1:5173',
    'http://localhost:8000',
    'http://127.0.0.1:8000',
];

function getAllowedCorsOrigins() {
    const raw = process.env.CORS_ORIGIN;
    if (typeof raw === 'string' && raw.trim() !== '') {
        return new Set(
            raw
                .split(',')
                .map((s) => s.trim().replace(/\/+$/, ''))
                .filter((s) => s && s !== '*')
        );
    }
    if (process.env.NODE_ENV === 'production') {
        return new Set();
    }
    return new Set(DEV_DEFAULT_CORS_ORIGINS);
}

app.use((_req, res, next) => {
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('X-Frame-Options', 'DENY');
    res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
    next();
});

app.use(cors({
    origin(origin, callback) {
        if (!origin) {
            return callback(null, true);
        }
        const normalized = String(origin).trim().replace(/\/+$/, '');
        const allowed = getAllowedCorsOrigins();
        return callback(null, allowed.has(normalized));
    },
    credentials: true,
    optionsSuccessStatus: 204,
}));

app.use(express.json({
    limit: '16kb',
}));

app.use(express.static('public'));

app.use(express.urlencoded({
    extended: true,
    limit: '16kb',
}));

app.use(cookieParser());

// Health routes
app.use('/api/health', healthRoutes);

// Teammate authentication & RBAC routes
app.use('/api/v1/users', authRoutes);

// Checkpoint 4.5 policy PDF upload & ingestion job routes (both /api and /api/v1 for frontend compatibility)
app.use('/api', ingestionRoutes);
app.use('/api/v1', ingestionRoutes);

// Checkpoint 4.9 evidence-grounded Q&A proxy & history routes
app.use('/api', qaRoutes);
app.use('/api/v1', qaRoutes);

// Checkpoint 5.0 evidence-grounded M5 policy recommendation routes
app.use('/api', recommendationRoutes);
app.use('/api/v1', recommendationRoutes);

// Global sanitized error handler
app.use(errorHandler);

module.exports = app;
