const express = require('express');
const cors = require('cors');
const cookieParser = require('cookie-parser');

const healthRoutes = require('./routes/health.routes');
const authRoutes = require('./routes/auth.routes');
const errorHandler = require('./middlewares/error.middleware');

const allowedOrigins = (
    process.env.CORS_ORIGIN || 'http://localhost:5173,http://127.0.0.1:5173'
)
    .split(',')
    .map((origin) => origin.trim())
    .filter(Boolean);

const app = express();

app.use(cors({
    origin(origin, callback) {
        if (!origin || allowedOrigins.includes(origin)) {
            return callback(null, true);
        }
        return callback(new Error(`CORS blocked for origin: ${origin}`));
    },
    credentials: true,
}));

app.use(express.json({
    limit: "16kb"
}));

app.use(express.static("public"));

app.use(express.urlencoded({
    extended: true,
    limit: "16kb"
}));

app.use(cookieParser());

// Health routes
app.use('/api/health', healthRoutes);

// Authentication routes
app.use('/api/v1/users', authRoutes);

// Global error handler
app.use(errorHandler);

module.exports = app;