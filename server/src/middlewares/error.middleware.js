const apiError = require('../utils/apiError');

function errorHandler(err, _req, res, _next) {
    if (err instanceof apiError) {
        return res.status(err.statusCode).json({
            success: false,
            statusCode: err.statusCode,
            message: err.message,
            errors: err.errors,
            data: null,
        });
    }

    // PostgreSQL unique constraint violation
    if (err && err.code === '23505') {
        return res.status(409).json({
            success: false,
            statusCode: 409,
            message: 'Resource already exists',
            errors: [],
            data: null,
        });
    }

    // PostgreSQL invalid UUID / input syntax
    if (err && err.code === '22P02') {
        return res.status(400).json({
            success: false,
            statusCode: 400,
            message: 'Invalid identifier format',
            errors: [],
            data: null,
        });
    }

    const statusCode = err.statusCode || 500;
    return res.status(statusCode).json({
        success: false,
        statusCode,
        message: err.message || 'Internal server error',
        errors: [],
        data: null,
    });
}

module.exports = errorHandler;
