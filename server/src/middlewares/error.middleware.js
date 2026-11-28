const multer = require('multer');
const apiError = require('../utils/apiError');

function sanitizePublicMessage(message) {
    if (typeof message !== 'string' || !message.trim()) {
        return 'Internal server error';
    }
    return message
        .replace(/postgres(?:ql)?:\/\/[^\s'"]+/gi, 'postgresql://<redacted>')
        .replace(/Bearer\s+[A-Za-z0-9._-]+/gi, 'Bearer <redacted>')
        .slice(0, 300);
}

function errorHandler(err, _req, res, _next) {
    if (err instanceof apiError) {
        return res.status(err.statusCode).json({
            success: false,
            statusCode: err.statusCode,
            message: sanitizePublicMessage(err.message),
            errors: Array.isArray(err.errors) ? err.errors : [],
            data: null,
        });
    }

    if (err instanceof multer.MulterError) {
        if (err.code === 'LIMIT_FILE_SIZE') {
            return res.status(413).json({
                success: false,
                statusCode: 413,
                message: 'Uploaded PDF exceeds the maximum allowed size limit',
                errors: [{ code: 'UPLOAD_TOO_LARGE' }],
                data: null,
            });
        }
        return res.status(400).json({
            success: false,
            statusCode: 400,
            message: 'Invalid multipart upload request',
            errors: [{ code: err.code || 'UPLOAD_ERROR' }],
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

    const statusCode = Number(err?.statusCode) >= 400 && Number(err?.statusCode) < 600
        ? Number(err.statusCode)
        : 500;

    return res.status(statusCode).json({
        success: false,
        statusCode,
        message: statusCode >= 500 ? 'Internal server error' : sanitizePublicMessage(err?.message),
        errors: [],
        data: null,
    });
}

module.exports = errorHandler;
