const jwt = require('jsonwebtoken');
const apiError = require('../utils/apiError');
const asyncHandler = require('../utils/asyncHandler');
const { findUserById } = require('../models/user.model');

const verifyJWT = asyncHandler(async (req, _res, next) => {
    const authHeader = req.headers.authorization;
    const bearerToken =
        typeof authHeader === 'string' && authHeader.startsWith('Bearer ')
            ? authHeader.slice(7).trim()
            : null;

    const token = req.cookies?.accessToken || bearerToken;

    if (!token) {
        throw new apiError(401, 'Authentication required');
    }

    let decoded;
    try {
        decoded = jwt.verify(token, process.env.ACCESS_TOKEN_SECRET);
    } catch (_error) {
        throw new apiError(401, 'Invalid or expired access token');
    }

    const user = await findUserById(decoded.sub);

    if (!user || !user.is_active) {
        throw new apiError(401, 'User account is unavailable or inactive');
    }

    req.user = user;
    next();
});

module.exports = {
    verifyJWT,
};
