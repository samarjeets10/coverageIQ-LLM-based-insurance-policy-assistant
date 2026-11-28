const crypto = require('node:crypto');
const jwt = require('jsonwebtoken');
const asyncHandler = require('../utils/asyncHandler');
const apiResponse = require('../utils/apiResponse');
const apiError = require('../utils/apiError');
const {
    PLATFORM_ROLES,
    VALID_PLATFORM_ROLES,
} = require('../constants/roles');
const {
    createUser,
    findUserByEmail,
    verifyPassword,
    saveRefreshToken,
    revokeRefreshToken,
    consumeAndRotateRefreshToken,
    updateUserRole,
} = require('../models/user.model');

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

const COOKIE_OPTIONS = {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    path: '/',
};

function signAccessToken(user) {
    const accessSecret = process.env.ACCESS_TOKEN_SECRET;
    if (!accessSecret) {
        throw new apiError(500, 'JWT secrets are not configured on the server');
    }
    return jwt.sign(
        {
            sub: user.id,
            email: user.email,
            role: user.platform_role,
        },
        accessSecret,
        {
            expiresIn: process.env.ACCESS_TOKEN_EXPIRY || '15m',
        }
    );
}

function signRefreshToken(userId) {
    const refreshSecret = process.env.REFRESH_TOKEN_SECRET;
    if (!refreshSecret) {
        throw new apiError(500, 'JWT secrets are not configured on the server');
    }
    const refreshToken = jwt.sign(
        {
            sub: userId,
            jti: crypto.randomUUID(),
        },
        refreshSecret,
        {
            expiresIn: process.env.REFRESH_TOKEN_EXPIRY || '1d',
        }
    );
    const decodedRefresh = jwt.decode(refreshToken);
    return {
        refreshToken,
        expiresAt: new Date(decodedRefresh.exp * 1000),
    };
}

function sanitizeUser(user) {
    if (!user) return null;
    const { password_hash: _passwordHash, ...safeUser } = user;
    return safeUser;
}

async function generateAccessAndRefreshToken(user) {
    const accessToken = signAccessToken(user);
    const { refreshToken, expiresAt } = signRefreshToken(user.id);

    const savedToken = await saveRefreshToken({
        userId: user.id,
        token: refreshToken,
        expiresAt,
    });

    return {
        accessToken,
        refreshToken,
        refreshTokenRecord: savedToken,
    };
}

const userRegister = asyncHandler(async (req, res) => {
    const {
        fullName,
        email,
        password,
        role,
    } = req.body || {};

    if (
        [fullName, email, password].some(
            (field) => typeof field !== 'string' || field.trim() === ''
        )
    ) {
        throw new apiError(400, 'fullName, email, and password are required');
    }

    if (password.length < 8) {
        throw new apiError(400, 'Password must be at least 8 characters long');
    }

    const requestedRole =
        typeof role === 'string' && role.trim() !== ''
            ? role.trim().toLowerCase()
            : PLATFORM_ROLES.USER;

    if (!VALID_PLATFORM_ROLES.includes(requestedRole)) {
        throw new apiError(
            400,
            `Invalid role. Allowed roles: ${VALID_PLATFORM_ROLES.join(', ')}`
        );
    }

    if (requestedRole === PLATFORM_ROLES.ADMIN) {
        throw new apiError(
            403,
            'Admin role cannot be self-assigned during registration'
        );
    }

    const existingUser = await findUserByEmail(email);
    if (existingUser) {
        throw new apiError(409, 'User with this email already exists');
    }

    const user = await createUser({
        fullName,
        email,
        password,
        platformRole: requestedRole,
    });

    const { accessToken, refreshToken } =
        await generateAccessAndRefreshToken(user);

    return res
        .status(201)
        .cookie('accessToken', accessToken, COOKIE_OPTIONS)
        .cookie('refreshToken', refreshToken, COOKIE_OPTIONS)
        .json(
            new apiResponse(
                201,
                {
                    user: sanitizeUser(user),
                    accessToken,
                    refreshToken,
                },
                'User registered successfully'
            )
        );
});

const userLogin = asyncHandler(async (req, res) => {
    const { email, password } = req.body || {};

    if (
        [email, password].some(
            (field) => typeof field !== 'string' || field.trim() === ''
        )
    ) {
        throw new apiError(400, 'Email and password are required');
    }

    const user = await findUserByEmail(email);
    if (!user || !user.is_active) {
        throw new apiError(401, 'Invalid email or password');
    }

    const isPasswordValid = await verifyPassword(password, user.password_hash);
    if (!isPasswordValid) {
        throw new apiError(401, 'Invalid email or password');
    }

    const { accessToken, refreshToken } =
        await generateAccessAndRefreshToken(user);

    return res
        .status(200)
        .cookie('accessToken', accessToken, COOKIE_OPTIONS)
        .cookie('refreshToken', refreshToken, COOKIE_OPTIONS)
        .json(
            new apiResponse(
                200,
                {
                    user: sanitizeUser(user),
                    accessToken,
                    refreshToken,
                },
                'User logged in successfully'
            )
        );
});

const userLogout = asyncHandler(async (req, res) => {
    const refreshToken = req.cookies?.refreshToken || req.body?.refreshToken;

    if (refreshToken) {
        await revokeRefreshToken(refreshToken);
    }

    return res
        .status(200)
        .clearCookie('accessToken', COOKIE_OPTIONS)
        .clearCookie('refreshToken', COOKIE_OPTIONS)
        .json(new apiResponse(200, {}, 'User logged out successfully'));
});

const refreshAccessToken = asyncHandler(async (req, res) => {
    const incomingRefreshToken =
        req.cookies?.refreshToken || req.body?.refreshToken;

    if (!incomingRefreshToken || typeof incomingRefreshToken !== 'string') {
        throw new apiError(401, 'Refresh token is required');
    }

    if (!process.env.ACCESS_TOKEN_SECRET || !process.env.REFRESH_TOKEN_SECRET) {
        throw new apiError(500, 'JWT secrets are not configured on the server');
    }

    let decoded;
    try {
        decoded = jwt.verify(
            incomingRefreshToken,
            process.env.REFRESH_TOKEN_SECRET
        );
    } catch (_error) {
        throw new apiError(401, 'Invalid or expired refresh token');
    }

    const { refreshToken: nextRefreshToken, expiresAt: nextExpiresAt } =
        signRefreshToken(decoded.sub);

    const rotation = await consumeAndRotateRefreshToken({
        incomingToken: incomingRefreshToken,
        expectedUserId: decoded.sub,
        newToken: nextRefreshToken,
        newExpiresAt: nextExpiresAt,
    });

    if (rotation.status === 'user_inactive') {
        throw new apiError(401, 'User account is unavailable or inactive');
    }

    if (rotation.status !== 'rotated') {
        res.clearCookie('accessToken', COOKIE_OPTIONS);
        res.clearCookie('refreshToken', COOKIE_OPTIONS);
        throw new apiError(401, 'Refresh token has been revoked or expired');
    }

    const accessToken = signAccessToken(rotation.user);

    return res
        .status(200)
        .cookie('accessToken', accessToken, COOKIE_OPTIONS)
        .cookie('refreshToken', nextRefreshToken, COOKIE_OPTIONS)
        .json(
            new apiResponse(
                200,
                {
                    accessToken,
                    refreshToken: nextRefreshToken,
                },
                'Access token refreshed successfully'
            )
        );
});

const getCurrentUser = asyncHandler(async (req, res) => {
    return res
        .status(200)
        .json(
            new apiResponse(
                200,
                { user: sanitizeUser(req.user) },
                'Current user fetched successfully'
            )
        );
});

const updateUserPlatformRole = asyncHandler(async (req, res) => {
    const { userId } = req.params;
    const { role } = req.body || {};

    if (typeof userId !== 'string' || !UUID_RE.test(userId.trim())) {
        throw new apiError(400, 'Invalid userId: must be a valid UUID');
    }

    if (typeof role !== 'string' || !VALID_PLATFORM_ROLES.includes(role.trim().toLowerCase())) {
        throw new apiError(
            400,
            `Invalid role. Allowed roles: ${VALID_PLATFORM_ROLES.join(', ')}`
        );
    }

    const updatedUser = await updateUserRole(userId.trim().toLowerCase(), role.trim().toLowerCase());
    if (!updatedUser) {
        throw new apiError(404, 'User not found');
    }

    return res
        .status(200)
        .json(
            new apiResponse(
                200,
                { user: sanitizeUser(updatedUser) },
                'User role updated successfully'
            )
        );
});

module.exports = {
    userRegister,
    userLogin,
    userLogout,
    refreshAccessToken,
    getCurrentUser,
    updateUserPlatformRole,
};
