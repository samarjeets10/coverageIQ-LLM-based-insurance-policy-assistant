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
    findUserById,
    verifyPassword,
    saveRefreshToken,
    findValidRefreshToken,
    revokeRefreshToken,
    updateUserRole,
} = require('../models/user.model');

const COOKIE_OPTIONS = {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
};

function sanitizeUser(user) {
    if (!user) return null;
    const { password_hash: _passwordHash, ...safeUser } = user;
    return safeUser;
}

async function generateAccessAndRefreshToken(user) {
    const accessSecret = process.env.ACCESS_TOKEN_SECRET;
    const refreshSecret = process.env.REFRESH_TOKEN_SECRET;

    if (!accessSecret || !refreshSecret) {
        throw new apiError(
            500,
            'JWT secrets are not configured on the server'
        );
    }

    const accessToken = jwt.sign(
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

    const refreshToken = jwt.sign(
        {
            sub: user.id,
            jti: crypto.randomUUID(),
        },
        refreshSecret,
        {
            expiresIn: process.env.REFRESH_TOKEN_EXPIRY || '1d',
        }
    );

    const decodedRefresh = jwt.decode(refreshToken);
    const expiresAt = new Date(decodedRefresh.exp * 1000);

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

    if (!incomingRefreshToken) {
        throw new apiError(401, 'Refresh token is required');
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

    const storedToken = await findValidRefreshToken(incomingRefreshToken);
    if (!storedToken) {
        throw new apiError(401, 'Refresh token has been revoked or expired');
    }

    const user = await findUserById(decoded.sub);
    if (!user || !user.is_active) {
        throw new apiError(401, 'User account is unavailable or inactive');
    }

    const { accessToken, refreshToken, refreshTokenRecord } =
        await generateAccessAndRefreshToken(user);

    await revokeRefreshToken(incomingRefreshToken, {
        replacedBy: refreshTokenRecord.id,
    });

    return res
        .status(200)
        .cookie('accessToken', accessToken, COOKIE_OPTIONS)
        .cookie('refreshToken', refreshToken, COOKIE_OPTIONS)
        .json(
            new apiResponse(
                200,
                {
                    accessToken,
                    refreshToken,
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

    if (typeof role !== 'string' || !VALID_PLATFORM_ROLES.includes(role.trim().toLowerCase())) {
        throw new apiError(
            400,
            `Invalid role. Allowed roles: ${VALID_PLATFORM_ROLES.join(', ')}`
        );
    }

    const updatedUser = await updateUserRole(userId, role.trim().toLowerCase());
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
