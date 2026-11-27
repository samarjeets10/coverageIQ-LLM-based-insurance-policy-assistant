const crypto = require('node:crypto');
const bcrypt = require('bcrypt');
const { query } = require('../config/db');
const { PLATFORM_ROLES } = require('../constants/roles');

const SALT_ROUNDS = 12;

const PUBLIC_USER_COLUMNS = `
    id,
    email,
    full_name,
    platform_role,
    is_active,
    email_verified_at,
    created_at,
    updated_at
`;

function hashRefreshToken(token) {
    return crypto.createHash('sha256').update(token).digest('hex');
}

async function createUser({
    fullName,
    email,
    password,
    platformRole = PLATFORM_ROLES.USER,
}) {
    const passwordHash = await bcrypt.hash(password, SALT_ROUNDS);

    const result = await query(
        `
        INSERT INTO users (full_name, email, password_hash, platform_role)
        VALUES ($1, $2, $3, $4)
        RETURNING ${PUBLIC_USER_COLUMNS}
        `,
        [fullName.trim(), email.trim().toLowerCase(), passwordHash, platformRole]
    );

    return result.rows[0];
}

async function findUserByEmail(email) {
    const result = await query(
        `
        SELECT
            ${PUBLIC_USER_COLUMNS},
            password_hash
        FROM users
        WHERE LOWER(email) = LOWER($1)
        `,
        [email.trim()]
    );

    return result.rows[0] || null;
}

async function findUserById(id) {
    const result = await query(
        `
        SELECT ${PUBLIC_USER_COLUMNS}
        FROM users
        WHERE id = $1
        `,
        [id]
    );

    return result.rows[0] || null;
}

async function verifyPassword(plainPassword, passwordHash) {
    return bcrypt.compare(plainPassword, passwordHash);
}

async function saveRefreshToken({ userId, token, expiresAt }) {
    const tokenHash = hashRefreshToken(token);

    const result = await query(
        `
        INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
        VALUES ($1, $2, $3)
        RETURNING id, user_id, expires_at, created_at
        `,
        [userId, tokenHash, expiresAt]
    );

    return result.rows[0];
}

async function findValidRefreshToken(token) {
    const tokenHash = hashRefreshToken(token);

    const result = await query(
        `
        SELECT id, user_id, expires_at, revoked_at, created_at
        FROM refresh_tokens
        WHERE token_hash = $1
          AND revoked_at IS NULL
          AND expires_at > NOW()
        `,
        [tokenHash]
    );

    return result.rows[0] || null;
}

async function revokeRefreshToken(token, { replacedBy = null } = {}) {
    const tokenHash = hashRefreshToken(token);

    const result = await query(
        `
        UPDATE refresh_tokens
        SET revoked_at = NOW(),
            replaced_by = COALESCE($2, replaced_by)
        WHERE token_hash = $1
          AND revoked_at IS NULL
        RETURNING id
        `,
        [tokenHash, replacedBy]
    );

    return result.rows[0] || null;
}

async function revokeAllUserRefreshTokens(userId) {
    await query(
        `
        UPDATE refresh_tokens
        SET revoked_at = NOW()
        WHERE user_id = $1
          AND revoked_at IS NULL
        `,
        [userId]
    );
}

async function updateUserRole(userId, platformRole) {
    const result = await query(
        `
        UPDATE users
        SET platform_role = $2,
            updated_at = NOW()
        WHERE id = $1
        RETURNING ${PUBLIC_USER_COLUMNS}
        `,
        [userId, platformRole]
    );

    return result.rows[0] || null;
}

module.exports = {
    createUser,
    findUserByEmail,
    findUserById,
    verifyPassword,
    saveRefreshToken,
    findValidRefreshToken,
    revokeRefreshToken,
    revokeAllUserRefreshTokens,
    updateUserRole,
};
