const PLATFORM_ROLES = Object.freeze({
    ADMIN: 'admin',
    USER: 'user',
    ORGANISATION: 'organisation',
});

const VALID_PLATFORM_ROLES = Object.freeze(Object.values(PLATFORM_ROLES));

const PERMISSIONS = Object.freeze({
    USER_READ_SELF: 'user:read:self',
    USER_UPDATE_SELF: 'user:update:self',
    USER_MANAGE_ALL: 'user:manage:all',
    ORG_CREATE: 'org:create',
    ORG_MANAGE: 'org:manage',
    POLICY_CREATE: 'policy:create',
    POLICY_READ: 'policy:read',
    POLICY_UPDATE: 'policy:update',
    POLICY_DELETE: 'policy:delete',
    CHAT_QUERY: 'chat:query',
    AUDIT_READ: 'audit:read',
});

const ROLE_PERMISSIONS = Object.freeze({
    [PLATFORM_ROLES.ADMIN]: Object.freeze(Object.values(PERMISSIONS)),
    [PLATFORM_ROLES.ORGANISATION]: Object.freeze([
        PERMISSIONS.USER_READ_SELF,
        PERMISSIONS.USER_UPDATE_SELF,
        PERMISSIONS.ORG_CREATE,
        PERMISSIONS.ORG_MANAGE,
        PERMISSIONS.POLICY_CREATE,
        PERMISSIONS.POLICY_READ,
        PERMISSIONS.POLICY_UPDATE,
        PERMISSIONS.POLICY_DELETE,
        PERMISSIONS.CHAT_QUERY,
    ]),
    [PLATFORM_ROLES.USER]: Object.freeze([
        PERMISSIONS.USER_READ_SELF,
        PERMISSIONS.USER_UPDATE_SELF,
        PERMISSIONS.POLICY_CREATE,
        PERMISSIONS.POLICY_READ,
        PERMISSIONS.POLICY_UPDATE,
        PERMISSIONS.POLICY_DELETE,
        PERMISSIONS.CHAT_QUERY,
    ]),
});

module.exports = {
    PLATFORM_ROLES,
    VALID_PLATFORM_ROLES,
    PERMISSIONS,
    ROLE_PERMISSIONS,
};
