const apiError = require('../utils/apiError');
const { ROLE_PERMISSIONS } = require('../constants/roles');

function authorizeRoles(...allowedRoles) {
    return (req, _res, next) => {
        if (!req.user) {
            return next(new apiError(401, 'Authentication required'));
        }

        if (!allowedRoles.includes(req.user.platform_role)) {
            return next(
                new apiError(
                    403,
                    'Forbidden: insufficient role privileges'
                )
            );
        }

        return next();
    };
}

function authorizePermissions(...requiredPermissions) {
    return (req, _res, next) => {
        if (!req.user) {
            return next(new apiError(401, 'Authentication required'));
        }

        const userPermissions = ROLE_PERMISSIONS[req.user.platform_role] || [];
        const hasAllPermissions = requiredPermissions.every((permission) =>
            userPermissions.includes(permission)
        );

        if (!hasAllPermissions) {
            return next(
                new apiError(
                    403,
                    'Forbidden: missing required permissions'
                )
            );
        }

        return next();
    };
}

module.exports = {
    authorizeRoles,
    authorizePermissions,
};
