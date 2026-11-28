const express = require('express');
const {
    userRegister,
    userLogin,
    userLogout,
    refreshAccessToken,
    getCurrentUser,
    updateUserPlatformRole,
} = require('../controllers/user.controller');
const { verifyJWT } = require('../middlewares/auth.middleware');
const { authorizeRoles } = require('../middlewares/rbac.middleware');
const { PLATFORM_ROLES } = require('../constants/roles');

const router = express.Router();

// Public routes
router.post('/register', userRegister);
router.post('/login', userLogin);
router.post('/refresh-token', refreshAccessToken);

// Authenticated routes
router.post('/logout', verifyJWT, userLogout);
router.get('/me', verifyJWT, getCurrentUser);

// RBAC: Admin-only routes
router.patch(
    '/:userId/role',
    verifyJWT,
    authorizeRoles(PLATFORM_ROLES.ADMIN),
    updateUserPlatformRole
);

module.exports = router;
