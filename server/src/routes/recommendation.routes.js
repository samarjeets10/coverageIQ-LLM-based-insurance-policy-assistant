const express = require('express');
const { verifyJWT } = require('../middlewares/auth.middleware');
const { authorizePermissions } = require('../middlewares/rbac.middleware');
const { PERMISSIONS } = require('../constants/roles');
const { handlePolicyRecommendations } = require('../controllers/recommendation.controller');

const router = express.Router();

router.post(
    '/recommendations',
    verifyJWT,
    authorizePermissions(PERMISSIONS.CHAT_QUERY),
    handlePolicyRecommendations
);

module.exports = router;
