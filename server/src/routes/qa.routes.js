const express = require('express');
const { verifyJWT } = require('../middlewares/auth.middleware');
const { authorizePermissions } = require('../middlewares/rbac.middleware');
const { PERMISSIONS } = require('../constants/roles');
const {
    handleAskPolicyQuestion,
    handleListPolicyQaHistory,
} = require('../controllers/qa.controller');

const router = express.Router();

router.post(
    '/policies/:policyId/ask',
    verifyJWT,
    authorizePermissions(PERMISSIONS.CHAT_QUERY),
    handleAskPolicyQuestion
);

router.get(
    '/policies/:policyId/qa-history',
    verifyJWT,
    authorizePermissions(PERMISSIONS.CHAT_QUERY),
    handleListPolicyQaHistory
);

module.exports = router;
