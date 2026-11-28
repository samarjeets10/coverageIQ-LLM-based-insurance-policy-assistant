const express = require('express');
const { verifyJWT } = require('../middlewares/auth.middleware');
const { authorizePermissions } = require('../middlewares/rbac.middleware');
const { uploadRateLimiter } = require('../middlewares/rateLimit.middleware');
const { uploadSinglePolicyPdf } = require('../middlewares/upload.middleware');
const { PERMISSIONS } = require('../constants/roles');
const {
    handleListPolicies,
    handleCreatePolicy,
    handleUpdatePolicyVisibility,
    authorizePolicyUploadTarget,
    handleUploadPolicyDocument,
    handleGetIngestionJob,
    handleListIngestionJobs,
} = require('../controllers/ingestion.controller');

const router = express.Router();

// Authenticated policy listing & draft policy creation for UI policy selector
router.get(
    '/policies',
    verifyJWT,
    authorizePermissions(PERMISSIONS.POLICY_READ),
    handleListPolicies
);

router.post(
    '/policies',
    verifyJWT,
    authorizePermissions(PERMISSIONS.POLICY_CREATE),
    handleCreatePolicy
);

router.patch(
    '/policies/:policyId/visibility',
    verifyJWT,
    authorizePermissions(PERMISSIONS.POLICY_UPDATE),
    handleUpdatePolicyVisibility
);

// Authenticated PDF upload + versioned document + ingestion job dispatch
router.post(
    '/policies/:policyId/documents',
    verifyJWT,
    authorizePermissions(PERMISSIONS.POLICY_UPDATE),
    authorizePolicyUploadTarget,
    uploadRateLimiter,
    uploadSinglePolicyPdf,
    handleUploadPolicyDocument
);

// Authenticated ingestion job status & paginated job list
router.get(
    '/ingestion/jobs',
    verifyJWT,
    authorizePermissions(PERMISSIONS.POLICY_READ),
    handleListIngestionJobs
);

router.get(
    '/ingestion/jobs/:jobId',
    verifyJWT,
    authorizePermissions(PERMISSIONS.POLICY_READ),
    handleGetIngestionJob
);

module.exports = router;
