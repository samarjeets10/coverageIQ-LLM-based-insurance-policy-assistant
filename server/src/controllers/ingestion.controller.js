const crypto = require('node:crypto');
const apiResponse = require('../utils/apiResponse');
const asyncHandler = require('../utils/asyncHandler');
const { validateUploadedPdf } = require('../services/pdfValidation.service');
const {
    verifyPolicyWriteAccess,
    uploadPolicyDocument,
    getAuthorizedIngestionJob,
    listAuthorizedIngestionJobs,
    listAuthorizedPolicies,
    createAuthorizedPolicy,
    updateAuthorizedPolicyVisibility,
} = require('../services/ingestionGateway.service');

const handleListPolicies = asyncHandler(async (req, res) => {
    const data = await listAuthorizedPolicies({
        user: req.user,
        queryParams: req.query || {},
    });

    return res
        .status(200)
        .json(new apiResponse(200, data, 'Policies listed successfully'));
});

const handleCreatePolicy = asyncHandler(async (req, res) => {
    const policy = await createAuthorizedPolicy({
        user: req.user,
        body: req.body || {},
    });

    return res
        .status(201)
        .json(new apiResponse(201, { policy }, 'Policy created successfully'));
});

const handleUpdatePolicyVisibility = asyncHandler(async (req, res) => {
    const requestId = req.headers['x-request-id']
        ? String(req.headers['x-request-id']).slice(0, 128)
        : crypto.randomUUID();
    const policy = await updateAuthorizedPolicyVisibility({
        user: req.user,
        policyId: req.params.policyId,
        body: req.body || {},
        requestId,
        backfillOptions: req.app.locals?.recommendationOptions || {},
    });

    return res
        .status(200)
        .json(new apiResponse(200, { policy }, 'Policy visibility updated successfully'));
});

const authorizePolicyUploadTarget = asyncHandler(async (req, _res, next) => {
    const policy = await verifyPolicyWriteAccess({
        user: req.user,
        policyId: req.params.policyId,
    });
    req.authorizedPolicy = policy;
    next();
});

const handleUploadPolicyDocument = asyncHandler(async (req, res) => {
    const validatedPdf = validateUploadedPdf(req.file);
    const requestId = req.headers['x-request-id']
        ? String(req.headers['x-request-id']).slice(0, 128)
        : crypto.randomUUID();
    const versionLabel = req.body?.version_label || req.body?.versionLabel || null;

    const result = await uploadPolicyDocument({
        user: req.user,
        policyId: req.params.policyId,
        fileBuffer: req.file.buffer,
        validatedPdf,
        versionLabel,
        requestId,
        storageOptions: req.app.locals?.ingestionStorageOptions || {},
        enqueueOptions: req.app.locals?.ingestionEnqueueOptions || {},
    });

    const msg = result.duplicate
        ? 'Duplicate policy document detected; returning existing record'
        : result.httpStatus === 202
            ? 'Policy document saved; ingestion dispatch scheduled for recovery'
            : 'Policy document uploaded and ingestion job queued successfully';

    return res
        .status(result.httpStatus)
        .json(new apiResponse(result.httpStatus, result, msg));
});

const handleGetIngestionJob = asyncHandler(async (req, res) => {
    const job = await getAuthorizedIngestionJob({
        user: req.user,
        jobId: req.params.jobId,
    });

    return res
        .status(200)
        .json(new apiResponse(200, { job }, 'Ingestion job fetched successfully'));
});

const handleListIngestionJobs = asyncHandler(async (req, res) => {
    const data = await listAuthorizedIngestionJobs({
        user: req.user,
        queryParams: req.query || {},
    });

    return res
        .status(200)
        .json(new apiResponse(200, data, 'Ingestion jobs listed successfully'));
});

module.exports = {
    handleListPolicies,
    handleCreatePolicy,
    handleUpdatePolicyVisibility,
    authorizePolicyUploadTarget,
    handleUploadPolicyDocument,
    handleGetIngestionJob,
    handleListIngestionJobs,
};
