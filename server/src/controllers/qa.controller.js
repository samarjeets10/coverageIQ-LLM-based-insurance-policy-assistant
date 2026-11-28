const crypto = require('node:crypto');
const apiResponse = require('../utils/apiResponse');
const asyncHandler = require('../utils/asyncHandler');
const {
    askPolicyQuestion,
    listPolicyQaHistory,
} = require('../services/qaGateway.service');

const handleAskPolicyQuestion = asyncHandler(async (req, res) => {
    const requestId = req.headers['x-request-id']
        ? String(req.headers['x-request-id']).slice(0, 128)
        : crypto.randomUUID();

    const data = await askPolicyQuestion({
        user: req.user,
        policyId: req.params.policyId,
        body: req.body || {},
        requestId,
        qaOptions: req.app.locals?.qaServiceOptions || {},
    });

    return res
        .status(200)
        .json(new apiResponse(200, data, 'Policy coverage answer verified successfully'));
});

const handleListPolicyQaHistory = asyncHandler(async (req, res) => {
    const data = await listPolicyQaHistory({
        user: req.user,
        policyId: req.params.policyId,
        queryParams: req.query || {},
    });

    return res
        .status(200)
        .json(new apiResponse(200, data, 'Policy Q&A history fetched successfully'));
});

module.exports = {
    handleAskPolicyQuestion,
    handleListPolicyQaHistory,
};
