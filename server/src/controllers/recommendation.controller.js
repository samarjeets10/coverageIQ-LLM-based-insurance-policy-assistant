const crypto = require('node:crypto');
const apiResponse = require('../utils/apiResponse');
const asyncHandler = require('../utils/asyncHandler');
const { requestPolicyRecommendations } = require('../services/recommendationGateway.service');

const handlePolicyRecommendations = asyncHandler(async (req, res) => {
    const requestId = req.headers['x-request-id']
        ? String(req.headers['x-request-id']).replace(/[\r\n]/g, '').slice(0, 128)
        : crypto.randomUUID();

    const data = await requestPolicyRecommendations({
        user: req.user,
        body: req.body || {},
        requestId,
        recOptions: req.app.locals?.recommendationServiceOptions || {},
    });

    return res
        .status(200)
        .json(new apiResponse(200, data, 'Policy recommendations evaluated successfully'));
});

module.exports = {
    handlePolicyRecommendations,
};
