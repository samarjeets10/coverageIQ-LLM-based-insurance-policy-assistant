const crypto = require('node:crypto');
const apiError = require('../utils/apiError');
const { query } = require('../config/db');
const { PLATFORM_ROLES } = require('../constants/roles');
const { listUserOrganizationScopes } = require('../models/user.model');
const { sanitizePublicErrorMessage } = require('./ingestionGateway.service');

const ALLOWED_M5_POLICY_TYPES = new Set(['health', 'car', 'life', 'term']);
const ALLOWED_RANKING_MODELS = new Set(['auto', 'lightgbm', 'weighted']);
const ALLOWED_STATUSES = new Set(['RECOMMENDATIONS_READY', 'NO_MATCHING_POLICIES']);
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function getAiServiceBaseUrl(overrideUrl = null) {
    const raw = overrideUrl || process.env.AI_SERVICE_URL || 'http://127.0.0.1:8001';
    return String(raw).trim().replace(/\/+$/, '');
}

function mapPlatformRoleToAiRole(platformRole) {
    const clean = String(platformRole || '').trim().toLowerCase();
    if (clean === PLATFORM_ROLES.ORGANISATION || clean === 'organization') {
        return 'organization';
    }
    if (clean === PLATFORM_ROLES.ADMIN) {
        return 'admin';
    }
    return 'user';
}

function sanitizeRecommendationPayload(body = {}) {
    if (!body || typeof body !== 'object' || Array.isArray(body)) {
        throw new apiError(400, 'Recommendation request body must be a JSON object');
    }

    // Never trust client-supplied authorization/tenant scope
    const forbiddenTopKeys = [
        'scope',
        'owner_scope',
        'ownerScope',
        'user_id',
        'userId',
        'organization_id',
        'organizationId',
    ];
    for (const k of forbiddenTopKeys) {
        if (body[k] !== undefined) {
            throw new apiError(400, 'Client-supplied authorization scope is not allowed');
        }
    }

    const rawProfile = body.profile && typeof body.profile === 'object' && !Array.isArray(body.profile)
        ? { ...body.profile }
        : { ...body };
    delete rawProfile.top_n;
    delete rawProfile.ranking_model;

    for (const k of forbiddenTopKeys) {
        if (rawProfile[k] !== undefined) {
            throw new apiError(400, 'Client-supplied authorization scope is not allowed');
        }
    }

    const policyType = typeof rawProfile.policy_type === 'string'
        ? rawProfile.policy_type.trim().toLowerCase()
        : '';
    if (!ALLOWED_M5_POLICY_TYPES.has(policyType)) {
        throw new apiError(
            400,
            `policy_type is required and must be one of: ${Array.from(ALLOWED_M5_POLICY_TYPES).join(', ')}`
        );
    }

    const age = Number(rawProfile.age);
    if (!Number.isInteger(age) || age < 0 || age > 120) {
        throw new apiError(400, 'age is required and must be an integer between 0 and 120');
    }

    const budgetMax = Number(rawProfile.budget_max);
    if (!Number.isFinite(budgetMax) || budgetMax <= 0) {
        throw new apiError(400, 'budget_max is required and must be greater than 0');
    }

    const budgetMin = rawProfile.budget_min === undefined || rawProfile.budget_min === null || rawProfile.budget_min === ''
        ? 0
        : Number(rawProfile.budget_min);
    if (!Number.isFinite(budgetMin) || budgetMin < 0 || budgetMin > budgetMax) {
        throw new apiError(400, 'budget_min must be a finite non-negative number <= budget_max');
    }

    const minSumInsured = Number(rawProfile.min_sum_insured);
    if (!Number.isFinite(minSumInsured) || minSumInsured <= 0) {
        throw new apiError(400, 'min_sum_insured is required and must be greater than 0');
    }

    if (rawProfile.max_sum_insured !== undefined && rawProfile.max_sum_insured !== null && rawProfile.max_sum_insured !== '') {
        const maxSi = Number(rawProfile.max_sum_insured);
        if (!Number.isFinite(maxSi) || maxSi < minSumInsured) {
            throw new apiError(400, 'max_sum_insured must be >= min_sum_insured');
        }
    }

    const coverageMode = typeof rawProfile.coverage_mode === 'string' && rawProfile.coverage_mode.trim()
        ? rawProfile.coverage_mode.trim().toLowerCase()
        : 'individual';
    const familySize = rawProfile.family_size === undefined || rawProfile.family_size === null || rawProfile.family_size === ''
        ? 1
        : Number(rawProfile.family_size);
    if (!Number.isInteger(familySize) || familySize < 1 || familySize > 15) {
        throw new apiError(400, 'family_size must be an integer between 1 and 15');
    }
    if (coverageMode === 'individual' && familySize > 1) {
        throw new apiError(400, 'family_size > 1 requires coverage_mode "family_floater"');
    }

    const topN = body.top_n === undefined || body.top_n === null ? 5 : Number(body.top_n);
    if (!Number.isInteger(topN) || topN < 1 || topN > 20) {
        throw new apiError(400, 'top_n must be an integer between 1 and 20');
    }

    const rankingModel = body.ranking_model === undefined || body.ranking_model === null
        ? 'auto'
        : String(body.ranking_model).trim().toLowerCase();
    if (!ALLOWED_RANKING_MODELS.has(rankingModel)) {
        throw new apiError(
            400,
            `ranking_model must be one of: ${Array.from(ALLOWED_RANKING_MODELS).join(', ')}`
        );
    }

    return {
        profile: {
            ...rawProfile,
            policy_type: policyType,
            age,
            budget_min: budgetMin,
            budget_max: budgetMax,
            min_sum_insured: minSumInsured,
            coverage_mode: coverageMode,
            family_size: familySize,
        },
        top_n: topN,
        ranking_model: rankingModel,
    };
}

async function verifyCatalogAuthorizationAndCitations(payload) {
    if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
        throw new apiError(502, 'AI recommendation service returned an invalid payload');
    }
    if (!ALLOWED_STATUSES.has(payload.status)) {
        throw new apiError(502, 'AI recommendation service returned an unrecognized status');
    }
    if (!Array.isArray(payload.recommendations)) {
        throw new apiError(502, 'AI recommendation service returned invalid recommendations list');
    }
    if (!payload.filter_report || typeof payload.filter_report !== 'object') {
        throw new apiError(502, 'AI recommendation service returned missing filter_report');
    }

    const policyIds = payload.recommendations
        .map((rec) => String(rec?.policy_id || '').trim());

    for (const pid of policyIds) {
        if (!UUID_RE.test(pid)) {
            throw new apiError(
                502,
                'Recommendation verification failed: upstream AI service returned malformed policy identifier'
            );
        }
    }

    // Confirm at the Node gateway boundary that every returned policy is an authorized active catalog policy
    let verifiedCatalogMap = new Map();
    if (policyIds.length > 0) {
        const res = await query(
            `
            SELECT p.id, p.visibility, p.status, p.active_version_id, pd.version_number
            FROM policies p
            JOIN policy_documents pd
              ON pd.policy_id = p.id AND pd.id = p.active_version_id
            WHERE p.id = ANY($1::uuid[])
              AND p.visibility = 'catalog'
              AND p.status = 'active'
              AND pd.is_archived = FALSE
              AND pd.ingestion_status = 'completed'
            `,
            [policyIds]
        );
        verifiedCatalogMap = new Map(res.rows.map((r) => [String(r.id), r]));

        for (const rec of payload.recommendations) {
            const pid = String(rec?.policy_id || '').trim();
            if (!verifiedCatalogMap.has(pid)) {
                throw new apiError(
                    502,
                    'Recommendation verification failed: unindexed or non-catalog policy returned by upstream AI service'
                );
            }
            const verifiedRow = verifiedCatalogMap.get(pid);
            if (
                rec.document_id !== undefined
                && rec.document_id !== null
                && String(rec.document_id).trim() !== String(verifiedRow.active_version_id)
            ) {
                throw new apiError(
                    502,
                    'Recommendation verification failed: upstream returned non-active document version'
                );
            }
            const citList = Array.isArray(rec.citations) ? rec.citations : [];
            const chunkKeys = citList
                .map((c) => (typeof c?.chunk_id === 'string' ? c.chunk_id.trim() : ''))
                .filter(Boolean);
            if (chunkKeys.length > 0) {
                const chunkRes = await query(
                    `
                    SELECT chunk_key
                    FROM policy_chunks
                    WHERE policy_id = $1
                      AND document_id = $2
                      AND chunk_key = ANY($3::text[])
                    `,
                    [pid, verifiedRow.active_version_id, chunkKeys]
                );
                const validChunkSet = new Set(chunkRes.rows.map((r) => String(r.chunk_key)));
                for (const cit of citList) {
                    if (!validChunkSet.has(String(cit?.chunk_id || '').trim())) {
                        throw new apiError(
                            502,
                            'Recommendation citation provenance failed validation against policy chunks'
                        );
                    }
                }
            }
        }
    }

    return payload;
}

async function requestPolicyRecommendations({
    user,
    body = {},
    requestId = null,
    recOptions = {},
}) {
    const validatedBody = sanitizeRecommendationPayload(body);

    const { readScopes } = await listUserOrganizationScopes(user.id);
    const preferredScope = readScopes.find((s) => s.startsWith('org:')) || `user:${user.id}`;
    const trustedScope = {
        user_id: String(user.id),
        role: mapPlatformRoleToAiRole(user.platform_role),
        owner_scope: preferredScope,
        allow_catalog_read: true,
    };

    const baseUrl = getAiServiceBaseUrl(recOptions.aiServiceUrl);
    const serviceToken = recOptions.serviceToken !== undefined
        ? recOptions.serviceToken
        : process.env.INTERNAL_SERVICE_TOKEN || '';
    const fetchImpl = recOptions.fetchImpl || global.fetch;
    const timeoutMs = Number(
        recOptions.timeoutMs || process.env.RECOMMENDATION_HTTP_TIMEOUT_MS || 45000
    );

    if (!serviceToken) {
        throw new apiError(503, 'Internal AI service authentication is not configured');
    }

    const effectiveRequestId = requestId || crypto.randomUUID();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    let response;
    let payload = null;
    try {
        response = await fetchImpl(`${baseUrl}/ai/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Service-Token': serviceToken,
                'X-Request-Id': effectiveRequestId,
            },
            body: JSON.stringify({
                ...validatedBody,
                scope: trustedScope,
            }),
            signal: controller.signal,
        });
        payload = await response.json().catch(() => null);
    } catch (err) {
        if (err && err.name === 'AbortError') {
            throw new apiError(504, 'AI recommendation service timed out');
        }
        throw new apiError(503, 'AI recommendation service is temporarily unavailable');
    } finally {
        clearTimeout(timer);
    }

    if (!response.ok) {
        const rawDetail = typeof payload?.detail === 'string'
            ? payload.detail
            : Array.isArray(payload?.detail)
                ? payload.detail.map((d) => d?.msg || '').filter(Boolean).join('; ')
                : payload?.message;
        const safeDetail = sanitizePublicErrorMessage(rawDetail);

        if (response.status === 401 || response.status === 403) {
            throw new apiError(403, safeDetail || 'AI service rejected authorization credentials');
        }
        if (response.status === 400 || response.status === 422) {
            throw new apiError(400, safeDetail || 'Invalid recommendation request parameters');
        }
        if (response.status === 501) {
            throw new apiError(501, safeDetail || 'Requested ranking model is not implemented');
        }
        if (response.status === 503) {
            throw new apiError(503, safeDetail || 'Recommendation service is currently unavailable');
        }
        if (response.status === 504) {
            throw new apiError(504, safeDetail || 'Recommendation service timed out');
        }
        throw new apiError(502, safeDetail || 'Upstream AI service failed to compute recommendations');
    }

    return verifyCatalogAuthorizationAndCitations(payload);
}

module.exports = {
    sanitizeRecommendationPayload,
    verifyCatalogAuthorizationAndCitations,
    requestPolicyRecommendations,
};
