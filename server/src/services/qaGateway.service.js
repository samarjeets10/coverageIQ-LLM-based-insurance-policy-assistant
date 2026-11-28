const crypto = require('node:crypto');
const apiError = require('../utils/apiError');
const { query, withTransaction } = require('../config/db');
const { PLATFORM_ROLES } = require('../constants/roles');
const { listUserOrganizationScopes } = require('../models/user.model');
const {
    assertValidUuid,
    sanitizePublicErrorMessage,
} = require('./ingestionGateway.service');

const ALLOWED_DECISIONS = new Set([
    'COVERED',
    'NOT COVERED',
    'CONDITIONAL',
    'INSUFFICIENT EVIDENCE',
]);
const ORG_SCOPE_PREFIX = 'org:';

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

function extractOrganizationIdFromScope(ownerScope) {
    if (typeof ownerScope !== 'string' || !ownerScope.startsWith(ORG_SCOPE_PREFIX)) {
        return null;
    }
    const candidate = ownerScope.slice(ORG_SCOPE_PREFIX.length).trim();
    try {
        return assertValidUuid(candidate, 'organization_id');
    } catch {
        return null;
    }
}

async function authorizePolicyForQa({ user, policyId }) {
    const cleanPolicyId = assertValidUuid(policyId, 'policyId');
    const polRes = await query(
        `
        SELECT p.id, p.external_policy_key, p.title, p.policy_type, p.visibility,
               p.owner_id, p.owner_scope, p.status, p.active_version_id
        FROM policies p
        WHERE p.id = $1
        `,
        [cleanPolicyId]
    );

    if (polRes.rowCount === 0) {
        throw new apiError(404, 'Policy not found');
    }

    const policy = polRes.rows[0];
    if (policy.status === 'archived') {
        throw new apiError(409, 'Archived policies cannot be queried');
    }

    const isAdmin = user.platform_role === PLATFORM_ROLES.ADMIN;
    const { readScopes } = await listUserOrganizationScopes(user.id);
    const ownsPolicyScope = readScopes.includes(policy.owner_scope);
    const isActiveCatalog = policy.visibility === 'catalog' && policy.status === 'active';

    if (!isAdmin && !ownsPolicyScope && !isActiveCatalog) {
        throw new apiError(403, 'Forbidden: you do not have permission to query this policy');
    }

    // Determine authoritative tenant owner_scope forwarded to FastAPI
    const trustedOwnerScope = ownsPolicyScope
        ? policy.owner_scope
        : isAdmin
            ? policy.owner_scope
            : `user:${user.id}`;

    return {
        policy,
        trustedScope: {
            user_id: String(user.id),
            role: mapPlatformRoleToAiRole(user.platform_role),
            owner_scope: trustedOwnerScope,
            allow_catalog_read: isActiveCatalog,
        },
        organizationId: extractOrganizationIdFromScope(policy.owner_scope),
    };
}

async function verifyActiveDocumentReadyForQa(policy) {
    if (!policy.active_version_id) {
        throw new apiError(
            409,
            'This policy has no indexed document yet. Upload a PDF and wait for ingestion completion first.',
            [{ code: 'POLICY_NOT_INGESTED' }]
        );
    }

    const docRes = await query(
        `
        SELECT d.id, d.version_number, d.file_name, d.page_count, d.ingestion_status,
               d.is_archived,
               EXISTS (
                   SELECT 1
                   FROM policy_chunks pc
                   WHERE pc.document_id = d.id AND pc.policy_id = d.policy_id
               ) AS has_chunks
        FROM policy_documents d
        WHERE d.id = $1 AND d.policy_id = $2
        `,
        [policy.active_version_id, policy.id]
    );

    if (docRes.rowCount === 0 || docRes.rows[0].is_archived) {
        throw new apiError(
            409,
            'Active policy document version is unavailable for Q&A.',
            [{ code: 'ACTIVE_DOCUMENT_UNAVAILABLE' }]
        );
    }

    const doc = docRes.rows[0];
    if (doc.ingestion_status !== 'completed' || !doc.has_chunks) {
        throw new apiError(
            409,
            'Policy document ingestion has not completed yet; please wait for COMPLETED status.',
            [{ code: 'POLICY_INGESTION_INCOMPLETE' }]
        );
    }

    return doc;
}

function validateAndNormalizeAiAnswer(rawBody) {
    if (!rawBody || typeof rawBody !== 'object' || Array.isArray(rawBody)) {
        throw new apiError(502, 'AI service returned an invalid answer format');
    }

    const decision = typeof rawBody.decision === 'string' ? rawBody.decision.trim() : '';
    if (!ALLOWED_DECISIONS.has(decision)) {
        throw new apiError(502, 'AI service returned an unrecognized coverage decision');
    }

    const reason = typeof rawBody.reason === 'string' && rawBody.reason.trim()
        ? rawBody.reason.trim()
        : '';
    if (!reason) {
        throw new apiError(502, 'AI service returned an empty answer explanation');
    }

    if (rawBody.confidence_score !== undefined && rawBody.confidence_score !== null) {
        const score = Number(rawBody.confidence_score);
        if (!Number.isFinite(score) || score < 0 || score > 1) {
            throw new apiError(502, 'AI service returned an invalid confidence score');
        }
    }
    const confidenceScore = rawBody.confidence_score !== undefined && rawBody.confidence_score !== null
        ? Number(rawBody.confidence_score)
        : null;

    if (!Array.isArray(rawBody.conditions)) {
        throw new apiError(502, 'AI service returned invalid conditions payload');
    }
    const conditions = rawBody.conditions
        .filter((item) => typeof item === 'string' && item.trim() !== '')
        .map((item) => item.trim());

    if (!Array.isArray(rawBody.citations)) {
        throw new apiError(502, 'AI service returned invalid citations payload');
    }
    const citations = rawBody.citations.map((cit) => {
        if (!cit || typeof cit !== 'object') {
            throw new apiError(502, 'AI service returned malformed citation item');
        }
        const chunkId = typeof cit.chunk_id === 'string' ? cit.chunk_id.trim() : '';
        const clause = typeof cit.clause === 'string' ? cit.clause.trim() : '';
        const page = Number(cit.page);
        if (!chunkId || !clause || !Number.isInteger(page) || page < 1) {
            throw new apiError(502, 'AI service returned incomplete citation provenance');
        }
        return {
            chunk_id: chunkId,
            page,
            clause,
            category: typeof cit.category === 'string' ? cit.category : null,
            snippet: typeof cit.snippet === 'string' ? cit.snippet : null,
            page_end: Number.isInteger(Number(cit.page_end)) && Number(cit.page_end) >= page
                ? Number(cit.page_end)
                : null,
            page_range: typeof cit.page_range === 'string' && cit.page_range.trim()
                ? cit.page_range.trim()
                : String(page),
        };
    });

    const evaluatorStatus = typeof rawBody.evaluator_status === 'string' && rawBody.evaluator_status.trim()
        ? rawBody.evaluator_status.trim()
        : 'UNKNOWN';

    const retryCount = Number.isInteger(Number(rawBody.retry_count)) && Number(rawBody.retry_count) >= 0
        ? Number(rawBody.retry_count)
        : 0;

    const meta = rawBody.metadata && typeof rawBody.metadata === 'object' ? rawBody.metadata : {};
    if (
        process.env.NODE_ENV === 'production'
        && (meta.mock_mode === true || meta.storage_mode === 'mock_memory')
    ) {
        throw new apiError(503, 'Mock AI storage responses cannot be served in production');
    }

    const sanitizedProviderMeta = {
        storage_mode: typeof meta.storage_mode === 'string' ? meta.storage_mode : null,
        mock_mode: Boolean(meta.mock_mode),
        retrieved_chunks_count: Number.isFinite(Number(meta.retrieved_chunks_count))
            ? Number(meta.retrieved_chunks_count)
            : citations.length,
        verified_claims_count: Number.isFinite(Number(meta.verified_claims_count))
            ? Number(meta.verified_claims_count)
            : null,
        verifier_llm_skipped: Boolean(meta.verifier_llm_skipped),
        generation_llm: meta.generation_llm && typeof meta.generation_llm === 'object'
            ? {
                provider: typeof meta.generation_llm.provider === 'string'
                    ? meta.generation_llm.provider
                    : null,
                model: typeof meta.generation_llm.model === 'string'
                    ? meta.generation_llm.model
                    : null,
                finish_reason: typeof meta.generation_llm.finish_reason === 'string'
                    ? meta.generation_llm.finish_reason
                    : null,
                request_id: typeof meta.generation_llm.request_id === 'string'
                    ? meta.generation_llm.request_id
                    : null,
            }
            : null,
    };

    return {
        decision,
        confidence_score: confidenceScore,
        conditions,
        reason,
        citations,
        evaluator_status: evaluatorStatus,
        retry_count: retryCount,
        provider_metadata: sanitizedProviderMeta,
    };
}

function serializeChatMessage(row) {
    if (!row) return null;
    return {
        id: row.id,
        user_id: row.user_id,
        organization_id: row.organization_id || null,
        policy_id: row.policy_id,
        document_id: row.document_id || null,
        conversation_id: row.conversation_id,
        role: row.role,
        content: row.content,
        decision: row.decision || null,
        confidence_score: row.confidence_score !== null && row.confidence_score !== undefined
            ? Number(row.confidence_score)
            : null,
        conditions: Array.isArray(row.conditions) ? row.conditions : [],
        citations: Array.isArray(row.citations) ? row.citations : [],
        evaluator_status: row.evaluator_status || null,
        retry_count: row.retry_count !== null && row.retry_count !== undefined
            ? Number(row.retry_count)
            : null,
        provider_metadata: row.provider_metadata && typeof row.provider_metadata === 'object'
            ? row.provider_metadata
            : {},
        created_at: row.created_at,
    };
}

async function invokeFastApiAsk({ policyId, question, trustedScope, requestId }, options = {}) {
    const baseUrl = getAiServiceBaseUrl(options.aiServiceUrl);
    const serviceToken = options.serviceToken !== undefined
        ? options.serviceToken
        : process.env.INTERNAL_SERVICE_TOKEN || '';
    const fetchImpl = options.fetchImpl || global.fetch;
    const timeoutMs = Number(options.timeoutMs || process.env.QA_HTTP_TIMEOUT_MS || 90000);

    if (!serviceToken) {
        throw new apiError(503, 'Internal AI service authentication is not configured');
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    let response;
    let payload = null;
    try {
        response = await fetchImpl(`${baseUrl}/ai/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Service-Token': serviceToken,
                'X-Request-Id': requestId || crypto.randomUUID(),
            },
            body: JSON.stringify({
                policy_id: policyId,
                question,
                scope: trustedScope,
            }),
            signal: controller.signal,
        });
        payload = await response.json().catch(() => null);
    } catch (err) {
        if (err && err.name === 'AbortError') {
            throw new apiError(504, 'AI service request timed out while analyzing your policy');
        }
        throw new apiError(503, 'AI service is temporarily unavailable; please retry shortly');
    } finally {
        clearTimeout(timer);
    }

    if (!response.ok) {
        const safeDetail = sanitizePublicErrorMessage(payload?.detail || payload?.message);
        if (response.status === 403) {
            throw new apiError(403, safeDetail || 'Access denied for requested policy scope');
        }
        if (response.status === 429) {
            throw new apiError(429, safeDetail || 'AI provider rate limit exceeded; please retry shortly');
        }
        if (response.status === 504) {
            throw new apiError(504, safeDetail || 'AI provider timed out while processing the query');
        }
        if (response.status === 503) {
            throw new apiError(503, safeDetail || 'AI service is currently unavailable');
        }
        if (response.status === 422 || response.status === 400) {
            throw new apiError(400, safeDetail || 'Invalid policy question payload');
        }
        throw new apiError(502, safeDetail || 'Upstream AI service failed to answer the policy question');
    }

    return validateAndNormalizeAiAnswer(payload);
}

async function askPolicyQuestion({
    user,
    policyId,
    body = {},
    requestId = null,
    qaOptions = {},
}) {
    if (
        body.scope !== undefined
        || body.owner_scope !== undefined
        || body.ownerScope !== undefined
        || body.user_id !== undefined
    ) {
        throw new apiError(400, 'Client-supplied authorization scope is not allowed');
    }

    const rawQuestion = typeof body.question === 'string' ? body.question.trim() : '';
    if (rawQuestion.length < 3) {
        throw new apiError(400, 'Question is required and must be at least 3 characters long');
    }
    if (rawQuestion.length > 2000) {
        throw new apiError(400, 'Question must be at most 2000 characters long');
    }

    let conversationId;
    if (body.conversation_id !== undefined && body.conversation_id !== null && String(body.conversation_id).trim() !== '') {
        conversationId = assertValidUuid(String(body.conversation_id), 'conversation_id');
    } else if (body.conversationId !== undefined && body.conversationId !== null && String(body.conversationId).trim() !== '') {
        conversationId = assertValidUuid(String(body.conversationId), 'conversationId');
    } else {
        conversationId = crypto.randomUUID();
    }

    const { policy, trustedScope, organizationId } = await authorizePolicyForQa({
        user,
        policyId,
    });
    const activeDoc = await verifyActiveDocumentReadyForQa(policy);

    const validatedAnswer = await invokeFastApiAsk(
        {
            policyId: policy.id,
            question: rawQuestion,
            trustedScope,
            requestId: requestId || crypto.randomUUID(),
        },
        qaOptions
    );

    const { userMessageRow, assistantMessageRow } = await withTransaction(async (client) => {
        const userInsert = await client.query(
            `
            INSERT INTO chat_messages (
                user_id, organization_id, policy_id, document_id,
                owner_scope, conversation_id, role, content, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, 'user', $7, clock_timestamp())
            RETURNING *
            `,
            [
                user.id,
                organizationId,
                policy.id,
                activeDoc.id,
                trustedScope.owner_scope,
                conversationId,
                rawQuestion,
            ]
        );
        const userCreatedAt = userInsert.rows[0].created_at;

        const assistantInsert = await client.query(
            `
            INSERT INTO chat_messages (
                user_id, organization_id, policy_id, document_id,
                owner_scope, conversation_id, role, content,
                decision, confidence_score, conditions, citations,
                evaluator_status, retry_count, provider_metadata, created_at
            )
            VALUES (
                $1, $2, $3, $4,
                $5, $6, 'assistant', $7,
                $8, $9, $10::jsonb, $11::jsonb,
                $12, $13, $14::jsonb,
                GREATEST(clock_timestamp(), $15::timestamptz + interval '1 millisecond')
            )
            RETURNING *
            `,
            [
                user.id,
                organizationId,
                policy.id,
                activeDoc.id,
                trustedScope.owner_scope,
                conversationId,
                validatedAnswer.reason,
                validatedAnswer.decision,
                validatedAnswer.confidence_score,
                JSON.stringify(validatedAnswer.conditions),
                JSON.stringify(validatedAnswer.citations),
                validatedAnswer.evaluator_status,
                validatedAnswer.retry_count,
                JSON.stringify(validatedAnswer.provider_metadata),
                userCreatedAt,
            ]
        );

        return {
            userMessageRow: userInsert.rows[0],
            assistantMessageRow: assistantInsert.rows[0],
        };
    });

    const serializedUserMsg = serializeChatMessage(userMessageRow);
    const serializedAssistantMsg = serializeChatMessage(assistantMessageRow);

    return {
        conversation_id: conversationId,
        policy: {
            id: policy.id,
            external_policy_key: policy.external_policy_key,
            title: policy.title,
            policy_type: policy.policy_type,
            status: policy.status,
            active_version_id: activeDoc.id,
        },
        document: {
            id: activeDoc.id,
            version_number: activeDoc.version_number,
            file_name: activeDoc.file_name,
        },
        user_message: serializedUserMsg,
        assistant_message: serializedAssistantMsg,
        answer: {
            query: rawQuestion,
            decision: validatedAnswer.decision,
            confidence_score: validatedAnswer.confidence_score,
            conditions: validatedAnswer.conditions,
            reason: validatedAnswer.reason,
            citations: validatedAnswer.citations,
            evaluator_status: validatedAnswer.evaluator_status,
            retry_count: validatedAnswer.retry_count,
            metadata: validatedAnswer.provider_metadata,
        },
    };
}

async function listPolicyQaHistory({ user, policyId, queryParams = {} }) {
    if (queryParams.user_id !== undefined || queryParams.userId !== undefined) {
        throw new apiError(400, 'Clients cannot supply user_id filters');
    }

    const { policy } = await authorizePolicyForQa({ user, policyId });

    const rawPage = queryParams.page === undefined ? 1 : Number(queryParams.page);
    const rawLimit = queryParams.limit === undefined ? 40 : Number(queryParams.limit);
    if (!Number.isInteger(rawPage) || rawPage < 1) {
        throw new apiError(400, 'Query parameter "page" must be a positive integer');
    }
    if (!Number.isInteger(rawLimit) || rawLimit < 1 || rawLimit > 100) {
        throw new apiError(400, 'Query parameter "limit" must be an integer between 1 and 100');
    }

    let conversationFilter = null;
    const rawConvId = queryParams.conversation_id ?? queryParams.conversationId;
    if (rawConvId !== undefined && String(rawConvId).trim() !== '') {
        conversationFilter = assertValidUuid(String(rawConvId), 'conversationId');
    }

    // Strict per-user conversation isolation: User A never sees User B's questions or answers.
    const params = [policy.id, user.id];
    const whereClauses = ['cm.policy_id = $1', 'cm.user_id = $2'];
    if (conversationFilter !== null) {
        params.push(conversationFilter);
        whereClauses.push(`cm.conversation_id = $${params.length}`);
    }
    const whereSql = `WHERE ${whereClauses.join(' AND ')}`;

    const countRes = await query(
        `
        SELECT
            COUNT(*)::int AS total_messages,
            COUNT(*) FILTER (WHERE cm.role = 'user')::int AS total_questions
        FROM chat_messages cm
        ${whereSql}
        `,
        params
    );

    const totalItems = Number(countRes.rows[0]?.total_messages || 0);
    const totalQuestions = Number(countRes.rows[0]?.total_questions || 0);
    const totalPages = Math.max(1, Math.ceil(totalItems / rawLimit));
    const offset = (rawPage - 1) * rawLimit;

    const listParams = [...params, rawLimit, offset];
    const rowsRes = await query(
        `
        SELECT cm.*
        FROM chat_messages cm
        ${whereSql}
        ORDER BY cm.created_at ASC,
                 CASE WHEN cm.role = 'user' THEN 0 ELSE 1 END ASC,
                 cm.id ASC
        LIMIT $${listParams.length - 1} OFFSET $${listParams.length}
        `,
        listParams
    );

    return {
        policy_id: policy.id,
        items: rowsRes.rows.map(serializeChatMessage),
        total_questions: totalQuestions,
        pagination: {
            page: rawPage,
            limit: rawLimit,
            total_items: totalItems,
            total_pages: totalPages,
            has_next: rawPage < totalPages && totalItems > 0,
            has_prev: rawPage > 1,
        },
    };
}

module.exports = {
    authorizePolicyForQa,
    verifyActiveDocumentReadyForQa,
    validateAndNormalizeAiAnswer,
    askPolicyQuestion,
    listPolicyQaHistory,
};
