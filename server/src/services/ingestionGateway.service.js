const crypto = require('node:crypto');
const apiError = require('../utils/apiError');
const { query, withTransaction } = require('../config/db');
const { PLATFORM_ROLES } = require('../constants/roles');
const { listUserOrganizationScopes } = require('../models/user.model');
const {
    buildRelativeStorageKey,
    writeDocumentBytes,
    deleteDocumentIfExists,
} = require('./documentStorage.service');

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const ALLOWED_JOB_STATUSES = new Set(['queued', 'processing', 'completed', 'failed', 'cancelled']);

function assertValidUuid(value, fieldName = 'id') {
    if (typeof value !== 'string' || !UUID_RE.test(value.trim())) {
        throw new apiError(400, `Invalid ${fieldName}: must be a valid UUID`);
    }
    return value.trim().toLowerCase();
}

function sanitizePublicErrorMessage(message) {
    if (typeof message !== 'string' || !message.trim()) {
        return null;
    }
    return message
        .replace(/postgres(?:ql)?:\/\/[^\s'"]+/gi, 'postgresql://<redacted>')
        .replace(/redis:\/\/[^\s'"]+/gi, 'redis://<redacted>')
        .replace(/\b(?:nvapi-[A-Za-z0-9_-]+|AIza[0-9A-Za-z_-]+|sk-[A-Za-z0-9_-]+)\b/gi, '<redacted-secret>')
        .replace(/Bearer\s+[A-Za-z0-9._~+/=-]+/gi, 'Bearer <redacted>')
        .replace(/(?:\/[A-Za-z0-9._-]+){2,}/g, '<redacted-path>')
        .replace(/\s+/g, ' ')
        .trim()
        .slice(0, 300);
}

function serializeDocument(docRow) {
    if (!docRow) return null;
    return {
        id: docRow.id,
        policy_id: docRow.policy_id,
        version_number: docRow.version_number,
        version_label: docRow.version_label,
        file_name: docRow.file_name,
        mime_type: docRow.mime_type,
        file_size_bytes: Number(docRow.file_size_bytes),
        file_hash_sha256: docRow.file_hash_sha256,
        page_count: Number(docRow.page_count || 0),
        ingestion_status: docRow.ingestion_status,
        created_at: docRow.created_at,
        updated_at: docRow.updated_at,
    };
}

function serializeJob(jobRow, extras = {}) {
    if (!jobRow) return null;
    const totalChunks = Number(jobRow.total_chunks || 0);
    const processedChunks = Number(jobRow.processed_chunks || 0);
    const percent = jobRow.status === 'completed'
        ? 100
        : totalChunks > 0
            ? Math.min(99, Math.round((processedChunks / totalChunks) * 100))
            : 0;

    const sanitizedErrMsg = sanitizePublicErrorMessage(jobRow.error_message);
    const errorInfo = jobRow.error_code || sanitizedErrMsg
        ? {
            code: jobRow.error_code || null,
            message: sanitizedErrMsg,
        }
        : null;

    const payload = {
        id: jobRow.id,
        policy_id: jobRow.policy_id,
        document_id: jobRow.document_id,
        external_policy_key: jobRow.external_policy_key,
        document_version_snapshot: jobRow.document_version_snapshot,
        status: jobRow.status,
        current_stage: jobRow.current_stage,
        progress: {
            total_chunks: totalChunks,
            processed_chunks: processedChunks,
            percent,
        },
        attempt_count: Number(jobRow.attempt_count || 0),
        error: errorInfo,
        started_at: jobRow.started_at,
        completed_at: jobRow.completed_at,
        created_at: jobRow.created_at,
        updated_at: jobRow.updated_at,
    };

    if (extras.policy_title !== undefined || extras.policy_status !== undefined) {
        payload.policy = jobRow.policy_id
            ? {
                id: jobRow.policy_id,
                external_policy_key: jobRow.external_policy_key,
                title: extras.policy_title || null,
                status: extras.policy_status || null,
            }
            : null;
    }

    if (extras.file_name !== undefined || extras.document_ingestion_status !== undefined) {
        payload.document = jobRow.document_id
            ? {
                id: jobRow.document_id,
                version_number: jobRow.document_version_snapshot,
                file_name: extras.file_name || null,
                ingestion_status: extras.document_ingestion_status || null,
            }
            : null;
    }

    return payload;
}

async function verifyPolicyWriteAccess({ user, policyId, dbClient = null, forUpdate = false }) {
    const cleanPolicyId = assertValidUuid(policyId, 'policyId');
    const runner = dbClient ? dbClient.query.bind(dbClient) : query;
    const lockClause = forUpdate ? ' FOR UPDATE' : '';

    const polRes = await runner(
        `
        SELECT id, external_policy_key, title, policy_type, visibility,
               owner_id, owner_scope, status, active_version_id
        FROM policies
        WHERE id = $1
        ${lockClause}
        `,
        [cleanPolicyId]
    );

    if (polRes.rowCount === 0) {
        throw new apiError(404, 'Policy not found');
    }

    const policy = polRes.rows[0];
    if (user.platform_role !== PLATFORM_ROLES.ADMIN) {
        const { writeScopes } = await listUserOrganizationScopes(user.id, dbClient);
        if (!writeScopes.includes(policy.owner_scope)) {
            throw new apiError(403, 'Forbidden: you do not have upload permission for this policy');
        }
    }

    if (policy.status === 'archived') {
        throw new apiError(409, 'Archived policies cannot accept new document uploads');
    }

    return policy;
}

function getAiServiceBaseUrl(overrideUrl = null) {
    const raw = overrideUrl || process.env.AI_SERVICE_URL || 'http://127.0.0.1:8001';
    return String(raw).trim().replace(/\/+$/, '');
}

async function invokeInternalEnqueue({ jobId, policyId, documentId, requestId }, options = {}) {
    const baseUrl = getAiServiceBaseUrl(options.aiServiceUrl);
    const serviceToken = options.serviceToken !== undefined
        ? options.serviceToken
        : process.env.INTERNAL_SERVICE_TOKEN || '';
    const fetchImpl = options.fetchImpl || global.fetch;
    const timeoutMs = Number(options.timeoutMs || process.env.ENQUEUE_HTTP_TIMEOUT_MS || 5000);
    const maxAttempts = Number(options.maxInlineAttempts || process.env.ENQUEUE_INLINE_ATTEMPTS || 2);

    let lastError = null;
    for (let attempt = 1; attempt <= Math.max(1, maxAttempts); attempt += 1) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs);
        try {
            const response = await fetchImpl(`${baseUrl}/internal/ingestion/enqueue`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Service-Token': serviceToken,
                    'X-Request-Id': requestId || crypto.randomUUID(),
                },
                body: JSON.stringify({
                    job_id: jobId,
                    policy_id: policyId,
                    document_id: documentId,
                    request_id: requestId || null,
                }),
                signal: controller.signal,
            });
            clearTimeout(timer);

            const body = await response.json().catch(() => ({}));
            if (response.ok) {
                return {
                    ok: true,
                    statusCode: response.status,
                    data: body,
                };
            }

            const err = new Error(body?.detail || `Internal enqueue rejected with status ${response.status}`);
            err.statusCode = response.status;
            err.retryable = response.status >= 500;
            lastError = err;

            if (!err.retryable) {
                return {
                    ok: false,
                    retryable: false,
                    statusCode: response.status,
                    errorCode: 'ENQUEUE_REJECTED',
                    message: body?.detail || 'Internal ingestion enqueue rejected request',
                };
            }
        } catch (err) {
            clearTimeout(timer);
            err.retryable = true;
            lastError = err;
        }

        if (attempt < maxAttempts) {
            await new Promise((resolve) => setTimeout(resolve, 100 * attempt));
        }
    }

    return {
        ok: false,
        retryable: lastError?.retryable !== false,
        statusCode: lastError?.statusCode || 503,
        errorCode: 'ENQUEUE_SERVICE_UNAVAILABLE',
        message: 'Background ingestion dispatch delayed; automatic reconciliation scheduled',
    };
}

async function markJobEnqueueRetryPending({ jobId, errorCode, errorMessage, requestId, retryDelayMs = 1500 }) {
    const nextRetryAt = new Date(Date.now() + retryDelayMs).toISOString();
    const res = await query(
        `
        UPDATE ingestion_jobs
        SET current_stage = CASE
                WHEN status = 'queued' THEN 'ENQUEUE_RETRY_PENDING'
                ELSE current_stage
            END,
            error_code = CASE WHEN status = 'queued' THEN $2 ELSE error_code END,
            error_message = CASE WHEN status = 'queued' THEN $3 ELSE error_message END,
            metadata = metadata || jsonb_build_object(
                'dispatch_state', 'retry_scheduled',
                'dispatch_attempts', COALESCE((metadata->>'dispatch_attempts')::int, 0) + 1,
                'next_retry_at', $4::text,
                'last_enqueue_error_at', NOW()::text,
                'request_id', COALESCE($5::text, metadata->>'request_id')
            ),
            updated_at = NOW()
        WHERE id = $1
          AND status = 'queued'
          AND celery_task_id IS NULL
        RETURNING *
        `,
        [jobId, errorCode, errorMessage, nextRetryAt, requestId || null]
    );
    return res.rows[0] || null;
}

async function markJobEnqueueExhausted({ jobId, documentId, errorCode, errorMessage }) {
    return withTransaction(async (client) => {
        const jobRes = await client.query(
            `
            UPDATE ingestion_jobs
            SET status = 'failed',
                current_stage = 'ENQUEUE_EXHAUSTED',
                error_code = $2,
                error_message = $3,
                completed_at = COALESCE(completed_at, NOW()),
                metadata = metadata || jsonb_build_object(
                    'dispatch_state', 'exhausted',
                    'exhausted_at', NOW()::text
                ),
                updated_at = NOW()
            WHERE id = $1
              AND status = 'queued'
              AND celery_task_id IS NULL
            RETURNING *
            `,
            [jobId, errorCode, errorMessage]
        );
        if (jobRes.rowCount > 0 && documentId) {
            await client.query(
                `
                UPDATE policy_documents
                SET ingestion_status = 'failed',
                    updated_at = NOW()
                WHERE id = $1
                  AND ingestion_status IN ('pending', 'queued')
                `,
                [documentId]
            );
        }
        return jobRes.rows[0] || null;
    });
}

async function uploadPolicyDocument({
    user,
    policyId,
    fileBuffer,
    validatedPdf,
    versionLabel = null,
    requestId = null,
    storageOptions = {},
    enqueueOptions = {},
}) {
    const reqId = requestId || crypto.randomUUID();
    const candidateDocId = crypto.randomUUID();
    const candidateJobId = crypto.randomUUID();
    let newlyWrittenStorageKey = null;

    let txResult;
    try {
        txResult = await withTransaction(async (client) => {
            const policy = await verifyPolicyWriteAccess({
                user,
                policyId,
                dbClient: client,
                forUpdate: true,
            });

            // Optional hook for transactional failure/cleanup verification tests
            if (typeof storageOptions.beforeTransactionCommit === 'function') {
                await storageOptions.beforeTransactionCommit({ policyId: policy.id });
            }

            const dupRes = await client.query(
                `
                SELECT *
                FROM policy_documents
                WHERE policy_id = $1
                  AND file_hash_sha256 = $2
                  AND is_archived = FALSE
                ORDER BY version_number DESC
                LIMIT 1
                `,
                [policy.id, validatedPdf.sha256]
            );

            if (dupRes.rowCount > 0 && dupRes.rows[0].ingestion_status !== 'failed') {
                const existingDoc = dupRes.rows[0];
                const jobRes = await client.query(
                    `
                    SELECT *
                    FROM ingestion_jobs
                    WHERE document_id = $1
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    `,
                    [existingDoc.id]
                );
                return {
                    duplicate: true,
                    policy,
                    document: existingDoc,
                    job: jobRes.rows[0] || null,
                    createdNewFile: false,
                };
            }

            const verRes = await client.query(
                `
                SELECT COALESCE(MAX(version_number), 0)::int + 1 AS next_version
                FROM policy_documents
                WHERE policy_id = $1
                `,
                [policy.id]
            );
            const nextVersion = Number(verRes.rows[0].next_version);
            const storageKey = buildRelativeStorageKey({
                ownerScope: policy.owner_scope,
                policyId: policy.id,
                versionNumber: nextVersion,
                documentId: candidateDocId,
            });

            // Persist file outside PostgreSQL/Redis cleanly; rollback catch removes orphaned file if TX fails
            await writeDocumentBytes(storageKey, fileBuffer, storageOptions);
            newlyWrittenStorageKey = storageKey;

            const cleanVersionLabel = typeof versionLabel === 'string' && versionLabel.trim()
                ? versionLabel.trim().slice(0, 64)
                : `v${nextVersion}`;

            const docMetadata = {
                uploaded_by_user_id: user.id,
                request_id: reqId,
                original_filename: validatedPdf.sanitizedFilename,
            };

            const docRes = await client.query(
                `
                INSERT INTO policy_documents (
                    id, policy_id, version_number, version_label, file_name,
                    storage_path, mime_type, file_size_bytes, file_hash_sha256,
                    page_count, is_encrypted, ingestion_status, document_metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 0, FALSE, 'queued', $10::jsonb)
                RETURNING *
                `,
                [
                    candidateDocId,
                    policy.id,
                    nextVersion,
                    cleanVersionLabel,
                    validatedPdf.sanitizedFilename,
                    storageKey,
                    validatedPdf.mimeType,
                    validatedPdf.fileSizeBytes,
                    validatedPdf.sha256,
                    JSON.stringify(docMetadata),
                ]
            );

            const jobMetadata = {
                request_id: reqId,
                uploaded_by_user_id: user.id,
                dispatch_state: 'pending_enqueue',
                dispatch_attempts: 0,
            };

            const jobRes = await client.query(
                `
                INSERT INTO ingestion_jobs (
                    id, policy_id, document_id, external_policy_key,
                    document_version_snapshot, file_hash_sha256, owner_scope,
                    status, current_stage, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, 'queued', 'QUEUED', $8::jsonb)
                RETURNING *
                `,
                [
                    candidateJobId,
                    policy.id,
                    candidateDocId,
                    policy.external_policy_key,
                    nextVersion,
                    validatedPdf.sha256,
                    policy.owner_scope,
                    JSON.stringify(jobMetadata),
                ]
            );

            await client.query(
                `
                INSERT INTO audit_logs (
                    actor_user_id, action, resource_type, resource_id, request_id, details
                )
                VALUES ($1, 'policy_document.uploaded', 'policy_document', $2, $3, $4::jsonb)
                `,
                [
                    user.id,
                    candidateDocId,
                    reqId,
                    JSON.stringify({
                        policy_id: policy.id,
                        job_id: candidateJobId,
                        version_number: nextVersion,
                        file_hash_sha256: validatedPdf.sha256,
                    }),
                ]
            ).catch(() => {
                // Non-fatal if audit_logs insert fails in standalone partial schemas
            });

            if (typeof storageOptions.afterInsertBeforeCommit === 'function') {
                await storageOptions.afterInsertBeforeCommit({
                    policyId: policy.id,
                    documentId: candidateDocId,
                    jobId: candidateJobId,
                    storageKey,
                });
            }

            return {
                duplicate: false,
                policy,
                document: docRes.rows[0],
                job: jobRes.rows[0],
                createdNewFile: true,
            };
        });
    } catch (txErr) {
        if (newlyWrittenStorageKey) {
            await deleteDocumentIfExists(newlyWrittenStorageKey, storageOptions);
        }
        throw txErr;
    }

    // Duplicate upload case: do not re-create job or file; if an un-dispatched job still exists, enqueue idempotently
    if (txResult.duplicate) {
        let currentJob = txResult.job;
        if (currentJob && currentJob.status === 'queued' && !currentJob.celery_task_id) {
            const enq = await invokeInternalEnqueue(
                {
                    jobId: currentJob.id,
                    policyId: txResult.policy.id,
                    documentId: txResult.document.id,
                    requestId: reqId,
                },
                enqueueOptions
            );
            if (enq.ok) {
                const refreshed = await query('SELECT * FROM ingestion_jobs WHERE id = $1', [currentJob.id]);
                if (refreshed.rowCount > 0) currentJob = refreshed.rows[0];
            }
        }
        return {
            httpStatus: 200,
            duplicate: true,
            enqueued: false,
            policy: {
                id: txResult.policy.id,
                external_policy_key: txResult.policy.external_policy_key,
                title: txResult.policy.title,
                status: txResult.policy.status,
            },
            document: serializeDocument(txResult.document),
            job: serializeJob(currentJob),
        };
    }

    // Dispatch committed job to Python FastAPI internal enqueue endpoint
    const enqResult = await invokeInternalEnqueue(
        {
            jobId: txResult.job.id,
            policyId: txResult.policy.id,
            documentId: txResult.document.id,
            requestId: reqId,
        },
        enqueueOptions
    );

    if (enqResult.ok) {
        const refreshed = await query('SELECT * FROM ingestion_jobs WHERE id = $1', [txResult.job.id]);
        const latestJob = refreshed.rowCount > 0 ? refreshed.rows[0] : txResult.job;
        return {
            httpStatus: 201,
            duplicate: false,
            enqueued: Boolean(enqResult.data?.enqueued),
            dispatch: {
                state: 'dispatched',
                task_id: enqResult.data?.task_id || null,
            },
            policy: {
                id: txResult.policy.id,
                external_policy_key: txResult.policy.external_policy_key,
                title: txResult.policy.title,
                status: txResult.policy.status,
            },
            document: serializeDocument(txResult.document),
            job: serializeJob(latestJob),
        };
    }

    if (!enqResult.retryable) {
        const exhaustedJob = await markJobEnqueueExhausted({
            jobId: txResult.job.id,
            documentId: txResult.document.id,
            errorCode: enqResult.errorCode || 'ENQUEUE_REJECTED',
            errorMessage: enqResult.message || 'Background ingestion enqueue failed',
        });
        const refreshedDoc = await query('SELECT * FROM policy_documents WHERE id = $1', [txResult.document.id]);
        return {
            httpStatus: 202,
            duplicate: false,
            enqueued: false,
            dispatch: {
                state: 'failed',
            },
            policy: {
                id: txResult.policy.id,
                external_policy_key: txResult.policy.external_policy_key,
                title: txResult.policy.title,
                status: txResult.policy.status,
            },
            document: serializeDocument(refreshedDoc.rows[0] || txResult.document),
            job: serializeJob(exhaustedJob || txResult.job),
        };
    }

    const pendingJob = await markJobEnqueueRetryPending({
        jobId: txResult.job.id,
        errorCode: enqResult.errorCode,
        errorMessage: enqResult.message,
        requestId: reqId,
    });

    return {
        httpStatus: 202,
        duplicate: false,
        enqueued: false,
        dispatch: {
            state: 'retry_scheduled',
        },
        policy: {
            id: txResult.policy.id,
            external_policy_key: txResult.policy.external_policy_key,
            title: txResult.policy.title,
            status: txResult.policy.status,
        },
        document: serializeDocument(txResult.document),
        job: serializeJob(pendingJob || txResult.job),
    };
}

async function reconcilePendingIngestionJobs(options = {}) {
    const staleThresholdMs = Number(options.staleThresholdMs ?? process.env.ENQUEUE_STALE_MS ?? 3000);
    const maxAttempts = Number(options.maxDispatchAttempts ?? process.env.ENQUEUE_MAX_DISPATCH_ATTEMPTS ?? 5);
    const limit = Math.max(1, Math.min(50, Number(options.limit || 20)));
    const staleCutoff = new Date(Date.now() - staleThresholdMs).toISOString();

    const candidates = await query(
        `
        SELECT j.id, j.policy_id, j.document_id, j.current_stage, j.metadata, j.created_at
        FROM ingestion_jobs j
        WHERE j.status = 'queued'
          AND j.celery_task_id IS NULL
          AND j.policy_id IS NOT NULL
          AND j.document_id IS NOT NULL
          AND (
              j.current_stage = 'ENQUEUE_RETRY_PENDING'
              OR j.created_at <= $1::timestamptz
          )
        ORDER BY j.created_at ASC
        LIMIT $2
        `,
        [staleCutoff, limit]
    );

    const summary = {
        scanned: candidates.rowCount,
        dispatched: 0,
        rescheduled: 0,
        exhausted: 0,
    };

    for (const row of candidates.rows) {
        const meta = row.metadata || {};
        if (!options.ignoreBackoff && meta.next_retry_at) {
            const nextAtMs = Date.parse(meta.next_retry_at);
            if (!Number.isNaN(nextAtMs) && nextAtMs > Date.now()) {
                continue;
            }
        }

        const attemptsSoFar = Number(meta.dispatch_attempts || 0);
        const reqId = meta.request_id || `reconcile-${row.id}`;
        const enq = await invokeInternalEnqueue(
            {
                jobId: row.id,
                policyId: row.policy_id,
                documentId: row.document_id,
                requestId: reqId,
            },
            {
                ...options.enqueueOptions,
                maxInlineAttempts: 1,
            }
        );

        if (enq.ok) {
            summary.dispatched += 1;
            continue;
        }

        const nextAttempt = attemptsSoFar + 1;
        if (!enq.retryable || nextAttempt >= maxAttempts) {
            await markJobEnqueueExhausted({
                jobId: row.id,
                documentId: row.document_id,
                errorCode: 'ENQUEUE_DISPATCH_FAILED',
                errorMessage: 'Unable to dispatch ingestion job to background worker after maximum attempts.',
            });
            summary.exhausted += 1;
        } else {
            const backoffMs = Math.min(60000, 1000 * 2 ** (nextAttempt - 1));
            await markJobEnqueueRetryPending({
                jobId: row.id,
                errorCode: enq.errorCode,
                errorMessage: enq.message,
                requestId: reqId,
                retryDelayMs: backoffMs,
            });
            summary.rescheduled += 1;
        }
    }

    return summary;
}

function startIngestionReconciler(options = {}) {
    const intervalMs = Number(options.intervalMs || process.env.ENQUEUE_RECONCILE_INTERVAL_MS || 15000);
    if (!Number.isFinite(intervalMs) || intervalMs <= 0) {
        return { stop: () => {} };
    }
    let running = false;
    const timer = setInterval(async () => {
        if (running) return;
        running = true;
        try {
            await reconcilePendingIngestionJobs(options);
        } catch (err) {
            console.error('Ingestion reconciler cycle failed:', sanitizePublicErrorMessage(err.message));
        } finally {
            running = false;
        }
    }, intervalMs);
    if (typeof timer.unref === 'function') {
        timer.unref();
    }
    return {
        stop: () => clearInterval(timer),
    };
}

async function getAuthorizedIngestionJob({ user, jobId }) {
    const cleanJobId = assertValidUuid(jobId, 'jobId');
    const isAdmin = user.platform_role === PLATFORM_ROLES.ADMIN;

    let result;
    if (isAdmin) {
        result = await query(
            `
            SELECT j.*,
                   p.title AS policy_title,
                   p.status AS policy_status,
                   d.file_name AS doc_file_name,
                   d.ingestion_status AS doc_ingestion_status
            FROM ingestion_jobs j
            LEFT JOIN policies p ON p.id = j.policy_id
            LEFT JOIN policy_documents d ON d.id = j.document_id
            WHERE j.id = $1
            `,
            [cleanJobId]
        );
    } else {
        const { readScopes } = await listUserOrganizationScopes(user.id);
        result = await query(
            `
            SELECT j.*,
                   p.title AS policy_title,
                   p.status AS policy_status,
                   d.file_name AS doc_file_name,
                   d.ingestion_status AS doc_ingestion_status
            FROM ingestion_jobs j
            LEFT JOIN policies p ON p.id = j.policy_id
            LEFT JOIN policy_documents d ON d.id = j.document_id
            WHERE j.id = $1
              AND j.owner_scope = ANY($2::text[])
            `,
            [cleanJobId, readScopes]
        );
    }

    if (result.rowCount === 0) {
        // Deliberately return 404 without revealing whether another tenant owns this job ID
        throw new apiError(404, 'Ingestion job not found');
    }

    const row = result.rows[0];
    return serializeJob(row, {
        policy_title: row.policy_title,
        policy_status: row.policy_status,
        file_name: row.doc_file_name,
        document_ingestion_status: row.doc_ingestion_status,
    });
}

async function listAuthorizedIngestionJobs({ user, queryParams = {} }) {
    if (queryParams.owner_scope !== undefined || queryParams.ownerScope !== undefined) {
        throw new apiError(400, 'Clients cannot supply owner_scope filters');
    }

    const rawPage = queryParams.page === undefined ? 1 : Number(queryParams.page);
    const rawLimit = queryParams.limit === undefined ? 20 : Number(queryParams.limit);
    if (!Number.isInteger(rawPage) || rawPage < 1) {
        throw new apiError(400, 'Query parameter "page" must be a positive integer');
    }
    if (!Number.isInteger(rawLimit) || rawLimit < 1 || rawLimit > 100) {
        throw new apiError(400, 'Query parameter "limit" must be an integer between 1 and 100');
    }

    let statusFilter = null;
    if (queryParams.status !== undefined && String(queryParams.status).trim() !== '') {
        const s = String(queryParams.status).trim().toLowerCase();
        if (!ALLOWED_JOB_STATUSES.has(s)) {
            throw new apiError(
                400,
                `Invalid status filter. Allowed: ${Array.from(ALLOWED_JOB_STATUSES).join(', ')}`
            );
        }
        statusFilter = s;
    }

    let policyIdFilter = null;
    if (queryParams.policyId !== undefined && String(queryParams.policyId).trim() !== '') {
        policyIdFilter = assertValidUuid(String(queryParams.policyId), 'policyId');
    }

    const isAdmin = user.platform_role === PLATFORM_ROLES.ADMIN;
    const whereClauses = [];
    const params = [];

    if (!isAdmin) {
        const { readScopes } = await listUserOrganizationScopes(user.id);
        params.push(readScopes);
        whereClauses.push(`j.owner_scope = ANY($${params.length}::text[])`);
    }
    if (statusFilter !== null) {
        params.push(statusFilter);
        whereClauses.push(`j.status = $${params.length}`);
    }
    if (policyIdFilter !== null) {
        params.push(policyIdFilter);
        whereClauses.push(`j.policy_id = $${params.length}`);
    }

    const whereSql = whereClauses.length > 0 ? `WHERE ${whereClauses.join(' AND ')}` : '';
    const countRes = await query(
        `SELECT COUNT(*)::int AS total FROM ingestion_jobs j ${whereSql}`,
        params
    );
    const totalItems = Number(countRes.rows[0]?.total || 0);
    const totalPages = Math.max(1, Math.ceil(totalItems / rawLimit));
    const offset = (rawPage - 1) * rawLimit;

    const listParams = [...params, rawLimit, offset];
    const itemsRes = await query(
        `
        SELECT j.*,
               p.title AS policy_title,
               p.status AS policy_status,
               d.file_name AS doc_file_name,
               d.ingestion_status AS doc_ingestion_status
        FROM ingestion_jobs j
        LEFT JOIN policies p ON p.id = j.policy_id
        LEFT JOIN policy_documents d ON d.id = j.document_id
        ${whereSql}
        ORDER BY j.created_at DESC, j.id DESC
        LIMIT $${listParams.length - 1} OFFSET $${listParams.length}
        `,
        listParams
    );

    const items = itemsRes.rows.map((row) =>
        serializeJob(row, {
            policy_title: row.policy_title,
            policy_status: row.policy_status,
            file_name: row.doc_file_name,
            document_ingestion_status: row.doc_ingestion_status,
        })
    );

    return {
        items,
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

const ALLOWED_POLICY_TYPES = new Set(['health', 'car', 'life', 'term', 'travel', 'motor', 'pension', 'general']);
const ALLOWED_POLICY_STATUSES = new Set(['draft', 'active', 'inactive', 'archived']);

function serializePolicy(row, canUpload = false) {
    if (!row) return null;
    return {
        id: row.id,
        external_policy_key: row.external_policy_key,
        title: row.title,
        insurer_name: row.insurer_name || null,
        policy_type: row.policy_type,
        visibility: row.visibility,
        status: row.status,
        active_version_id: row.active_version_id || null,
        attribute_extraction_status: row.attribute_extraction_status || null,
        attribute_extraction_source: row.attribute_extraction_source || null,
        is_catalog_eligible: row.is_catalog_eligible !== undefined && row.is_catalog_eligible !== null
            ? Boolean(row.is_catalog_eligible)
            : row.visibility === 'catalog',
        can_upload: Boolean(canUpload),
        created_at: row.created_at,
        updated_at: row.updated_at,
    };
}

async function listAuthorizedPolicies({ user, queryParams = {} }) {
    if (queryParams.owner_scope !== undefined || queryParams.ownerScope !== undefined) {
        throw new apiError(400, 'Clients cannot supply owner_scope filters');
    }

    const isAdmin = user.platform_role === PLATFORM_ROLES.ADMIN;
    const { readScopes, writeScopes } = await listUserOrganizationScopes(user.id);
    const writableOnly = String(queryParams.writable || '').toLowerCase() === 'true';
    const includeArchived = String(queryParams.includeArchived || '').toLowerCase() === 'true';

    let statusFilter = null;
    if (queryParams.status !== undefined && String(queryParams.status).trim() !== '') {
        const s = String(queryParams.status).trim().toLowerCase();
        if (!ALLOWED_POLICY_STATUSES.has(s)) {
            throw new apiError(
                400,
                `Invalid status filter. Allowed: ${Array.from(ALLOWED_POLICY_STATUSES).join(', ')}`
            );
        }
        statusFilter = s;
    }

    const whereClauses = [];
    const params = [];

    if (!isAdmin) {
        const targetScopes = writableOnly ? writeScopes : readScopes;
        params.push(targetScopes);
        whereClauses.push(`p.owner_scope = ANY($${params.length}::text[])`);
    }

    if (statusFilter !== null) {
        params.push(statusFilter);
        whereClauses.push(`p.status = $${params.length}`);
    } else if (!includeArchived) {
        whereClauses.push("p.status <> 'archived'");
    }

    const whereSql = whereClauses.length > 0 ? `WHERE ${whereClauses.join(' AND ')}` : '';
    const res = await query(
        `
        SELECT
            p.*,
            pa.extraction_status AS attribute_extraction_status,
            pa.extraction_source AS attribute_extraction_source,
            pa.is_catalog_eligible AS is_catalog_eligible
        FROM policies p
        LEFT JOIN policy_attributes pa
          ON pa.policy_id = p.id AND pa.document_id = p.active_version_id
        ${whereSql}
        ORDER BY p.updated_at DESC, p.created_at DESC, p.id DESC
        LIMIT 100
        `,
        params
    );

    return {
        items: res.rows.map((row) =>
            serializePolicy(row, isAdmin || writeScopes.includes(row.owner_scope))
        ),
    };
}

async function createAuthorizedPolicy({ user, body = {} }) {
    const rawTitle = typeof body.title === 'string' ? body.title.trim() : '';
    if (!rawTitle || rawTitle.length > 250) {
        throw new apiError(400, 'Policy title is required and must be at most 250 characters');
    }

    const policyType = typeof body.policy_type === 'string' && body.policy_type.trim() !== ''
        ? body.policy_type.trim().toLowerCase()
        : 'general';
    if (!ALLOWED_POLICY_TYPES.has(policyType)) {
        throw new apiError(
            400,
            `Invalid policy_type. Allowed: ${Array.from(ALLOWED_POLICY_TYPES).join(', ')}`
        );
    }

    const insurerName = typeof body.insurer_name === 'string' && body.insurer_name.trim() !== ''
        ? body.insurer_name.trim().slice(0, 200)
        : null;

    const requestedVisibility = typeof body.visibility === 'string' && body.visibility.trim() !== ''
        ? body.visibility.trim().toLowerCase()
        : 'private';
    if (!['private', 'catalog'].includes(requestedVisibility)) {
        throw new apiError(400, 'Visibility must be either "private" or "catalog"');
    }

    let ownerId = user.id;
    let ownerScope = `user:${user.id}`;
    const orgIdRaw = body.organization_id;

    if (orgIdRaw !== undefined && orgIdRaw !== null && String(orgIdRaw).trim() !== '') {
        const orgId = assertValidUuid(String(orgIdRaw), 'organization_id');
        if (user.platform_role !== PLATFORM_ROLES.ADMIN) {
            const memberRes = await query(
                `
                SELECT role
                FROM organization_members
                WHERE organization_id = $1 AND user_id = $2
                `,
                [orgId, user.id]
            );
            if (memberRes.rowCount === 0 || !['owner', 'admin'].includes(memberRes.rows[0].role)) {
                throw new apiError(403, 'Forbidden: organization owner or admin membership is required');
            }
        }
        ownerId = orgId;
        ownerScope = `org:${orgId}`;
    } else if (
        requestedVisibility === 'catalog'
        && ![PLATFORM_ROLES.ADMIN, PLATFORM_ROLES.ORGANISATION].includes(user.platform_role)
    ) {
        throw new apiError(403, 'Forbidden: only organisation or admin accounts can publish catalog policies');
    }

    let externalKey = '';
    if (typeof body.external_policy_key === 'string' && body.external_policy_key.trim() !== '') {
        externalKey = body.external_policy_key.trim().replace(/[^a-zA-Z0-9_.-]/g, '_').slice(0, 64);
    }
    if (!externalKey) {
        const slug = rawTitle.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 24) || 'policy';
        externalKey = `pol_${slug}_${crypto.randomBytes(4).toString('hex')}`;
    }

    const res = await query(
        `
        INSERT INTO policies (
            external_policy_key, title, insurer_name, policy_type,
            visibility, owner_id, owner_scope, status
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, 'draft')
        RETURNING *
        `,
        [externalKey, rawTitle, insurerName, policyType, requestedVisibility, ownerId, ownerScope]
    );

    return serializePolicy(res.rows[0], true);
}

async function invokeInternalExtractAttributes(
    { policyId, documentId, isCatalogEligible = true, requestId = null },
    options = {}
) {
    const baseUrl = getAiServiceBaseUrl(options.aiServiceUrl);
    const serviceToken = options.serviceToken !== undefined
        ? options.serviceToken
        : process.env.INTERNAL_SERVICE_TOKEN || '';
    const fetchImpl = options.fetchImpl || global.fetch;
    const timeoutMs = Number(options.timeoutMs || process.env.EXTRACT_ATTRIBUTES_TIMEOUT_MS || 45000);

    if (!serviceToken) {
        throw new apiError(503, 'Internal AI service authentication is not configured');
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    let response;
    let body = null;
    try {
        response = await fetchImpl(`${baseUrl}/internal/recommendations/extract-attributes`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Service-Token': serviceToken,
                'X-Request-Id': requestId || crypto.randomUUID(),
            },
            body: JSON.stringify({
                policy_id: policyId,
                document_id: documentId,
                is_catalog_eligible: Boolean(isCatalogEligible),
            }),
            signal: controller.signal,
        });
        body = await response.json().catch(() => ({}));
    } catch (err) {
        if (err && err.name === 'AbortError') {
            throw new apiError(504, 'AI attribute extraction timed out while publishing policy');
        }
        throw new apiError(503, 'AI attribute extraction service is temporarily unavailable');
    } finally {
        clearTimeout(timer);
    }

    if (!response.ok) {
        const detail = sanitizePublicErrorMessage(body?.detail || body?.message);
        if (response.status === 409) {
            throw new apiError(409, detail || 'Policy document is not ready for catalog extraction');
        }
        if (response.status === 404) {
            throw new apiError(404, detail || 'Policy or document not found for attribute extraction');
        }
        if (response.status === 429) {
            throw new apiError(429, detail || 'AI provider rate limit exceeded during attribute extraction');
        }
        if (response.status === 504) {
            throw new apiError(504, detail || 'AI attribute extraction timed out');
        }
        if (response.status === 503) {
            throw new apiError(503, detail || 'AI attribute extraction service unavailable');
        }
        throw new apiError(502, detail || 'AI attribute extraction failed while publishing policy');
    }

    return body;
}

async function updateAuthorizedPolicyVisibility({
    user,
    policyId,
    body = {},
    requestId = null,
    backfillOptions = {},
}) {
    if (![PLATFORM_ROLES.ADMIN, PLATFORM_ROLES.ORGANISATION].includes(user.platform_role)) {
        throw new apiError(
            403,
            'Forbidden: only organisation or admin accounts can manage catalog visibility'
        );
    }

    const targetVisibility = typeof body.visibility === 'string'
        ? body.visibility.trim().toLowerCase()
        : '';
    if (!['private', 'catalog'].includes(targetVisibility)) {
        throw new apiError(400, 'Visibility must be either "private" or "catalog"');
    }

    const reqId = requestId || crypto.randomUUID();
    const policy = await verifyPolicyWriteAccess({ user, policyId });

    let activeDocId = policy.active_version_id || null;
    if (targetVisibility === 'catalog') {
        if (policy.status !== 'active' || !activeDocId) {
            throw new apiError(
                409,
                'Only active policies with a completed document version can be published to the recommendations catalog'
            );
        }

        const docRes = await query(
            `
            SELECT id, ingestion_status, is_archived
            FROM policy_documents
            WHERE id = $1 AND policy_id = $2
            `,
            [activeDocId, policy.id]
        );
        if (
            docRes.rowCount === 0
            || docRes.rows[0].is_archived
            || docRes.rows[0].ingestion_status !== 'completed'
        ) {
            throw new apiError(
                409,
                'Only active policies with a completed document version can be published to the recommendations catalog'
            );
        }

        const attrRes = await query(
            `
            SELECT extraction_status
            FROM policy_attributes
            WHERE policy_id = $1 AND document_id = $2
            `,
            [policy.id, activeDocId]
        );
        const existingStatus = attrRes.rowCount > 0 ? attrRes.rows[0].extraction_status : null;
        if (!existingStatus || existingStatus === 'pending' || existingStatus === 'failed') {
            await invokeInternalExtractAttributes(
                {
                    policyId: policy.id,
                    documentId: activeDocId,
                    isCatalogEligible: true,
                    requestId: reqId,
                },
                backfillOptions
            );
        }
    }

    const updatedRow = await withTransaction(async (client) => {
        const locked = await verifyPolicyWriteAccess({
            user,
            policyId: policy.id,
            dbClient: client,
            forUpdate: true,
        });
        activeDocId = locked.active_version_id || null;

        if (targetVisibility === 'catalog') {
            if (locked.status !== 'active' || !activeDocId) {
                throw new apiError(
                    409,
                    'Only active policies with a completed document version can be published to the recommendations catalog'
                );
            }
            const docRes = await client.query(
                `
                SELECT id
                FROM policy_documents
                WHERE id = $1 AND policy_id = $2 AND is_archived = FALSE AND ingestion_status = 'completed'
                `,
                [activeDocId, locked.id]
            );
            if (docRes.rowCount === 0) {
                throw new apiError(
                    409,
                    'Only active policies with a completed document version can be published to the recommendations catalog'
                );
            }
        }

        const polUpdateRes = await client.query(
            `
            UPDATE policies
            SET visibility = $2,
                updated_at = NOW()
            WHERE id = $1
            RETURNING *
            `,
            [locked.id, targetVisibility]
        );
        const updatedPolicy = polUpdateRes.rows[0];

        let attrRow = null;
        if (activeDocId) {
            const attrUpdateRes = await client.query(
                `
                UPDATE policy_attributes
                SET is_catalog_eligible = $3,
                    updated_at = NOW()
                WHERE policy_id = $1 AND document_id = $2
                RETURNING extraction_status, extraction_source, is_catalog_eligible
                `,
                [locked.id, activeDocId, targetVisibility === 'catalog']
            );
            if (attrUpdateRes.rowCount > 0) {
                attrRow = attrUpdateRes.rows[0];
            }
        }

        await client.query(
            `
            INSERT INTO audit_logs (
                actor_user_id, action, resource_type, resource_id, request_id, details
            )
            VALUES ($1, 'policy.visibility.updated', 'policy', $2, $3, $4::jsonb)
            `,
            [
                user.id,
                locked.id,
                reqId,
                JSON.stringify({
                    previous_visibility: locked.visibility,
                    visibility: targetVisibility,
                    active_version_id: activeDocId,
                    is_catalog_eligible: attrRow ? Boolean(attrRow.is_catalog_eligible) : targetVisibility === 'catalog',
                }),
            ]
        );

        return {
            ...updatedPolicy,
            attribute_extraction_status: attrRow?.extraction_status || null,
            attribute_extraction_source: attrRow?.extraction_source || null,
            is_catalog_eligible: attrRow
                ? Boolean(attrRow.is_catalog_eligible)
                : targetVisibility === 'catalog',
        };
    });

    return serializePolicy(updatedRow, true);
}

module.exports = {
    assertValidUuid,
    sanitizePublicErrorMessage,
    serializeDocument,
    serializeJob,
    serializePolicy,
    verifyPolicyWriteAccess,
    invokeInternalEnqueue,
    invokeInternalExtractAttributes,
    markJobEnqueueRetryPending,
    markJobEnqueueExhausted,
    uploadPolicyDocument,
    reconcilePendingIngestionJobs,
    startIngestionReconciler,
    getAuthorizedIngestionJob,
    listAuthorizedIngestionJobs,
    listAuthorizedPolicies,
    createAuthorizedPolicy,
    updateAuthorizedPolicyVisibility,
};
