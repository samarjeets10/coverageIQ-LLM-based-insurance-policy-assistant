const { describe, it, before, after, beforeEach } = require('node:test');
const assert = require('node:assert/strict');
const { execFileSync, spawn } = require('node:child_process');
const crypto = require('node:crypto');
const fsPromises = require('node:fs/promises');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const { URL } = require('node:url');

const { Pool } = require('pg');

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const SERVER_ROOT = path.resolve(__dirname, '..');
const PYTHON_BIN = process.env.PYTHON_BIN || path.join(os.homedir(), 'ai-env-py311', 'bin', 'python');

const RUNTIME_DSN = process.env.TEST_DATABASE_URL
    || 'postgresql://coverageiq_runtime:change_me_runtime@127.0.0.1:55432/coverageiq_db';
const MIGRATION_DSN = process.env.TEST_MIGRATION_DATABASE_URL
    || 'postgresql://coverageiq_migrator:change_me_migrator@127.0.0.1:55432/coverageiq_db';
const REDIS_URL = process.env.TEST_REDIS_URL || 'redis://127.0.0.1:56379/0';

async function findFreePort() {
    return new Promise((resolve, reject) => {
        const srv = net.createServer();
        srv.unref();
        srv.on('error', reject);
        srv.listen(0, '127.0.0.1', () => {
            const addr = srv.address();
            srv.close(() => resolve(addr.port));
        });
    });
}

async function isTcpReachable(host, port) {
    return new Promise((resolve) => {
        const socket = net.createConnection({ host, port });
        socket.setTimeout(1500);
        socket.on('connect', () => {
            socket.destroy();
            resolve(true);
        });
        socket.on('error', () => {
            socket.destroy();
            resolve(false);
        });
        socket.on('timeout', () => {
            socket.destroy();
            resolve(false);
        });
    });
}

function buildMultiPagePdfBuffer(pagesText) {
    return Buffer.from(
        execFileSync(
            PYTHON_BIN,
            [
                '-c',
                'import fitz, json, sys; d=fitz.open();\n'
                + 'for txt in json.loads(sys.argv[1]):\n'
                + '    p=d.new_page(width=595,height=842); p.insert_text((50,72), txt)\n'
                + 'sys.stdout.buffer.write(d.tobytes())',
                JSON.stringify(pagesText),
            ],
            { encoding: null }
        )
    );
}

async function waitForHttpHealth(url, timeoutMs = 25000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        try {
            const res = await fetch(url);
            if (res.ok) return;
        } catch (_e) {
            // waiting
        }
        await new Promise((r) => setTimeout(r, 250));
    }
    throw new Error(`Timed out waiting for HTTP service at ${url}`);
}

async function waitForJobTerminal(nodeBaseUrl, token, jobId, expectedStatus = 'completed', timeoutMs = 45000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        const res = await fetch(`${nodeBaseUrl}/api/v1/ingestion/jobs/${jobId}`, {
            headers: { Authorization: `Bearer ${token}` },
        });
        assert.equal(res.status, 200);
        const body = await res.json();
        const job = body.data.job;
        if (job.status === expectedStatus) {
            return job;
        }
        if (job.status === 'failed' && expectedStatus !== 'failed') {
            throw new Error(`Job ${jobId} failed unexpectedly: ${JSON.stringify(job)}`);
        }
        await new Promise((r) => setTimeout(r, 300));
    }
    throw new Error(`Timed out waiting for job ${jobId} to reach '${expectedStatus}'`);
}

describe('CoverageIQ Checkpoint 5.0 End-to-End Recommendations, Visibility & Attribute Extraction', () => {
    let pgPool;
    let adminPool;
    let uploadRoot;
    let aiPort;
    let nodePort;
    let aiBaseUrl;
    let nodeBaseUrl;
    let aiProc;
    let workerProc;
    let httpServer;
    let app;
    let dbMod;
    let redisMod;
    let rateLimitMod;
    let internalToken;
    let eligibleHealthPdf;
    let ineligibleExpensivePdf;
    let privateHealthPdf;

    before(async (t) => {
        const pgParsed = new URL(RUNTIME_DSN);
        const redisParsed = new URL(REDIS_URL);
        const pgUp = await isTcpReachable(pgParsed.hostname || '127.0.0.1', Number(pgParsed.port || 5432));
        const redisUp = await isTcpReachable(redisParsed.hostname || '127.0.0.1', Number(redisParsed.port || 6379));
        if (!pgUp || !redisUp) {
            t.skip('Real PostgreSQL 16 + pgvector and Redis 7 are required for Checkpoint 5.0 integration tests.');
            return;
        }

        execFileSync('npm', ['--prefix', SERVER_ROOT, 'run', 'migrate:up'], {
            env: { ...process.env, MIGRATION_DATABASE_URL: MIGRATION_DSN },
            stdio: 'pipe',
        });

        uploadRoot = await fsPromises.mkdtemp(path.join(os.tmpdir(), 'ciq-rec-e2e-'));
        internalToken = 'ciq-rec-py-token-50-' + crypto.randomBytes(6).toString('hex');
        aiPort = await findFreePort();
        nodePort = await findFreePort();
        aiBaseUrl = `http://127.0.0.1:${aiPort}`;
        nodeBaseUrl = `http://127.0.0.1:${nodePort}`;

        process.env.DATABASE_URL = RUNTIME_DSN;
        process.env.REDIS_URL = REDIS_URL;
        process.env.AI_SERVICE_URL = aiBaseUrl;
        process.env.INTERNAL_SERVICE_TOKEN = internalToken;
        process.env.POLICY_UPLOAD_ROOT = uploadRoot;
        process.env.ACCESS_TOKEN_SECRET = 'test-access-secret-50';
        process.env.REFRESH_TOKEN_SECRET = 'test-refresh-secret-50';
        process.env.ACCESS_TOKEN_EXPIRY = '15m';
        process.env.REFRESH_TOKEN_EXPIRY = '1d';
        process.env.MAX_PDF_SIZE_BYTES = String(5 * 1024 * 1024);
        process.env.UPLOAD_RATE_LIMIT_MAX = '100';
        process.env.RECOMMENDATION_HTTP_TIMEOUT_MS = '30000';

        adminPool = new Pool({ connectionString: MIGRATION_DSN, max: 4 });
        dbMod = require('../src/config/db');
        redisMod = require('../src/config/redis');
        pgPool = dbMod.pool;
        rateLimitMod = require('../src/middlewares/rateLimit.middleware');
        app = require('../src/app');

        eligibleHealthPdf = buildMultiPagePdfBuffer([
            'SECTION 1 PLAN SCHEDULE & ELIGIBILITY\n'
            + 'Entry Age: 18 to 65 years.\n'
            + 'Lifelong renewal available.\n'
            + 'Coverage Mode: Individual and Family Floater up to 4 family members.\n'
            + 'Eligible applicants: Salaried and Self-Employed.\n'
            + 'Annual Premium: INR 12,000 to 18,000.\n'
            + 'Sum Insured: INR 500,000 to 1,500,000.\n'
            + 'Policy Tenure: 1 to 3 years.',
            'SECTION 2 COVERAGE BENEFITS & EXCLUSIONS\n'
            + 'In-patient hospitalization and daycare procedures covered with cashless facility\n'
            + 'across 7,500 network hospitals Pan-India. Initial waiting period: 30 days.\n'
            + 'Pre-existing conditions covered after 24 months.\n'
            + 'EXCLUSIONS: Cosmetic surgery is excluded and not covered.',
        ]);

        ineligibleExpensivePdf = buildMultiPagePdfBuffer([
            'SECTION 1 LUXURY HEALTH PLAN\n'
            + 'Entry Age: 18 to 65 years.\n'
            + 'Coverage Mode: Individual and Family Floater.\n'
            + 'Eligible applicants: Salaried.\n'
            + 'Annual Premium: INR 80,000 to 120,000.\n'
            + 'Sum Insured: INR 500,000 to 1,000,000.\n'
            + 'Policy Tenure: 1 to 2 years.',
        ]);

        privateHealthPdf = buildMultiPagePdfBuffer([
            'SECTION 1 INTERNAL PRIVATE HEALTH PLAN\n'
            + 'Entry Age: 18 to 65 years.\n'
            + 'Coverage Mode: Individual.\n'
            + 'Eligible applicants: Salaried.\n'
            + 'Annual Premium: INR 9,000 to 11,000.\n'
            + 'Sum Insured: INR 500,000 to 1,000,000.\n'
            + 'Policy Tenure: 1 to 2 years.',
        ]);

        const childEnv = {
            ...process.env,
            APP_ENV: 'development',
            DEBUG: 'true',
            USE_MOCK_STORAGE: 'false',
            DATABASE_URL: RUNTIME_DSN,
            INTERNAL_SERVICE_TOKEN: internalToken,
            REDIS_URL,
            CELERY_BROKER_URL: REDIS_URL,
            CELERY_RESULT_BACKEND: REDIS_URL,
            CELERY_INGESTION_QUEUE: 'ingestion',
            CELERY_OCR_QUEUE: 'ocr',
            CELERY_WORKER_CONCURRENCY: '1',
            CELERY_INGESTION_EMBEDDING_BATCH_SIZE: '1',
            POLICY_UPLOAD_ROOT: uploadRoot,
            EMBEDDING_PROVIDER: 'offline',
            LLM_PROVIDER: 'offline',
        };

        aiProc = spawn(
            PYTHON_BIN,
            ['-m', 'uvicorn', 'ai_service.main:app', '--host', '127.0.0.1', '--port', String(aiPort)],
            { cwd: REPO_ROOT, env: childEnv, stdio: ['ignore', 'pipe', 'pipe'] }
        );
        await waitForHttpHealth(`${aiBaseUrl}/ai/health`, 25000);

        workerProc = spawn(
            PYTHON_BIN,
            [
                '-m',
                'celery',
                '-A',
                'ai_service.workers.celery_app:celery_app',
                'worker',
                '-P',
                'solo',
                '-c',
                '1',
                '-Q',
                'ingestion,ocr',
                '--loglevel=WARNING',
                '-n',
                `ciq-rec-e2e-${process.pid}@localhost`,
            ],
            { cwd: REPO_ROOT, env: childEnv, stdio: ['ignore', 'pipe', 'pipe'] }
        );
        await new Promise((r) => setTimeout(r, 2500));

        await new Promise((resolve) => {
            httpServer = app.listen(nodePort, '127.0.0.1', resolve);
        });
    });

    after(async () => {
        if (httpServer) {
            await new Promise((resolve) => httpServer.close(resolve));
        }
        if (workerProc && workerProc.exitCode === null) {
            workerProc.kill('SIGTERM');
            await new Promise((r) => workerProc.once('exit', r));
        }
        if (aiProc && aiProc.exitCode === null) {
            aiProc.kill('SIGTERM');
            await new Promise((r) => aiProc.once('exit', r));
        }
        if (adminPool) await adminPool.end();
        if (pgPool) await pgPool.end();
        if (redisMod) await redisMod.closeRedisClients();
        if (uploadRoot) {
            await fsPromises.rm(uploadRoot, { recursive: true, force: true });
        }
    });

    beforeEach(async () => {
        if (!adminPool) return;
        await adminPool.query(
            'TRUNCATE TABLE chat_messages, audit_logs, refresh_tokens, organization_members, organizations, users, '
            + 'chunk_embeddings, policy_chunks, ingestion_jobs, policy_documents, policies CASCADE'
        );
        await rateLimitMod.uploadRateLimiter.reset();
        app.locals.ingestionStorageOptions = {};
        app.locals.ingestionEnqueueOptions = {};
        app.locals.recommendationServiceOptions = {};
        app.locals.recommendationOptions = {};
    });

    async function registerUser({ fullName, email, password = 'Password123!', role = 'user' }) {
        const res = await fetch(`${nodeBaseUrl}/api/v1/users/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fullName, email, password, role }),
        });
        assert.equal(res.status, 201, `register failed for ${email}`);
        const payload = await res.json();
        return {
            user: payload.data.user,
            token: payload.data.accessToken,
        };
    }

    async function createPolicyViaApi(token, { title, insurer_name = 'Shield Test Insurer', policy_type = 'health', visibility }) {
        const res = await fetch(`${nodeBaseUrl}/api/v1/policies`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                title,
                insurer_name,
                policy_type,
                ...(visibility ? { visibility } : {}),
            }),
        });
        return { status: res.status, body: await res.json() };
    }

    async function uploadAndWaitCompleted({ policyId, token, pdfBuffer, filename = 'policy.pdf' }) {
        const form = new FormData();
        form.append('file', new Blob([pdfBuffer], { type: 'application/pdf' }), filename);
        const upRes = await fetch(`${nodeBaseUrl}/api/v1/policies/${policyId}/documents`, {
            method: 'POST',
            headers: { Authorization: `Bearer ${token}` },
            body: form,
        });
        assert.equal(upRes.status, 201);
        const upBody = await upRes.json();
        const job = await waitForJobTerminal(nodeBaseUrl, token, upBody.data.job.id, 'completed');
        return { job, document: upBody.data.document };
    }

    it('verifies policy_attributes migration down/up redo and runtime least-privilege grants', async () => {
        execFileSync('npm', ['--prefix', SERVER_ROOT, 'run', 'migrate:down'], {
            env: { ...process.env, MIGRATION_DATABASE_URL: MIGRATION_DSN },
            stdio: 'pipe',
        });
        execFileSync('npm', ['--prefix', SERVER_ROOT, 'run', 'migrate:up'], {
            env: { ...process.env, MIGRATION_DATABASE_URL: MIGRATION_DSN },
            stdio: 'pipe',
        });

        const tableCheck = await pgPool.query(
            `
            SELECT table_name,
                   has_table_privilege('coverageiq_runtime', table_name, 'SELECT') AS can_select,
                   has_table_privilege('coverageiq_runtime', table_name, 'INSERT') AS can_insert,
                   has_table_privilege('coverageiq_runtime', table_name, 'UPDATE') AS can_update,
                   has_table_privilege('coverageiq_runtime', table_name, 'DELETE') AS can_delete
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name IN ('policy_attributes', 'ranking_features')
            ORDER BY table_name
            `
        );
        assert.equal(tableCheck.rowCount, 2);
        for (const row of tableCheck.rows) {
            assert.equal(row.can_select, true);
            assert.equal(row.can_insert, true);
            assert.equal(row.can_update, true);
            assert.equal(row.can_delete, true);
        }
    });

    it('enforces auth, scope spoof blocking, RBAC visibility guards, and resilient AI proxy errors', async () => {
        // 1. Unauthenticated /recommendations -> 401
        const unauthRes = await fetch(`${nodeBaseUrl}/api/v1/recommendations`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                profile: { policy_type: 'health', age: 30, budget_max: 20000, min_sum_insured: 500000 },
            }),
        });
        assert.equal(unauthRes.status, 401);

        const regularUser = await registerUser({
            fullName: 'Regular Buyer',
            email: 'buyer@rec.test',
            role: 'user',
        });
        const orgA = await registerUser({
            fullName: 'Org Alpha',
            email: 'orga@rec.test',
            role: 'organisation',
        });
        const orgB = await registerUser({
            fullName: 'Org Beta',
            email: 'orgb@rec.test',
            role: 'organisation',
        });

        // 2. Client-spoofed scope rejected at Express boundary -> 400
        const spoofRes = await fetch(`${nodeBaseUrl}/api/v1/recommendations`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${regularUser.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                scope: { user_id: 'forged-admin', role: 'admin', owner_scope: 'org:any' },
                profile: { policy_type: 'health', age: 30, budget_max: 20000, min_sum_insured: 500000 },
            }),
        });
        assert.equal(spoofRes.status, 400);

        // 3. Regular user cannot create catalog policy (403) and cannot toggle catalog visibility (403)
        const userCatalogCreate = await createPolicyViaApi(regularUser.token, {
            title: 'User Fake Catalog',
            visibility: 'catalog',
        });
        assert.equal(userCatalogCreate.status, 403);

        const orgDraft = await createPolicyViaApi(orgA.token, {
            title: 'Org Draft Plan',
        });
        assert.equal(orgDraft.status, 201);
        assert.equal(orgDraft.body.data.policy.visibility, 'private');

        const userToggle = await fetch(
            `${nodeBaseUrl}/api/v1/policies/${orgDraft.body.data.policy.id}/visibility`,
            {
                method: 'PATCH',
                headers: {
                    Authorization: `Bearer ${regularUser.token}`,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ visibility: 'catalog' }),
            }
        );
        assert.equal(userToggle.status, 403);

        const orgBToggle = await fetch(
            `${nodeBaseUrl}/api/v1/policies/${orgDraft.body.data.policy.id}/visibility`,
            {
                method: 'PATCH',
                headers: {
                    Authorization: `Bearer ${orgB.token}`,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ visibility: 'catalog' }),
            }
        );
        assert.equal(orgBToggle.status, 403);

        // Draft policy without active completed document cannot be published -> 409
        const draftPublish = await fetch(
            `${nodeBaseUrl}/api/v1/policies/${orgDraft.body.data.policy.id}/visibility`,
            {
                method: 'PATCH',
                headers: {
                    Authorization: `Bearer ${orgA.token}`,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ visibility: 'catalog' }),
            }
        );
        assert.equal(draftPublish.status, 409);

        // 4. Upstream AI timeout -> 504
        app.locals.recommendationServiceOptions = {
            fetchImpl: async (_url, init) => new Promise((_resolve, reject) => {
                init.signal.addEventListener('abort', () => {
                    const err = new Error('aborted');
                    err.name = 'AbortError';
                    reject(err);
                });
            }),
            timeoutMs: 25,
        };
        const timeoutRes = await fetch(`${nodeBaseUrl}/api/v1/recommendations`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${regularUser.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                profile: { policy_type: 'health', age: 30, budget_max: 20000, min_sum_insured: 500000 },
            }),
        });
        assert.equal(timeoutRes.status, 504);

        // 5. Upstream connection refused -> 503
        const deadPort = await findFreePort();
        app.locals.recommendationServiceOptions = { aiServiceUrl: `http://127.0.0.1:${deadPort}` };
        const downRes = await fetch(`${nodeBaseUrl}/api/v1/recommendations`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${regularUser.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                profile: { policy_type: 'health', age: 30, budget_max: 20000, min_sum_insured: 500000 },
            }),
        });
        assert.equal(downRes.status, 503);
    });

    it('executes full PDF ingestion -> attribute extraction -> catalog publish -> recommendations -> strict NO_MATCHING_POLICIES & citation check', async () => {
        const orgUser = await registerUser({
            fullName: 'Insurer Catalog Admin',
            email: 'catalog@rec.test',
            role: 'organisation',
        });
        const buyer = await registerUser({
            fullName: 'End Consumer',
            email: 'consumer@rec.test',
            role: 'user',
        });

        // 1. Create eligible plan (initially private, then published explicitly after ingestion)
        const eligibleCreate = await createPolicyViaApi(orgUser.token, {
            title: 'Shield Prime Health',
            insurer_name: 'Shield General Insurance',
            policy_type: 'health',
        });
        assert.equal(eligibleCreate.status, 201);
        const eligibleId = eligibleCreate.body.data.policy.id;
        assert.equal(eligibleCreate.body.data.policy.visibility, 'private');

        const { document: eligibleDoc } = await uploadAndWaitCompleted({
            policyId: eligibleId,
            token: orgUser.token,
            pdfBuffer: eligibleHealthPdf,
            filename: 'shield-prime-health.pdf',
        });

        // Verify Celery pipeline populated policy_attributes automatically (still not catalog-eligible while private!)
        const prePubAttrRes = await pgPool.query(
            `
            SELECT extraction_status, is_catalog_eligible, annual_premium_min, copay_pct,
                   source_chunk_ids, exclusions
            FROM policy_attributes
            WHERE policy_id = $1 AND document_id = $2
            `,
            [eligibleId, eligibleDoc.id]
        );
        assert.equal(prePubAttrRes.rowCount, 1);
        assert.equal(prePubAttrRes.rows[0].extraction_status, 'completed');
        assert.equal(prePubAttrRes.rows[0].is_catalog_eligible, false);
        assert.equal(Number(prePubAttrRes.rows[0].annual_premium_min), 12000);
        assert.equal(prePubAttrRes.rows[0].copay_pct, null);
        assert.ok(Array.isArray(prePubAttrRes.rows[0].exclusions));
        assert.ok(prePubAttrRes.rows[0].exclusions.includes('cosmetic_surgery'));

        // Publish eligible policy to catalog via PATCH visibility
        const pubRes = await fetch(`${nodeBaseUrl}/api/v1/policies/${eligibleId}/visibility`, {
            method: 'PATCH',
            headers: {
                Authorization: `Bearer ${orgUser.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ visibility: 'catalog' }),
        });
        assert.equal(pubRes.status, 200);
        const pubBody = await pubRes.json();
        assert.equal(pubBody.data.policy.visibility, 'catalog');
        assert.equal(pubBody.data.policy.is_catalog_eligible, true);

        // 2. Create & publish expensive ineligible catalog plan
        const expensiveCreate = await createPolicyViaApi(orgUser.token, {
            title: 'Luxury Executive Health',
            insurer_name: 'Luxury Assurance',
            policy_type: 'health',
            visibility: 'catalog',
        });
        assert.equal(expensiveCreate.status, 201);
        const expensiveId = expensiveCreate.body.data.policy.id;
        await uploadAndWaitCompleted({
            policyId: expensiveId,
            token: orgUser.token,
            pdfBuffer: ineligibleExpensivePdf,
            filename: 'luxury-health.pdf',
        });

        // 3. Create & upload private org plan (stays visibility='private'!)
        const privateCreate = await createPolicyViaApi(orgUser.token, {
            title: 'Internal Employee Only Plan',
            insurer_name: 'Internal Health',
            policy_type: 'health',
        });
        assert.equal(privateCreate.status, 201);
        const privateId = privateCreate.body.data.policy.id;
        await uploadAndWaitCompleted({
            policyId: privateId,
            token: orgUser.token,
            pdfBuffer: privateHealthPdf,
            filename: 'private-health.pdf',
        });

        // 4. Query recommendations as unrelated regular user
        const recRes = await fetch(`${nodeBaseUrl}/api/v1/recommendations`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${buyer.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                profile: {
                    policy_type: 'health',
                    age: 32,
                    budget_min: 5000,
                    budget_max: 25000,
                    min_sum_insured: 500000,
                    applicant_type: 'salaried',
                    coverage_mode: 'individual',
                },
                top_n: 5,
                ranking_model: 'auto',
            }),
        });
        assert.equal(recRes.status, 200);
        const recBody = await recRes.json();
        const data = recBody.data;
        assert.equal(data.status, 'RECOMMENDATIONS_READY');
        // Private policy excluded completely -> total_candidates = 2 (eligible + expensive)
        assert.equal(data.filter_report.total_candidates, 2);
        assert.equal(data.filter_report.passed_count, 1);
        assert.equal(data.filter_report.eliminated_count, 1);
        assert.equal(data.recommendations.length, 1);

        const top = data.recommendations[0];
        assert.equal(top.policy_id, eligibleId);
        assert.equal(top.policy_name, 'Shield Prime Health');
        assert.ok(top.missing_attributes.includes('copay_pct'));
        assert.ok(top.exclusions.includes('cosmetic_surgery'));
        assert.ok(Array.isArray(top.citations) && top.citations.length >= 1);

        // Verify every citation maps to real persisted policy_chunks
        const chunkCheck = await pgPool.query(
            'SELECT chunk_key FROM policy_chunks WHERE policy_id = $1 AND document_id = $2',
            [eligibleId, eligibleDoc.id]
        );
        const validChunks = new Set(chunkCheck.rows.map((r) => r.chunk_key));
        for (const cit of top.citations) {
            assert.ok(validChunks.has(cit.chunk_id), `Expected persisted chunk_id ${cit.chunk_id}`);
        }

        // Verify eliminated plan recorded accurately and private plan leaked nowhere
        const elimIds = data.filter_report.eliminations.map((e) => e.policy_id);
        assert.ok(elimIds.includes(expensiveId));
        assert.ok(!elimIds.includes(privateId));

        // 5. Query with too-tight budget -> deterministic NO_MATCHING_POLICIES
        const noMatchRes = await fetch(`${nodeBaseUrl}/api/v1/recommendations`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${buyer.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                profile: {
                    policy_type: 'health',
                    age: 32,
                    budget_min: 1000,
                    budget_max: 4000,
                    min_sum_insured: 500000,
                    applicant_type: 'salaried',
                    coverage_mode: 'individual',
                },
            }),
        });
        assert.equal(noMatchRes.status, 200);
        const noMatchBody = await noMatchRes.json();
        assert.equal(noMatchBody.data.status, 'NO_MATCHING_POLICIES');
        assert.deepEqual(noMatchBody.data.recommendations, []);
        assert.equal(noMatchBody.data.filter_report.eliminated_count, 2);

        // 6. Audit logs persisted visibility change
        const auditRes = await pgPool.query(
            "SELECT action, details FROM audit_logs WHERE resource_id = $1 AND action = 'policy.visibility.updated'",
            [eligibleId]
        );
        assert.equal(auditRes.rowCount, 1);

        // 7. Toggle back to private & verify it disappears from catalog candidates
        const unpubRes = await fetch(`${nodeBaseUrl}/api/v1/policies/${eligibleId}/visibility`, {
            method: 'PATCH',
            headers: {
                Authorization: `Bearer ${orgUser.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ visibility: 'private' }),
        });
        assert.equal(unpubRes.status, 200);
        assert.equal((await unpubRes.json()).data.policy.visibility, 'private');

        // 8. Delete policy_attributes row and re-publish -> Express backfills via /internal/recommendations/extract-attributes
        await adminPool.query(
            'DELETE FROM policy_attributes WHERE policy_id = $1 AND document_id = $2',
            [eligibleId, eligibleDoc.id]
        );
        const repubRes = await fetch(`${nodeBaseUrl}/api/v1/policies/${eligibleId}/visibility`, {
            method: 'PATCH',
            headers: {
                Authorization: `Bearer ${orgUser.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ visibility: 'catalog' }),
        });
        assert.equal(repubRes.status, 200);
        assert.equal((await repubRes.json()).data.policy.attribute_extraction_status, 'completed');
    });
});
