const { describe, it, before, after, beforeEach } = require('node:test');
const assert = require('node:assert/strict');
const { execFileSync, spawn } = require('node:child_process');
const crypto = require('node:crypto');
const fs = require('node:fs');
const fsPromises = require('node:fs/promises');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const { URL } = require('node:url');

const { Pool } = require('pg');

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const SERVER_ROOT = path.resolve(__dirname, '..');
const PY311_BIN = path.join(os.homedir(), 'ai-env-py311', 'bin', 'python');
const PYTHON_BIN = process.env.PYTHON_BIN || (fs.existsSync(PY311_BIN) ? PY311_BIN : path.join(os.homedir(), 'ai-env', 'bin', 'python'));

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
                'import fitz, json, sys\n'
                + 'd = fitz.open()\n'
                + 'for text in json.loads(sys.argv[1]):\n'
                + '    p = d.new_page(width=595, height=842)\n'
                + '    p.insert_text((50, 72), text)\n'
                + 'sys.stdout.buffer.write(d.tobytes())\n',
                JSON.stringify(pagesText),
            ],
            { encoding: null }
        )
    );
}

async function waitForHttpHealth(url, timeoutMs = 20000) {
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
        const payload = await res.json();
        const job = payload.data.job;
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

describe('CoverageIQ Checkpoint 4.9 End-to-End Q&A Proxy, Persistence & Security', () => {
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
    let policyPdfBuffer;

    before(async (t) => {
        const pgParsed = new URL(RUNTIME_DSN);
        const redisParsed = new URL(REDIS_URL);
        const pgUp = await isTcpReachable(pgParsed.hostname || '127.0.0.1', Number(pgParsed.port || 5432));
        const redisUp = await isTcpReachable(redisParsed.hostname || '127.0.0.1', Number(redisParsed.port || 6379));
        if (!pgUp || !redisUp) {
            t.skip('Real PostgreSQL 16 + pgvector and Redis 7 are required for Checkpoint 4.9 integration tests.');
            return;
        }

        execFileSync('npm', ['--prefix', SERVER_ROOT, 'run', 'migrate:up'], {
            env: { ...process.env, MIGRATION_DATABASE_URL: MIGRATION_DSN },
            stdio: 'pipe',
        });

        uploadRoot = await fsPromises.mkdtemp(path.join(os.tmpdir(), 'ciq-qa-e2e-'));
        internalToken = 'ciq-qa-py-token-49-' + crypto.randomBytes(6).toString('hex');
        aiPort = await findFreePort();
        nodePort = await findFreePort();
        aiBaseUrl = `http://127.0.0.1:${aiPort}`;
        nodeBaseUrl = `http://127.0.0.1:${nodePort}`;

        process.env.DATABASE_URL = RUNTIME_DSN;
        process.env.REDIS_URL = REDIS_URL;
        process.env.AI_SERVICE_URL = aiBaseUrl;
        process.env.INTERNAL_SERVICE_TOKEN = internalToken;
        process.env.POLICY_UPLOAD_ROOT = uploadRoot;
        process.env.ACCESS_TOKEN_SECRET = 'test-access-secret-49';
        process.env.REFRESH_TOKEN_SECRET = 'test-refresh-secret-49';
        process.env.ACCESS_TOKEN_EXPIRY = '15m';
        process.env.REFRESH_TOKEN_EXPIRY = '1d';
        process.env.MAX_PDF_SIZE_BYTES = String(5 * 1024 * 1024);
        process.env.UPLOAD_RATE_LIMIT_MAX = '100';
        process.env.QA_HTTP_TIMEOUT_MS = '30000';

        adminPool = new Pool({ connectionString: MIGRATION_DSN, max: 4 });
        dbMod = require('../src/config/db');
        redisMod = require('../src/config/redis');
        pgPool = dbMod.pool;
        rateLimitMod = require('../src/middlewares/rateLimit.middleware');
        app = require('../src/app');

        policyPdfBuffer = buildMultiPagePdfBuffer([
            'SECTION 2 COVERAGE BENEFITS\n'
            + 'Section 2.1 In-Patient Hospitalization Coverage\n'
            + 'In-patient hospitalization expenses for minimum 24 consecutive\n'
            + 'hours are covered up to Sum Insured INR 500,000.\n'
            + 'Pre-hospitalization medical expenses up to 30 days\n'
            + 'are covered.',
            'SECTION 3 EXCLUSIONS\n'
            + 'Section 3.1 Cosmetic Surgery\n'
            + 'Cosmetic surgery, aesthetic procedures, and non-medically\n'
            + 'necessary enhancements are excluded and not covered.',
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
                `ciq-qa-e2e-${process.pid}@localhost`,
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
        app.locals.qaServiceOptions = {};
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

    async function seedPolicy({ externalKey, title, ownerId, ownerScope, visibility = 'private', status = 'draft' }) {
        const res = await pgPool.query(
            `
            INSERT INTO policies (external_policy_key, title, policy_type, visibility, owner_id, owner_scope, status)
            VALUES ($1, $2, 'health', $3, $4, $5, $6)
            RETURNING *
            `,
            [externalKey, title, visibility, ownerId, ownerScope, status]
        );
        return res.rows[0];
    }

    async function uploadAndWaitCompleted({ policyId, token, filename = 'policy.pdf' }) {
        const form = new FormData();
        form.append('file', new Blob([policyPdfBuffer], { type: 'application/pdf' }), filename);
        const upRes = await fetch(`${nodeBaseUrl}/api/v1/policies/${policyId}/documents`, {
            method: 'POST',
            headers: { Authorization: `Bearer ${token}` },
            body: form,
        });
        assert.equal(upRes.status, 201);
        const upBody = await upRes.json();
        return waitForJobTerminal(nodeBaseUrl, token, upBody.data.job.id, 'completed');
    }

    it('verifies chat_messages migration down/up redo and runtime least-privilege grants', async () => {
        execFileSync('npx', ['node-pg-migrate', 'down', '2', '-d', 'MIGRATION_DATABASE_URL', '-m', 'migrations', '-t', 'pgmigrations'], {
            cwd: SERVER_ROOT,
            env: { ...process.env, MIGRATION_DATABASE_URL: MIGRATION_DSN },
            stdio: 'pipe',
        });

        try {
            const afterDown = await adminPool.query(
                "SELECT to_regclass('public.chat_messages') AS reg"
            );
            assert.equal(afterDown.rows[0].reg, null);
        } finally {
            execFileSync('npm', ['--prefix', SERVER_ROOT, 'run', 'migrate:up'], {
                env: { ...process.env, MIGRATION_DATABASE_URL: MIGRATION_DSN },
                stdio: 'pipe',
            });
        }

        const afterUp = await adminPool.query(
            "SELECT to_regclass('public.chat_messages') AS reg"
        );
        assert.equal(afterUp.rows[0].reg, 'chat_messages');

        // Runtime DB role must NOT have DDL privileges on chat_messages
        await assert.rejects(
            pgPool.query('DROP TABLE chat_messages'),
            (err) => err && /must be owner of table|permission denied/i.test(err.message)
        );
    });

    it('enforces auth, input validation, draft/archived guards, and rejects client-supplied scope', async () => {
        const alice = await registerUser({ fullName: 'Alice One', email: 'alice.sec@coverageiq.test' });
        const bob = await registerUser({ fullName: 'Bob One', email: 'bob.sec@coverageiq.test' });

        const draftPol = await seedPolicy({
            externalKey: 'pol_qa_draft',
            title: 'Draft Only Policy',
            ownerId: alice.user.id,
            ownerScope: `user:${alice.user.id}`,
            status: 'draft',
        });
        const archivedPol = await seedPolicy({
            externalKey: 'pol_qa_arch',
            title: 'Archived Policy',
            ownerId: alice.user.id,
            ownerScope: `user:${alice.user.id}`,
            status: 'archived',
        });

        const unauthAsk = await fetch(`${nodeBaseUrl}/api/v1/policies/${draftPol.id}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: 'Is hospitalization covered?' }),
        });
        assert.equal(unauthAsk.status, 401);

        const unauthHist = await fetch(`${nodeBaseUrl}/api/v1/policies/${draftPol.id}/qa-history`);
        assert.equal(unauthHist.status, 401);

        const badUuid = await fetch(`${nodeBaseUrl}/api/v1/policies/not-a-uuid/ask`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${alice.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: 'Is hospitalization covered?' }),
        });
        assert.equal(badUuid.status, 400);

        const missingPol = await fetch(`${nodeBaseUrl}/api/v1/policies/${crypto.randomUUID()}/ask`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${alice.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: 'Is hospitalization covered?' }),
        });
        assert.equal(missingPol.status, 404);

        const scopeSpoof = await fetch(`${nodeBaseUrl}/api/v1/policies/${draftPol.id}/ask`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${alice.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: 'Is hospitalization covered?',
                scope: { role: 'admin', owner_scope: 'org:fake' },
            }),
        });
        assert.equal(scopeSpoof.status, 400);

        const userFilterSpoof = await fetch(
            `${nodeBaseUrl}/api/v1/policies/${draftPol.id}/qa-history?user_id=${bob.user.id}`,
            { headers: { Authorization: `Bearer ${alice.token}` } }
        );
        assert.equal(userFilterSpoof.status, 400);

        const shortQ = await fetch(`${nodeBaseUrl}/api/v1/policies/${draftPol.id}/ask`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${alice.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: 'hi' }),
        });
        assert.equal(shortQ.status, 400);

        const crossTenantAsk = await fetch(`${nodeBaseUrl}/api/v1/policies/${draftPol.id}/ask`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${bob.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: 'Is hospitalization covered?' }),
        });
        assert.equal(crossTenantAsk.status, 403);

        const draftAsk = await fetch(`${nodeBaseUrl}/api/v1/policies/${draftPol.id}/ask`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${alice.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: 'Is hospitalization covered?' }),
        });
        assert.equal(draftAsk.status, 409);

        const archivedAsk = await fetch(`${nodeBaseUrl}/api/v1/policies/${archivedPol.id}/ask`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${alice.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: 'Is hospitalization covered?' }),
        });
        assert.equal(archivedAsk.status, 409);
    });

    it('executes full Upload -> Celery -> Ask -> PG chat_messages persistence -> Chronological History & Catalog Isolation', async () => {
        const alice = await registerUser({ fullName: 'Alice Owner', email: 'alice.e2e@coverageiq.test' });
        const bob = await registerUser({ fullName: 'Bob Peer', email: 'bob.e2e@coverageiq.test' });
        const orgOwner = await registerUser({
            fullName: 'Carol Org',
            email: 'carol.org@coverageiq.test',
            role: 'organisation',
        });

        // 1. Alice's private policy upload + Celery ingestion
        const alicePolicy = await seedPolicy({
            externalKey: 'pol_alice_qa_1',
            title: 'Alice Gold Health',
            ownerId: alice.user.id,
            ownerScope: `user:${alice.user.id}`,
            status: 'draft',
        });
        const aliceJob = await uploadAndWaitCompleted({
            policyId: alicePolicy.id,
            token: alice.token,
            filename: 'alice_gold.pdf',
        });

        // 2. Ask covered question -> verified answer + PG persistence
        const ask1Res = await fetch(`${nodeBaseUrl}/api/v1/policies/${alicePolicy.id}/ask`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${alice.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: 'Is in-patient hospitalization covered?' }),
        });
        assert.equal(ask1Res.status, 200);
        const ask1 = (await ask1Res.json()).data;

        assert.ok(ask1.conversation_id);
        assert.equal(ask1.document.id, aliceJob.document_id);
        assert.equal(ask1.answer.decision, 'COVERED');
        assert.ok(Array.isArray(ask1.answer.citations) && ask1.answer.citations.length > 0);

        const firstCitation = ask1.answer.citations[0];
        assert.equal(firstCitation.page, 1);

        // Cross-check cited chunk_id (policy_chunks.chunk_key) truly belongs to Alice's ingested document in PostgreSQL
        const chunkCheck = await pgPool.query(
            'SELECT chunk_key, page_number FROM policy_chunks WHERE document_id = $1 AND chunk_key = $2',
            [aliceJob.document_id, firstCitation.chunk_id]
        );
        assert.equal(chunkCheck.rowCount, 1);
        assert.equal(chunkCheck.rows[0].page_number, 1);

        // 3. Follow-up question on same conversation thread
        const ask2Res = await fetch(`${nodeBaseUrl}/api/v1/policies/${alicePolicy.id}/ask`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${alice.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: 'Can I claim cosmetic surgery under this policy?',
                conversation_id: ask1.conversation_id,
            }),
        });
        assert.equal(ask2Res.status, 200);
        const ask2 = (await ask2Res.json()).data;
        assert.equal(ask2.conversation_id, ask1.conversation_id);
        assert.equal(ask2.answer.decision, 'NOT COVERED');

        // 4. Verify chat_messages database rows and chronological history API
        const dbRows = await pgPool.query(
            'SELECT role, decision, document_id, owner_scope, conversation_id FROM chat_messages WHERE policy_id = $1 ORDER BY created_at ASC, id ASC',
            [alicePolicy.id]
        );
        assert.equal(dbRows.rowCount, 4);
        assert.deepEqual(
            dbRows.rows.map((r) => r.role),
            ['user', 'assistant', 'user', 'assistant']
        );
        assert.ok(dbRows.rows.every((r) => r.owner_scope === `user:${alice.user.id}`));
        assert.ok(dbRows.rows.every((r) => r.document_id === aliceJob.document_id));

        const histRes = await fetch(
            `${nodeBaseUrl}/api/v1/policies/${alicePolicy.id}/qa-history?page=1&limit=10`,
            { headers: { Authorization: `Bearer ${alice.token}` } }
        );
        assert.equal(histRes.status, 200);
        const histBody = (await histRes.json()).data;
        assert.equal(histBody.total_questions, 2);
        assert.equal(histBody.items.length, 4);
        assert.equal(histBody.items[0].content, 'Is in-patient hospitalization covered?');
        assert.equal(histBody.items[1].decision, 'COVERED');
        assert.equal(histBody.items[2].content, 'Can I claim cosmetic surgery under this policy?');
        assert.equal(histBody.items[3].decision, 'NOT COVERED');

        // Bob cannot read Alice's private policy Q&A history
        const bobPrivateHist = await fetch(
            `${nodeBaseUrl}/api/v1/policies/${alicePolicy.id}/qa-history`,
            { headers: { Authorization: `Bearer ${bob.token}` } }
        );
        assert.equal(bobPrivateHist.status, 403);

        // 5. Organization catalog policy tests ('organisation' -> 'organization' mapping + per-user history isolation)
        const orgRes = await pgPool.query(
            `INSERT INTO organizations (name, slug, created_by) VALUES ('ShieldOrg', 'shield-org', $1) RETURNING *`,
            [orgOwner.user.id]
        );
        const orgId = orgRes.rows[0].id;
        await pgPool.query(
            `INSERT INTO organization_members (organization_id, user_id, role) VALUES ($1, $2, 'owner')`,
            [orgId, orgOwner.user.id]
        );
        const catalogPolicy = await seedPolicy({
            externalKey: 'pol_org_catalog_1',
            title: 'Public Catalog Health Plan',
            ownerId: orgOwner.user.id,
            ownerScope: `org:${orgId}`,
            visibility: 'catalog',
            status: 'draft',
        });
        await uploadAndWaitCompleted({
            policyId: catalogPolicy.id,
            token: orgOwner.token,
            filename: 'catalog_health.pdf',
        });

        const aliceCatalogAsk = await fetch(`${nodeBaseUrl}/api/v1/policies/${catalogPolicy.id}/ask`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${alice.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: 'Is hospitalization covered in catalog plan?' }),
        });
        assert.equal(aliceCatalogAsk.status, 200);

        const bobCatalogAsk = await fetch(`${nodeBaseUrl}/api/v1/policies/${catalogPolicy.id}/ask`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${bob.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: 'Does catalog plan cover cosmetic surgery?' }),
        });
        assert.equal(bobCatalogAsk.status, 200);

        // Carol ('organisation' role) asks as org owner too -> exercises 'organisation' -> 'organization' role map
        const carolOrgAsk = await fetch(`${nodeBaseUrl}/api/v1/policies/${catalogPolicy.id}/ask`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${orgOwner.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: 'Are pre-hospitalization expenses covered?' }),
        });
        assert.equal(carolOrgAsk.status, 200);

        const bobHistRes = await fetch(
            `${nodeBaseUrl}/api/v1/policies/${catalogPolicy.id}/qa-history`,
            { headers: { Authorization: `Bearer ${bob.token}` } }
        );
        assert.equal(bobHistRes.status, 200);
        const bobHist = (await bobHistRes.json()).data;
        assert.equal(bobHist.total_questions, 1);
        assert.equal(bobHist.items.length, 2);
        assert.equal(bobHist.items[0].content, 'Does catalog plan cover cosmetic surgery?');
        assert.ok(
            !bobHist.items.some((m) => m.content.includes('catalog plan') && m.content.includes('hospitalization')),
            'Bob must never see Alice chat history on shared catalog policy'
        );
    });

    it('persists ZERO chat_messages when AI service fails, times out, lacks auth, or leaks mock_mode in production', async () => {
        const alice = await registerUser({ fullName: 'Alice FailClosed', email: 'alice.fc@coverageiq.test' });
        const alicePolicy = await seedPolicy({
            externalKey: 'pol_alice_fc',
            title: 'Fail Closed Policy',
            ownerId: alice.user.id,
            ownerScope: `user:${alice.user.id}`,
            status: 'draft',
        });
        await uploadAndWaitCompleted({
            policyId: alicePolicy.id,
            token: alice.token,
            filename: 'failclosed.pdf',
        });

        // 1. Mismatched internal service token -> FastAPI returns 403
        app.locals.qaServiceOptions = { serviceToken: 'wrong-internal-token' };
        const badTokenRes = await fetch(`${nodeBaseUrl}/api/v1/policies/${alicePolicy.id}/ask`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${alice.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: 'Is hospitalization covered?' }),
        });
        assert.equal(badTokenRes.status, 403);

        // 2. Upstream timeout (AbortError) -> Express returns 504
        app.locals.qaServiceOptions = {
            fetchImpl: async () => {
                const err = new Error('The operation was aborted');
                err.name = 'AbortError';
                throw err;
            },
        };
        const timeoutRes = await fetch(`${nodeBaseUrl}/api/v1/policies/${alicePolicy.id}/ask`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${alice.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: 'Is hospitalization covered?' }),
        });
        assert.equal(timeoutRes.status, 504);

        // 3. Upstream 503 with sensitive secret -> sanitized 503
        app.locals.qaServiceOptions = {
            fetchImpl: async () => ({
                ok: false,
                status: 503,
                json: async () => ({ detail: 'nvapi-secret-leak database overload' }),
            }),
        };
        const upstream503 = await fetch(`${nodeBaseUrl}/api/v1/policies/${alicePolicy.id}/ask`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${alice.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: 'Is hospitalization covered?' }),
        });
        assert.equal(upstream503.status, 503);
        const body503 = await upstream503.text();
        assert.ok(!body503.includes('nvapi-secret-leak'));

        // 4. Malformed answer lacking valid citation provenance -> 502
        app.locals.qaServiceOptions = {
            fetchImpl: async () => ({
                ok: true,
                status: 200,
                json: async () => ({
                    decision: 'COVERED',
                    reason: 'Covered.',
                    conditions: [],
                    citations: [{ chunk_id: 'c1' }], // missing page & clause
                    evaluator_status: 'VERIFIED_PASS',
                    retry_count: 0,
                }),
            }),
        };
        const badShapeRes = await fetch(`${nodeBaseUrl}/api/v1/policies/${alicePolicy.id}/ask`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${alice.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: 'Is hospitalization covered?' }),
        });
        assert.equal(badShapeRes.status, 502);

        // 5. Production mock_mode rejection -> 503
        const prevNodeEnv = process.env.NODE_ENV;
        process.env.NODE_ENV = 'production';
        try {
            app.locals.qaServiceOptions = {
                fetchImpl: async () => ({
                    ok: true,
                    status: 200,
                    json: async () => ({
                        decision: 'COVERED',
                        reason: 'Mock response',
                        conditions: [],
                        citations: [],
                        evaluator_status: 'VERIFIED_PASS',
                        retry_count: 0,
                        metadata: { mock_mode: true, storage_mode: 'mock_memory' },
                    }),
                }),
            };
            const prodMockRes = await fetch(`${nodeBaseUrl}/api/v1/policies/${alicePolicy.id}/ask`, {
                method: 'POST',
                headers: {
                    Authorization: `Bearer ${alice.token}`,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: 'Is hospitalization covered?' }),
            });
            assert.equal(prodMockRes.status, 503);
        } finally {
            process.env.NODE_ENV = prevNodeEnv;
        }

        const chatCountRes = await pgPool.query(
            'SELECT COUNT(*)::int AS cnt FROM chat_messages WHERE policy_id = $1',
            [alicePolicy.id]
        );
        assert.equal(chatCountRes.rows[0].cnt, 0, 'Zero messages must persist when AI fails');
    });
});
