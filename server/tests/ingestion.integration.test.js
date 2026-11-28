const { describe, it, before, after, beforeEach } = require('node:test');
const assert = require('node:assert/strict');
const { execFileSync, spawn } = require('node:child_process');
const crypto = require('node:crypto');
const fs = require('node:fs/promises');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const { URL } = require('node:url');

const { Pool } = require('pg');

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const SERVER_ROOT = path.resolve(__dirname, '..');
const PYTHON_BIN = process.env.PYTHON_BIN || path.join(os.homedir(), 'ai-env', 'bin', 'python');

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

function buildValidPdfBuffer(bodyText) {
    return Buffer.from(
        execFileSync(
            PYTHON_BIN,
            [
                '-c',
                'import fitz, sys; d=fitz.open(); p=d.new_page(width=595,height=842); '
                + 'p.insert_text((50,72), sys.argv[1]); sys.stdout.buffer.write(d.tobytes())',
                bodyText,
            ],
            { encoding: null }
        )
    );
}

function buildEncryptedPdfBuffer() {
    return Buffer.from(
        execFileSync(
            PYTHON_BIN,
            [
                '-c',
                'import fitz, sys; d=fitz.open(); p=d.new_page(); p.insert_text((50,72), "Secret Policy"); '
                + 'sys.stdout.buffer.write(d.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="adm", user_pw="usr"))',
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
        const res = await fetch(`${nodeBaseUrl}/api/ingestion/jobs/${jobId}`, {
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

async function countFilesRecursive(dir) {
    let count = 0;
    const entries = await fs.readdir(dir, { withFileTypes: true }).catch(() => []);
    for (const entry of entries) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            count += await countFilesRecursive(full);
        } else if (entry.isFile()) {
            count += 1;
        }
    }
    return count;
}

describe('CoverageIQ Checkpoint 4.5 Node.js Upload + Job API + Real FastAPI/Celery/Redis/PG16', () => {
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
    let gatewaySvc;
    let rateLimitMod;
    let internalToken;
    let validPdf1;
    let validPdf2;
    let validPdf3;
    let encryptedPdf;

    before(async (t) => {
        const pgParsed = new URL(RUNTIME_DSN);
        const redisParsed = new URL(REDIS_URL);
        const pgUp = await isTcpReachable(pgParsed.hostname || '127.0.0.1', Number(pgParsed.port || 5432));
        const redisUp = await isTcpReachable(redisParsed.hostname || '127.0.0.1', Number(redisParsed.port || 6379));
        if (!pgUp || !redisUp) {
            t.skip('Real PostgreSQL 16 + pgvector and Redis 7 are required for Checkpoint 4.5 integration tests.');
            return;
        }

        // Run Node-owned migrations using schema owner role
        execFileSync('npm', ['--prefix', SERVER_ROOT, 'run', 'migrate:up'], {
            env: { ...process.env, MIGRATION_DATABASE_URL: MIGRATION_DSN },
            stdio: 'pipe',
        });

        uploadRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'ciq-node-uploads-'));
        internalToken = 'ciq-node-py-token-45-' + crypto.randomBytes(6).toString('hex');
        aiPort = await findFreePort();
        nodePort = await findFreePort();
        aiBaseUrl = `http://127.0.0.1:${aiPort}`;
        nodeBaseUrl = `http://127.0.0.1:${nodePort}`;

        // Set server environment BEFORE loading app, Redis & db pool
        process.env.DATABASE_URL = RUNTIME_DSN;
        process.env.REDIS_URL = REDIS_URL;
        process.env.AI_SERVICE_URL = aiBaseUrl;
        process.env.INTERNAL_SERVICE_TOKEN = internalToken;
        process.env.POLICY_UPLOAD_ROOT = uploadRoot;
        process.env.ACCESS_TOKEN_SECRET = 'test-access-secret-45';
        process.env.REFRESH_TOKEN_SECRET = 'test-refresh-secret-45';
        process.env.ACCESS_TOKEN_EXPIRY = '15m';
        process.env.REFRESH_TOKEN_EXPIRY = '1d';
        process.env.MAX_PDF_SIZE_BYTES = String(5 * 1024 * 1024);
        process.env.UPLOAD_RATE_LIMIT_MAX = '100';

        adminPool = new Pool({ connectionString: MIGRATION_DSN, max: 4 });
        dbMod = require('../src/config/db');
        redisMod = require('../src/config/redis');
        pgPool = dbMod.pool;
        gatewaySvc = require('../src/services/ingestionGateway.service');
        rateLimitMod = require('../src/middlewares/rateLimit.middleware');
        app = require('../src/app');

        validPdf1 = buildValidPdfBuffer(
            'SECTION 2 COVERAGE BENEFITS\n'
            + 'Section 2.1 In-Patient Hospitalization\n'
            + 'Hospitalization expenses for minimum 24 consecutive hours are covered up to Sum Insured.'
        );
        validPdf2 = buildValidPdfBuffer(
            'SECTION 2 COVERAGE BENEFITS - VERSION 2\n'
            + 'Section 2.1 Updated In-Patient Hospitalization\n'
            + 'Updated room rent limit is 2% of Sum Insured per day and ICU limit is 4% per day.'
        );
        validPdf3 = buildValidPdfBuffer(
            'SECTION 2 COVERAGE BENEFITS - VERSION 3\n'
            + 'Section 2.2 Emergency Ambulance Coverage\n'
            + 'Road ambulance charges up to INR 5,000 per hospitalization event are covered.'
        );
        encryptedPdf = buildEncryptedPdfBuffer();

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
                `ciq-node-e2e-${process.pid}@localhost`,
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
            await fs.rm(uploadRoot, { recursive: true, force: true });
        }
    });

    beforeEach(async () => {
        if (!adminPool) return;
        await adminPool.query(
            'TRUNCATE TABLE audit_logs, refresh_tokens, organization_members, organizations, users, '
            + 'chunk_embeddings, policy_chunks, ingestion_jobs, policy_documents, policies CASCADE'
        );
        await rateLimitMod.uploadRateLimiter.reset();
        app.locals.ingestionStorageOptions = {};
        app.locals.ingestionEnqueueOptions = {};
        const entries = await fs.readdir(uploadRoot).catch(() => []);
        for (const e of entries) {
            await fs.rm(path.join(uploadRoot, e), { recursive: true, force: true });
        }
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
            refreshToken: payload.data.refreshToken,
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

    async function uploadPdfForm({ policyId, token, buffer, filename = 'policy.pdf', mimeType = 'application/pdf', extraFields = {}, prefix = '/api' }) {
        const form = new FormData();
        const blob = new Blob([buffer], { type: mimeType });
        form.append('file', blob, filename);
        for (const [k, v] of Object.entries(extraFields)) {
            form.append(k, String(v));
        }
        return fetch(`${nodeBaseUrl}${prefix}/policies/${policyId}/documents`, {
            method: 'POST',
            headers: token ? { Authorization: `Bearer ${token}` } : {},
            body: form,
        });
    }

    it('enforces JWT auth, RBAC, and rejects cross-tenant or malformed policy uploads', async () => {
        const alice = await registerUser({ fullName: 'Alice User', email: 'alice@coverageiq.test' });
        const bob = await registerUser({ fullName: 'Bob User', email: 'bob@coverageiq.test' });
        const alicePolicy = await seedPolicy({
            externalKey: 'pol_alice_sec',
            title: 'Alice Health Policy',
            ownerId: alice.user.id,
            ownerScope: `user:${alice.user.id}`,
        });

        // 1. Unauthenticated upload & job requests -> 401
        const unauthUpload = await uploadPdfForm({ policyId: alicePolicy.id, token: null, buffer: validPdf1 });
        assert.equal(unauthUpload.status, 401);

        const unauthJobs = await fetch(`${nodeBaseUrl}/api/ingestion/jobs`);
        assert.equal(unauthJobs.status, 401);

        // 2. Unauthorized user (Bob) uploading to Alice's policy -> 403 and zero files written
        const crossUpload = await uploadPdfForm({ policyId: alicePolicy.id, token: bob.token, buffer: validPdf1 });
        assert.equal(crossUpload.status, 403);
        assert.equal(await countFilesRecursive(uploadRoot), 0);

        // 3. Invalid policy UUID -> 400; nonexistent UUID -> 404; archived policy -> 409
        const badUuid = await uploadPdfForm({ policyId: 'not-a-uuid', token: alice.token, buffer: validPdf1 });
        assert.equal(badUuid.status, 400);

        const missingPol = await uploadPdfForm({ policyId: crypto.randomUUID(), token: alice.token, buffer: validPdf1 });
        assert.equal(missingPol.status, 404);

        const archivedPol = await seedPolicy({
            externalKey: 'pol_alice_arch',
            title: 'Archived',
            ownerId: alice.user.id,
            ownerScope: `user:${alice.user.id}`,
            status: 'archived',
        });
        const archResp = await uploadPdfForm({ policyId: archivedPol.id, token: alice.token, buffer: validPdf1 });
        assert.equal(archResp.status, 409);

        // 4. Organization membership write access vs non-member
        const orgOwner = await registerUser({
            fullName: 'Carol Org',
            email: 'carol@org.test',
            role: 'organisation',
        });
        const orgRes = await pgPool.query(
            `INSERT INTO organizations (name, slug, created_by) VALUES ('Acme Insure', 'acme-insure', $1) RETURNING *`,
            [orgOwner.user.id]
        );
        const orgId = orgRes.rows[0].id;
        await pgPool.query(
            `INSERT INTO organization_members (organization_id, user_id, role) VALUES ($1, $2, 'owner')`,
            [orgId, orgOwner.user.id]
        );
        const orgPolicy = await seedPolicy({
            externalKey: 'pol_org_acme',
            title: 'Acme Group Plan',
            ownerId: orgId,
            ownerScope: `org:${orgId}`,
            visibility: 'catalog',
        });

        const nonMemberUpload = await uploadPdfForm({ policyId: orgPolicy.id, token: bob.token, buffer: validPdf1 });
        assert.equal(nonMemberUpload.status, 403);

        const orgUpload = await uploadPdfForm({ policyId: orgPolicy.id, token: orgOwner.token, buffer: validPdf1 });
        assert.equal(orgUpload.status, 201);
        const orgData = await orgUpload.json();
        await waitForJobTerminal(nodeBaseUrl, orgOwner.token, orgData.data.job.id, 'completed');
    });

    it('rejects invalid extensions, MIME mismatches, bad signatures, encrypted, corrupted, and oversized PDFs cleanly', async () => {
        const alice = await registerUser({ fullName: 'Alice Validation', email: 'val@coverageiq.test' });
        const pol = await seedPolicy({
            externalKey: 'pol_validation',
            title: 'Validation Plan',
            ownerId: alice.user.id,
            ownerScope: `user:${alice.user.id}`,
        });

        // 1. Extension mismatch
        const rExt = await uploadPdfForm({
            policyId: pol.id,
            token: alice.token,
            buffer: validPdf1,
            filename: 'notes.txt',
            mimeType: 'application/pdf',
        });
        assert.equal(rExt.status, 400);

        // 2. MIME mismatch
        const rMime = await uploadPdfForm({
            policyId: pol.id,
            token: alice.token,
            buffer: validPdf1,
            filename: 'policy.pdf',
            mimeType: 'text/plain',
        });
        assert.equal(rMime.status, 415);

        // 3. Empty 0-byte PDF
        const rEmpty = await uploadPdfForm({
            policyId: pol.id,
            token: alice.token,
            buffer: Buffer.alloc(0),
            filename: 'empty.pdf',
        });
        assert.equal(rEmpty.status, 400);

        // 4. Invalid %PDF- signature
        const rSig = await uploadPdfForm({
            policyId: pol.id,
            token: alice.token,
            buffer: Buffer.from('Plain text pretending to be PDF'),
            filename: 'fake.pdf',
        });
        assert.equal(rSig.status, 400);

        // 5. Password-protected / encrypted PDF
        const rEnc = await uploadPdfForm({
            policyId: pol.id,
            token: alice.token,
            buffer: encryptedPdf,
            filename: 'locked.pdf',
        });
        assert.equal(rEnc.status, 422);
        const encBody = await rEnc.json();
        assert.equal(encBody.errors[0].code, 'PASSWORD_PROTECTED_PDF');

        // 6. Corrupted / truncated PDF
        const rCorrupt = await uploadPdfForm({
            policyId: pol.id,
            token: alice.token,
            buffer: Buffer.from('%PDF-1.4\ncorrupted truncated payload without xref or eof'),
            filename: 'corrupt.pdf',
        });
        assert.equal(rCorrupt.status, 422);

        // 7. Oversize limit
        const prevMax = process.env.MAX_PDF_SIZE_BYTES;
        process.env.MAX_PDF_SIZE_BYTES = '256';
        try {
            const rLarge = await uploadPdfForm({
                policyId: pol.id,
                token: alice.token,
                buffer: validPdf1,
                filename: 'big.pdf',
            });
            assert.equal(rLarge.status, 413);
        } finally {
            process.env.MAX_PDF_SIZE_BYTES = prevMax;
        }

        // Verify zero documents, zero jobs, zero stored files created across all failed validations
        const docsCount = await pgPool.query('SELECT COUNT(*)::int AS c FROM policy_documents WHERE policy_id = $1', [pol.id]);
        const jobsCount = await pgPool.query('SELECT COUNT(*)::int AS c FROM ingestion_jobs WHERE policy_id = $1', [pol.id]);
        assert.equal(docsCount.rows[0].c, 0);
        assert.equal(jobsCount.rows[0].c, 0);
        assert.equal(await countFilesRecursive(uploadRoot), 0);
    });

    it('completes real upload -> FastAPI enqueue -> Redis -> Celery -> PG16, handles duplicates, versioning, isolation, pagination & filters', async () => {
        const alice = await registerUser({ fullName: 'Alice E2E', email: 'alice_e2e@coverageiq.test' });
        const bob = await registerUser({ fullName: 'Bob Other', email: 'bob_other@coverageiq.test' });
        const pol1 = await seedPolicy({
            externalKey: 'pol_e2e_1',
            title: 'E2E Primary Plan',
            ownerId: alice.user.id,
            ownerScope: `user:${alice.user.id}`,
        });
        const pol2 = await seedPolicy({
            externalKey: 'pol_e2e_2',
            title: 'E2E Secondary Plan',
            ownerId: alice.user.id,
            ownerScope: `user:${alice.user.id}`,
        });

        // 1. Valid v1 PDF upload -> 201 -> Real Celery processes to COMPLETED in PostgreSQL
        const up1 = await uploadPdfForm({ policyId: pol1.id, token: alice.token, buffer: validPdf1, filename: 'health_v1.pdf' });
        assert.equal(up1.status, 201);
        const body1 = await up1.json();
        assert.equal(body1.data.duplicate, false);
        assert.equal(body1.data.enqueued, true);
        assert.equal(body1.data.document.version_number, 1);
        assert.equal(body1.data.document.ingestion_status, 'queued');
        assert.equal(body1.data.job.status, 'queued');
        assert.equal('storage_path' in body1.data.document, false);
        assert.equal(JSON.stringify(body1).includes(internalToken), false);
        assert.equal(JSON.stringify(body1).includes('change_me_runtime'), false);

        const job1Done = await waitForJobTerminal(nodeBaseUrl, alice.token, body1.data.job.id, 'completed');
        assert.equal(job1Done.current_stage, 'COMPLETED');
        assert.equal(job1Done.progress.percent, 100);
        assert.ok(job1Done.progress.total_chunks >= 1);
        assert.equal(job1Done.document.ingestion_status, 'completed');

        // Confirm chunks & embeddings persisted in PG and policy promoted to active v1
        const polRow1 = (await pgPool.query('SELECT status, active_version_id FROM policies WHERE id = $1', [pol1.id])).rows[0];
        assert.equal(polRow1.status, 'active');
        assert.equal(polRow1.active_version_id, body1.data.document.id);

        // 2. Duplicate SHA-256 upload -> 200 OK idempotent, zero duplicate files/docs/jobs
        const dupResp = await uploadPdfForm({ policyId: pol1.id, token: alice.token, buffer: validPdf1, filename: 'duplicate_name.pdf' });
        assert.equal(dupResp.status, 200);
        const dupBody = await dupResp.json();
        assert.equal(dupBody.data.duplicate, true);
        assert.equal(dupBody.data.enqueued, false);
        assert.equal(dupBody.data.document.id, body1.data.document.id);
        assert.equal(dupBody.data.job.id, body1.data.job.id);
        assert.equal(await countFilesRecursive(uploadRoot), 1);

        // 3. New version v2 upload -> creates version_number=2, advances active_version_id on completion
        const up2 = await uploadPdfForm({
            policyId: pol1.id,
            token: alice.token,
            buffer: validPdf2,
            filename: 'health_v2.pdf',
            extraFields: { version_label: '2026-rider' },
            prefix: '/api/v1', // Also verify /api/v1 compatibility mount
        });
        assert.equal(up2.status, 201);
        const body2 = await up2.json();
        assert.equal(body2.data.document.version_number, 2);
        assert.equal(body2.data.document.version_label, '2026-rider');
        await waitForJobTerminal(nodeBaseUrl, alice.token, body2.data.job.id, 'completed');

        const polRow2 = (await pgPool.query('SELECT active_version_id FROM policies WHERE id = $1', [pol1.id])).rows[0];
        assert.equal(polRow2.active_version_id, body2.data.document.id);
        assert.equal(await countFilesRecursive(uploadRoot), 2);

        // Upload another policy document to pol2 for filtering tests
        const up3 = await uploadPdfForm({ policyId: pol2.id, token: alice.token, buffer: validPdf3, filename: 'pol2_v1.pdf' });
        assert.equal(up3.status, 201);
        const body3 = await up3.json();
        await waitForJobTerminal(nodeBaseUrl, alice.token, body3.data.job.id, 'completed');

        // 4. Cross-tenant job lookup isolation (Bob cannot inspect Alice's job -> 404, no existence leak)
        const crossJob = await fetch(`${nodeBaseUrl}/api/ingestion/jobs/${body1.data.job.id}`, {
            headers: { Authorization: `Bearer ${bob.token}` },
        });
        assert.equal(crossJob.status, 404);

        // 5. Reject client-supplied owner_scope override -> 400
        const badScopeList = await fetch(
            `${nodeBaseUrl}/api/ingestion/jobs?owner_scope=${encodeURIComponent(`user:${alice.user.id}`)}`,
            { headers: { Authorization: `Bearer ${bob.token}` } }
        );
        assert.equal(badScopeList.status, 400);

        // Bob's job list sees zero Alice jobs
        const bobList = await (await fetch(`${nodeBaseUrl}/api/ingestion/jobs`, {
            headers: { Authorization: `Bearer ${bob.token}` },
        })).json();
        assert.equal(bobList.data.pagination.total_items, 0);
        assert.deepEqual(bobList.data.items, []);

        // 6. Paginated + status + policyId filtering for Alice
        const page1 = await (await fetch(
            `${nodeBaseUrl}/api/ingestion/jobs?page=1&limit=2&status=completed`,
            { headers: { Authorization: `Bearer ${alice.token}` } }
        )).json();
        assert.equal(page1.data.items.length, 2);
        assert.equal(page1.data.pagination.total_items, 3);
        assert.equal(page1.data.pagination.total_pages, 2);
        assert.equal(page1.data.pagination.has_next, true);
        assert.equal(page1.data.pagination.has_prev, false);

        const page2 = await (await fetch(
            `${nodeBaseUrl}/api/v1/ingestion/jobs?page=2&limit=2&status=completed&policyId=${pol1.id}`,
            { headers: { Authorization: `Bearer ${alice.token}` } }
        )).json();
        assert.equal(page2.data.pagination.total_items, 2);
        assert.equal(page2.data.items.length, 0);

        const filteredPol2 = await (await fetch(
            `${nodeBaseUrl}/api/ingestion/jobs?policyId=${pol2.id}`,
            { headers: { Authorization: `Bearer ${alice.token}` } }
        )).json();
        assert.equal(filteredPol2.data.pagination.total_items, 1);
        assert.equal(filteredPol2.data.items[0].id, body3.data.job.id);
    });

    it('cleans up orphaned PDF file when database transaction fails and sanitizes error responses', async () => {
        const alice = await registerUser({ fullName: 'Alice Cleanup', email: 'cleanup@coverageiq.test' });
        const pol = await seedPolicy({
            externalKey: 'pol_cleanup',
            title: 'Cleanup Plan',
            ownerId: alice.user.id,
            ownerScope: `user:${alice.user.id}`,
        });

        app.locals.ingestionStorageOptions = {
            afterInsertBeforeCommit: async () => {
                throw new Error('Simulated DB commit outage postgresql://coverageiq_runtime:secret@127.0.0.1:55432/coverageiq_db');
            },
        };

        const failedResp = await uploadPdfForm({ policyId: pol.id, token: alice.token, buffer: validPdf1, filename: 'rollback.pdf' });
        assert.equal(failedResp.status, 500);
        const failBody = await failedResp.json();
        assert.equal(JSON.stringify(failBody).includes('secret'), false);
        assert.equal(JSON.stringify(failBody).includes(uploadRoot), false);

        // Crucially: zero files left behind and zero DB rows left behind!
        assert.equal(await countFilesRecursive(uploadRoot), 0);
        const docs = await pgPool.query('SELECT COUNT(*)::int AS c FROM policy_documents WHERE policy_id = $1', [pol.id]);
        const jobs = await pgPool.query('SELECT COUNT(*)::int AS c FROM ingestion_jobs WHERE policy_id = $1', [pol.id]);
        assert.equal(docs.rows[0].c, 0);
        assert.equal(jobs.rows[0].c, 0);
    });

    it('recovers from enqueue failure via reconciler and marks exhausted dispatch failures cleanly as failed', async () => {
        const alice = await registerUser({ fullName: 'Alice Recovery', email: 'recover@coverageiq.test' });
        const pol = await seedPolicy({
            externalKey: 'pol_recover',
            title: 'Recovery Plan',
            ownerId: alice.user.id,
            ownerScope: `user:${alice.user.id}`,
        });

        // 1. Simulate Python internal enqueue temporarily unreachable -> 202 retry_scheduled
        app.locals.ingestionEnqueueOptions = {
            aiServiceUrl: 'http://127.0.0.1:59997',
            maxInlineAttempts: 1,
            timeoutMs: 1000,
        };
        const recUpload = await uploadPdfForm({ policyId: pol.id, token: alice.token, buffer: validPdf1, filename: 'delayed.pdf' });
        assert.equal(recUpload.status, 202);
        const recBody = await recUpload.json();
        assert.equal(recBody.data.dispatch.state, 'retry_scheduled');
        assert.equal(recBody.data.job.status, 'queued');
        assert.equal(recBody.data.job.current_stage, 'ENQUEUE_RETRY_PENDING');
        assert.equal(recBody.data.job.error.code, 'ENQUEUE_SERVICE_UNAVAILABLE');

        // Restore live FastAPI URL -> run reconciler -> Celery processes job to COMPLETED!
        app.locals.ingestionEnqueueOptions = {};
        const reconSummary = await gatewaySvc.reconcilePendingIngestionJobs({ ignoreBackoff: true });
        assert.equal(reconSummary.dispatched, 1);

        const completed = await waitForJobTerminal(nodeBaseUrl, alice.token, recBody.data.job.id, 'completed');
        assert.equal(completed.current_stage, 'COMPLETED');

        // 2. Exhaustion scenario: unreachable enqueue hits maxDispatchAttempts -> transitions job & doc to failed (never left queued!)
        app.locals.ingestionEnqueueOptions = {
            aiServiceUrl: 'http://127.0.0.1:59997',
            maxInlineAttempts: 1,
            timeoutMs: 500,
        };
        const exUpload = await uploadPdfForm({ policyId: pol.id, token: alice.token, buffer: validPdf2, filename: 'exhaust.pdf' });
        assert.equal(exUpload.status, 202);
        const exBody = await exUpload.json();

        const exSummary = await gatewaySvc.reconcilePendingIngestionJobs({
            ignoreBackoff: true,
            maxDispatchAttempts: 2,
            enqueueOptions: { aiServiceUrl: 'http://127.0.0.1:59997', maxInlineAttempts: 1, timeoutMs: 500 },
        });
        assert.equal(exSummary.exhausted, 1);

        const failedStatus = await (await fetch(`${nodeBaseUrl}/api/ingestion/jobs/${exBody.data.job.id}`, {
            headers: { Authorization: `Bearer ${alice.token}` },
        })).json();
        assert.equal(failedStatus.data.job.status, 'failed');
        assert.equal(failedStatus.data.job.current_stage, 'ENQUEUE_EXHAUSTED');
        assert.equal(failedStatus.data.job.error.code, 'ENQUEUE_DISPATCH_FAILED');
        assert.equal(failedStatus.data.job.document.ingestion_status, 'failed');
    });

    it('enforces upload rate limits cleanly with 429 and Retry-After', async () => {
        const alice = await registerUser({ fullName: 'Alice RL', email: 'rl@coverageiq.test' });
        const pol = await seedPolicy({
            externalKey: 'pol_rl',
            title: 'Rate Limit Plan',
            ownerId: alice.user.id,
            ownerScope: `user:${alice.user.id}`,
        });

        // Lower limit temporarily for rate-limit check
        const prevMax = process.env.UPLOAD_RATE_LIMIT_MAX;
        process.env.UPLOAD_RATE_LIMIT_MAX = '1';
        try {
            const first = await uploadPdfForm({ policyId: pol.id, token: alice.token, buffer: validPdf1, filename: 'rl1.pdf' });
            assert.equal(first.status, 201);
            const firstData = await first.json();
            await waitForJobTerminal(nodeBaseUrl, alice.token, firstData.data.job.id, 'completed');

            const second = await uploadPdfForm({ policyId: pol.id, token: alice.token, buffer: validPdf2, filename: 'rl2.pdf' });
            assert.equal(second.status, 429);
            assert.ok(second.headers.get('retry-after'));
        } finally {
            process.env.UPLOAD_RATE_LIMIT_MAX = prevMax;
        }
    });

    it('supports health checks, cookie-based auth compatible with client, and redacts sensitive substrings in failed jobs', async () => {
        const healthResp = await fetch(`${nodeBaseUrl}/api/health`);
        assert.equal(healthResp.status, 200);
        const healthBody = await healthResp.json();
        assert.equal(healthBody.status, 'healthy');
        assert.equal(healthBody.services.database, 'up');
        assert.equal(healthBody.services.redis, 'up');

        await registerUser({ fullName: 'Cookie User', email: 'cookie@coverageiq.test', password: 'Password123!' });
        const loginResp = await fetch(`${nodeBaseUrl}/api/v1/users/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: 'cookie@coverageiq.test', password: 'Password123!' }),
        });
        assert.equal(loginResp.status, 200);
        const loginData = await loginResp.json();
        const meCookie = `accessToken=${loginData.data.accessToken}`;
        const userId = loginData.data.user.id;

        const pol = await seedPolicy({
            externalKey: 'pol_cookie',
            title: 'Cookie Auth Plan',
            ownerId: userId,
            ownerScope: `user:${userId}`,
        });

        // Insert a failed job containing an internal path & DSN to confirm job serialization redacts them
        const failJobRes = await pgPool.query(
            `
            INSERT INTO ingestion_jobs (
                policy_id, external_policy_key, owner_scope, status,
                current_stage, error_code, error_message
            )
            VALUES ($1, $2, $3, 'failed', 'FAILED', 'WORKER_ERR', $4)
            RETURNING id
            `,
            [
                pol.id,
                pol.external_policy_key,
                pol.owner_scope,
                'Worker failure at /var/lib/uploads/secret.pdf connecting postgresql://runtime:pw@db:5432/cov',
            ]
        );

        const jobResp = await fetch(`${nodeBaseUrl}/api/v1/ingestion/jobs/${failJobRes.rows[0].id}`, {
            headers: { Cookie: meCookie },
        });
        assert.equal(jobResp.status, 200);
        const jobBody = await jobResp.json();
        const errMsg = jobBody.data.job.error.message;
        assert.equal(errMsg.includes('pw@'), false);
        assert.equal(errMsg.includes('/var/lib/uploads/secret.pdf'), false);
        assert.ok(errMsg.includes('postgresql://<redacted>'));
        assert.ok(errMsg.includes('<redacted-path>'));
    });

    it('verifies full Checkpoint 4.6 frontend /api/v1 lifecycle: login, policy selector/create, upload, enqueue, poll completed/failed, duplicate, cross-tenant rejection', async () => {
        // 1. Register & Login (using HttpOnly cookie auth matching React AuthContext/api.js)
        await registerUser({ fullName: 'Alice FE', email: 'alice_fe@coverageiq.test', password: 'Password123!' });
        const loginAlice = await fetch(`${nodeBaseUrl}/api/v1/users/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: 'alice_fe@coverageiq.test', password: 'Password123!' }),
        });
        assert.equal(loginAlice.status, 200);
        const aliceCookie = `accessToken=${(await loginAlice.json()).data.accessToken}`;

        const bob = await registerUser({ fullName: 'Bob FE', email: 'bob_fe@coverageiq.test', password: 'Password123!' });
        const bobCookie = `accessToken=${bob.token}`;

        // 2. Policy authorization & selector endpoints (/api/v1/policies)
        // Regular user cannot publish catalog policy
        const forbiddenCat = await fetch(`${nodeBaseUrl}/api/v1/policies`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Cookie: aliceCookie },
            body: JSON.stringify({ title: 'Fake Catalog', policy_type: 'health', visibility: 'catalog' }),
        });
        assert.equal(forbiddenCat.status, 403);

        const createPol = await fetch(`${nodeBaseUrl}/api/v1/policies`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Cookie: aliceCookie },
            body: JSON.stringify({ title: 'Alice UI Plan', policy_type: 'health', insurer_name: 'CoverageIQ' }),
        });
        assert.equal(createPol.status, 201);
        const createdPol = (await createPol.json()).data.policy;
        assert.equal(createdPol.can_upload, true);

        const alicePolicies = await (await fetch(`${nodeBaseUrl}/api/v1/policies`, {
            headers: { Cookie: aliceCookie },
        })).json();
        assert.equal(alicePolicies.data.items.some((p) => p.id === createdPol.id), true);

        const bobPolicies = await (await fetch(`${nodeBaseUrl}/api/v1/policies`, {
            headers: { Cookie: bobCookie },
        })).json();
        assert.equal(bobPolicies.data.items.length, 0);

        // 3–7. Upload -> Job creation -> Celery enqueue -> Job status retrieval -> Successful ingestion
        const form1 = new FormData();
        form1.append('file', new Blob([validPdf1], { type: 'application/pdf' }), 'alice_ui_v1.pdf');
        const upRes = await fetch(`${nodeBaseUrl}/api/v1/policies/${createdPol.id}/documents`, {
            method: 'POST',
            headers: { Cookie: aliceCookie },
            body: form1,
        });
        assert.equal(upRes.status, 201);
        const upJson = await upRes.json();
        const jobId = upJson.data.job.id;
        assert.equal(upJson.data.enqueued, true);

        const doneJob = await waitForJobTerminal(nodeBaseUrl, (await (await fetch(`${nodeBaseUrl}/api/v1/users/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: 'alice_fe@coverageiq.test', password: 'Password123!' }),
        })).json()).data.accessToken, jobId, 'completed');
        assert.equal(doneJob.current_stage, 'COMPLETED');
        assert.equal(doneJob.progress.percent, 100);

        // 8. Duplicate upload via /api/v1 returns 200 OK
        const dupForm = new FormData();
        dupForm.append('file', new Blob([validPdf1], { type: 'application/pdf' }), 'alice_ui_v1_copy.pdf');
        const dupRes = await fetch(`${nodeBaseUrl}/api/v1/policies/${createdPol.id}/documents`, {
            method: 'POST',
            headers: { Cookie: aliceCookie },
            body: dupForm,
        });
        assert.equal(dupRes.status, 200);
        assert.equal((await dupRes.json()).data.duplicate, true);

        // 9. Failed ingestion path visible through GET /api/v1/ingestion/jobs/:jobId cleanly
        app.locals.ingestionEnqueueOptions = { aiServiceUrl: 'http://127.0.0.1:59997', maxInlineAttempts: 1, timeoutMs: 500 };
        const formFail = new FormData();
        formFail.append('file', new Blob([validPdf2], { type: 'application/pdf' }), 'tampered_disk.pdf');
        const failUpRes = await fetch(`${nodeBaseUrl}/api/v1/policies/${createdPol.id}/documents`, {
            method: 'POST',
            headers: { Cookie: aliceCookie },
            body: formFail,
        });
        assert.equal(failUpRes.status, 202);
        const failJobId = (await failUpRes.json()).data.job.id;

        // Corrupt stored PDF bytes on disk before reconciliation dispatches worker
        const storageRow = (await pgPool.query('SELECT storage_path FROM policy_documents WHERE id = (SELECT document_id FROM ingestion_jobs WHERE id = $1)', [failJobId])).rows[0];
        await fs.writeFile(path.join(uploadRoot, storageRow.storage_path), Buffer.from('not-a-pdf-bytes'));
        app.locals.ingestionEnqueueOptions = {};
        await gatewaySvc.reconcilePendingIngestionJobs({ ignoreBackoff: true });

        const deadline = Date.now() + 30000;
        let failedJob = null;
        while (Date.now() < deadline) {
            const pollResp = await fetch(`${nodeBaseUrl}/api/v1/ingestion/jobs/${failJobId}`, {
                headers: { Cookie: aliceCookie },
            });
            const pollData = await pollResp.json();
            if (pollData.data.job.status === 'failed') {
                failedJob = pollData.data.job;
                break;
            }
            await new Promise((r) => setTimeout(r, 250));
        }
        assert.ok(failedJob, 'Expected Celery worker to persist failed state');
        assert.equal(failedJob.current_stage, 'FAILED');
        assert.equal(failedJob.error.code, 'INVALID_PDF_SIGNATURE');
        assert.equal(JSON.stringify(failedJob).includes(uploadRoot), false);

        // 10. Cross-tenant rejection for Bob
        const bobUpload = new FormData();
        bobUpload.append('file', new Blob([validPdf3], { type: 'application/pdf' }), 'bob.pdf');
        const bobUpResp = await fetch(`${nodeBaseUrl}/api/v1/policies/${createdPol.id}/documents`, {
            method: 'POST',
            headers: { Cookie: bobCookie },
            body: bobUpload,
        });
        assert.equal(bobUpResp.status, 403);

        const bobJobResp = await fetch(`${nodeBaseUrl}/api/v1/ingestion/jobs/${jobId}`, {
            headers: { Cookie: bobCookie },
        });
        assert.equal(bobJobResp.status, 404);
    });
});
