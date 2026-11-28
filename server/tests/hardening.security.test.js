const { describe, it, before, after, beforeEach } = require('node:test');
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const crypto = require('node:crypto');
const fs = require('node:fs/promises');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const { URL } = require('node:url');

const jwt = require('jsonwebtoken');
const { Pool } = require('pg');

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

describe('CoverageIQ Checkpoint 4.7 Node.js Security, Concurrency & Hardening', () => {
    let pgPool;
    let adminPool;
    let uploadRoot;
    let nodeBaseUrl;
    let httpServer;
    let app;
    let dbMod;
    let redisMod;
    let documentStorageSvc;
    let rateLimitMod;
    let origCorsOrigin;
    let origNodeEnv;

    before(async (t) => {
        const pgParsed = new URL(RUNTIME_DSN);
        const redisParsed = new URL(REDIS_URL);
        const pgUp = await isTcpReachable(pgParsed.hostname || '127.0.0.1', Number(pgParsed.port || 5432));
        const redisUp = await isTcpReachable(redisParsed.hostname || '127.0.0.1', Number(redisParsed.port || 6379));
        if (!pgUp || !redisUp) {
            t.skip('Real PostgreSQL 16 + pgvector and Redis 7 required for Checkpoint 4.7/4.8 security tests.');
            return;
        }

        execFileSync('npm', ['--prefix', SERVER_ROOT, 'run', 'migrate:up'], {
            env: { ...process.env, MIGRATION_DATABASE_URL: MIGRATION_DSN },
            stdio: 'pipe',
        });

        origCorsOrigin = process.env.CORS_ORIGIN;
        origNodeEnv = process.env.NODE_ENV;

        uploadRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'ciq-sec-uploads-'));
        const nodePort = await findFreePort();
        nodeBaseUrl = `http://127.0.0.1:${nodePort}`;

        process.env.DATABASE_URL = RUNTIME_DSN;
        process.env.REDIS_URL = REDIS_URL;
        process.env.AI_SERVICE_URL = 'http://127.0.0.1:59994';
        process.env.INTERNAL_SERVICE_TOKEN = 'sec-test-internal-token';
        process.env.POLICY_UPLOAD_ROOT = uploadRoot;
        process.env.ACCESS_TOKEN_SECRET = 'sec-test-access-secret';
        process.env.REFRESH_TOKEN_SECRET = 'sec-test-refresh-secret';
        process.env.ACCESS_TOKEN_EXPIRY = '15m';
        process.env.REFRESH_TOKEN_EXPIRY = '1d';
        process.env.MAX_PDF_SIZE_BYTES = String(5 * 1024 * 1024);
        process.env.UPLOAD_RATE_LIMIT_MAX = '100';

        adminPool = new Pool({ connectionString: MIGRATION_DSN, max: 4 });
        dbMod = require('../src/config/db');
        redisMod = require('../src/config/redis');
        pgPool = dbMod.pool;
        documentStorageSvc = require('../src/services/documentStorage.service');
        rateLimitMod = require('../src/middlewares/rateLimit.middleware');
        app = require('../src/app');

        await new Promise((resolve) => {
            httpServer = app.listen(nodePort, '127.0.0.1', resolve);
        });
    });

    after(async () => {
        if (origCorsOrigin === undefined) {
            delete process.env.CORS_ORIGIN;
        } else {
            process.env.CORS_ORIGIN = origCorsOrigin;
        }
        if (origNodeEnv === undefined) {
            delete process.env.NODE_ENV;
        } else {
            process.env.NODE_ENV = origNodeEnv;
        }

        if (httpServer) {
            await new Promise((resolve) => httpServer.close(resolve));
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
        delete process.env.CORS_ORIGIN;
        process.env.NODE_ENV = 'test';
        process.env.REDIS_URL = REDIS_URL;
        process.env.UPLOAD_RATE_LIMIT_MAX = '100';
        await adminPool.query(
            'TRUNCATE TABLE audit_logs, refresh_tokens, organization_members, organizations, users, '
            + 'chunk_embeddings, policy_chunks, ingestion_jobs, policy_documents, policies CASCADE'
        );
        await rateLimitMod.uploadRateLimiter.reset();
        app.locals.ingestionStorageOptions = {};
        app.locals.ingestionEnqueueOptions = {};
        const entries = await fs.readdir(uploadRoot).catch(() => []);
        for (const entry of entries) {
            await fs.rm(path.join(uploadRoot, entry), { recursive: true, force: true });
        }
    });

    async function registerAndLogin({ fullName, email, password = 'Password123!' }) {
        const res = await fetch(`${nodeBaseUrl}/api/v1/users/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fullName, email, password, role: 'user' }),
        });
        assert.equal(res.status, 201);
        const body = await res.json();
        return {
            user: body.data.user,
            accessToken: body.data.accessToken,
            refreshToken: body.data.refreshToken,
            setCookies: res.headers.getSetCookie(),
        };
    }

    it('enforces strict CORS allowlist (never reflects arbitrary Origin with credentials) and security headers', async () => {
        // 1. Dev default rejects untrusted external Origin
        const evilRes = await fetch(`${nodeBaseUrl}/api/health`, {
            headers: { Origin: 'https://evil-attacker.example' },
        });
        assert.equal(evilRes.status, 200);
        assert.equal(evilRes.headers.get('access-control-allow-origin'), null);
        assert.equal(evilRes.headers.get('access-control-allow-credentials'), null);
        assert.equal(evilRes.headers.get('x-content-type-options'), 'nosniff');
        assert.equal(evilRes.headers.get('x-frame-options'), 'DENY');
        assert.equal(evilRes.headers.get('x-powered-by'), null);

        // 2. Production mode without CORS_ORIGIN denies cross-origin even for localhost
        process.env.NODE_ENV = 'production';
        const prodLocalRes = await fetch(`${nodeBaseUrl}/api/health`, {
            headers: { Origin: 'http://localhost:5173' },
        });
        assert.equal(prodLocalRes.headers.get('access-control-allow-origin'), null);

        // 3. Explicit CORS_ORIGIN ignores wildcard '*' and honors only exact configured origins
        process.env.CORS_ORIGIN = '*, http://127.0.0.1:5173, https://portal.coverageiq.test/';
        const allowedRes = await fetch(`${nodeBaseUrl}/api/health`, {
            headers: { Origin: 'https://portal.coverageiq.test' },
        });
        assert.equal(allowedRes.headers.get('access-control-allow-origin'), 'https://portal.coverageiq.test');
        assert.equal(allowedRes.headers.get('access-control-allow-credentials'), 'true');

        const blockedWildcardRes = await fetch(`${nodeBaseUrl}/api/health`, {
            headers: { Origin: 'https://another-unlisted.example' },
        });
        assert.equal(blockedWildcardRes.headers.get('access-control-allow-origin'), null);
    });

    it('rejects expired/tampered JWTs, self-assigned admin roles, non-admin role edits, and malformed userId UUIDs', async () => {
        const regAdmin = await fetch(`${nodeBaseUrl}/api/v1/users/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                fullName: 'Privilege Escalator',
                email: 'priv@coverageiq.test',
                password: 'Password123!',
                role: 'admin',
            }),
        });
        assert.equal(regAdmin.status, 403);

        const alice = await registerAndLogin({
            fullName: 'Alice Sec',
            email: 'alice.sec@coverageiq.test',
        });

        const expiredToken = jwt.sign(
            { sub: alice.user.id, email: alice.user.email, role: 'user' },
            process.env.ACCESS_TOKEN_SECRET,
            { expiresIn: -10 }
        );
        const expiredRes = await fetch(`${nodeBaseUrl}/api/v1/users/me`, {
            headers: { Authorization: `Bearer ${expiredToken}` },
        });
        assert.equal(expiredRes.status, 401);

        const forgedToken = jwt.sign(
            { sub: alice.user.id, email: alice.user.email, role: 'admin' },
            'wrong-signing-secret',
            { expiresIn: '15m' }
        );
        const forgedRes = await fetch(`${nodeBaseUrl}/api/v1/users/me`, {
            headers: { Authorization: `Bearer ${forgedToken}` },
        });
        assert.equal(forgedRes.status, 401);

        // Non-admin cannot promote themselves or others
        const nonAdminRoleRes = await fetch(`${nodeBaseUrl}/api/v1/users/${alice.user.id}/role`, {
            method: 'PATCH',
            headers: {
                Authorization: `Bearer ${alice.accessToken}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ role: 'admin' }),
        });
        assert.equal(nonAdminRoleRes.status, 403);

        // Promote Alice in DB strictly to verify admin UUID validation boundary
        await adminPool.query(`UPDATE users SET platform_role = 'admin' WHERE id = $1`, [alice.user.id]);
        const badUuidRes = await fetch(`${nodeBaseUrl}/api/v1/users/not-a-valid-uuid/role`, {
            method: 'PATCH',
            headers: {
                Authorization: `Bearer ${alice.accessToken}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ role: 'user' }),
        });
        assert.equal(badUuidRes.status, 400);
        const badUuidJson = await badUuidRes.json();
        assert.match(badUuidJson.message, /Invalid userId/i);
    });

    it('rotates refresh tokens atomically, detects reuse to revoke session family, and blocks post-logout refresh', async () => {
        const alice = await registerAndLogin({
            fullName: 'Alice Token',
            email: 'alice.token@coverageiq.test',
        });

        assert.equal(alice.setCookies.some((c) => c.startsWith('accessToken=') && c.includes('HttpOnly') && c.includes('Path=/')), true);
        assert.equal(alice.setCookies.some((c) => c.startsWith('refreshToken=') && c.includes('HttpOnly') && c.includes('Path=/')), true);

        const r1 = alice.refreshToken;
        // 1. Single valid rotation r1 -> r2
        const rot1Res = await fetch(`${nodeBaseUrl}/api/v1/users/refresh-token`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ refreshToken: r1 }),
        });
        assert.equal(rot1Res.status, 200);
        const rot1Body = await rot1Res.json();
        const r2 = rot1Body.data.refreshToken;
        assert.ok(r2 && r2 !== r1);

        // 2. Replay attack with already-used r1 -> 401 AND revokes active family (r2 must also fail)
        const replayRes = await fetch(`${nodeBaseUrl}/api/v1/users/refresh-token`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ refreshToken: r1 }),
        });
        assert.equal(replayRes.status, 401);

        const r2AfterReplay = await fetch(`${nodeBaseUrl}/api/v1/users/refresh-token`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ refreshToken: r2 }),
        });
        assert.equal(r2AfterReplay.status, 401);

        // 3. Concurrent racing refresh requests with fresh login token r3 -> at most 1 succeeds
        const login2Res = await fetch(`${nodeBaseUrl}/api/v1/users/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: 'alice.token@coverageiq.test', password: 'Password123!' }),
        });
        assert.equal(login2Res.status, 200);
        const login2Json = await login2Res.json();
        const r3 = login2Json.data.refreshToken;

        const [raceA, raceB] = await Promise.all([
            fetch(`${nodeBaseUrl}/api/v1/users/refresh-token`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ refreshToken: r3 }),
            }),
            fetch(`${nodeBaseUrl}/api/v1/users/refresh-token`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ refreshToken: r3 }),
            }),
        ]);
        const raceStatuses = [raceA.status, raceB.status].sort((a, b) => a - b);
        assert.deepEqual(raceStatuses, [200, 401]);

        // 4. Logout revokes refresh token cleanly
        const login3Res = await fetch(`${nodeBaseUrl}/api/v1/users/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: 'alice.token@coverageiq.test', password: 'Password123!' }),
        });
        const login3Data = (await login3Res.json()).data;
        const logoutRes = await fetch(`${nodeBaseUrl}/api/v1/users/logout`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${login3Data.accessToken}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ refreshToken: login3Data.refreshToken }),
        });
        assert.equal(logoutRes.status, 200);

        const afterLogoutRefresh = await fetch(`${nodeBaseUrl}/api/v1/users/refresh-token`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ refreshToken: login3Data.refreshToken }),
        });
        assert.equal(afterLogoutRefresh.status, 401);
    });

    it('serializes concurrent uploads to the same policy, strips spoofed ownership, cleans up failed TX files, and rejects path traversal', async () => {
        const alice = await registerAndLogin({
            fullName: 'Alice Concurrent',
            email: 'alice.conc@coverageiq.test',
        });

        // Spoofed owner_id/owner_scope in POST /api/policies must be ignored
        const createRes = await fetch(`${nodeBaseUrl}/api/v1/policies`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${alice.accessToken}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                title: 'Concurrency Test Policy',
                policy_type: 'health',
                owner_id: 'attacker-user-id',
                owner_scope: 'org:spoofed-org',
            }),
        });
        assert.equal(createRes.status, 201);
        const policy = (await createRes.json()).data.policy;
        assert.equal(policy.owner_id, undefined);
        assert.equal(policy.owner_scope, undefined);
        const dbPolicy = (await pgPool.query('SELECT owner_id, owner_scope FROM policies WHERE id = $1', [policy.id])).rows[0];
        assert.equal(dbPolicy.owner_id, alice.user.id);
        assert.equal(dbPolicy.owner_scope, `user:${alice.user.id}`);

        // Mock enqueue so uploads persist and settle synchronously at 202 without needing external Celery
        app.locals.ingestionEnqueueOptions = {
            aiServiceUrl: 'http://127.0.0.1:59994',
            maxInlineAttempts: 1,
            timeoutMs: 100,
        };

        // 1. Path traversal & symlink escape validation in documentStorage.service.js
        for (const badRef of ['../escape.pdf', '/etc/passwd', 'a\0b.pdf', '..\\win\\escape.pdf']) {
            await assert.rejects(
                () => documentStorageSvc.resolveSafePath(badRef, { rootDir: uploadRoot }),
                (err) => err && err.statusCode === 400
            );
        }

        const outsideDir = await fs.mkdtemp(path.join(os.tmpdir(), 'ciq-outside-symlink-'));
        try {
            const symlinkPath = path.join(uploadRoot, 'evil-link');
            await fs.symlink(outsideDir, symlinkPath);
            await assert.rejects(
                () => documentStorageSvc.resolveSafePath('evil-link/doc.pdf', { rootDir: uploadRoot }),
                (err) => err && err.statusCode === 400
            );
        } finally {
            await fs.rm(outsideDir, { recursive: true, force: true });
        }

        // 2. Concurrent uploads to the SAME policy -> strictly sequential version_number 1, 2, 3
        const pdfA = buildValidPdfBuffer('Policy Concurrent Version A content');
        const pdfB = buildValidPdfBuffer('Policy Concurrent Version B content');
        const pdfC = buildValidPdfBuffer('Policy Concurrent Version C content');

        const uploadOne = (buf, name) => {
            const form = new FormData();
            form.append('file', new Blob([buf], { type: 'application/pdf' }), name);
            return fetch(`${nodeBaseUrl}/api/v1/policies/${policy.id}/documents`, {
                method: 'POST',
                headers: { Authorization: `Bearer ${alice.accessToken}` },
                body: form,
            });
        };

        const responses = await Promise.all([
            uploadOne(pdfA, 'vA.pdf'),
            uploadOne(pdfB, 'vB.pdf'),
            uploadOne(pdfC, 'vC.pdf'),
        ]);
        const payloads = await Promise.all(responses.map(async (r) => ({ status: r.status, json: await r.json() })));
        for (const item of payloads) {
            assert.ok([201, 202].includes(item.status), `Unexpected concurrent upload status: ${JSON.stringify(item)}`);
        }
        const versions = payloads
            .map((p) => Number(p.json.data.document.version_number))
            .sort((a, b) => a - b);
        assert.deepEqual(versions, [1, 2, 3]);

        // 3. DB transaction failure after disk write rolls back DB AND removes written PDF cleanly
        const pdfD = buildValidPdfBuffer('Rollback test content');
        app.locals.ingestionStorageOptions = {
            afterInsertBeforeCommit: async ({ storageKey }) => {
                // Verify temp file exists mid-tx, then abort
                await fs.access(path.join(uploadRoot, storageKey));
                throw new Error('Simulated commit-time failure');
            },
        };
        const rollbackRes = await uploadOne(pdfD, 'rollback.pdf');
        assert.equal(rollbackRes.status, 500);
        const docCountRes = await pgPool.query('SELECT COUNT(*)::int AS cnt FROM policy_documents WHERE policy_id = $1', [policy.id]);
        assert.equal(docCountRes.rows[0].cnt, 3);
        const filesLeft = await fs.readdir(path.join(uploadRoot, `user_${alice.user.id}`, policy.id));
        assert.equal(filesLeft.length, 3);

        // 4. Rate limiter returns 429 + Retry-After when burst limit exceeded
        app.locals.ingestionStorageOptions = {};
        process.env.UPLOAD_RATE_LIMIT_MAX = '1';
        await rateLimitMod.uploadRateLimiter.reset();
        const firstBurst = await uploadOne(pdfD, 'burst1.pdf');
        assert.ok([201, 202].includes(firstBurst.status));
        const secondBurst = await uploadOne(buildValidPdfBuffer('Burst 2 content'), 'burst2.pdf');
        assert.equal(secondBurst.status, 429);
        assert.ok(secondBurst.headers.get('retry-after'));
        assert.equal(secondBurst.headers.get('x-ratelimit-remaining'), '0');
    });

    it('enforces least-privilege runtime DB role (coverageiq_runtime cannot execute DDL)', async () => {
        await assert.rejects(
            () => pgPool.query('CREATE TABLE runtime_ddl_probe (id INT)'),
            (err) => err && err.code === '42501'
        );
        await assert.rejects(
            () => pgPool.query('ALTER TABLE policies ADD COLUMN unauthorized_col TEXT'),
            (err) => err && err.code === '42501'
        );
    });

    it('shares atomic Redis rate limits across multiple instances, resists X-Forwarded-For spoofing, sets TTL, and fails closed on outage/bad config', async () => {
        const express = require('express');
        const errorHandler = require('../src/middlewares/error.middleware');
        const prefix = `ciq:rl:multi:${crypto.randomBytes(4).toString('hex')}:`;

        const buildNodeInstance = (limiterOpts) => {
            const inst = express();
            const limiter = rateLimitMod.createRateLimiter(limiterOpts);
            inst.post('/limited', limiter, (_req, res) => res.status(200).json({ ok: true }));
            inst.use(errorHandler);
            return { inst, limiter };
        };

        const { inst: appA, limiter: limiterA } = buildNodeInstance({ keyPrefix: prefix, max: 3, windowMs: 30000 });
        const { inst: appB } = buildNodeInstance({ keyPrefix: prefix, max: 3, windowMs: 30000 });

        const portA = await findFreePort();
        const portB = await findFreePort();
        const srvA = await new Promise((resolve) => {
            const s = appA.listen(portA, '127.0.0.1', () => resolve(s));
        });
        const srvB = await new Promise((resolve) => {
            const s = appB.listen(portB, '127.0.0.1', () => resolve(s));
        });

        try {
            // 1. Concurrent burst across TWO separate Express instances -> strictly 3 allowed, 3 rejected 429
            const burstUrls = [
                { url: `http://127.0.0.1:${portA}/limited`, xff: '198.51.100.1' },
                { url: `http://127.0.0.1:${portB}/limited`, xff: '198.51.100.2' },
                { url: `http://127.0.0.1:${portA}/limited`, xff: '198.51.100.3' },
                { url: `http://127.0.0.1:${portB}/limited`, xff: '198.51.100.4' },
                { url: `http://127.0.0.1:${portA}/limited`, xff: '198.51.100.5' },
                { url: `http://127.0.0.1:${portB}/limited`, xff: '198.51.100.6' },
            ];
            // Also proves untrusted X-Forwarded-For spoofing cannot bypass rate limit
            const burstResults = await Promise.all(
                burstUrls.map((item) =>
                    fetch(item.url, {
                        method: 'POST',
                        headers: {
                            'X-Forwarded-For': item.xff,
                            'X-Real-IP': item.xff,
                            Forwarded: `for=${item.xff}`,
                        },
                    })
                )
            );
            const codes = burstResults.map((r) => r.status).sort((a, b) => a - b);
            assert.deepEqual(codes, [200, 200, 200, 429, 429, 429]);

            const client = await redisMod.getConnectedRedisClient(REDIS_URL);
            const matchingKeys = [];
            let cur = '0';
            do {
                const scanRes = await client.scan(cur, { MATCH: `${prefix}*`, COUNT: 50 });
                cur = String(scanRes.cursor);
                matchingKeys.push(...scanRes.keys);
            } while (cur !== '0');
            assert.equal(matchingKeys.length, 1);
            const pttl = await client.pTTL(matchingKeys[0]);
            assert.ok(pttl > 0 && pttl <= 30000, `Expected positive TTL <= 30000ms, got ${pttl}`);

            // 2. Fail-closed on unreachable Redis (503, never silently disabled)
            const { inst: failApp } = buildNodeInstance({
                keyPrefix: `${prefix}down:`,
                max: 5,
                windowMs: 10000,
                redisUrl: 'redis://127.0.0.1:59991/0',
                commandTimeoutMs: 350,
            });
            const failPort = await findFreePort();
            const failSrv = await new Promise((resolve) => {
                const s = failApp.listen(failPort, '127.0.0.1', () => resolve(s));
            });
            try {
                const downRes = await fetch(`http://127.0.0.1:${failPort}/limited`, { method: 'POST' });
                assert.equal(downRes.status, 503);
                const downBody = await downRes.json();
                assert.equal(downBody.errors?.[0]?.code, 'RATE_LIMIT_UNAVAILABLE');
            } finally {
                await new Promise((r) => failSrv.close(r));
            }

            // 3. Invalid config fails closed with 500 (never silently disabled)
            const { inst: badCfgApp } = buildNodeInstance({ keyPrefix: prefix, max: 0, windowMs: 1000 });
            const badPort = await findFreePort();
            const badSrv = await new Promise((resolve) => {
                const s = badCfgApp.listen(badPort, '127.0.0.1', () => resolve(s));
            });
            try {
                const badRes = await fetch(`http://127.0.0.1:${badPort}/limited`, { method: 'POST' });
                assert.equal(badRes.status, 500);
                const badBody = await badRes.json();
                assert.equal(badBody.errors?.[0]?.code, 'RATE_LIMIT_CONFIG_INVALID');
            } finally {
                await new Promise((r) => badSrv.close(r));
            }
        } finally {
            await limiterA.reset();
            await new Promise((r) => srvA.close(r));
            await new Promise((r) => srvB.close(r));
        }
    });

    it('verifies shared storage permissions and cross-runtime readability between Node gateway and Python worker storage', async () => {
        const alice = await registerAndLogin({
            fullName: 'Alice Storage',
            email: 'alice.storage@coverageiq.test',
        });
        const createRes = await fetch(`${nodeBaseUrl}/api/v1/policies`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${alice.accessToken}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ title: 'Shared Storage Audit Policy', policy_type: 'health' }),
        });
        assert.equal(createRes.status, 201);
        const policy = (await createRes.json()).data.policy;

        app.locals.ingestionEnqueueOptions = {
            aiServiceUrl: 'http://127.0.0.1:59994',
            maxInlineAttempts: 1,
            timeoutMs: 100,
        };
        const pdfBuf = buildValidPdfBuffer('Shared storage cross-service verification document.');
        const expectedSha = crypto.createHash('sha256').update(pdfBuf).digest('hex');

        const form = new FormData();
        form.append('file', new Blob([pdfBuf], { type: 'application/pdf' }), 'shared_check.pdf');
        const upRes = await fetch(`${nodeBaseUrl}/api/v1/policies/${policy.id}/documents`, {
            method: 'POST',
            headers: { Authorization: `Bearer ${alice.accessToken}` },
            body: form,
        });
        assert.equal(upRes.status, 202);
        const docId = (await upRes.json()).data.document.id;

        const row = (await pgPool.query('SELECT storage_path FROM policy_documents WHERE id = $1', [docId])).rows[0];
        const absPath = path.join(uploadRoot, row.storage_path);
        const stat = await fs.stat(absPath);
        assert.equal(stat.isFile(), true);
        assert.equal(stat.size, pdfBuf.length);

        // Verify independent Python worker storage backend reads exact same file and SHA-256
        const pyCheckOut = execFileSync(
            PYTHON_BIN,
            [
                '-c',
                [
                    'import hashlib, sys',
                    'from ai_service.storage import LocalFilesystemDocumentStorage',
                    's = LocalFilesystemDocumentStorage(root_dir=sys.argv[1])',
                    'with s.resolve_for_reading(sys.argv[2]) as h:',
                    '    data = h.local_path.read_bytes()',
                    '    print(hashlib.sha256(data).hexdigest(), h.file_size_bytes)',
                ].join('\n'),
                uploadRoot,
                row.storage_path,
            ],
            { encoding: 'utf8', env: { ...process.env, PYTHONPATH: path.resolve(SERVER_ROOT, '..') } }
        ).trim();
        assert.equal(pyCheckOut, `${expectedSha} ${pdfBuf.length}`);
    });
});
