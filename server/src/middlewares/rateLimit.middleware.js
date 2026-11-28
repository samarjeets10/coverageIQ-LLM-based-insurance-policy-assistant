const crypto = require('node:crypto');
const apiError = require('../utils/apiError');
const {
    getConnectedRedisClient,
    sanitizeRedisUrl,
    withTimeout,
} = require('../config/redis');

// Single atomic Lua script executing on Redis's authoritative server clock:
// trims stale timestamps older than window_ms, rejects if count >= max_req with exact Retry-After,
// or appends a unique member for this request and sets key PEXPIRE.
const SLIDING_WINDOW_RATE_LIMIT_LUA = `
local t = redis.call('TIME')
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local window_ms = tonumber(ARGV[1])
local max_req = tonumber(ARGV[2])
local member = ARGV[3]
local cutoff = now_ms - window_ms

redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', cutoff)
local count = redis.call('ZCARD', KEYS[1])

if count >= max_req then
    local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
    local oldest_ms = now_ms
    if oldest and #oldest >= 2 then
        oldest_ms = tonumber(oldest[2])
    end
    local retry_ms = math.max(1000, (oldest_ms + window_ms) - now_ms)
    redis.call('PEXPIRE', KEYS[1], window_ms)
    return {0, count, retry_ms}
end

redis.call('ZADD', KEYS[1], now_ms, member)
redis.call('PEXPIRE', KEYS[1], window_ms)
return {1, count + 1, 0}
`;

function resolveRateLimitClientKey(req, keyGenerator = null) {
    if (typeof keyGenerator === 'function') {
        const custom = String(keyGenerator(req) || '').trim();
        if (!custom) {
            throw new apiError(500, 'Rate limiter key generator returned an empty key');
        }
        return custom;
    }

    // Prefer verified authenticated JWT subject; fall back strictly to Express proxy-aware req.ip
    // (never trusting unvalidated X-Forwarded-For / X-Real-IP headers directly).
    if (req.user && req.user.id) {
        return `user:${String(req.user.id).trim()}`;
    }

    const remoteIp = String(req.ip || req.socket?.remoteAddress || 'unknown').trim();
    return `ip:${remoteIp || 'unknown'}`;
}

function createRateLimiter(options = {}) {
    const keyPrefix = String(options.keyPrefix || 'ciq:ratelimit:upload:').trim() || 'ciq:ratelimit:upload:';
    const message = options.message || 'Too many requests, please retry later';

    const getWindowMs = () => {
        if (typeof options.windowMs === 'function') return Number(options.windowMs());
        if (options.windowMs !== undefined) return Number(options.windowMs);
        return Number(process.env.UPLOAD_RATE_LIMIT_WINDOW_MS || 15 * 60 * 1000);
    };
    const getMax = () => {
        if (typeof options.max === 'function') return Number(options.max());
        if (options.max !== undefined) return Number(options.max);
        return Number(process.env.UPLOAD_RATE_LIMIT_MAX || 30);
    };
    const getRedisUrl = () => (typeof options.redisUrl === 'function' ? options.redisUrl() : options.redisUrl ?? null);
    const getCommandTimeoutMs = () => Number(
        options.commandTimeoutMs || process.env.REDIS_COMMAND_TIMEOUT_MS || 2000
    );

    async function middleware(req, res, next) {
        const windowMs = Math.floor(getWindowMs());
        const max = Math.floor(getMax());

        // Never silently disable rate limiting on invalid configuration.
        if (!Number.isFinite(max) || max <= 0 || !Number.isFinite(windowMs) || windowMs <= 0) {
            return next(
                new apiError(500, 'Upload rate limit configuration is invalid', [
                    { code: 'RATE_LIMIT_CONFIG_INVALID' },
                ])
            );
        }

        let identityKey;
        try {
            identityKey = resolveRateLimitClientKey(req, options.keyGenerator);
        } catch (keyErr) {
            return next(keyErr);
        }

        const redisKey = `${keyPrefix}${identityKey}`;
        const memberId = `${Date.now()}:${crypto.randomUUID()}`;
        const redisUrl = getRedisUrl();

        let evalResult;
        try {
            const client = await getConnectedRedisClient(redisUrl);
            evalResult = await withTimeout(
                client.eval(SLIDING_WINDOW_RATE_LIMIT_LUA, {
                    keys: [redisKey],
                    arguments: [String(windowMs), String(max), memberId],
                }),
                Math.max(250, getCommandTimeoutMs()),
                'Redis rate-limit EVAL'
            );
        } catch (err) {
            console.error(
                `Upload rate limiter Redis failure (${sanitizeRedisUrl(redisUrl || process.env.REDIS_URL)}): ${err?.message || 'error'}`
            );
            return next(
                new apiError(503, 'Upload rate-limit verification unavailable; please retry shortly', [
                    { code: 'RATE_LIMIT_UNAVAILABLE' },
                ])
            );
        }

        const allowed = Number(evalResult?.[0] ?? 0) === 1;
        const count = Number(evalResult?.[1] ?? 0);
        const retryAfterMs = Number(evalResult?.[2] ?? 1000);

        if (!allowed) {
            const retryAfterSec = Math.max(1, Math.ceil(retryAfterMs / 1000));
            res.setHeader('Retry-After', String(retryAfterSec));
            res.setHeader('X-RateLimit-Limit', String(max));
            res.setHeader('X-RateLimit-Remaining', '0');
            return next(
                new apiError(429, message, [
                    { code: 'RATE_LIMIT_EXCEEDED', retryAfterSeconds: retryAfterSec },
                ])
            );
        }

        res.setHeader('X-RateLimit-Limit', String(max));
        res.setHeader('X-RateLimit-Remaining', String(Math.max(0, max - count)));
        return next();
    }

    middleware.reset = async () => {
        try {
            const client = await getConnectedRedisClient(getRedisUrl());
            let cursor = '0';
            do {
                const reply = await client.scan(cursor, {
                    MATCH: `${keyPrefix}*`,
                    COUNT: 100,
                });
                cursor = String(reply.cursor);
                if (reply.keys && reply.keys.length > 0) {
                    await client.del(reply.keys);
                }
            } while (cursor !== '0');
        } catch (_err) {
            // Best-effort test cleanup only; enforcement paths fail closed above.
        }
    };

    return middleware;
}

const uploadRateLimiter = createRateLimiter();

module.exports = {
    SLIDING_WINDOW_RATE_LIMIT_LUA,
    resolveRateLimitClientKey,
    createRateLimiter,
    uploadRateLimiter,
};
