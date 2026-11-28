const { URL } = require('node:url');
const { createClient } = require('redis');

const DEFAULT_DEV_REDIS_URL = 'redis://127.0.0.1:6379/0';
const clientsByUrl = new Map();

function sanitizeRedisUrl(rawUrl) {
    if (!rawUrl || typeof rawUrl !== 'string') {
        return 'redis://<unset>';
    }
    try {
        const parsed = new URL(rawUrl.trim());
        if (parsed.password) {
            parsed.password = '***';
        }
        return parsed.toString();
    } catch (_err) {
        return 'redis://<invalid>';
    }
}

function resolveRedisUrl(explicitUrl = null) {
    const candidate = explicitUrl !== null && explicitUrl !== undefined
        ? String(explicitUrl).trim()
        : (process.env.REDIS_URL || '').trim();

    if (candidate) {
        return candidate;
    }

    if (process.env.NODE_ENV === 'production') {
        throw new Error('REDIS_URL is required in production for shared rate limiting and health verification.');
    }

    return DEFAULT_DEV_REDIS_URL;
}

function withTimeout(promise, timeoutMs, label) {
    let timer = null;
    const timeoutPromise = new Promise((_, reject) => {
        timer = setTimeout(() => {
            const err = new Error(`${label} timed out after ${timeoutMs}ms`);
            err.code = 'REDIS_TIMEOUT';
            reject(err);
        }, timeoutMs);
        if (typeof timer.unref === 'function') {
            timer.unref();
        }
    });
    return Promise.race([promise, timeoutPromise]).finally(() => {
        if (timer !== null) {
            clearTimeout(timer);
        }
    });
}

function createManagedEntry(redisUrl) {
    const connectTimeout = Math.max(250, Number(process.env.REDIS_CONNECT_TIMEOUT_MS || 2000));
    const client = createClient({
        url: redisUrl,
        disableOfflineQueue: true,
        socket: {
            connectTimeout,
            reconnectStrategy(retries) {
                if (process.env.REDIS_DISABLE_RECONNECT === 'true' || retries >= 10) {
                    return false;
                }
                return Math.min(2000, 100 * (retries + 1));
            },
        },
    });

    const entry = {
        url: redisUrl,
        client,
        connectPromise: null,
        lastError: null,
    };

    client.on('error', (err) => {
        entry.lastError = err?.message || 'Redis client error';
    });
    client.on('ready', () => {
        entry.lastError = null;
    });

    return entry;
}

async function getConnectedRedisClient(explicitUrl = null) {
    const redisUrl = resolveRedisUrl(explicitUrl);
    let entry = clientsByUrl.get(redisUrl);
    if (!entry) {
        entry = createManagedEntry(redisUrl);
        clientsByUrl.set(redisUrl, entry);
    }

    const { client } = entry;
    if (client.isReady) {
        return client;
    }

    const connectTimeoutMs = Math.max(250, Number(process.env.REDIS_CONNECT_TIMEOUT_MS || 2000));
    if (!entry.connectPromise) {
        const doConnect = async () => {
            if (!client.isOpen) {
                await client.connect();
            }
        };
        entry.connectPromise = withTimeout(doConnect(), connectTimeoutMs, `Redis connect (${sanitizeRedisUrl(redisUrl)})`)
            .catch(async (err) => {
                clientsByUrl.delete(redisUrl);
                try {
                    client.destroy();
                } catch (_destroyErr) {
                    // ignore cleanup failure
                }
                throw err;
            })
            .finally(() => {
                entry.connectPromise = null;
            });
    }

    await entry.connectPromise;
    if (!client.isReady) {
        throw new Error(`Redis client not ready (${sanitizeRedisUrl(redisUrl)}): ${entry.lastError || 'disconnected'}`);
    }
    return client;
}

async function checkRedisConnection(explicitUrl = null) {
    const started = Date.now();
    const timeoutMs = Math.max(250, Number(process.env.REDIS_COMMAND_TIMEOUT_MS || 2000));
    const client = await getConnectedRedisClient(explicitUrl);
    const pong = await withTimeout(client.ping(), timeoutMs, 'Redis PING');
    if (pong !== 'PONG') {
        throw new Error(`Unexpected Redis PING response: ${String(pong)}`);
    }
    return {
        connected: true,
        latencyMs: Date.now() - started,
    };
}

async function closeRedisClients() {
    const entries = Array.from(clientsByUrl.values());
    clientsByUrl.clear();
    await Promise.all(
        entries.map(async ({ client }) => {
            try {
                if (client.isOpen) {
                    await withTimeout(client.quit(), 1500, 'Redis QUIT').catch(() => client.destroy());
                } else {
                    client.destroy();
                }
            } catch (_err) {
                try {
                    client.destroy();
                } catch (_innerErr) {
                    // ignore
                }
            }
        })
    );
}

module.exports = {
    sanitizeRedisUrl,
    resolveRedisUrl,
    withTimeout,
    getConnectedRedisClient,
    checkRedisConnection,
    closeRedisClients,
};
