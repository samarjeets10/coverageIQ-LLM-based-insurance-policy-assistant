const crypto = require('node:crypto');
const fs = require('node:fs/promises');
const path = require('node:path');
const apiError = require('../utils/apiError');

function getUploadRoot(explicitRoot = null) {
    const raw = explicitRoot !== null ? explicitRoot : process.env.POLICY_UPLOAD_ROOT || './uploads/policies';
    if (!raw || String(raw).trim() === '') {
        throw new apiError(500, 'Document storage root is not configured');
    }
    return path.resolve(String(raw).trim());
}

function sanitizeScopeSlug(ownerScope) {
    return String(ownerScope || 'unknown')
        .trim()
        .replace(/[^a-zA-Z0-9_-]/g, '_')
        .slice(0, 80);
}

function buildRelativeStorageKey({ ownerScope, policyId, versionNumber, documentId }) {
    const scopeSlug = sanitizeScopeSlug(ownerScope);
    return path.posix.join(
        scopeSlug,
        String(policyId),
        `v${Number(versionNumber)}_${String(documentId)}.pdf`
    );
}

async function resolveSafePath(storageRef, options = {}) {
    if (typeof storageRef !== 'string' || !storageRef.trim()) {
        throw new apiError(400, 'Invalid document storage reference');
    }
    const cleaned = storageRef.trim();
    if (cleaned.includes('\0')) {
        throw new apiError(400, 'Invalid character in document storage reference');
    }
    if (path.isAbsolute(cleaned) || cleaned.split(/[\\/]+/).includes('..')) {
        throw new apiError(400, 'Storage path traversal is forbidden');
    }

    const rootDir = getUploadRoot(options.rootDir ?? null);
    const realRoot = await fs.realpath(rootDir).catch(async () => {
        await fs.mkdir(rootDir, { recursive: true });
        return fs.realpath(rootDir);
    });

    const candidate = path.resolve(realRoot, cleaned);
    const rel = path.relative(realRoot, candidate);
    if (rel === '' || rel.startsWith('..') || path.isAbsolute(rel)) {
        throw new apiError(400, 'Resolved storage path escapes POLICY_UPLOAD_ROOT');
    }

    // Verify no symlinked ancestor or target escapes storage root
    const parentReal = await fs.realpath(path.dirname(candidate)).catch(() => null);
    if (parentReal !== null) {
        const parentRel = path.relative(realRoot, parentReal);
        if (parentRel.startsWith('..') || path.isAbsolute(parentRel)) {
            throw new apiError(400, 'Storage directory symlink escapes POLICY_UPLOAD_ROOT');
        }
    }

    return {
        rootDir: realRoot,
        absolutePath: candidate,
        relativeKey: cleaned.replace(/\\/g, '/'),
    };
}

async function writeDocumentBytes(relativeKey, buffer, options = {}) {
    const { absolutePath, relativeKey: cleanKey } = await resolveSafePath(relativeKey, options);
    await fs.mkdir(path.dirname(absolutePath), { recursive: true });

    const tmpPath = `${absolutePath}.${crypto.randomBytes(8).toString('hex')}.tmp`;
    try {
        await fs.writeFile(tmpPath, buffer, { flag: 'wx' });
        await fs.rename(tmpPath, absolutePath);
    } catch (err) {
        await fs.unlink(tmpPath).catch(() => {});
        throw err;
    }
    return {
        relativeKey: cleanKey,
        sizeBytes: buffer.length,
    };
}

async function deleteDocumentIfExists(relativeKey, options = {}) {
    if (!relativeKey) return false;
    try {
        const { absolutePath } = await resolveSafePath(relativeKey, options);
        await fs.unlink(absolutePath);
        return true;
    } catch (err) {
        if (err && err.code === 'ENOENT') {
            return false;
        }
        return false;
    }
}

async function documentExists(relativeKey, options = {}) {
    try {
        const { absolutePath } = await resolveSafePath(relativeKey, options);
        const stat = await fs.stat(absolutePath);
        return stat.isFile();
    } catch (_err) {
        return false;
    }
}

module.exports = {
    getUploadRoot,
    buildRelativeStorageKey,
    resolveSafePath,
    writeDocumentBytes,
    deleteDocumentIfExists,
    documentExists,
};
