const crypto = require('node:crypto');
const path = require('node:path');
const apiError = require('../utils/apiError');

const DEFAULT_MAX_PDF_SIZE_BYTES = 25 * 1024 * 1024;
const MAGIC_SEARCH_WINDOW = 1024;
const TAIL_SEARCH_WINDOW = 4096;

function getMaxPdfSizeBytes() {
    const raw = process.env.MAX_PDF_SIZE_BYTES;
    if (raw === undefined || raw === null || String(raw).trim() === '') {
        return DEFAULT_MAX_PDF_SIZE_BYTES;
    }
    const parsed = Number(raw);
    if (!Number.isFinite(parsed) || parsed <= 0) {
        return DEFAULT_MAX_PDF_SIZE_BYTES;
    }
    return Math.floor(parsed);
}

function sanitizeOriginalFilename(rawName) {
    const base = path.basename(String(rawName || 'policy.pdf'));
    const cleaned = base.replace(/[\x00-\x1f\x7f\\/:*?"<>|]/g, '_').trim();
    return cleaned || 'policy.pdf';
}

function containsPdfEncryptionDictionary(buffer) {
    // Detect standard PDF /Encrypt indirect reference or inline dictionary entries
    const searchText = buffer.toString('latin1');
    return /\/Encrypt\s+\d+\s+\d+\s+R\b/.test(searchText)
        || /\/Encrypt\s*<</.test(searchText);
}

function hasValidPdfStructure(buffer) {
    const len = buffer.length;
    const tailStart = Math.max(0, len - TAIL_SEARCH_WINDOW);
    const tail = buffer.subarray(tailStart).toString('latin1');

    const hasEof = tail.includes('%%EOF');
    const hasStartXref = tail.includes('startxref');
    if (!hasEof || !hasStartXref) {
        return false;
    }

    const headAndTrailer = buffer.toString('latin1');
    const hasObject = /\b\d+\s+\d+\s+obj\b/.test(headAndTrailer) && /\bendobj\b/.test(headAndTrailer);
    const hasXrefOrStream = /\bxref\b/.test(headAndTrailer) || /\/Type\s*\/XRef\b/.test(headAndTrailer);
    return hasObject && hasXrefOrStream;
}

function validateUploadedPdf(file, options = {}) {
    if (!file || !Buffer.isBuffer(file.buffer)) {
        throw new apiError(400, 'A PDF file upload field named "file" is required', [
            { code: 'FILE_REQUIRED' },
        ]);
    }

    const maxBytes = options.maxSizeBytes || getMaxPdfSizeBytes();
    const buffer = file.buffer;
    const sanitizedFilename = sanitizeOriginalFilename(file.originalname);

    if (path.extname(sanitizedFilename).toLowerCase() !== '.pdf') {
        throw new apiError(400, 'Invalid file extension: only .pdf files are supported', [
            { code: 'INVALID_FILE_EXTENSION' },
        ]);
    }

    const mime = String(file.mimetype || '').trim().toLowerCase();
    if (mime !== 'application/pdf') {
        throw new apiError(415, 'Unsupported media type: Content-Type must be application/pdf', [
            { code: 'UNSUPPORTED_MEDIA_TYPE' },
        ]);
    }

    if (buffer.length === 0) {
        throw new apiError(400, 'Uploaded PDF file is empty (0 bytes)', [
            { code: 'EMPTY_PDF' },
        ]);
    }

    if (buffer.length > maxBytes) {
        throw new apiError(413, 'Uploaded PDF exceeds the maximum allowed size limit', [
            { code: 'UPLOAD_TOO_LARGE', maxSizeBytes: maxBytes },
        ]);
    }

    const magicWindow = buffer.subarray(0, Math.min(MAGIC_SEARCH_WINDOW, buffer.length));
    if (!magicWindow.includes(Buffer.from('%PDF-', 'ascii'))) {
        throw new apiError(400, 'Invalid PDF signature: missing %PDF- file header', [
            { code: 'INVALID_PDF_SIGNATURE' },
        ]);
    }

    if (containsPdfEncryptionDictionary(buffer)) {
        throw new apiError(422, 'Encrypted or password-protected PDF documents are not supported', [
            { code: 'PASSWORD_PROTECTED_PDF' },
        ]);
    }

    if (!hasValidPdfStructure(buffer)) {
        throw new apiError(422, 'Malformed or corrupted PDF structure detected', [
            { code: 'CORRUPTED_PDF' },
        ]);
    }

    const sha256 = crypto.createHash('sha256').update(buffer).digest('hex');

    return {
        sha256,
        fileSizeBytes: buffer.length,
        sanitizedFilename,
        mimeType: 'application/pdf',
    };
}

module.exports = {
    DEFAULT_MAX_PDF_SIZE_BYTES,
    getMaxPdfSizeBytes,
    sanitizeOriginalFilename,
    validateUploadedPdf,
};
