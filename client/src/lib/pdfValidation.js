import { DEFAULT_MAX_PDF_SIZE_BYTES } from '@/lib/api';

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 1024) return `${bytes || 0} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function validateClientPdfFile(file, maxSizeBytes = DEFAULT_MAX_PDF_SIZE_BYTES) {
  if (!file) {
    return 'Please select a PDF file to upload.';
  }
  const name = String(file.name || '').toLowerCase();
  if (!name.endsWith('.pdf')) {
    return 'Invalid file type: only .pdf files are accepted.';
  }
  const mime = String(file.type || '').trim().toLowerCase();
  if (mime && mime !== 'application/pdf') {
    return `Unsupported file MIME type "${mime}": expected application/pdf.`;
  }
  if (file.size === 0) {
    return 'The selected PDF file is empty (0 bytes).';
  }
  if (file.size > maxSizeBytes) {
    return `PDF size (${formatBytes(file.size)}) exceeds the maximum limit (${formatBytes(maxSizeBytes)}).`;
  }
  return null;
}
