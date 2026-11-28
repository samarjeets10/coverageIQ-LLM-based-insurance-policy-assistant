import { useRef, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Copy,
  FileText,
  FolderPlus,
  Info,
  UploadCloud,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import {
  DEFAULT_MAX_PDF_SIZE_BYTES,
  ingestionApi,
  policiesApi,
} from '@/lib/api';
import { formatBytes, validateClientPdfFile } from '@/lib/pdfValidation';

const POLICY_TYPES = [
  { value: 'health', label: 'Health' },
  { value: 'car', label: 'Car / Motor' },
  { value: 'life', label: 'Life' },
  { value: 'term', label: 'Term' },
  { value: 'travel', label: 'Travel' },
  { value: 'pension', label: 'Pension' },
  { value: 'general', label: 'General' },
];

export function PolicyUploadPanel({
  policies = [],
  policiesLoading = false,
  userRole = 'user',
  maxSizeBytes = DEFAULT_MAX_PDF_SIZE_BYTES,
  onPolicyCreated,
  onUploadCompleted,
}) {
  const fileInputRef = useRef(null);

  const writablePolicies = policies.filter((p) => p.can_upload !== false);

  const [manualPolicyMode, setManualPolicyMode] = useState(false);
  const [selectedPolicyId, setSelectedPolicyId] = useState('');
  const [manualPolicyId, setManualPolicyId] = useState('');

  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [newPolicyType, setNewPolicyType] = useState('health');
  const [newInsurer, setNewInsurer] = useState('');
  const [publishToCatalog, setPublishToCatalog] = useState(false);
  const [creatingPolicy, setCreatingPolicy] = useState(false);

  const canPublishCatalog = userRole === 'organisation' || userRole === 'admin';

  const [selectedFile, setSelectedFile] = useState(null);
  const [versionLabel, setVersionLabel] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const [errorBanner, setErrorBanner] = useState(null);
  const [noticeBanner, setNoticeBanner] = useState(null);

  const effectivePolicyId = manualPolicyMode
    ? manualPolicyId.trim()
    : selectedPolicyId || writablePolicies[0]?.id || '';

  function handleFileChange(event) {
    setErrorBanner(null);
    setNoticeBanner(null);
    const file = event.target.files?.[0] || null;
    if (!file) {
      setSelectedFile(null);
      return;
    }
    const validationError = validateClientPdfFile(file, maxSizeBytes);
    if (validationError) {
      setSelectedFile(null);
      setErrorBanner({
        title: 'Invalid PDF selected',
        message: validationError,
      });
      return;
    }
    setSelectedFile(file);
  }

  async function handleCreatePolicy(event) {
    event.preventDefault();
    setErrorBanner(null);
    setCreatingPolicy(true);
    try {
      const res = await policiesApi.create({
        title: newTitle,
        policy_type: newPolicyType,
        insurer_name: newInsurer.trim() || undefined,
        ...(canPublishCatalog && publishToCatalog ? { visibility: 'catalog' } : {}),
      });
      const created = res.data?.policy;
      if (created) {
        setSelectedPolicyId(created.id);
        setManualPolicyMode(false);
        setShowCreateForm(false);
        setNewTitle('');
        setNewInsurer('');
        setPublishToCatalog(false);
        if (typeof onPolicyCreated === 'function') {
          onPolicyCreated(created);
        }
      }
    } catch (err) {
      setErrorBanner({
        title: `Could not create policy${err.statusCode ? ` (HTTP ${err.statusCode})` : ''}`,
        code: err.code || null,
        message: err.message || 'Failed to create policy.',
      });
    } finally {
      setCreatingPolicy(false);
    }
  }

  async function handleUploadSubmit(event) {
    event.preventDefault();
    setErrorBanner(null);
    setNoticeBanner(null);

    if (!effectivePolicyId) {
      setErrorBanner({
        title: 'Policy required',
        message: 'Select or create a policy before uploading a PDF.',
      });
      return;
    }

    const validationError = validateClientPdfFile(selectedFile, maxSizeBytes);
    if (validationError) {
      setErrorBanner({
        title: 'File validation failed',
        message: validationError,
      });
      return;
    }

    setUploading(true);
    setUploadProgress(0);
    try {
      const res = await ingestionApi.uploadDocument(effectivePolicyId, {
        file: selectedFile,
        versionLabel,
        onProgress: (pct) => setUploadProgress(pct),
      });

      const payload = res.data || {};
      const job = payload.job || null;
      const doc = payload.document || null;
      const isDuplicate = Boolean(payload.duplicate);
      const isDelayed = payload.dispatch?.state === 'retry_scheduled' || res.statusCode === 202;

      setNoticeBanner({
        kind: isDuplicate ? 'duplicate' : isDelayed ? 'delayed' : 'queued',
        message: res.message,
        jobId: job?.id || null,
        jobStatus: job?.status || null,
        currentStage: job?.current_stage || null,
        documentVersion: doc?.version_number || null,
        fileName: doc?.file_name || selectedFile.name,
      });

      setSelectedFile(null);
      setVersionLabel('');
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }

      if (typeof onUploadCompleted === 'function') {
        onUploadCompleted({
          response: res,
          job,
          document: doc,
          policyId: effectivePolicyId,
        });
      }
    } catch (err) {
      setErrorBanner({
        title: `Upload failed${err.statusCode ? ` (HTTP ${err.statusCode})` : ''}`,
        code: err.code || null,
        message: err.message || 'Upload failed.',
      });
    } finally {
      setUploading(false);
    }
  }

  return (
    <Card className="p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-slate-900">
            Upload Policy PDF
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            Validated PDFs are stored cleanly and queued for asynchronous BGE-M3
            ingestion and indexing.
          </p>
        </div>
        <Badge variant="default">Max {formatBytes(maxSizeBytes)}</Badge>
      </div>

      {/* Policy selector + New Policy creator */}
      <div className="mt-5 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <label
            htmlFor="policy-selector"
            className="text-sm font-medium text-slate-700"
          >
            Target Policy
          </label>
          <div className="flex items-center gap-3 text-xs font-medium">
            <button
              type="button"
              onClick={() => setManualPolicyMode((v) => !v)}
              className="text-slate-500 underline-offset-2 hover:text-slate-900 hover:underline cursor-pointer"
            >
              {manualPolicyMode ? 'Use policy list' : 'Enter Policy UUID'}
            </button>
            <button
              type="button"
              onClick={() => setShowCreateForm((v) => !v)}
              className="inline-flex items-center gap-1 text-blue-600 hover:text-blue-700 cursor-pointer"
            >
              <FolderPlus className="h-3.5 w-3.5" />
              {showCreateForm ? 'Cancel' : 'New policy'}
            </button>
          </div>
        </div>

        {manualPolicyMode ? (
          <Input
            id="policy-selector"
            placeholder="Enter target Policy UUID (e.g. b6d9f3a8-...)"
            value={manualPolicyId}
            onChange={(e) => setManualPolicyId(e.target.value)}
            disabled={uploading}
          />
        ) : (
          <select
            id="policy-selector"
            aria-label="Target Policy"
            value={effectivePolicyId}
            onChange={(e) => setSelectedPolicyId(e.target.value)}
            disabled={uploading || policiesLoading}
            className="flex h-11 w-full rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-sm text-slate-900 focus:border-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-600/15 disabled:opacity-50"
          >
            {policiesLoading ? (
              <option value="">Loading policies...</option>
            ) : writablePolicies.length === 0 ? (
              <option value="">No policies yet — create one to upload</option>
            ) : (
              writablePolicies.map((pol) => (
                <option key={pol.id} value={pol.id}>
                  {pol.title} ({pol.policy_type} · {pol.status} · {pol.external_policy_key})
                </option>
              ))
            )}
          </select>
        )}

        {showCreateForm && (
          <form
            onSubmit={handleCreatePolicy}
            className="rounded-xl border border-slate-200 bg-[#FAFAFA] p-4 space-y-3"
          >
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-600">
              Create Draft Policy ({userRole})
            </p>
            <div className="grid gap-3 sm:grid-cols-2">
              <Input
                placeholder="Policy title (e.g. Health Shield Plan)"
                required
                value={newTitle}
                onChange={(e) => setNewTitle(e.target.value)}
              />
              <select
                aria-label="Policy Type"
                value={newPolicyType}
                onChange={(e) => setNewPolicyType(e.target.value)}
                className="flex h-11 w-full rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-sm text-slate-900"
              >
                {POLICY_TYPES.map((t) => (
                  <option key={t.value} value={t.value}>
                    {t.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <Input
                className="flex-1 min-w-45"
                placeholder="Insurer name (optional)"
                value={newInsurer}
                onChange={(e) => setNewInsurer(e.target.value)}
              />
              <Button type="submit" size="sm" disabled={creatingPolicy}>
                {creatingPolicy ? 'Creating...' : 'Create Policy'}
              </Button>
            </div>
            {canPublishCatalog && (
              <label className="flex items-start gap-2.5 rounded-lg border border-slate-200 bg-white px-3 py-2.5 text-xs text-slate-700 cursor-pointer">
                <input
                  type="checkbox"
                  checked={publishToCatalog}
                  onChange={(e) => setPublishToCatalog(e.target.checked)}
                  className="mt-0.5 h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                />
                <span>
                  <span className="font-medium text-slate-900">Publish to Recommendations Catalog</span>
                  <span className="block text-slate-500">
                    Leave unchecked to keep this policy private to your organisation. When checked, completed PDF uploads become eligible for catalog recommendations.
                  </span>
                </span>
              </label>
            )}
          </form>
        )}
      </div>

      {/* Upload Form */}
      <form onSubmit={handleUploadSubmit} className="mt-5 space-y-4">
        <div className="space-y-1.5">
          <label
            htmlFor="policy-pdf-input"
            className="text-sm font-medium text-slate-700"
          >
            Policy Document (.pdf)
          </label>
          <label
            htmlFor="policy-pdf-input"
            className="flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-slate-300 bg-[#FAFAFA] px-4 py-6 text-center transition-colors hover:border-slate-400 hover:bg-slate-50"
          >
            <UploadCloud className="h-6 w-6 text-blue-600" />
            <span className="mt-2 text-sm font-medium text-slate-800">
              {selectedFile
                ? selectedFile.name
                : 'Choose a PDF file from your computer'}
            </span>
            <span className="mt-0.5 text-xs text-slate-500">
              {selectedFile
                ? `${formatBytes(selectedFile.size)} · Ready to validate and upload`
                : 'Must be an unencrypted application/pdf document'}
            </span>
            <input
              ref={fileInputRef}
              id="policy-pdf-input"
              type="file"
              accept=".pdf,application/pdf"
              onChange={handleFileChange}
              disabled={uploading}
              className="sr-only"
            />
          </label>
        </div>

        <div className="space-y-1.5">
          <label
            htmlFor="version-label-input"
            className="text-sm font-medium text-slate-700"
          >
            Version Label <span className="text-slate-400">(optional)</span>
          </label>
          <Input
            id="version-label-input"
            placeholder="e.g. 2026-annual-renewal"
            maxLength={64}
            value={versionLabel}
            onChange={(e) => setVersionLabel(e.target.value)}
            disabled={uploading}
          />
        </div>

        {uploading && (
          <div className="space-y-1.5" role="status" aria-live="polite">
            <div className="flex items-center justify-between text-xs font-medium text-slate-600">
              <span>Uploading PDF to gateway...</span>
              <span>{uploadProgress}%</span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
              <div
                className="h-full bg-blue-600 transition-all duration-150"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          </div>
        )}

        {errorBanner && (
          <div
            role="alert"
            className="flex items-start gap-2.5 rounded-xl border border-red-200 bg-red-50 px-3.5 py-3 text-sm text-red-800"
          >
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-600" />
            <div className="space-y-0.5">
              <p className="font-semibold">
                {errorBanner.title}
                {errorBanner.code ? ` · ${errorBanner.code}` : ''}
              </p>
              <p>{errorBanner.message}</p>
            </div>
          </div>
        )}

        {noticeBanner && (
          <div
            role="status"
            className={
              noticeBanner.kind === 'duplicate'
                ? 'rounded-xl border border-blue-200 bg-blue-50 px-3.5 py-3 text-sm text-blue-900'
                : noticeBanner.kind === 'delayed'
                  ? 'rounded-xl border border-amber-200 bg-amber-50 px-3.5 py-3 text-sm text-amber-900'
                  : 'rounded-xl border border-emerald-200 bg-emerald-50 px-3.5 py-3 text-sm text-emerald-900'
            }
          >
            <div className="flex items-start gap-2.5">
              {noticeBanner.kind === 'duplicate' ? (
                <Copy className="mt-0.5 h-4 w-4 shrink-0 text-blue-600" />
              ) : noticeBanner.kind === 'delayed' ? (
                <Info className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
              ) : (
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" />
              )}
              <div className="space-y-1">
                <p className="font-semibold">
                  {noticeBanner.kind === 'duplicate'
                    ? 'Duplicate PDF Detected (Already Indexed / Queued)'
                    : noticeBanner.kind === 'delayed'
                      ? 'Upload Saved — Background Dispatch Pending'
                      : 'Upload Complete — Ingestion Job Queued'}
                </p>
                <p>
                  {noticeBanner.fileName}
                  {noticeBanner.documentVersion
                    ? ` (v${noticeBanner.documentVersion})`
                    : ''}{' '}
                  {noticeBanner.kind === 'duplicate'
                    ? 'matched an existing SHA-256 upload.'
                    : 'has been queued for parsing and embedding — monitor job completion below.'}
                </p>
                {noticeBanner.jobId && (
                  <p className="flex flex-wrap items-center gap-2 text-xs font-mono">
                    <span>Job ID: {noticeBanner.jobId}</span>
                    {noticeBanner.jobStatus && (
                      <Badge variant="default" className="py-0 text-xs">
                        {noticeBanner.jobStatus}
                      </Badge>
                    )}
                    {noticeBanner.currentStage && (
                      <span>Stage: {noticeBanner.currentStage}</span>
                    )}
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        <Button
          type="submit"
          className="w-full"
          disabled={uploading || !selectedFile || !effectivePolicyId}
        >
          <FileText className="h-4 w-4" />
          {uploading ? `Uploading (${uploadProgress}%)...` : 'Upload & Queue Ingestion'}
        </Button>
      </form>
    </Card>
  );
}
