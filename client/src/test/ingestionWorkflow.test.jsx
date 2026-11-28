import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AuthContext } from '@/context/useAuth';
import { ProtectedRoute } from '@/components/ProtectedRoute';
import { PolicyUploadPanel } from '@/components/PolicyUploadPanel';
import { IngestionJobsPanel } from '@/components/IngestionJobsPanel';
import { ingestionApi } from '@/lib/api';

function makeAuthWrapper(authValue) {
  return function Wrapper({ children }) {
    return (
      <AuthContext.Provider value={authValue}>
        {children}
      </AuthContext.Provider>
    );
  };
}

function makePdfFile(name = 'health_policy.pdf', size = 256, type = 'application/pdf') {
  const bytes = new Uint8Array(size);
  bytes.set([0x25, 0x50, 0x44, 0x46, 0x2d]); // %PDF-
  const file = new File([bytes], name, { type });
  Object.defineProperty(file, 'size', { value: size });
  return file;
}

describe('CoverageIQ Checkpoint 4.6 Frontend API, Upload & Job Tracking', () => {
  const originalXhr = globalThis.XMLHttpRequest;
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    globalThis.XMLHttpRequest = originalXhr;
    globalThis.fetch = originalFetch;
  });

  it('sends FormData via XHR without setting Content-Type, reports progress, handles 401 refresh & HTTP error codes', async () => {
    const requestRecords = [];

    class FakeXHR {
      constructor() {
        this.headers = {};
        this.upload = {};
        this.withCredentials = false;
      }
      open(method, url) {
        this.method = method;
        this.url = url;
      }
      setRequestHeader(k, v) {
        this.headers[k] = v;
      }
      getResponseHeader(name) {
        if (name.toLowerCase() === 'retry-after') {
          return this.retryAfter || null;
        }
        return null;
      }
      send(body) {
        requestRecords.push({
          method: this.method,
          url: this.url,
          headers: { ...this.headers },
          withCredentials: this.withCredentials,
          isFormData: body instanceof FormData,
        });

        if (requestRecords.length === 1) {
          // Simulate 401 first to confirm refresh-token retry
          this.status = 401;
          this.responseText = JSON.stringify({ message: 'Expired token' });
          this.onload();
          return;
        }

        if (typeof this.upload.onprogress === 'function') {
          this.upload.onprogress({ lengthComputable: true, loaded: 50, total: 100 });
          this.upload.onprogress({ lengthComputable: true, loaded: 100, total: 100 });
        }
        this.status = 201;
        this.responseText = JSON.stringify({
          statusCode: 201,
          success: true,
          message: 'Policy document uploaded and ingestion job queued successfully',
          data: {
            duplicate: false,
            document: { id: 'doc-1', version_number: 1, file_name: 'pol.pdf' },
            job: { id: 'job-1', status: 'queued', current_stage: 'QUEUED' },
          },
        });
        this.onload();
      }
    }

    globalThis.XMLHttpRequest = FakeXHR;
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: true });

    const progressEvents = [];
    const file = makePdfFile('pol.pdf', 100);
    const res = await ingestionApi.uploadDocument('pol-uuid-1', {
      file,
      versionLabel: '2026-v1',
      onProgress: (pct) => progressEvents.push(pct),
    });

    expect(res.data.job.id).toBe('job-1');
    expect(globalThis.fetch).toHaveBeenCalledWith(
      '/api/v1/users/refresh-token',
      expect.objectContaining({ method: 'POST', credentials: 'include' })
    );
    expect(requestRecords).toHaveLength(2);
    expect(requestRecords[0].url).toBe('/api/v1/policies/pol-uuid-1/documents');
    expect(requestRecords[0].withCredentials).toBe(true);
    expect(requestRecords[0].isFormData).toBe(true);
    expect(requestRecords[0].headers['Content-Type']).toBeUndefined();
    expect(progressEvents).toContain(50);
    expect(progressEvents).toContain(100);

    // Verify 403 / 409 / 413 / 415 / 422 / 429 parsing on fetch-based & XHR APIs
    for (const [status, expectedMsg, extra] of [
      [403, 'Policy upload denied', {}],
      [409, 'Archived policy', {}],
      [413, 'Payload too large', {}],
      [415, 'Unsupported MIME type', {}],
      [422, 'Encrypted PDF unsupported', { code: 'PASSWORD_PROTECTED_PDF' }],
      [429, 'Rate limit exceeded', { code: 'RATE_LIMIT_EXCEEDED', retryAfter: '12' }],
    ]) {
      globalThis.XMLHttpRequest = class {
        constructor() {
          this.upload = {};
        }
        open() {}
        getResponseHeader(h) {
          return h.toLowerCase() === 'retry-after' ? extra.retryAfter || null : null;
        }
        send() {
          this.status = status;
          this.responseText = JSON.stringify({
            message: expectedMsg,
            errors: extra.code ? [{ code: extra.code }] : [],
          });
          this.onload();
        }
      };
      await expect(
        ingestionApi.uploadDocument('pol-1', { file: makePdfFile() })
      ).rejects.toMatchObject({
        statusCode: status,
        message: expectedMsg,
        ...(extra.code ? { code: extra.code } : {}),
        ...(extra.retryAfter ? { retryAfterSeconds: 12 } : {}),
      });
    }
  });

  it('redirects unauthenticated visitors away from protected /dashboard route to /login', () => {
    const Wrapper = makeAuthWrapper({
      user: null,
      loading: false,
      login: vi.fn(),
      register: vi.fn(),
      logout: vi.fn(),
    });

    render(
      <Wrapper>
        <MemoryRouter initialEntries={['/dashboard']}>
          <Routes>
            <Route path="/login" element={<div>Login Screen</div>} />
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <div>Secret Workspace</div>
                </ProtectedRoute>
              }
            />
          </Routes>
        </MemoryRouter>
      </Wrapper>
    );

    expect(screen.getByText('Login Screen')).toBeInTheDocument();
    expect(screen.queryByText('Secret Workspace')).not.toBeInTheDocument();
  });

  it('validates files client-side, uploads PDF with progress, handles duplicates, and displays server errors', async () => {
    const samplePolicies = [
      {
        id: '11111111-1111-4111-8111-111111111111',
        external_policy_key: 'pol_health_1',
        title: 'Health Shield Plan',
        policy_type: 'health',
        status: 'draft',
        can_upload: true,
      },
    ];

    const uploadSpy = vi.spyOn(ingestionApi, 'uploadDocument');
    const onUploadCompleted = vi.fn();

    render(
      <PolicyUploadPanel
        policies={samplePolicies}
        userRole="user"
        maxSizeBytes={1024}
        onUploadCompleted={onUploadCompleted}
      />
    );

    const fileInput = screen.getByLabelText(/Policy Document/i);

    // 1. Reject non-PDF extension client-side
    const txtFile = new File(['hello'], 'notes.txt', { type: 'text/plain' });
    fireEvent.change(fileInput, { target: { files: [txtFile] } });
    expect(await screen.findByRole('alert')).toHaveTextContent(/only \.pdf files are accepted/i);
    expect(uploadSpy).not.toHaveBeenCalled();

    // 2. Reject oversized PDF client-side
    const bigPdf = makePdfFile('huge.pdf', 4096, 'application/pdf');
    fireEvent.change(fileInput, { target: { files: [bigPdf] } });
    expect(await screen.findByRole('alert')).toHaveTextContent(/exceeds the maximum limit/i);

    // 3. Valid upload queues job cleanly (does NOT claim ingestion completed)
    uploadSpy.mockImplementationOnce(async (_policyId, { onProgress }) => {
      onProgress(60);
      onProgress(100);
      return {
        statusCode: 201,
        success: true,
        message: 'Policy document uploaded and ingestion job queued successfully',
        data: {
          duplicate: false,
          document: { id: 'doc-10', version_number: 1, file_name: 'health_policy.pdf' },
          job: { id: 'job-10', status: 'queued', current_stage: 'QUEUED' },
        },
      };
    });

    fireEvent.change(fileInput, { target: { files: [makePdfFile('health_policy.pdf', 512)] } });
    fireEvent.click(screen.getByRole('button', { name: /Upload & Queue Ingestion/i }));

    expect(await screen.findByText(/Upload Complete — Ingestion Job Queued/i)).toBeInTheDocument();
    expect(screen.getByText(/Job ID: job-10/i)).toBeInTheDocument();
    expect(onUploadCompleted).toHaveBeenCalledTimes(1);

    // 4. Duplicate upload displays idempotent notice
    uploadSpy.mockResolvedValueOnce({
      statusCode: 200,
      success: true,
      message: 'Duplicate policy document detected',
      data: {
        duplicate: true,
        document: { id: 'doc-10', version_number: 1, file_name: 'health_policy.pdf' },
        job: { id: 'job-10', status: 'completed', current_stage: 'COMPLETED' },
      },
    });

    fireEvent.change(fileInput, { target: { files: [makePdfFile('health_policy.pdf', 512)] } });
    fireEvent.click(screen.getByRole('button', { name: /Upload & Queue Ingestion/i }));
    expect(await screen.findByText(/Duplicate PDF Detected/i)).toBeInTheDocument();

    // 5. Server validation error (422 encrypted PDF) surfaced clearly
    const encErr = new Error('Encrypted or password-protected PDF documents are not supported');
    encErr.statusCode = 422;
    encErr.code = 'PASSWORD_PROTECTED_PDF';
    uploadSpy.mockRejectedValueOnce(encErr);

    fireEvent.change(fileInput, { target: { files: [makePdfFile('locked.pdf', 512)] } });
    fireEvent.click(screen.getByRole('button', { name: /Upload & Queue Ingestion/i }));
    const errAlert = await screen.findByRole('alert');
    expect(errAlert).toHaveTextContent(/HTTP 422/);
    expect(errAlert).toHaveTextContent(/PASSWORD_PROTECTED_PDF/);
    expect(errAlert).toHaveTextContent(/password-protected PDF/i);
  });

  it('tracks jobs via controlled polling, displays sanitized failure info, filters/paginates, and stops polling at terminal state', async () => {
    let pollStep = 0;
    const jobStates = [
      {
        id: 'job-poll-1',
        policy_id: 'pol-1',
        document_id: 'doc-1',
        external_policy_key: 'pol_health_1',
        document_version_snapshot: 1,
        status: 'queued',
        current_stage: 'QUEUED',
        progress: { total_chunks: 0, processed_chunks: 0, percent: 0 },
        error: null,
        created_at: '2026-11-27T15:00:00Z',
        started_at: null,
        completed_at: null,
        policy: { id: 'pol-1', title: 'Health Shield Plan', status: 'draft' },
        document: { id: 'doc-1', file_name: 'health.pdf', ingestion_status: 'queued' },
      },
      {
        id: 'job-poll-1',
        policy_id: 'pol-1',
        document_id: 'doc-1',
        external_policy_key: 'pol_health_1',
        document_version_snapshot: 1,
        status: 'processing',
        current_stage: 'EMBEDDING',
        progress: { total_chunks: 8, processed_chunks: 4, percent: 50 },
        error: null,
        created_at: '2026-11-27T15:00:00Z',
        started_at: '2026-11-27T15:00:02Z',
        completed_at: null,
        policy: { id: 'pol-1', title: 'Health Shield Plan', status: 'draft' },
        document: { id: 'doc-1', file_name: 'health.pdf', ingestion_status: 'processing' },
      },
      {
        id: 'job-poll-1',
        policy_id: 'pol-1',
        document_id: 'doc-1',
        external_policy_key: 'pol_health_1',
        document_version_snapshot: 1,
        status: 'completed',
        current_stage: 'COMPLETED',
        progress: { total_chunks: 8, processed_chunks: 8, percent: 100 },
        error: null,
        created_at: '2026-11-27T15:00:00Z',
        started_at: '2026-11-27T15:00:02Z',
        completed_at: '2026-11-27T15:00:05Z',
        policy: { id: 'pol-1', title: 'Health Shield Plan', status: 'active' },
        document: { id: 'doc-1', file_name: 'health.pdf', ingestion_status: 'completed' },
      },
    ];

    const listSpy = vi.spyOn(ingestionApi, 'listJobs').mockImplementation(async ({ status, page }) => {
      const idx = Math.min(pollStep, jobStates.length - 1);
      if (status === 'failed') {
        return {
          data: {
            items: [
              {
                id: 'job-fail-1',
                policy_id: 'pol-1',
                document_id: 'doc-2',
                external_policy_key: 'pol_health_1',
                document_version_snapshot: 2,
                status: 'failed',
                current_stage: 'FAILED',
                progress: { total_chunks: 0, processed_chunks: 0, percent: 0 },
                error: { code: 'INVALID_PDF_SIGNATURE', message: 'Invalid PDF header signature.' },
                created_at: '2026-11-27T15:05:00Z',
                started_at: '2026-11-27T15:05:01Z',
                completed_at: '2026-11-27T15:05:01Z',
                policy: { id: 'pol-1', title: 'Health Shield Plan', status: 'active' },
                document: { id: 'doc-2', file_name: 'corrupt.pdf', ingestion_status: 'failed' },
              },
            ],
            pagination: {
              page: page || 1,
              limit: 6,
              total_items: 8,
              total_pages: 2,
              has_next: (page || 1) < 2,
              has_prev: (page || 1) > 1,
            },
          },
        };
      }
      return {
        data: {
          items: [jobStates[idx]],
          pagination: {
            page: page || 1,
            limit: 6,
            total_items: 1,
            total_pages: 1,
            has_next: false,
            has_prev: false,
          },
        },
      };
    });

    const getSpy = vi.spyOn(ingestionApi, 'getJob').mockImplementation(async (id) => {
      if (id === 'job-fail-1') {
        return {
          data: {
            job: {
              id: 'job-fail-1',
              policy_id: 'pol-1',
              document_id: 'doc-2',
              external_policy_key: 'pol_health_1',
              document_version_snapshot: 2,
              status: 'failed',
              current_stage: 'FAILED',
              progress: { total_chunks: 0, processed_chunks: 0, percent: 0 },
              error: { code: 'INVALID_PDF_SIGNATURE', message: 'Invalid PDF header signature.' },
              created_at: '2026-11-27T15:05:00Z',
              started_at: '2026-11-27T15:05:01Z',
              completed_at: '2026-11-27T15:05:01Z',
              policy: { id: 'pol-1', title: 'Health Shield Plan', status: 'active' },
              document: { id: 'doc-2', file_name: 'corrupt.pdf', ingestion_status: 'failed' },
            },
          },
        };
      }
      const cur = jobStates[Math.min(pollStep, jobStates.length - 1)];
      pollStep += 1;
      return { data: { job: cur } };
    });

    render(
      <IngestionJobsPanel
        policies={[{ id: 'pol-1', external_policy_key: 'pol_health_1', title: 'Health Shield Plan' }]}
        activeJobId="job-poll-1"
        pollIntervalMs={40}
        maxPollCycles={20}
      />
    );

    // Wait for job to progress through queued -> processing -> completed
    await waitFor(
      () => {
        expect(screen.getByText('COMPLETED')).toBeInTheDocument();
        expect(screen.getByText('Chunks: 8 / 8')).toBeInTheDocument();
      },
      { timeout: 2500 }
    );

    // Confirm polling actually stops after terminal state (Idle badge shown, call count stabilizes)
    expect(screen.getByText('Idle')).toBeInTheDocument();
    const callsAtCompletion = getSpy.mock.calls.length;
    await new Promise((r) => setTimeout(r, 180));
    expect(getSpy.mock.calls.length).toBe(callsAtCompletion);

    // Filter to failed jobs and click failed job row to inspect sanitized error info
    fireEvent.change(screen.getByLabelText(/Filter jobs by status/i), {
      target: { value: 'failed' },
    });
    const failRowBtn = await screen.findByRole('button', { name: /corrupt\.pdf/i });
    fireEvent.click(failRowBtn);

    const errBanner = await screen.findByRole('alert');
    expect(errBanner).toHaveTextContent(/INVALID_PDF_SIGNATURE/);
    expect(errBanner).toHaveTextContent(/Invalid PDF header signature/);
    expect(listSpy).toHaveBeenCalledWith(expect.objectContaining({ status: 'failed' }));

    // Verify pagination Next advances to page 2
    fireEvent.click(screen.getByRole('button', { name: /^Next$/i }));
    await waitFor(() => {
      expect(listSpy).toHaveBeenCalledWith(expect.objectContaining({ status: 'failed', page: 2 }));
    });
  });
});
