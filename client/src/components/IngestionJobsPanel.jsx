import { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  Loader2,
  RefreshCw,
  XCircle,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { ingestionApi } from '@/lib/api';

const TERMINAL_STATUSES = new Set(['completed', 'failed', 'cancelled']);

function isTerminalStatus(status) {
  return TERMINAL_STATUSES.has(String(status || '').toLowerCase());
}

function getStatusBadgeVariant(status) {
  switch (String(status || '').toLowerCase()) {
    case 'completed':
      return 'success';
    case 'processing':
      return 'accent';
    case 'queued':
      return 'warning';
    case 'failed':
      return 'danger';
    default:
      return 'default';
  }
}

function formatTimestamp(ts) {
  if (!ts) return '—';
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) return '—';
  return date.toLocaleString();
}

export function IngestionJobsPanel({
  policies = [],
  activeJobId = null,
  onSelectJob,
  onJobCompleted,
  refreshNonce = 0,
  pollIntervalMs = 2500,
  maxPollCycles = 120,
}) {
  const [jobs, setJobs] = useState([]);
  const [pagination, setPagination] = useState({
    page: 1,
    limit: 6,
    total_items: 0,
    total_pages: 1,
    has_next: false,
    has_prev: false,
  });
  const [page, setPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState('');
  const [policyFilter, setPolicyFilter] = useState('');
  const [loading, setLoading] = useState(true);
  const [listError, setListError] = useState('');

  const [localSelection, setLocalSelection] = useState(null);
  const [selectedJob, setSelectedJob] = useState(null);
  const [detailError, setDetailError] = useState('');
  const [manualTick, setManualTick] = useState(0);
  const [pollExhausted, setPollExhausted] = useState(false);

  const pollCycleRef = useRef(0);
  const trackedInFlightRef = useRef(new Set());
  const notifiedCompletedRef = useRef(new Set());
  const onJobCompletedRef = useRef(onJobCompleted);

  useEffect(() => {
    onJobCompletedRef.current = onJobCompleted;
  }, [onJobCompleted]);

  useEffect(() => {
    if (activeJobId && !notifiedCompletedRef.current.has(activeJobId)) {
      trackedInFlightRef.current.add(activeJobId);
    }
  }, [activeJobId]);

  function checkAndNotifyCompletedJobs(jobCandidates = []) {
    for (const candidate of jobCandidates) {
      if (!candidate || !candidate.id) continue;
      const st = String(candidate.status || '').toLowerCase();
      if (!isTerminalStatus(st)) {
        trackedInFlightRef.current.add(candidate.id);
      } else if (
        st === 'completed'
        && trackedInFlightRef.current.has(candidate.id)
        && !notifiedCompletedRef.current.has(candidate.id)
      ) {
        notifiedCompletedRef.current.add(candidate.id);
        trackedInFlightRef.current.delete(candidate.id);
        if (typeof onJobCompletedRef.current === 'function') {
          onJobCompletedRef.current(candidate);
        }
      }
    }
  }
  const userPickedId =
    localSelection && localSelection.forActiveJobId === activeJobId
      ? localSelection.jobId
      : null;
  const effectiveSelectedId = userPickedId || activeJobId || jobs[0]?.id || null;

  async function syncSnapshot({ targetJobId = effectiveSelectedId } = {}) {
    try {
      const res = await ingestionApi.listJobs({
        page,
        limit: 6,
        status: statusFilter,
        policyId: policyFilter,
      });
      const items = res.data?.items || [];
      const pag = res.data?.pagination || {};
      setJobs(items);
      checkAndNotifyCompletedJobs(items);
      setPagination({
        page: pag.page || page,
        limit: pag.limit || 6,
        total_items: Number(pag.total_items || 0),
        total_pages: Number(pag.total_pages || 1),
        has_next: Boolean(pag.has_next),
        has_prev: Boolean(pag.has_prev),
      });
      setListError('');

      const lookupId = targetJobId || items[0]?.id || null;
      if (lookupId) {
        try {
          const detailRes = await ingestionApi.getJob(lookupId);
          setSelectedJob(detailRes.data?.job || null);
          checkAndNotifyCompletedJobs(detailRes.data?.job ? [detailRes.data.job] : []);
          setDetailError('');
        } catch (detailErr) {
          setDetailError(detailErr.message || 'Could not load job details.');
        }
      } else {
        setSelectedJob(null);
        setDetailError('');
      }
    } catch (err) {
      setListError(err.message || 'Could not fetch ingestion jobs.');
    } finally {
      setLoading(false);
    }
  }

  // Async effect subscription: all setState calls happen asynchronously after await
  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const res = await ingestionApi.listJobs({
          page,
          limit: 6,
          status: statusFilter,
          policyId: policyFilter,
        });
        if (cancelled) return;

        const items = res.data?.items || [];
        const pag = res.data?.pagination || {};
        setJobs(items);
        checkAndNotifyCompletedJobs(items);
        setPagination({
          page: pag.page || page,
          limit: pag.limit || 6,
          total_items: Number(pag.total_items || 0),
          total_pages: Number(pag.total_pages || 1),
          has_next: Boolean(pag.has_next),
          has_prev: Boolean(pag.has_prev),
        });
        setListError('');

        const isFiltered = Boolean(statusFilter || policyFilter);
        const activeMatchesFilter = activeJobId && items.some((j) => j.id === activeJobId);
        const lookupId =
          userPickedId ||
          (isFiltered ? (activeMatchesFilter ? activeJobId : items[0]?.id) : activeJobId || items[0]?.id) ||
          null;
        if (lookupId) {
          try {
            const detailRes = await ingestionApi.getJob(lookupId);
            if (!cancelled) {
              setSelectedJob(detailRes.data?.job || null);
              checkAndNotifyCompletedJobs(detailRes.data?.job ? [detailRes.data.job] : []);
              setDetailError('');
            }
          } catch (detailErr) {
            if (!cancelled) {
              setDetailError(detailErr.message || 'Could not load job details.');
            }
          }
        } else if (!cancelled) {
          setSelectedJob(null);
          setDetailError('');
        }
      } catch (err) {
        if (!cancelled) {
          setListError(err.message || 'Could not fetch ingestion jobs.');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [page, statusFilter, policyFilter, activeJobId, userPickedId, refreshNonce, manualTick]);

  const hasNonTerminal =
    (selectedJob && !isTerminalStatus(selectedJob.status)) ||
    jobs.some((j) => !isTerminalStatus(j.status));

  // Controlled bounded polling: automatically stops when all tracked jobs reach terminal states
  useEffect(() => {
    if (!hasNonTerminal || pollExhausted || pollIntervalMs <= 0) {
      return undefined;
    }

    let cancelled = false;
    const timerId = setInterval(async () => {
      if (cancelled) return;
      if (pollCycleRef.current >= maxPollCycles) {
        setPollExhausted(true);
        return;
      }
      pollCycleRef.current += 1;
      try {
        const res = await ingestionApi.listJobs({
          page,
          limit: 6,
          status: statusFilter,
          policyId: policyFilter,
        });
        if (cancelled) return;
        const items = res.data?.items || [];
        const pag = res.data?.pagination || {};
        setJobs(items);
        checkAndNotifyCompletedJobs(items);
        setPagination({
          page: pag.page || page,
          limit: pag.limit || 6,
          total_items: Number(pag.total_items || 0),
          total_pages: Number(pag.total_pages || 1),
          has_next: Boolean(pag.has_next),
          has_prev: Boolean(pag.has_prev),
        });
        const isFiltered = Boolean(statusFilter || policyFilter);
        const activeMatchesFilter = activeJobId && items.some((j) => j.id === activeJobId);
        const lookupId =
          userPickedId ||
          (isFiltered ? (activeMatchesFilter ? activeJobId : items[0]?.id) : activeJobId || items[0]?.id) ||
          null;
        if (lookupId) {
          const detailRes = await ingestionApi.getJob(lookupId);
          if (!cancelled) {
            setSelectedJob(detailRes.data?.job || null);
            checkAndNotifyCompletedJobs(detailRes.data?.job ? [detailRes.data.job] : []);
            setDetailError('');
          }
        }
      } catch (err) {
        if (!cancelled) {
          setListError(err.message || 'Could not refresh ingestion jobs.');
        }
      }
    }, pollIntervalMs);

    return () => {
      cancelled = true;
      clearInterval(timerId);
    };
  }, [
    hasNonTerminal,
    pollExhausted,
    pollIntervalMs,
    maxPollCycles,
    page,
    statusFilter,
    policyFilter,
    activeJobId,
    userPickedId,
  ]);

  async function handleSelectJob(jobId) {
    setLocalSelection({ forActiveJobId: activeJobId, jobId });
    setPollExhausted(false);
    pollCycleRef.current = 0;
    if (typeof onSelectJob === 'function') {
      onSelectJob(jobId);
    }
    await syncSnapshot({ targetJobId: jobId });
  }

  function handleManualRefresh() {
    setLoading(true);
    pollCycleRef.current = 0;
    setPollExhausted(false);
    setManualTick((t) => t + 1);
  }

  return (
    <Card className="p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-slate-900">
            Ingestion Jobs
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            Track validation, text/OCR extraction, chunking, embedding progress,
            and errors per policy version.
          </p>
        </div>

        <div className="flex items-center gap-2">
          {hasNonTerminal && !pollExhausted ? (
            <Badge variant="accent">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Polling active job
            </Badge>
          ) : (
            <Badge variant="default">Idle</Badge>
          )}
          <Button
            variant="secondary"
            size="sm"
            onClick={handleManualRefresh}
            disabled={loading}
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Filters */}
      <div className="mt-5 grid gap-3 sm:grid-cols-2">
        <select
          aria-label="Filter jobs by status"
          value={statusFilter}
          onChange={(e) => {
            setLoading(true);
            setStatusFilter(e.target.value);
            setPage(1);
          }}
          className="flex h-10 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm text-slate-900"
        >
          <option value="">All statuses</option>
          <option value="queued">Queued</option>
          <option value="processing">Processing</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
          <option value="cancelled">Cancelled</option>
        </select>

        <select
          aria-label="Filter jobs by policy"
          value={policyFilter}
          onChange={(e) => {
            setLoading(true);
            setPolicyFilter(e.target.value);
            setPage(1);
          }}
          className="flex h-10 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm text-slate-900"
        >
          <option value="">All authorized policies</option>
          {policies.map((pol) => (
            <option key={pol.id} value={pol.id}>
              {pol.title} ({pol.external_policy_key})
            </option>
          ))}
        </select>
      </div>

      {listError && (
        <div
          role="alert"
          className="mt-4 rounded-xl border border-red-200 bg-red-50 px-3.5 py-2.5 text-sm text-red-700"
        >
          {listError}
        </div>
      )}

      {detailError && (
        <div
          role="alert"
          className="mt-4 rounded-xl border border-red-200 bg-red-50 px-3.5 py-2.5 text-sm text-red-700"
        >
          {detailError}
        </div>
      )}

      {/* Selected Job Inspector */}
      {selectedJob && (
        <div
          aria-label="Selected Job Details"
          className="mt-5 rounded-xl border border-slate-200 bg-[#FAFAFA] p-4 space-y-3"
        >
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="space-y-0.5">
              <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
                Job Detail
              </p>
              <p className="font-mono text-xs text-slate-800">{selectedJob.id}</p>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant={getStatusBadgeVariant(selectedJob.status)}>
                {selectedJob.status === 'completed' && (
                  <CheckCircle2 className="h-3.5 w-3.5" />
                )}
                {selectedJob.status === 'failed' && (
                  <XCircle className="h-3.5 w-3.5" />
                )}
                {selectedJob.status === 'queued' && (
                  <Clock className="h-3.5 w-3.5" />
                )}
                {selectedJob.status}
              </Badge>
              {selectedJob.current_stage && (
                <Badge variant="default">{selectedJob.current_stage}</Badge>
              )}
            </div>
          </div>

          <div className="grid gap-2 text-xs text-slate-600 sm:grid-cols-2">
            <div>
              <span className="font-medium text-slate-800">Policy: </span>
              {selectedJob.policy?.title || selectedJob.external_policy_key}
            </div>
            <div>
              <span className="font-medium text-slate-800">Document: </span>
              {selectedJob.document?.file_name || 'PDF'} (v
              {selectedJob.document_version_snapshot || 1})
            </div>
            <div>
              <span className="font-medium text-slate-800">Created: </span>
              {formatTimestamp(selectedJob.created_at)}
            </div>
            <div>
              <span className="font-medium text-slate-800">
                {selectedJob.completed_at ? 'Completed: ' : 'Started: '}
              </span>
              {formatTimestamp(selectedJob.completed_at || selectedJob.started_at)}
            </div>
          </div>

          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs text-slate-600">
              <span>
                Chunks: {selectedJob.progress?.processed_chunks ?? 0} /{' '}
                {selectedJob.progress?.total_chunks ?? 0}
              </span>
              <span>{selectedJob.progress?.percent ?? 0}%</span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200">
              <div
                className={
                  selectedJob.status === 'failed'
                    ? 'h-full bg-red-500 transition-all'
                    : selectedJob.status === 'completed'
                      ? 'h-full bg-emerald-600 transition-all'
                      : 'h-full bg-blue-600 transition-all'
                }
                style={{ width: `${selectedJob.progress?.percent ?? 0}%` }}
              />
            </div>
          </div>

          {selectedJob.error && (
            <div
              role="alert"
              className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800"
            >
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-red-600" />
              <div>
                <span className="font-semibold">
                  {selectedJob.error.code || 'INGESTION_ERROR'}:{' '}
                </span>
                <span>{selectedJob.error.message}</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Paginated Jobs List */}
      <div className="mt-5 space-y-2.5">
        {loading && jobs.length === 0 ? (
          <p className="py-6 text-center text-sm text-slate-500">
            Loading ingestion jobs...
          </p>
        ) : jobs.length === 0 ? (
          <p className="rounded-xl border border-dashed border-slate-200 py-8 text-center text-sm text-slate-500">
            No ingestion jobs found for the selected filter.
          </p>
        ) : (
          jobs.map((job) => {
            const isSelected = selectedJob?.id === job.id;
            return (
              <button
                key={job.id}
                type="button"
                onClick={() => handleSelectJob(job.id)}
                className={`flex w-full items-center justify-between gap-3 rounded-xl border px-3.5 py-3 text-left transition-colors cursor-pointer ${
                  isSelected
                    ? 'border-slate-900 bg-slate-50'
                    : 'border-slate-200 bg-white hover:border-slate-300'
                }`}
              >
                <div className="min-w-0 flex-1 space-y-0.5">
                  <p className="truncate text-sm font-medium text-slate-900">
                    {job.policy?.title || job.external_policy_key} · v
                    {job.document_version_snapshot || 1}
                  </p>
                  <p className="truncate font-mono text-xs text-slate-500">
                    {job.document?.file_name || 'policy.pdf'} ·{' '}
                    {job.current_stage || job.status}
                  </p>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <span className="text-xs text-slate-500">
                    {job.progress?.percent ?? 0}%
                  </span>
                  <Badge variant={getStatusBadgeVariant(job.status)}>
                    {job.status}
                  </Badge>
                </div>
              </button>
            );
          })
        )}
      </div>

      {/* Pagination Footer */}
      <div className="mt-5 flex items-center justify-between border-t border-slate-200 pt-4 text-xs text-slate-600">
        <span>
          Page {pagination.page} of {pagination.total_pages} (
          {pagination.total_items} total)
        </span>
        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            size="sm"
            disabled={!pagination.has_prev || loading}
            onClick={() => {
              setLoading(true);
              setPage((p) => Math.max(1, p - 1));
            }}
          >
            Previous
          </Button>
          <Button
            variant="secondary"
            size="sm"
            disabled={!pagination.has_next || loading}
            onClick={() => {
              setLoading(true);
              setPage((p) => p + 1);
            }}
          >
            Next
          </Button>
        </div>
      </div>
    </Card>
  );
}
