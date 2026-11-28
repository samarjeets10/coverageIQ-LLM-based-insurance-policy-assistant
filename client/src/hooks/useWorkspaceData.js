import { useEffect, useState } from 'react';
import { ingestionApi, policiesApi, qaApi } from '@/lib/api';

export function useWorkspaceData() {
  const [policies, setPolicies] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [qaTotalQuestions, setQaTotalQuestions] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [activeJobId, setActiveJobId] = useState(null);
  const [refreshNonce, setRefreshNonce] = useState(0);

  useEffect(() => {
    let active = true;

    Promise.all([
      policiesApi.list(),
      ingestionApi.listJobs({ page: 1, limit: 20 }).catch(() => ({ data: { items: [] } })),
    ])
      .then(async ([polRes, jobRes]) => {
        if (!active) return;
        const polItems = polRes?.data?.items || [];
        setPolicies(polItems);
        setJobs(jobRes?.data?.items || []);
        setError('');

        const activePols = polItems.filter((p) => p.status === 'active').slice(0, 10);
        if (activePols.length === 0) {
          setQaTotalQuestions(0);
          return;
        }
        const histCounts = await Promise.all(
          activePols.map((pol) =>
            qaApi
              .getHistory(pol.id, { page: 1, limit: 1 })
              .then((r) => Number(r?.data?.total_questions || 0))
              .catch(() => 0)
          )
        );
        if (active) {
          setQaTotalQuestions(histCounts.reduce((sum, n) => sum + n, 0));
        }
      })
      .catch((err) => {
        if (!active) return;
        setError(err.message || 'Could not load workspace data.');
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [refreshNonce]);

  function handlePolicyCreated(policy) {
    setPolicies((prev) => [policy, ...prev.filter((p) => p.id !== policy.id)]);
  }

  function handleUploadCompleted({ job }) {
    if (job?.id) {
      setActiveJobId(job.id);
    }
    setRefreshNonce((n) => n + 1);
  }

  function triggerRefresh() {
    setRefreshNonce((n) => n + 1);
  }

  return {
    policies,
    jobs,
    qaTotalQuestions,
    loading,
    error,
    activeJobId,
    setActiveJobId,
    refreshNonce,
    handlePolicyCreated,
    handleUploadCompleted,
    triggerRefresh,
  };
}
