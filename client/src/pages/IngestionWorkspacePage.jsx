import { useEffect, useState } from 'react';
import { Footer } from '@/components/Footer';
import { IngestionJobsPanel } from '@/components/IngestionJobsPanel';
import { Navbar } from '@/components/Navbar';
import { PolicyUploadPanel } from '@/components/PolicyUploadPanel';
import { Badge } from '@/components/ui/badge';
import { useAuth } from '@/context/useAuth';
import { policiesApi } from '@/lib/api';

export function IngestionWorkspacePage() {
  const { user } = useAuth();
  const [policies, setPolicies] = useState([]);
  const [policiesLoading, setPoliciesLoading] = useState(true);
  const [policiesError, setPoliciesError] = useState('');

  const [activeJobId, setActiveJobId] = useState(null);
  const [refreshNonce, setRefreshNonce] = useState(0);

  useEffect(() => {
    let active = true;

    policiesApi
      .list()
      .then((res) => {
        if (active) {
          setPolicies(res.data?.items || []);
          setPoliciesError('');
        }
      })
      .catch((err) => {
        if (active) {
          setPoliciesError(err.message || 'Could not load your policies.');
        }
      })
      .finally(() => {
        if (active) setPoliciesLoading(false);
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

  return (
    <div className="min-h-screen bg-[#FAFAFA] text-slate-900">
      <Navbar />

      <main className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
        <div className="mb-8 flex flex-wrap items-center justify-between gap-3">
          <div>
            <Badge variant="accent">Policy Ingestion Workspace</Badge>
            <h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-900 sm:text-3xl">
              Upload Policy PDFs & Track Ingestion Jobs
            </h1>
            <p className="mt-1 text-sm text-slate-500">
              Signed in as{' '}
              <span className="font-medium text-slate-700">{user?.full_name}</span>{' '}
              ({user?.platform_role}). All uploads are tenant-isolated.
            </p>
          </div>
        </div>

        {policiesError && (
          <div
            role="alert"
            className="mb-6 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
          >
            {policiesError}
          </div>
        )}

        <div className="grid gap-6 lg:grid-cols-12">
          <div className="lg:col-span-5">
            <PolicyUploadPanel
              policies={policies}
              policiesLoading={policiesLoading}
              userRole={user?.platform_role || 'user'}
              onPolicyCreated={handlePolicyCreated}
              onUploadCompleted={handleUploadCompleted}
            />
          </div>

          <div className="lg:col-span-7">
            <IngestionJobsPanel
              policies={policies}
              activeJobId={activeJobId}
              onSelectJob={setActiveJobId}
              refreshNonce={refreshNonce}
            />
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
