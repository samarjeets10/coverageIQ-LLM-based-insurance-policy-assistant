import { PolicyUploadPanel } from '@/components/PolicyUploadPanel';
import { IngestionJobsPanel } from '@/components/IngestionJobsPanel';

export function WorkspaceIngestionTab({
  policies = [],
  policiesLoading = false,
  policiesError = '',
  userRole = 'user',
  activeJobId = null,
  onSelectJob,
  refreshNonce = 0,
  onPolicyCreated,
  onUploadCompleted,
  onJobCompleted,
}) {
  return (
    <div className="space-y-6">
      {policiesError && (
        <div
          role="alert"
          className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
        >
          {policiesError}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-12">
        <div className="lg:col-span-5">
          <PolicyUploadPanel
            policies={policies}
            policiesLoading={policiesLoading}
            userRole={userRole}
            onPolicyCreated={onPolicyCreated}
            onUploadCompleted={onUploadCompleted}
          />
        </div>

        <div className="lg:col-span-7">
          <IngestionJobsPanel
            policies={policies}
            activeJobId={activeJobId}
            onSelectJob={onSelectJob}
            onJobCompleted={onJobCompleted}
            refreshNonce={refreshNonce}
          />
        </div>
      </div>
    </div>
  );
}
