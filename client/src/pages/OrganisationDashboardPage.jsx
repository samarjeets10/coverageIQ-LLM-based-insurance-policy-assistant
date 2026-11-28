import { useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Activity,
  GitBranch,
  Layers,
  LayoutDashboard,
  Plus,
  Upload,
  Users,
} from 'lucide-react';
import { useAuth } from '@/context/useAuth';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { DashboardCard } from '@/components/dashboard/DashboardWidgets';
import { KpiCard, HorizontalBarChart, DonutSummary } from '@/components/dashboard/DashboardCharts';
import { useWorkspaceData } from '@/hooks/useWorkspaceData';
import { WorkspaceIngestionTab } from '@/components/dashboard/WorkspaceIngestionTab';
import { IngestionJobsPanel } from '@/components/IngestionJobsPanel';
import { policiesApi } from '@/lib/api';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

const ORG_NAV_GROUPS = [
  {
    items: [
      { id: 'overview', label: 'Overview', icon: LayoutDashboard },
      { id: 'catalog', label: 'Catalog Policies', icon: Layers },
      { id: 'versions', label: 'Policy Versions', icon: GitBranch },
      { id: 'pipeline', label: 'Ingestion Queue', icon: Activity },
      { id: 'members', label: 'Team Members', icon: Users },
    ],
  },
];

const ORG_ACTIONS = [
  {
    id: 'catalog',
    title: 'Publish & Upload Policy PDF',
    description:
      'Create draft policies and upload PDF revisions to the ingestion pipeline.',
    icon: Layers,
    actionLabel: 'Manage Catalog',
  },
  {
    id: 'versions',
    title: 'Policy Inventory',
    description:
      'Inspect policy records, ownership scope, and active document statuses.',
    icon: GitBranch,
    actionLabel: 'View Policies',
  },
  {
    id: 'pipeline',
    title: 'Ingestion Queue',
    description:
      'Monitor PDF text extraction, OCR, and semantic chunking jobs live.',
    icon: Activity,
    actionLabel: 'Check Queue',
  },
  {
    id: 'members',
    title: 'Team Members',
    description:
      'Organisation team management UI (membership APIs planned).',
    icon: Users,
    actionLabel: 'View Team',
  },
];

const TAB_TITLES = {
  overview: 'Organisation Workspace',
  catalog: 'Catalog Policies & Upload',
  versions: 'Policy Inventory & Versions',
  pipeline: 'Ingestion Queue',
  members: 'Team Members',
};

function OverviewTab({ policies, jobs, onNavigate }) {
  const activePolicies = policies.filter((p) => p.status === 'active').length;
  const draftPolicies = policies.filter((p) => p.status === 'draft').length;
  const archivedPolicies = policies.filter((p) => p.status === 'archived').length;

  const completed = jobs.filter((j) => j.status === 'completed').length;
  const processing = jobs.filter((j) => j.status === 'processing').length;
  const queued = jobs.filter((j) => j.status === 'queued').length;
  const failed = jobs.filter((j) => j.status === 'failed').length;

  const kpis = [
    { label: 'Workspace Policies', value: String(policies.length), hint: 'Authorized organisation plans', icon: Layers },
    { label: 'Active Policies', value: String(activePolicies), hint: `${draftPolicies} draft(s)`, icon: GitBranch },
    { label: 'Ingestion Jobs', value: String(jobs.length), hint: `${completed} completed`, icon: Activity },
    { label: 'Queued / Active', value: String(queued + processing), hint: failed ? `${failed} failed` : 'No active failures', icon: Upload },
  ];

  const statusSegments = [
    { label: 'Active', value: activePolicies, stroke: '#16a34a' },
    { label: 'Draft', value: draftPolicies, stroke: '#f59e0b' },
    { label: 'Archived', value: archivedPolicies, stroke: '#94a3b8' },
  ];

  const ingestionBars = [
    { label: 'Completed', value: completed, color: 'bg-emerald-500' },
    { label: 'Processing', value: processing, color: 'bg-blue-500' },
    { label: 'Queued', value: queued, color: 'bg-amber-500' },
    { label: 'Failed', value: failed, color: 'bg-red-500' },
  ];

  return (
    <div className="space-y-8">
      <section>
        <h2 className="mb-4 text-base font-semibold text-slate-900">
          Organisation Overview
        </h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {kpis.map((kpi) => (
            <KpiCard key={kpi.label} {...kpi} />
          ))}
        </div>
      </section>

      <section className="grid gap-5 lg:grid-cols-2">
        <DonutSummary
          title="Policy Status"
          subtitle="Distribution of your organisation policy states"
          centerValue={String(policies.length)}
          centerLabel="Total"
          segments={statusSegments}
        />
        <HorizontalBarChart
          title="Ingestion Pipeline"
          subtitle="Status of PDF processing and chunking jobs"
          items={ingestionBars}
        />
      </section>

      <section>
        <h2 className="mb-4 text-base font-semibold text-slate-900">
          Quick Actions
        </h2>
        <div className="grid gap-5 sm:grid-cols-2">
          {ORG_ACTIONS.map((item) => (
            <DashboardCard
              key={item.id}
              title={item.title}
              description={item.description}
              icon={item.icon}
              actionLabel={item.actionLabel}
              onClick={() => onNavigate(item.id)}
            />
          ))}
        </div>
      </section>
    </div>
  );
}

function getAttributeStatusVariant(status) {
  if (status === 'completed') return 'success';
  if (status === 'partial') return 'warning';
  if (status === 'insufficient_data' || status === 'failed') return 'danger';
  return 'default';
}

function VersionsTab({ policies, onGoUpload, onRefresh }) {
  const [updatingId, setUpdatingId] = useState(null);
  const [visibilityError, setVisibilityError] = useState('');

  async function handleToggleVisibility(pol) {
    const nextVisibility = pol.visibility === 'catalog' ? 'private' : 'catalog';
    setVisibilityError('');
    setUpdatingId(pol.id);
    try {
      await policiesApi.updateVisibility(pol.id, nextVisibility);
      if (typeof onRefresh === 'function') {
        onRefresh();
      }
    } catch (err) {
      setVisibilityError(err?.message || 'Could not update policy visibility.');
    } finally {
      setUpdatingId(null);
    }
  }

  return (
    <div className="space-y-6">
      <Card className="p-6">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div>
            <h3 className="text-base font-semibold text-slate-900">Policy Records</h3>
            <p className="mt-1 text-sm text-slate-500">
              All authorized policies visible in your workspace. Upload a new PDF version in the Catalog tab, or publish active plans to the Recommendations Catalog.
            </p>
          </div>
          <Button size="sm" onClick={onGoUpload}>
            <Upload className="h-4 w-4" />
            Upload New Version
          </Button>
        </div>

        {visibilityError && (
          <div className="mb-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {visibilityError}
          </div>
        )}

        <div className="rounded-xl border border-slate-200/80 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200/80 bg-slate-50/80">
                <th className="px-4 py-3 text-left font-medium text-slate-600">Title</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Type</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Visibility</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Status</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Attributes</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Catalog Action</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Policy ID</th>
              </tr>
            </thead>
            <tbody>
              {policies.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-4 py-12 text-center text-slate-400">
                    No policies yet. Create a policy and upload a PDF in Catalog Policies.
                  </td>
                </tr>
              ) : (
                policies.map((pol) => {
                  const isCatalog = pol.visibility === 'catalog';
                  const canToggle = pol.can_upload !== false && pol.status !== 'archived' && (isCatalog || pol.status === 'active');
                  return (
                    <tr key={pol.id} className="border-b border-slate-100 last:border-0">
                      <td className="px-4 py-3 font-medium text-slate-900">{pol.title}</td>
                      <td className="px-4 py-3 capitalize text-slate-600">{pol.policy_type}</td>
                      <td className="px-4 py-3">
                        <Badge variant={isCatalog ? 'success' : 'default'}>
                          {pol.visibility}
                        </Badge>
                      </td>
                      <td className="px-4 py-3">
                        <Badge variant={pol.status === 'active' ? 'success' : 'default'}>
                          {pol.status}
                        </Badge>
                      </td>
                      <td className="px-4 py-3">
                        <Badge variant={getAttributeStatusVariant(pol.attribute_extraction_status)}>
                          {pol.attribute_extraction_status || 'pending'}
                        </Badge>
                      </td>
                      <td className="px-4 py-3">
                        <Button
                          type="button"
                          size="sm"
                          variant={isCatalog ? 'outline' : 'default'}
                          disabled={!canToggle || updatingId === pol.id}
                          onClick={() => handleToggleVisibility(pol)}
                        >
                          {updatingId === pol.id
                            ? 'Updating...'
                            : isCatalog
                              ? 'Make Private'
                              : 'Publish to Catalog'}
                        </Button>
                      </td>
                      <td className="px-4 py-3 font-mono text-xs text-slate-500">{pol.id}</td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function MembersTab({ user }) {
  return (
    <div className="space-y-6">
      <Card className="p-6">
        <h3 className="text-base font-semibold text-slate-900">Team Members</h3>
        <p className="mt-1 text-sm text-slate-500">
          Signed in as <span className="font-medium text-slate-800">{user?.full_name}</span> ({user?.email}). Team invite APIs are planned in the organisation management milestone.
        </p>
      </Card>
    </div>
  );
}

export function OrganisationDashboardPage() {
  const { user } = useAuth();
  const [searchParams, setSearchParams] = useSearchParams();
  const initialTab = ORG_NAV_GROUPS[0].items.some((i) => i.id === searchParams.get('tab'))
    ? searchParams.get('tab')
    : 'overview';
  const [activeTab, setActiveTabState] = useState(initialTab);

  const workspace = useWorkspaceData();

  function setActiveTab(tabId) {
    setActiveTabState(tabId);
    setSearchParams(tabId === 'overview' ? {} : { tab: tabId }, { replace: true });
  }

  const renderContent = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <OverviewTab
            policies={workspace.policies}
            jobs={workspace.jobs}
            onNavigate={setActiveTab}
          />
        );
      case 'catalog':
        return (
          <WorkspaceIngestionTab
            policies={workspace.policies}
            policiesLoading={workspace.loading}
            policiesError={workspace.error}
            userRole={user?.platform_role || 'organisation'}
            activeJobId={workspace.activeJobId}
            onSelectJob={workspace.setActiveJobId}
            refreshNonce={workspace.refreshNonce}
            onPolicyCreated={workspace.handlePolicyCreated}
            onUploadCompleted={workspace.handleUploadCompleted}
            onJobCompleted={workspace.triggerRefresh}
          />
        );
      case 'versions':
        return (
          <VersionsTab
            policies={workspace.policies}
            onGoUpload={() => setActiveTab('catalog')}
            onRefresh={workspace.triggerRefresh}
          />
        );
      case 'pipeline':
        return (
          <IngestionJobsPanel
            policies={workspace.policies}
            activeJobId={workspace.activeJobId}
            onSelectJob={workspace.setActiveJobId}
            onJobCompleted={workspace.triggerRefresh}
            refreshNonce={workspace.refreshNonce}
          />
        );
      case 'members':
        return <MembersTab user={user} />;
      default:
        return null;
    }
  };

  return (
    <DashboardLayout
      navGroups={ORG_NAV_GROUPS}
      activeTab={activeTab}
      onTabChange={setActiveTab}
      pageTitle={TAB_TITLES[activeTab] || 'Organisation Workspace'}
      primaryAction={{
        label: 'Add Catalog Policy',
        icon: Plus,
        onClick: () => setActiveTab('catalog'),
      }}
    >
      {renderContent()}
    </DashboardLayout>
  );
}
