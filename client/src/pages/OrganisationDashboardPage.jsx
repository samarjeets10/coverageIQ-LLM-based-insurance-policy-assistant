import { useState } from 'react';
import {
  Activity,
  CheckCircle,
  Clock,
  GitBranch,
  Layers,
  LayoutDashboard,
  Plus,
  Users,
  XCircle,
} from 'lucide-react';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { DashboardCard } from '@/components/dashboard/DashboardWidgets';
import { KpiCard, HorizontalBarChart, DonutSummary } from '@/components/dashboard/DashboardCharts';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';


/* ── Sidebar Navigation ── */
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

/* ── Quick Actions ── */
const ORG_ACTIONS = [
  {
    id: 'catalog',
    title: 'Publish Catalog Policy',
    description:
      'Add and maintain insurance plans in the shared recommendation catalog.',
    icon: Layers,
    actionLabel: 'Manage Catalog',
  },
  {
    id: 'versions',
    title: 'Policy Versions',
    description:
      'Track uploaded PDF revisions, SHA-256 hashes, and active versions.',
    icon: GitBranch,
    actionLabel: 'View Versions',
  },
  {
    id: 'pipeline',
    title: 'Ingestion Queue',
    description:
      'Monitor PDF text extraction, OCR, and semantic chunking jobs.',
    icon: Activity,
    actionLabel: 'Check Queue',
  },
  {
    id: 'members',
    title: 'Team Members',
    description:
      'Manage owners, admins, and members within your organisation workspace.',
    icon: Users,
    actionLabel: 'Manage Team',
  },
];

/* ── Mock KPI Data ── */
const MOCK_KPIS = [
  { label: 'Catalog Policies', value: '0', hint: 'Published plans in the catalog', icon: Layers },
  { label: 'Total Versions', value: '0', hint: 'Policy PDF revisions tracked', icon: GitBranch },
  { label: 'Team Members', value: '1', hint: 'Active members in your org', icon: Users },
  { label: 'Ingestion Jobs', value: '0', hint: 'PDF processing tasks', icon: Activity },
];

const CATALOG_STATUS_SEGMENTS = [
  { label: 'Active', value: 0, stroke: '#16a34a' },
  { label: 'Draft', value: 0, stroke: '#f59e0b' },
  { label: 'Archived', value: 0, stroke: '#94a3b8' },
];

const INGESTION_BARS = [
  { label: 'Completed', value: 0, color: 'bg-emerald-500' },
  { label: 'Processing', value: 0, color: 'bg-blue-500' },
  { label: 'Queued', value: 0, color: 'bg-amber-500' },
  { label: 'Failed', value: 0, color: 'bg-red-500' },
];

const TAB_TITLES = {
  overview: 'Organisation Workspace',
  catalog: 'Catalog Policies',
  versions: 'Policy Versions',
  pipeline: 'Ingestion Queue',
  members: 'Team Members',
};

/* ═══════════════════════════════════════════════════════════
   TAB CONTENT COMPONENTS
   ═══════════════════════════════════════════════════════════ */

function OverviewTab({ onNavigate }) {
  return (
    <div className="space-y-8">
      {/* KPI Row */}
      <section>
        <h2 className="mb-4 text-base font-semibold text-slate-900">
          Organisation Overview
        </h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {MOCK_KPIS.map((kpi) => (
            <KpiCard key={kpi.label} {...kpi} />
          ))}
        </div>
      </section>

      {/* Charts */}
      <section className="grid gap-5 lg:grid-cols-2">
        <DonutSummary
          title="Catalog Status"
          subtitle="Distribution of your catalog policy states"
          centerValue="0"
          centerLabel="Total"
          segments={CATALOG_STATUS_SEGMENTS}
        />
        <HorizontalBarChart
          title="Ingestion Pipeline"
          subtitle="Status of PDF processing and chunking jobs"
          items={INGESTION_BARS}
        />
      </section>

      {/* Quick Actions */}
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

function CatalogTab() {
  return (
    <div className="space-y-6">
      <Card className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-base font-semibold text-slate-900">
              Insurance Catalog
            </h3>
            <p className="mt-1 text-sm text-slate-500">
              Publish and manage insurance plans that users can discover through
              the recommendation engine.
            </p>
          </div>
          <Button size="sm">
            <Plus className="h-4 w-4" />
            New Policy
          </Button>
        </div>
      </Card>

      <Card className="p-6">
        <div className="rounded-xl border border-slate-200/80 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200/80 bg-slate-50/80">
                <th className="px-4 py-3 text-left font-medium text-slate-600">Policy Name</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Category</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Status</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Last Updated</th>
                <th className="px-4 py-3 text-right font-medium text-slate-600">Actions</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td colSpan={5} className="px-4 py-12 text-center text-slate-400">
                  No catalog policies published yet. Click "New Policy" to add your first plan.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function VersionsTab() {
  return (
    <div className="space-y-6">
      <Card className="p-6">
        <h3 className="text-base font-semibold text-slate-900">
          Policy Version History
        </h3>
        <p className="mt-1 text-sm text-slate-500">
          Track every uploaded PDF revision with SHA-256 integrity hashes. Compare
          versions and manage which revision is currently active.
        </p>
      </Card>

      <Card className="p-6">
        <div className="rounded-xl border border-slate-200/80 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200/80 bg-slate-50/80">
                <th className="px-4 py-3 text-left font-medium text-slate-600">Policy</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Version</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">SHA-256</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Uploaded</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td colSpan={5} className="px-4 py-12 text-center text-slate-400">
                  No version records yet. Versions are created when you upload PDFs to your catalog.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function PipelineTab() {
  return (
    <div className="space-y-6">
      <Card className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-base font-semibold text-slate-900">
              Ingestion Queue
            </h3>
            <p className="mt-1 text-sm text-slate-500">
              Monitor your PDF processing pipeline — text extraction, OCR,
              semantic chunking, and embedding generation jobs.
            </p>
          </div>
        </div>
      </Card>

      {/* Pipeline Status Cards */}
      <div className="grid gap-4 sm:grid-cols-4">
        {[
          { label: 'Queued', value: 0, icon: Clock, color: 'text-amber-600 bg-amber-50' },
          { label: 'Processing', value: 0, icon: Activity, color: 'text-blue-600 bg-blue-50' },
          { label: 'Completed', value: 0, icon: CheckCircle, color: 'text-emerald-600 bg-emerald-50' },
          { label: 'Failed', value: 0, icon: XCircle, color: 'text-red-600 bg-red-50' },
        ].map((s) => (
          <Card key={s.label} className="p-4">
            <div className="flex items-center gap-3">
              <span className={`flex h-9 w-9 items-center justify-center rounded-xl ${s.color}`}>
                <s.icon className="h-4 w-4" />
              </span>
              <div>
                <p className="text-xs text-slate-500">{s.label}</p>
                <p className="text-lg font-semibold text-slate-900">{s.value}</p>
              </div>
            </div>
          </Card>
        ))}
      </div>

      <Card className="p-6">
        <div className="rounded-xl border border-slate-200/80 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200/80 bg-slate-50/80">
                <th className="px-4 py-3 text-left font-medium text-slate-600">Job ID</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Policy</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Stage</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Status</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Started</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td colSpan={5} className="px-4 py-12 text-center text-slate-400">
                  No ingestion jobs yet. Upload a catalog policy PDF to start processing.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function MembersTab() {
  return (
    <div className="space-y-6">
      <Card className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-base font-semibold text-slate-900">
              Team Members
            </h3>
            <p className="mt-1 text-sm text-slate-500">
              Manage the people who can access and manage your organisation's
              catalog, versions, and ingestion pipeline.
            </p>
          </div>
          <Button size="sm">
            <Plus className="h-4 w-4" />
            Invite Member
          </Button>
        </div>
      </Card>

      <Card className="p-6">
        <div className="rounded-xl border border-slate-200/80 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200/80 bg-slate-50/80">
                <th className="px-4 py-3 text-left font-medium text-slate-600">Name</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Email</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Role</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Joined</th>
                <th className="px-4 py-3 text-right font-medium text-slate-600">Actions</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td colSpan={5} className="px-4 py-12 text-center text-slate-400">
                  You are the only member. Invite others to collaborate on your catalog.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   MAIN DASHBOARD PAGE
   ═══════════════════════════════════════════════════════════ */

export function OrganisationDashboardPage() {
  const [activeTab, setActiveTab] = useState('overview');

  const renderContent = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewTab onNavigate={setActiveTab} />;
      case 'catalog':
        return <CatalogTab />;
      case 'versions':
        return <VersionsTab />;
      case 'pipeline':
        return <PipelineTab />;
      case 'members':
        return <MembersTab />;
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
