import { useState } from 'react';
import {
  Building2,
  FileText,
  LayoutDashboard,
  ScrollText,
  Shield,
  UserCog,
  Users,
} from 'lucide-react';
import { useAuth } from '@/context/useAuth';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { DashboardCard } from '@/components/dashboard/DashboardWidgets';
import { KpiCard, HorizontalBarChart } from '@/components/dashboard/DashboardCharts';
import { useWorkspaceData } from '@/hooks/useWorkspaceData';
import { WorkspaceIngestionTab } from '@/components/dashboard/WorkspaceIngestionTab';
import { Card } from '@/components/ui/card';

const ADMIN_NAV_GROUPS = [
  {
    items: [
      { id: 'overview', label: 'Overview', icon: LayoutDashboard },
      { id: 'policies', label: 'Policies & Ingestion', icon: FileText },
      { id: 'users', label: 'Users & Roles', icon: Users },
      { id: 'organisations', label: 'Organisations', icon: Building2 },
      { id: 'audit', label: 'Audit Logs', icon: ScrollText },
    ],
  },
];

const ADMIN_ACTIONS = [
  {
    id: 'policies',
    title: 'Policies & Ingestion',
    description:
      'Inspect policy records and ingestion job statuses across the platform.',
    icon: FileText,
    actionLabel: 'Manage Policies & Jobs',
  },
  {
    id: 'users',
    title: 'Users & Roles',
    description:
      'Role update endpoint active at PATCH /api/v1/users/:userId/role.',
    icon: Users,
    actionLabel: 'View Users Info',
  },
  {
    id: 'organisations',
    title: 'Organisations',
    description:
      'Review registered organisations and oversee catalog publishing access.',
    icon: Building2,
    actionLabel: 'View Organisations',
  },
  {
    id: 'audit',
    title: 'Audit Logs',
    description:
      'Review security events, role updates, and system activity trails.',
    icon: ScrollText,
    actionLabel: 'View Logs',
  },
];

const TAB_TITLES = {
  overview: 'Admin Workspace',
  policies: 'Policies & Ingestion',
  users: 'Users & Roles',
  organisations: 'Organisations',
  audit: 'Audit Logs',
};

function OverviewTab({ policies, jobs, onNavigate }) {
  const completed = jobs.filter((j) => j.status === 'completed').length;
  const processing = jobs.filter((j) => j.status === 'processing' || j.status === 'queued').length;
  const failed = jobs.filter((j) => j.status === 'failed').length;

  const kpis = [
    { label: 'Visible Policies', value: String(policies.length), hint: 'Across accessible scopes', icon: FileText },
    { label: 'Ingestion Jobs', value: String(jobs.length), hint: `${completed} completed`, icon: ScrollText },
    { label: 'Active Queue', value: String(processing), hint: 'Queued or processing', icon: Users },
    { label: 'Failed Jobs', value: String(failed), hint: 'Sanitized error telemetry', icon: Shield },
  ];

  const jobBars = [
    { label: 'Completed', value: completed, color: 'bg-emerald-500' },
    { label: 'Queued / Processing', value: processing, color: 'bg-blue-600' },
    { label: 'Failed', value: failed, color: 'bg-red-500' },
  ];

  return (
    <div className="space-y-8">
      <section>
        <h2 className="mb-4 text-base font-semibold text-slate-900">
          Platform Overview
        </h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {kpis.map((kpi) => (
            <KpiCard key={kpi.label} {...kpi} />
          ))}
        </div>
      </section>

      <section>
        <HorizontalBarChart
          title="Ingestion Pipeline Telemetry"
          subtitle="Live job states returned from /api/v1/ingestion-jobs"
          items={jobBars}
        />
      </section>

      <section>
        <h2 className="mb-4 text-base font-semibold text-slate-900">
          Platform Governance
        </h2>
        <div className="grid gap-5 sm:grid-cols-2">
          {ADMIN_ACTIONS.map((item) => (
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

function PlaceholderGovernanceTab({ title, description }) {
  return (
    <Card className="p-6">
      <h3 className="text-base font-semibold text-slate-900">{title}</h3>
      <p className="mt-1 text-sm text-slate-500">{description}</p>
    </Card>
  );
}

export function AdminDashboardPage() {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState('overview');
  const workspace = useWorkspaceData();

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
      case 'policies':
        return (
          <WorkspaceIngestionTab
            policies={workspace.policies}
            policiesLoading={workspace.loading}
            policiesError={workspace.error}
            userRole={user?.platform_role || 'admin'}
            activeJobId={workspace.activeJobId}
            onSelectJob={workspace.setActiveJobId}
            refreshNonce={workspace.refreshNonce}
            onPolicyCreated={workspace.handlePolicyCreated}
            onUploadCompleted={workspace.handleUploadCompleted}
          />
        );
      case 'users':
        return (
          <PlaceholderGovernanceTab
            title="Platform Users"
            description="Role updates are supported via PATCH /api/v1/users/:userId/role. User directory listing API will connect in an upcoming checkpoint."
          />
        );
      case 'organisations':
        return (
          <PlaceholderGovernanceTab
            title="Registered Organisations"
            description="Organisation directory listing API will connect in an upcoming checkpoint."
          />
        );
      case 'audit':
        return (
          <PlaceholderGovernanceTab
            title="Audit Logs"
            description="Audit log query endpoint will connect in an upcoming checkpoint."
          />
        );
      default:
        return null;
    }
  };

  return (
    <DashboardLayout
      navGroups={ADMIN_NAV_GROUPS}
      activeTab={activeTab}
      onTabChange={setActiveTab}
      pageTitle={TAB_TITLES[activeTab] || 'Admin Workspace'}
      primaryAction={{
        label: 'Policies & Jobs',
        icon: UserCog,
        onClick: () => setActiveTab('policies'),
      }}
    >
      {renderContent()}
    </DashboardLayout>
  );
}
