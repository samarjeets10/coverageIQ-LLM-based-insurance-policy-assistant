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
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { DashboardCard } from '@/components/dashboard/DashboardWidgets';
import { KpiCard, HorizontalBarChart, DonutSummary } from '@/components/dashboard/DashboardCharts';
import { Card } from '@/components/ui/card';

/* ── Sidebar Navigation ── */
const ADMIN_NAV_GROUPS = [
  {
    items: [
      { id: 'overview', label: 'Overview', icon: LayoutDashboard },
      { id: 'users', label: 'Users & Roles', icon: Users },
      { id: 'organisations', label: 'Organisations', icon: Building2 },
      { id: 'policies', label: 'All Policies', icon: FileText },
      { id: 'audit', label: 'Audit Logs', icon: ScrollText },
    ],
  },
];

/* ── Quick Actions ── */
const ADMIN_ACTIONS = [
  {
    id: 'users',
    title: 'Users & Roles',
    description:
      'Assign and update platform roles across User, Organisation, and Admin accounts.',
    icon: Users,
    actionLabel: 'Manage Roles',
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
    id: 'policies',
    title: 'All Policies',
    description:
      'Inspect policy records and ingestion statuses across the platform.',
    icon: FileText,
    actionLabel: 'Browse Policies',
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

/* ── Mock KPI Data ── */
const MOCK_KPIS = [
  { label: 'Total Users', value: '1', hint: 'Registered platform accounts', icon: Users },
  { label: 'Organisations', value: '0', hint: 'Registered org accounts', icon: Building2 },
  { label: 'All Policies', value: '0', hint: 'Across all users and orgs', icon: FileText },
  { label: 'Audit Events', value: '0', hint: 'Recent system activity', icon: ScrollText },
];

const ROLE_DISTRIBUTION = [
  { label: 'User', value: 1, stroke: '#2563eb' },
  { label: 'Organisation', value: 0, stroke: '#16a34a' },
  { label: 'Admin', value: 1, stroke: '#0f172a' },
];

const REGISTRATION_BARS = [
  { label: 'This Week', value: 1, color: 'bg-blue-600' },
  { label: 'Last Week', value: 0, color: 'bg-slate-400' },
  { label: 'Two Weeks Ago', value: 0, color: 'bg-slate-300' },
  { label: 'Earlier', value: 0, color: 'bg-slate-200' },
];

const TAB_TITLES = {
  overview: 'Admin Workspace',
  users: 'Users & Roles',
  organisations: 'Organisations',
  policies: 'All Policies',
  audit: 'Audit Logs',
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
          Platform Overview
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
          title="Role Distribution"
          subtitle="Breakdown of registered accounts by platform role"
          centerValue="2"
          centerLabel="Accounts"
          segments={ROLE_DISTRIBUTION}
        />
        <HorizontalBarChart
          title="Registration Trend"
          subtitle="New account sign-ups over time"
          items={REGISTRATION_BARS}
        />
      </section>

      {/* Quick Actions */}
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

function UsersTab() {
  return (
    <div className="space-y-6">
      <Card className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-base font-semibold text-slate-900">
              Platform Users
            </h3>
            <p className="mt-1 text-sm text-slate-500">
              View all registered accounts and manage role assignments.
              Supported roles: User, Organisation, Admin.
            </p>
          </div>
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
                <th className="px-4 py-3 text-left font-medium text-slate-600">Registered</th>
                <th className="px-4 py-3 text-right font-medium text-slate-600">Actions</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td colSpan={5} className="px-4 py-12 text-center text-slate-400">
                  User data will load from the platform API once connected.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function OrganisationsTab() {
  return (
    <div className="space-y-6">
      <Card className="p-6">
        <h3 className="text-base font-semibold text-slate-900">
          Registered Organisations
        </h3>
        <p className="mt-1 text-sm text-slate-500">
          Oversee organisation accounts, their catalog publishing permissions,
          and team configurations.
        </p>
      </Card>

      <Card className="p-6">
        <div className="rounded-xl border border-slate-200/80 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200/80 bg-slate-50/80">
                <th className="px-4 py-3 text-left font-medium text-slate-600">Organisation</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Owner</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Catalog Policies</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Members</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td colSpan={5} className="px-4 py-12 text-center text-slate-400">
                  No organisations registered yet.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function AllPoliciesTab() {
  return (
    <div className="space-y-6">
      <Card className="p-6">
        <h3 className="text-base font-semibold text-slate-900">
          All Policies
        </h3>
        <p className="mt-1 text-sm text-slate-500">
          A consolidated view of every policy record across the platform —
          both user-uploaded documents and organisation catalog entries.
        </p>
      </Card>

      <Card className="p-6">
        <div className="rounded-xl border border-slate-200/80 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200/80 bg-slate-50/80">
                <th className="px-4 py-3 text-left font-medium text-slate-600">Policy</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Owner</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Type</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Source</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Ingestion Status</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Uploaded</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td colSpan={6} className="px-4 py-12 text-center text-slate-400">
                  No policy records found across the platform.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function AuditTab() {
  return (
    <div className="space-y-6">
      <Card className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-base font-semibold text-slate-900">
              Audit Logs
            </h3>
            <p className="mt-1 text-sm text-slate-500">
              Chronological record of security-relevant events — sign-ins,
              role changes, policy uploads, and failed access attempts.
            </p>
          </div>
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-slate-100 text-slate-600">
            <Shield className="h-5 w-5" />
          </span>
        </div>
      </Card>

      <Card className="p-6">
        <div className="rounded-xl border border-slate-200/80 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200/80 bg-slate-50/80">
                <th className="px-4 py-3 text-left font-medium text-slate-600">Timestamp</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Event</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Actor</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Target</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Details</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td colSpan={5} className="px-4 py-12 text-center text-slate-400">
                  No audit events recorded yet.
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

export function AdminDashboardPage() {
  const [activeTab, setActiveTab] = useState('overview');

  const renderContent = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewTab onNavigate={setActiveTab} />;
      case 'users':
        return <UsersTab />;
      case 'organisations':
        return <OrganisationsTab />;
      case 'policies':
        return <AllPoliciesTab />;
      case 'audit':
        return <AuditTab />;
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
        label: 'Manage Roles',
        icon: UserCog,
        onClick: () => setActiveTab('users'),
      }}
    >
      {renderContent()}
    </DashboardLayout>
  );
}
