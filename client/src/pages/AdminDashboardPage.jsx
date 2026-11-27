import { useMemo, useState } from 'react';
import {
  Activity,
  Building2,
  FileText,
  LayoutDashboard,
  ScrollText,
  ShieldCheck,
  UserCog,
  Users,
} from 'lucide-react';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import {
  DashboardCard,
  DashboardToolbar,
} from '@/components/dashboard/DashboardWidgets';
import { Badge } from '@/components/ui/badge';

const ADMIN_NAV_GROUPS = [
  {
    items: [
      { id: 'overview', label: 'Overview', icon: LayoutDashboard },
      { id: 'users', label: 'Users & Roles', icon: Users },
      { id: 'organisations', label: 'Organisations', icon: Building2 },
      { id: 'policies', label: 'All Policies', icon: FileText },
    ],
  },
  {
    label: 'System',
    items: [
      { id: 'jobs', label: 'Ingestion Jobs', icon: Activity },
      { id: 'audit', label: 'Audit Logs', icon: ScrollText },
    ],
  },
];

const ADMIN_MODULES = [
  {
    id: 'rbac-control',
    category: 'governance',
    title: 'RBAC Role Control',
    description: 'Assign or update platform roles across User, Organisation, and Admin.',
    badge: <Badge variant="accent">RBAC</Badge>,
    preview: (
      <>
        <div className="flex items-center justify-between text-xs text-slate-500">
          <span>Platform Roles</span>
          <span className="font-medium text-blue-600">3 roles</span>
        </div>
        <div className="flex flex-wrap gap-1.5 text-xs">
          <span className="rounded bg-white px-2.5 py-1 border border-slate-200">user</span>
          <span className="rounded bg-white px-2.5 py-1 border border-slate-200">organisation</span>
          <span className="rounded bg-slate-900 px-2.5 py-1 text-white">admin</span>
        </div>
      </>
    ),
  },
  {
    id: 'org-approvals',
    category: 'governance',
    title: 'Organisation Oversight',
    description: 'Review registered organisations and catalog publishing permissions.',
    badge: <Badge variant="success">Verified</Badge>,
    preview: (
      <>
        <div className="flex items-center justify-between text-xs text-slate-500">
          <span>Catalog Governance</span>
          <ShieldCheck className="h-4 w-4 text-emerald-600" />
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-2.5 text-xs text-slate-600">
          Approve or suspend organisation access
        </div>
      </>
    ),
  },
  {
    id: 'ingestion-monitor',
    category: 'system',
    title: 'Ingestion Monitor',
    description: 'Track queued, processing, and failed PDF ingestion jobs.',
    badge: <Badge variant="default">Jobs</Badge>,
    preview: (
      <>
        <div className="flex items-center justify-between text-xs text-slate-500">
          <span>Pipeline Health</span>
          <span className="font-medium text-emerald-700">Healthy</span>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-2.5 text-xs text-slate-600">
          PostgreSQL + API connected
        </div>
      </>
    ),
  },
  {
    id: 'audit-trail',
    category: 'system',
    title: 'Audit Logs',
    description: 'Inspect security events, role changes, and policy actions.',
    badge: <Badge variant="default">Logs</Badge>,
    preview: (
      <>
        <div className="flex items-center justify-between text-xs text-slate-500">
          <span>Audit Trail</span>
          <span className="font-medium text-slate-900">Recorded</span>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-2.5 text-xs text-slate-600">
          Every actor, resource, and timestamp logged
        </div>
      </>
    ),
  },
];

const TAB_TITLES = {
  overview: 'Admin Workspace',
  users: 'Users & Roles',
  organisations: 'Organisations',
  policies: 'All Policies',
  jobs: 'Ingestion Jobs',
  audit: 'Audit Logs',
  settings: 'Settings',
  help: 'Help & Docs',
};

export function AdminDashboardPage() {
  const [activeTab, setActiveTab] = useState('overview');
  const [filter, setFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');

  const filteredModules = useMemo(() => {
    return ADMIN_MODULES.filter((item) => {
      const matchesFilter = filter === 'all' || item.category === filter;
      const matchesSearch =
        item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.description.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesFilter && matchesSearch;
    });
  }, [filter, searchQuery]);

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
      <div className="space-y-8">
        <DashboardToolbar
          tabs={[
            { id: 'all', label: 'All Modules' },
            { id: 'governance', label: 'Governance' },
            { id: 'system', label: 'System' },
          ]}
          activeFilter={filter}
          onFilterChange={setFilter}
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          searchPlaceholder="Search admin tools..."
        />

        <section className="space-y-4">
          <h2 className="text-base font-semibold text-slate-900">
            Platform Governance
          </h2>
          <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
            {filteredModules.map((item) => (
              <DashboardCard
                key={item.id}
                title={item.title}
                description={item.description}
                badge={item.badge}
                preview={item.preview}
              />
            ))}
          </div>
        </section>
      </div>
    </DashboardLayout>
  );
}
