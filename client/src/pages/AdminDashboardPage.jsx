import { useState } from 'react';
import {
  Building2,
  FileText,
  LayoutDashboard,
  ScrollText,
  UserCog,
  Users,
} from 'lucide-react';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { DashboardCard } from '@/components/dashboard/DashboardWidgets';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

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

const TAB_TITLES = {
  overview: 'Admin Workspace',
  users: 'Users & Roles',
  organisations: 'Organisations',
  policies: 'All Policies',
  audit: 'Audit Logs',
};

export function AdminDashboardPage() {
  const [activeTab, setActiveTab] = useState('overview');

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
      {activeTab === 'overview' ? (
        <section className="space-y-4">
          <h2 className="text-base font-semibold text-slate-900">
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
                onClick={() => setActiveTab(item.id)}
              />
            ))}
          </div>
        </section>
      ) : (
        <Card className="p-8 text-center">
          <h2 className="text-lg font-semibold text-slate-900">
            {TAB_TITLES[activeTab]}
          </h2>
          <p className="mx-auto mt-2 max-w-md text-sm text-slate-500">
            No records yet. This section is ready for live platform data.
          </p>
          <div className="mt-5">
            <Button variant="secondary" size="sm" onClick={() => setActiveTab('overview')}>
              Back to Overview
            </Button>
          </div>
        </Card>
      )}
    </DashboardLayout>
  );
}
