import { useState } from 'react';
import {
  Activity,
  GitBranch,
  Layers,
  LayoutDashboard,
  Plus,
  Users,
} from 'lucide-react';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { DashboardCard } from '@/components/dashboard/DashboardWidgets';
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

const TAB_TITLES = {
  overview: 'Organisation Workspace',
  catalog: 'Catalog Policies',
  versions: 'Policy Versions',
  pipeline: 'Ingestion Queue',
  members: 'Team Members',
};

export function OrganisationDashboardPage() {
  const [activeTab, setActiveTab] = useState('overview');

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
      {activeTab === 'overview' ? (
        <section className="space-y-4">
          <h2 className="text-base font-semibold text-slate-900">
            Organisation Tools
          </h2>
          <div className="grid gap-5 sm:grid-cols-2">
            {ORG_ACTIONS.map((item) => (
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
            No records yet. This section is ready for your organisation data.
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
