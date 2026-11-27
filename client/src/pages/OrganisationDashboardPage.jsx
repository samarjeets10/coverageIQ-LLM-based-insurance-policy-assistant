import { useMemo, useState } from 'react';
import {
  Activity,
  Building2,
  FileCheck2,
  GitBranch,
  Layers,
  LayoutDashboard,
  Plus,
  Users,
} from 'lucide-react';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import {
  DashboardCard,
  DashboardToolbar,
} from '@/components/dashboard/DashboardWidgets';
import { Badge } from '@/components/ui/badge';

const ORG_NAV_GROUPS = [
  {
    items: [
      { id: 'overview', label: 'Overview', icon: LayoutDashboard },
      { id: 'catalog', label: 'Catalog Policies', icon: Layers },
      { id: 'versions', label: 'Policy Versions', icon: GitBranch },
      { id: 'pipeline', label: 'Ingestion Queue', icon: Activity },
    ],
  },
  {
    label: 'Organisation',
    items: [
      { id: 'members', label: 'Team Members', icon: Users },
      { id: 'profile', label: 'Org Profile', icon: Building2 },
    ],
  },
];

const ORG_MODULES = [
  {
    id: 'publish-policy',
    category: 'published',
    title: 'Publish Catalog Policy',
    description: 'Add a new insurance plan to the shared recommendation catalog.',
    badge: <Badge variant="accent">Catalog</Badge>,
    preview: (
      <>
        <div className="flex items-center justify-between text-xs text-slate-500">
          <span>Shared Catalog</span>
          <span className="font-medium text-blue-600">Public</span>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-2.5 text-xs text-slate-700">
          Structured metadata + PDF clause indexing
        </div>
      </>
    ),
  },
  {
    id: 'version-control',
    category: 'published',
    title: 'Version Control',
    description: 'Track policy PDF revisions, SHA-256 hashes, and active versions.',
    badge: <Badge variant="success">Active</Badge>,
    preview: (
      <>
        <div className="flex items-center justify-between text-xs font-medium text-slate-700">
          <span>Revisions</span>
          <span className="text-emerald-700">v3 Live</span>
        </div>
        <div className="space-y-1.5 text-xs">
          <div className="flex justify-between rounded bg-white px-2.5 py-1 border border-slate-200/70">
            <span className="text-slate-500">v3 · Current</span>
            <span className="font-medium text-slate-900">Active</span>
          </div>
          <div className="flex justify-between rounded bg-white px-2.5 py-1 border border-slate-200/70">
            <span className="text-slate-500">v2 · Archived</span>
            <span className="text-slate-400">Read-only</span>
          </div>
        </div>
      </>
    ),
  },
  {
    id: 'ingestion-pipeline',
    category: 'draft',
    title: 'OCR & Chunking Queue',
    description: 'Monitor PDF extraction, semantic chunking, and embedding status.',
    badge: <Badge variant="default">Queue</Badge>,
    preview: (
      <>
        <div className="flex items-center justify-between text-xs text-slate-500">
          <span>Worker Status</span>
          <span className="font-medium text-slate-900">Idle</span>
        </div>
        <div className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600">
          <FileCheck2 className="h-4 w-4 text-emerald-600" />
          <span>All catalog PDFs verified</span>
        </div>
      </>
    ),
  },
  {
    id: 'team-access',
    category: 'published',
    title: 'Organisation Members',
    description: 'Manage owners, admins, and members inside your organisation.',
    badge: <Badge variant="default">RBAC</Badge>,
    preview: (
      <>
        <div className="flex items-center justify-between text-xs text-slate-500">
          <span>Workspace Roles</span>
          <span className="font-medium text-slate-900">3 tiers</span>
        </div>
        <div className="flex gap-1.5 text-xs">
          <span className="rounded bg-white px-2 py-1 border border-slate-200">Owner</span>
          <span className="rounded bg-white px-2 py-1 border border-slate-200">Admin</span>
          <span className="rounded bg-white px-2 py-1 border border-slate-200">Member</span>
        </div>
      </>
    ),
  },
];

const CATALOG_PLANS = [
  {
    id: 'plan-1',
    title: 'Shield Health Plus',
    description: 'Health · Zero co-pay plan with day-care coverage.',
    badge: <Badge variant="success">Published</Badge>,
    preview: (
      <>
        <div className="flex items-center justify-between text-xs text-slate-500">
          <span>Version 2</span>
          <span className="font-medium text-slate-900">62 pages</span>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-2.5 text-xs text-slate-600">
          Indexed for recommendations &amp; Q&amp;A
        </div>
      </>
    ),
  },
  {
    id: 'plan-2',
    title: 'TermSecure 30',
    description: 'Term Life · Critical illness rider and claim settlement rules.',
    badge: <Badge variant="default">Draft</Badge>,
    preview: (
      <>
        <div className="flex items-center justify-between text-xs text-slate-500">
          <span>Version 1</span>
          <span className="font-medium text-slate-900">34 pages</span>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-2.5 text-xs text-slate-600">
          Awaiting final review before publishing
        </div>
      </>
    ),
  },
];

const TAB_TITLES = {
  overview: 'Organisation Workspace',
  catalog: 'Catalog Policies',
  versions: 'Policy Versions',
  pipeline: 'Ingestion Queue',
  members: 'Team Members',
  profile: 'Organisation Profile',
  settings: 'Settings',
  help: 'Help & Docs',
};

export function OrganisationDashboardPage() {
  const [activeTab, setActiveTab] = useState('overview');
  const [filter, setFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');

  const filteredModules = useMemo(() => {
    return ORG_MODULES.filter((item) => {
      const matchesFilter = filter === 'all' || item.category === filter;
      const matchesSearch =
        item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.description.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesFilter && matchesSearch;
    });
  }, [filter, searchQuery]);

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
      <div className="space-y-8">
        <DashboardToolbar
          tabs={[
            { id: 'all', label: 'All Plans' },
            { id: 'published', label: 'Published' },
            { id: 'draft', label: 'Drafts' },
          ]}
          activeFilter={filter}
          onFilterChange={setFilter}
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          searchPlaceholder="Search catalog policies..."
        />

        <section className="space-y-4">
          <h2 className="text-base font-semibold text-slate-900">
            Catalog Management
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

        <section className="space-y-4">
          <h2 className="text-base font-semibold text-slate-900">
            Published Plans
          </h2>
          <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
            {CATALOG_PLANS.map((plan) => (
              <DashboardCard
                key={plan.id}
                title={plan.title}
                description={plan.description}
                badge={plan.badge}
                preview={plan.preview}
              />
            ))}
          </div>
        </section>
      </div>
    </DashboardLayout>
  );
}
