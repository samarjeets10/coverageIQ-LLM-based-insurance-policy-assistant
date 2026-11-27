import { useMemo, useState } from 'react';
import {
  Bookmark,
  Clock,
  FileText,
  LayoutDashboard,
  MessageSquareText,
  Sparkles,
  Upload,
  CheckCircle2,
} from 'lucide-react';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import {
  DashboardCard,
  DashboardToolbar,
} from '@/components/dashboard/DashboardWidgets';
import { Badge } from '@/components/ui/badge';

const USER_NAV_GROUPS = [
  {
    items: [
      { id: 'overview', label: 'Overview', icon: LayoutDashboard },
      { id: 'policies', label: 'My Policies', icon: FileText },
      { id: 'qa', label: 'Coverage Q&A', icon: MessageSquareText },
      { id: 'summaries', label: 'Policy Summaries', icon: Sparkles },
      { id: 'recommendations', label: 'Recommendations', icon: CheckCircle2 },
    ],
  },
  {
    label: 'Library',
    items: [
      { id: 'citations', label: 'Saved Citations', icon: Bookmark },
      { id: 'jobs', label: 'Ingestion Jobs', icon: Clock },
    ],
  },
];

const USER_TOOLS = [
  {
    id: 'qa-tool',
    category: 'active',
    title: 'Ask Coverage Q&A',
    description: 'Ask any coverage question and verify the exact page and clause.',
    badge: <Badge variant="success">Ready</Badge>,
    preview: (
      <>
        <div className="flex items-center justify-between text-xs text-slate-500">
          <span>Day-care surgery?</span>
          <span className="font-medium text-emerald-700">COVERED</span>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600">
          Covered without 24h stay at network hospitals.
        </div>
        <span className="w-fit rounded bg-blue-50 px-2 py-0.5 text-xs font-medium text-blue-700">
          Page 12 · Clause 4.2
        </span>
      </>
    ),
  },
  {
    id: 'summary-tool',
    category: 'active',
    title: 'Policy Summarizer',
    description: 'Turn long policy PDFs into clear waiting-period and exclusion notes.',
    badge: <Badge variant="accent">Active</Badge>,
    preview: (
      <>
        <div className="flex items-center justify-between text-xs font-medium text-slate-700">
          <span>Key Terms</span>
          <span className="text-blue-600">3 sections</span>
        </div>
        <div className="space-y-1.5 text-xs">
          <div className="flex justify-between rounded bg-white px-2.5 py-1 border border-slate-200/70">
            <span className="text-slate-500">Waiting Period</span>
            <span className="font-medium text-slate-900">30 days</span>
          </div>
          <div className="flex justify-between rounded bg-white px-2.5 py-1 border border-slate-200/70">
            <span className="text-slate-500">Room Rent Cap</span>
            <span className="font-medium text-slate-900">1% / day</span>
          </div>
        </div>
      </>
    ),
  },
  {
    id: 'recommend-tool',
    category: 'active',
    title: 'Plan Recommendations',
    description: 'Compare verified organisation plans matched to your coverage needs.',
    badge: <Badge variant="default">Catalog</Badge>,
    preview: (
      <>
        <div className="flex items-center justify-between text-xs font-medium text-slate-700">
          <span>Top Match</span>
          <span className="text-blue-600">94%</span>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-2.5">
          <p className="text-xs font-semibold text-slate-900">Shield Health Plus</p>
          <p className="mt-0.5 text-xs text-slate-500">Zero co-pay · Clause 2.1</p>
        </div>
      </>
    ),
  },
  {
    id: 'upload-tool',
    category: 'recent',
    title: 'Upload Private PDF',
    description: 'Add a Car, Health, Life, or Term policy PDF to your private workspace.',
    badge: <Badge variant="default">Private</Badge>,
    preview: (
      <div className="flex h-full flex-col items-center justify-center rounded-lg border border-dashed border-slate-300 bg-white text-center">
        <Upload className="h-5 w-5 text-blue-600" />
        <span className="mt-1.5 text-xs font-medium text-slate-700">
          Drop PDF (OCR supported)
        </span>
      </div>
    ),
  },
];

const USER_POLICIES = [
  {
    id: 'pol-1',
    category: 'active',
    title: 'Health Shield 2026.pdf',
    description: 'Health insurance · 48 pages indexed with clause citations.',
    badge: <Badge variant="success">Active</Badge>,
    preview: (
      <>
        <div className="flex items-center justify-between text-xs text-slate-500">
          <span>Health · v1</span>
          <span className="text-slate-700 font-medium">48 pages</span>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-2.5 text-xs text-slate-600">
          Last query: Maternity waiting period (p.18)
        </div>
      </>
    ),
  },
  {
    id: 'pol-2',
    category: 'recent',
    title: 'Comprehensive Car Cover.pdf',
    description: 'Motor insurance · Zero-depreciation and roadside clauses indexed.',
    badge: <Badge variant="accent">Indexed</Badge>,
    preview: (
      <>
        <div className="flex items-center justify-between text-xs text-slate-500">
          <span>Car · v1</span>
          <span className="text-slate-700 font-medium">26 pages</span>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-2.5 text-xs text-slate-600">
          Summary generated · 14 clauses cited
        </div>
      </>
    ),
  },
];

const TAB_TITLES = {
  overview: 'User Workspace',
  policies: 'My Policies',
  qa: 'Coverage Q&A',
  summaries: 'Policy Summaries',
  recommendations: 'Recommendations',
  citations: 'Saved Citations',
  jobs: 'Ingestion Jobs',
  settings: 'Settings',
  help: 'Help & Docs',
};

export function UserDashboardPage() {
  const [activeTab, setActiveTab] = useState('overview');
  const [filter, setFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');

  const filteredTools = useMemo(() => {
    return USER_TOOLS.filter((item) => {
      const matchesFilter = filter === 'all' || item.category === filter;
      const matchesSearch =
        item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.description.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesFilter && matchesSearch;
    });
  }, [filter, searchQuery]);

  return (
    <DashboardLayout
      navGroups={USER_NAV_GROUPS}
      activeTab={activeTab}
      onTabChange={setActiveTab}
      pageTitle={TAB_TITLES[activeTab] || 'User Workspace'}
      primaryAction={{
        label: 'Upload Policy',
        icon: Upload,
        onClick: () => setActiveTab('policies'),
      }}
    >
      <div className="space-y-8">
        <DashboardToolbar
          tabs={[
            { id: 'all', label: 'All Tools' },
            { id: 'active', label: 'Active' },
            { id: 'recent', label: 'Recent' },
          ]}
          activeFilter={filter}
          onFilterChange={setFilter}
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          searchPlaceholder="Search tools or policies..."
        />

        <section className="space-y-4">
          <h2 className="text-base font-semibold text-slate-900">
            Policy Intelligence Tools
          </h2>
          <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
            {filteredTools.map((tool) => (
              <DashboardCard
                key={tool.id}
                title={tool.title}
                description={tool.description}
                badge={tool.badge}
                preview={tool.preview}
              />
            ))}
          </div>
        </section>

        <section className="space-y-4">
          <h2 className="text-base font-semibold text-slate-900">
            Your Uploaded Policies
          </h2>
          <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
            {USER_POLICIES.map((policy) => (
              <DashboardCard
                key={policy.id}
                title={policy.title}
                description={policy.description}
                badge={policy.badge}
                preview={policy.preview}
              />
            ))}
          </div>
        </section>
      </div>
    </DashboardLayout>
  );
}
