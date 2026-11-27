import { useState } from 'react';
import {
  CheckCircle2,
  FileText,
  LayoutDashboard,
  MessageSquareText,
  Sparkles,
  Upload,
  Search,
  FileUp,
  TrendingUp,
} from 'lucide-react';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { DashboardCard } from '@/components/dashboard/DashboardWidgets';
import { KpiCard, HorizontalBarChart, DonutSummary } from '@/components/dashboard/DashboardCharts';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

/* ── Sidebar Navigation ── */
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
];

/* ── Quick Action Cards ── */
const QUICK_ACTIONS = [
  {
    id: 'policies',
    title: 'Upload Policy PDF',
    description:
      'Add a Car, Health, Life, or Term policy PDF to your private workspace.',
    icon: Upload,
    actionLabel: 'Go to My Policies',
  },
  {
    id: 'qa',
    title: 'Coverage Q&A',
    description:
      'Ask coverage questions and get answers backed by exact page and clause citations.',
    icon: MessageSquareText,
    actionLabel: 'Open Q&A',
  },
  {
    id: 'summaries',
    title: 'Policy Summaries',
    description:
      'Generate plain-English breakdowns of coverage, exclusions, and waiting periods.',
    icon: Sparkles,
    actionLabel: 'View Summaries',
  },
  {
    id: 'recommendations',
    title: 'Plan Recommendations',
    description:
      'Compare verified organisation plans matched to your coverage requirements.',
    icon: CheckCircle2,
    actionLabel: 'Explore Plans',
  },
];

/* ── Mock KPI Data ── */
const MOCK_KPIS = [
  { label: 'Uploaded Policies', value: '0', hint: 'Upload your first policy to get started', icon: FileText },
  { label: 'Q&A Queries', value: '0', hint: 'Ask questions about your coverage', icon: MessageSquareText },
  { label: 'Summaries Generated', value: '0', hint: 'Summarize policy documents', icon: Sparkles },
  { label: 'Plans Reviewed', value: '0', hint: 'Explore recommended plans', icon: CheckCircle2 },
];

const DECISION_SEGMENTS = [
  { label: 'Covered', value: 0, stroke: '#16a34a' },
  { label: 'Not Covered', value: 0, stroke: '#dc2626' },
  { label: 'Conditional', value: 0, stroke: '#f59e0b' },
  { label: 'Insufficient Evidence', value: 0, stroke: '#94a3b8' },
];

const POLICY_TYPE_BARS = [
  { label: 'Health Insurance', value: 0, color: 'bg-blue-600' },
  { label: 'Car Insurance', value: 0, color: 'bg-emerald-500' },
  { label: 'Life Insurance', value: 0, color: 'bg-amber-500' },
  { label: 'Term Insurance', value: 0, color: 'bg-violet-500' },
];

const TAB_TITLES = {
  overview: 'User Workspace',
  policies: 'My Policies',
  qa: 'Coverage Q&A',
  summaries: 'Policy Summaries',
  recommendations: 'Recommendations',
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
          At a Glance
        </h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {MOCK_KPIS.map((kpi) => (
            <KpiCard key={kpi.label} {...kpi} />
          ))}
        </div>
      </section>

      {/* Charts Row */}
      <section className="grid gap-5 lg:grid-cols-2">
        <DonutSummary
          title="Query Decisions"
          subtitle="Distribution of your Q&A query outcomes"
          centerValue="0"
          centerLabel="Total"
          segments={DECISION_SEGMENTS}
        />
        <HorizontalBarChart
          title="Policies by Type"
          subtitle="Breakdown of your uploaded policy documents"
          items={POLICY_TYPE_BARS}
        />
      </section>

      {/* Quick Actions */}
      <section>
        <h2 className="mb-4 text-base font-semibold text-slate-900">
          Quick Actions
        </h2>
        <div className="grid gap-5 sm:grid-cols-2">
          {QUICK_ACTIONS.map((item) => (
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

function PoliciesTab() {
  return (
    <div className="space-y-6">
      {/* Upload Zone */}
      <Card className="border-dashed border-2 border-slate-300 bg-slate-50/50 p-8">
        <div className="flex flex-col items-center text-center">
          <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-blue-50 text-blue-600">
            <FileUp className="h-7 w-7" />
          </span>
          <h3 className="mt-4 text-base font-semibold text-slate-900">
            Upload a Policy PDF
          </h3>
          <p className="mx-auto mt-1.5 max-w-sm text-sm text-slate-500">
            Drag and drop your insurance policy document here, or click below to
            browse. Supported formats: PDF (up to 25 MB).
          </p>
          <Button size="sm" className="mt-5">
            <Upload className="h-4 w-4" />
            Browse Files
          </Button>
        </div>
      </Card>

      {/* Empty table placeholder */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-base font-semibold text-slate-900">Your Policies</h3>
        </div>
        <div className="rounded-xl border border-slate-200/80 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200/80 bg-slate-50/80">
                <th className="px-4 py-3 text-left font-medium text-slate-600">Policy Name</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Type</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Status</th>
                <th className="px-4 py-3 text-left font-medium text-slate-600">Uploaded</th>
                <th className="px-4 py-3 text-right font-medium text-slate-600">Actions</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td colSpan={5} className="px-4 py-12 text-center text-slate-400">
                  No policies uploaded yet. Upload your first PDF to get started.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function QATab() {
  return (
    <div className="space-y-6">
      {/* Query Input Area */}
      <Card className="p-6">
        <h3 className="text-base font-semibold text-slate-900">
          Ask a Coverage Question
        </h3>
        <p className="mt-1 text-sm text-slate-500">
          Select a policy, then type your question. You'll receive a grounded answer
          with citations pointing to the exact clause and page number.
        </p>
        <div className="mt-5 space-y-3">
          <div className="rounded-xl border border-slate-200 bg-slate-50/60 px-4 py-3 text-sm text-slate-400">
            Select a policy to query&hellip;
          </div>
          <div className="flex items-center gap-2">
            <div className="flex-1 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-400">
              e.g. "Is dental treatment covered under the waiting period?"
            </div>
            <Button size="sm" variant="default">
              <Search className="h-4 w-4" />
              Ask
            </Button>
          </div>
        </div>
      </Card>

      {/* Recent Queries placeholder */}
      <Card className="p-6">
        <h3 className="text-base font-semibold text-slate-900">Recent Queries</h3>
        <div className="mt-4 rounded-xl border border-slate-200/80 p-8 text-center text-sm text-slate-400">
          <MessageSquareText className="mx-auto h-8 w-8 text-slate-300" />
          <p className="mt-2">
            Your query history will appear here once you ask your first question.
          </p>
        </div>
      </Card>
    </div>
  );
}

function SummariesTab() {
  return (
    <div className="space-y-6">
      <Card className="p-6">
        <h3 className="text-base font-semibold text-slate-900">
          Generate Policy Summary
        </h3>
        <p className="mt-1 text-sm text-slate-500">
          Pick an uploaded policy and generate a plain-English breakdown of key
          sections — coverage details, exclusions, waiting periods, and claim
          procedures.
        </p>
        <div className="mt-5">
          <div className="rounded-xl border border-slate-200 bg-slate-50/60 px-4 py-3 text-sm text-slate-400">
            Select a policy to summarize&hellip;
          </div>
          <Button size="sm" className="mt-3">
            <Sparkles className="h-4 w-4" />
            Generate Summary
          </Button>
        </div>
      </Card>

      <Card className="p-6">
        <h3 className="text-base font-semibold text-slate-900">Past Summaries</h3>
        <div className="mt-4 rounded-xl border border-slate-200/80 p-8 text-center text-sm text-slate-400">
          <Sparkles className="mx-auto h-8 w-8 text-slate-300" />
          <p className="mt-2">
            Generated summaries will show up here with section-by-section breakdowns.
          </p>
        </div>
      </Card>
    </div>
  );
}

function RecommendationsTab() {
  return (
    <div className="space-y-6">
      <Card className="p-6">
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-base font-semibold text-slate-900">
              Plan Recommendations
            </h3>
            <p className="mt-1 text-sm text-slate-500">
              Verified organisation plans matched to your coverage needs, ranked by
              relevance and value. Powered by profile-based filtering and semantic
              similarity.
            </p>
          </div>
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600">
            <TrendingUp className="h-5 w-5" />
          </span>
        </div>
      </Card>

      <Card className="p-6">
        <h3 className="text-base font-semibold text-slate-900">Recommended Plans</h3>
        <div className="mt-4 rounded-xl border border-slate-200/80 p-8 text-center text-sm text-slate-400">
          <CheckCircle2 className="mx-auto h-8 w-8 text-slate-300" />
          <p className="mt-2">
            Upload at least one policy to unlock personalized plan recommendations.
          </p>
        </div>
      </Card>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   MAIN DASHBOARD PAGE
   ═══════════════════════════════════════════════════════════ */

export function UserDashboardPage() {
  const [activeTab, setActiveTab] = useState('overview');

  const renderContent = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewTab onNavigate={setActiveTab} />;
      case 'policies':
        return <PoliciesTab />;
      case 'qa':
        return <QATab />;
      case 'summaries':
        return <SummariesTab />;
      case 'recommendations':
        return <RecommendationsTab />;
      default:
        return null;
    }
  };

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
      {renderContent()}
    </DashboardLayout>
  );
}
