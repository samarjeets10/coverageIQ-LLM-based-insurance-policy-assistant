import { useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  CheckCircle2,
  FileText,
  LayoutDashboard,
  MessageSquareText,
  Sparkles,
  Upload,
} from 'lucide-react';
import { useAuth } from '@/context/useAuth';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { DashboardCard } from '@/components/dashboard/DashboardWidgets';
import { KpiCard, HorizontalBarChart, DonutSummary } from '@/components/dashboard/DashboardCharts';
import { useWorkspaceData } from '@/hooks/useWorkspaceData';
import { WorkspaceIngestionTab } from '@/components/dashboard/WorkspaceIngestionTab';
import { PolicyQAPanel } from '@/components/PolicyQAPanel';
import { RecommendationsPanel } from '@/components/RecommendationsPanel';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

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

const TAB_TITLES = {
  overview: 'User Workspace',
  policies: 'My Policies & Ingestion',
  qa: 'Coverage Q&A',
  summaries: 'Policy Summaries',
  recommendations: 'Recommendations',
};

function OverviewTab({ policies, jobs, qaTotalQuestions, onNavigate }) {
  const completedJobs = jobs.filter((j) => j.status === 'completed').length;
  const activeJobs = jobs.filter((j) => j.status === 'queued' || j.status === 'processing').length;
  const failedJobs = jobs.filter((j) => j.status === 'failed').length;

  const countByType = (type) =>
    policies.filter((p) => String(p.policy_type || '').toLowerCase() === type).length;

  const kpis = [
    {
      label: 'Uploaded Policies',
      value: String(policies.length),
      hint: policies.length ? 'Authorized workspace policies' : 'Upload your first policy to get started',
      icon: FileText,
    },
    {
      label: 'Completed Ingestions',
      value: String(completedJobs),
      hint: activeJobs ? `${activeJobs} job(s) in progress` : 'Chunked & indexed in pgvector',
      icon: CheckCircle2,
    },
    {
      label: 'Q&A Queries',
      value: String(qaTotalQuestions),
      hint: qaTotalQuestions > 0 ? 'Grounded questions verified' : 'Ask a question on an active policy',
      icon: MessageSquareText,
    },
    {
      label: 'Summaries Generated',
      value: '0',
      hint: 'Policy summarization UI coming soon',
      icon: Sparkles,
    },
  ];

  const jobSegments = [
    { label: 'Completed', value: completedJobs, stroke: '#16a34a' },
    { label: 'Queued / Processing', value: activeJobs, stroke: '#2563eb' },
    { label: 'Failed', value: failedJobs, stroke: '#dc2626' },
  ];

  const policyTypeBars = [
    { label: 'Health Insurance', value: countByType('health'), color: 'bg-blue-600' },
    {
      label: 'Car / Motor Insurance',
      value: countByType('car') + countByType('motor'),
      color: 'bg-emerald-500',
    },
    { label: 'Life / Term Insurance', value: countByType('life') + countByType('term'), color: 'bg-amber-500' },
    {
      label: 'Travel / Pension / General',
      value: countByType('travel') + countByType('pension') + countByType('general'),
      color: 'bg-violet-500',
    },
  ];

  return (
    <div className="space-y-8">
      <section>
        <h2 className="mb-4 text-base font-semibold text-slate-900">
          At a Glance
        </h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {kpis.map((kpi) => (
            <KpiCard key={kpi.label} {...kpi} />
          ))}
        </div>
      </section>

      <section className="grid gap-5 lg:grid-cols-2">
        <DonutSummary
          title="Ingestion Job Status"
          subtitle="Live status breakdown across your PDF ingestion jobs"
          centerValue={String(jobs.length)}
          centerLabel="Jobs"
          segments={jobSegments}
        />
        <HorizontalBarChart
          title="Policies by Type"
          subtitle="Breakdown of your workspace policy records"
          items={policyTypeBars}
        />
      </section>

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

function SummariesTab() {
  return (
    <div className="space-y-6">
      <Card className="p-6">
        <h3 className="text-base font-semibold text-slate-900">
          Generate Policy Summary
        </h3>
        <p className="mt-1 text-sm text-slate-500">
          Policy summarization UI will appear here once connected in the upcoming checkpoint.
        </p>
        <div className="mt-5">
          <Button size="sm" disabled>
            <Sparkles className="h-4 w-4" />
            Generate Summary
          </Button>
        </div>
      </Card>
    </div>
  );
}

export function UserDashboardPage() {
  const { user } = useAuth();
  const [searchParams, setSearchParams] = useSearchParams();
  const initialTab = USER_NAV_GROUPS[0].items.some((i) => i.id === searchParams.get('tab'))
    ? searchParams.get('tab')
    : 'overview';
  const [activeTab, setActiveTabState] = useState(initialTab);

  const workspace = useWorkspaceData();

  function setActiveTab(tabId) {
    setActiveTabState(tabId);
    setSearchParams(tabId === 'overview' ? {} : { tab: tabId }, { replace: true });
    if (tabId === 'qa' || tabId === 'overview') {
      workspace.triggerRefresh();
    }
  }

  const renderContent = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <OverviewTab
            policies={workspace.policies}
            jobs={workspace.jobs}
            qaTotalQuestions={workspace.qaTotalQuestions}
            onNavigate={setActiveTab}
          />
        );
      case 'policies':
        return (
          <WorkspaceIngestionTab
            policies={workspace.policies}
            policiesLoading={workspace.loading}
            policiesError={workspace.error}
            userRole={user?.platform_role || 'user'}
            activeJobId={workspace.activeJobId}
            onSelectJob={workspace.setActiveJobId}
            refreshNonce={workspace.refreshNonce}
            onPolicyCreated={workspace.handlePolicyCreated}
            onUploadCompleted={workspace.handleUploadCompleted}
            onJobCompleted={workspace.triggerRefresh}
          />
        );
      case 'qa':
        return (
          <PolicyQAPanel
            policies={workspace.policies}
            policiesLoading={workspace.loading}
            onGoUploadPolicies={() => setActiveTab('policies')}
            onQaCompleted={workspace.triggerRefresh}
          />
        );
      case 'summaries':
        return <SummariesTab />;
      case 'recommendations':
        return <RecommendationsPanel onGoPolicies={() => setActiveTab('policies')} />;
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
        label: 'Upload Policy PDF',
        icon: Upload,
        onClick: () => setActiveTab('policies'),
      }}
    >
      {renderContent()}
    </DashboardLayout>
  );
}
