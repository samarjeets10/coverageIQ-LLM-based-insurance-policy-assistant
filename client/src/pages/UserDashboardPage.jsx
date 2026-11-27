import { useState } from 'react';
import {
  CheckCircle2,
  FileText,
  LayoutDashboard,
  MessageSquareText,
  Sparkles,
  Upload,
} from 'lucide-react';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { DashboardCard } from '@/components/dashboard/DashboardWidgets';
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

const USER_ACTIONS = [
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
  policies: 'My Policies',
  qa: 'Coverage Q&A',
  summaries: 'Policy Summaries',
  recommendations: 'Recommendations',
};

export function UserDashboardPage() {
  const [activeTab, setActiveTab] = useState('overview');

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
      {activeTab === 'overview' ? (
        <section className="space-y-4">
          <h2 className="text-base font-semibold text-slate-900">
            Quick Actions
          </h2>
          <div className="grid gap-5 sm:grid-cols-2">
            {USER_ACTIONS.map((item) => (
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
            No items yet. This section is ready for your live policy data.
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
