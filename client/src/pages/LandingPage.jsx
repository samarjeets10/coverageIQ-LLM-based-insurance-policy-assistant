import { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  ArrowRight,
  Building2,
  CheckCircle2,
  ChevronDown,
  FileSearch,
  FileText,
  Layers,
  Lock,
  ScanText,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  User,
  UserCog,
} from 'lucide-react';
import { Navbar } from '@/components/Navbar';
import { Footer } from '@/components/Footer';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { useAuth } from '@/context/useAuth';
import { getDashboardPath } from '@/lib/roles';

const STACK_HIGHLIGHTS = [
  { icon: FileSearch, label: 'Hybrid Search (BM25 + Vector)' },
  { icon: ShieldCheck, label: 'Second-Pass Verifier' },
  { icon: FileText, label: 'Page & Clause Citations' },
  { icon: ScanText, label: 'OCR for Scanned PDFs' },
  { icon: Lock, label: 'Role-Based Isolation' },
];

const PROCESS_STEPS = [
  {
    icon: FileText,
    title: '1. Upload Your PDF',
    description:
      'Drop in any Car, Health, Life, or Term policy, including scanned copies. Every section, table, and page number stays intact.',
  },
  {
    icon: Sparkles,
    title: '2. Ask or Compare',
    description:
      'Ask specific coverage questions, generate a plain-English summary, or match your needs against plans in the catalog.',
  },
  {
    icon: ShieldCheck,
    title: '3. Check the Source',
    description:
      'Every claim is checked against the retrieved clause. If the document does not support an answer, we say so.',
  },
];

const RBAC_ROLES = [
  {
    role: 'user',
    title: 'Policyholder',
    icon: User,
    description:
      'Upload your own policy PDFs privately, get plain-English summaries, and ask coverage questions.',
    cta: 'Sign up as User',
    to: '/register?role=user',
  },
  {
    role: 'organisation',
    title: 'Organisation',
    icon: Building2,
    description:
      'Publish and maintain your insurance plans in the shared catalog with full version history.',
    cta: 'Sign up as Organisation',
    to: '/register?role=organisation',
  },
  {
    role: 'admin',
    title: 'Platform Admin',
    icon: UserCog,
    description:
      'Manage platform access, approve roles, and keep an eye on system activity.',
    cta: 'Admin Login',
    to: '/login',
  },
];

const FAQS = [
  {
    q: 'How do I know the answer is not made up?',
    a: 'Every statement is tied to a specific section of your PDF and checked by a separate verifier. If the document does not support an answer, CoverageIQ tells you there is not enough evidence instead of guessing.',
  },
  {
    q: 'Which insurance policies work with this?',
    a: 'You can upload Car, Health, Life, and Term insurance PDFs. Scanned documents are handled with OCR, and tables, waiting periods, and exclusions are preserved.',
  },
  {
    q: 'Are my personal policy PDFs private?',
    a: 'Yes. Files you upload as a User stay strictly in your own account. Only Organisation-published plans appear in the shared recommendation catalog.',
  },
  {
    q: 'Should I sign up as a User or an Organisation?',
    a: 'Pick User if you want to check your own policies or compare plans. Pick Organisation if you represent an insurer or team publishing plans to the catalog.',
  },
];

export function LandingPage() {
  const { user } = useAuth();
  const [openFaq, setOpenFaq] = useState(1);

  return (
    <div className="min-h-screen bg-[#FAFAFA] text-slate-900">
      <Navbar />

      <main>
        {/* Hero Section */}
        <section className="relative border-b border-dashed border-slate-200 bg-subtle-grid">
          <div className="mx-auto max-w-4xl px-4 py-20 text-center sm:px-6 sm:py-28">
            <Badge variant="accent" className="mb-6 bg-white shadow-xs">
              <span className="h-1.5 w-1.5 rounded-full bg-blue-600" />
              Bringing clarity to the details that matter most
            </Badge>

            <h1 className="mx-auto max-w-3xl text-4xl font-semibold tracking-tight text-slate-900 sm:text-5xl sm:leading-[1.12]">
              Read Every Insurance Clause Before You File a Claim
            </h1>

            <p className="mx-auto mt-5 max-w-2xl text-base leading-relaxed text-slate-500">
              Upload your Car, Health, Life, or Term policy PDF. Get plain-English
              summaries, plan recommendations, and coverage answers linked straight
              to the exact page and clause.
            </p>

            <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
              {user ? (
                <Link to={getDashboardPath(user.platform_role)}>
                  <Button size="lg">
                    Go to Dashboard ({user.platform_role})
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </Link>
              ) : (
                <>
                  <Link to="/register">
                    <Button size="lg">
                      Get Started
                      <ArrowRight className="h-4 w-4" />
                    </Button>
                  </Link>
                  <Link to="/login">
                    <Button variant="secondary" size="lg">
                      Login
                    </Button>
                  </Link>
                </>
              )}
            </div>
          </div>
        </section>

        {/* Architecture / Stack Strip */}
        <section className="border-b border-dashed border-slate-200 bg-white">
          <div className="mx-auto max-w-6xl px-4 py-12 text-center sm:px-6">
            <h2 className="text-xl font-semibold tracking-tight text-slate-900">
              Built to Never Guess on Coverage
            </h2>
            <p className="mx-auto mt-1.5 max-w-lg text-base text-slate-500">
              Every answer is checked against your actual policy text before it
              reaches you.
            </p>

            <div className="mt-8 flex flex-wrap items-center justify-center gap-x-8 gap-y-4">
              {STACK_HIGHLIGHTS.map((item) => {
                const Icon = item.icon;
                return (
                  <div
                    key={item.label}
                    className="flex items-center gap-2 text-sm font-medium text-slate-600"
                  >
                    <Icon className="h-4 w-4 text-blue-600" />
                    <span>{item.label}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        {/* Core Capabilities Bento Grid */}
        <section
          id="capabilities"
          className="border-b border-dashed border-slate-200 bg-[#FAFAFA]"
        >
          <div className="mx-auto max-w-5xl px-4 py-20 sm:px-6">
            <div className="text-center">
              <h2 className="text-2xl font-semibold tracking-tight text-slate-900 sm:text-3xl">
                What You Can Do
              </h2>
              <p className="mx-auto mt-2 max-w-lg text-base text-slate-500">
                Three tools built on one shared policy index, kept separate for
                each policy.
              </p>
            </div>

            <div className="mt-12 grid gap-6 md:grid-cols-2">
              {/* Card 1: Q&A */}
              <Card className="p-6">
                <div className="rounded-xl border border-slate-200/80 bg-[#FAFAFA] p-4">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-sm font-medium text-slate-500">
                      Is day-care surgery covered?
                    </span>
                    <Badge variant="success">COVERED</Badge>
                  </div>
                  <p className="mt-3 text-sm leading-relaxed text-slate-700">
                    Covered up to sum insured at a network hospital without a
                    24-hour inpatient stay.
                  </p>
                  <div className="mt-3 flex items-center gap-2.5">
                    <span className="rounded-md border border-blue-200 bg-blue-50 px-2.5 py-0.5 text-xs font-medium text-blue-700">
                      Page 12 · Clause 4.2
                    </span>
                    <span className="text-xs text-slate-400">
                      Confidence: 0.96
                    </span>
                  </div>
                </div>
                <h3 className="mt-5 text-lg font-semibold text-slate-900">
                  Ask About Your Coverage
                </h3>
                <p className="mt-1 text-sm leading-relaxed text-slate-500">
                  Get a clear verdict (Covered, Not Covered, Conditional, or Not
                  Enough Info) with the exact page and clause number.
                </p>
              </Card>

              {/* Card 2: Summarization */}
              <Card className="p-6">
                <div className="rounded-xl border border-slate-200/80 bg-[#FAFAFA] p-4">
                  <div className="flex items-center justify-between text-sm font-medium text-slate-700">
                    <span>Policy Summary</span>
                    <span className="text-blue-600">3 Verified Sections</span>
                  </div>
                  <div className="mt-3 space-y-2 text-sm">
                    <div className="flex items-center justify-between rounded-lg border border-slate-200/70 bg-white px-3 py-1.5">
                      <span className="text-slate-600">Waiting Period</span>
                      <span className="font-medium text-slate-900">30 days · p.6</span>
                    </div>
                    <div className="flex items-center justify-between rounded-lg border border-slate-200/70 bg-white px-3 py-1.5">
                      <span className="text-slate-600">Room Rent Cap</span>
                      <span className="font-medium text-slate-900">1% / day · p.9</span>
                    </div>
                  </div>
                </div>
                <h3 className="mt-5 text-lg font-semibold text-slate-900">
                  Plain-English Summaries
                </h3>
                <p className="mt-1 text-sm leading-relaxed text-slate-500">
                  Turn dense policy documents into a clear breakdown of coverage,
                  exclusions, and waiting periods.
                </p>
              </Card>

              {/* Card 3: Recommendation */}
              <Card className="p-6">
                <div className="rounded-xl border border-slate-200/80 bg-[#FAFAFA] p-4">
                  <div className="flex items-center justify-between text-sm font-medium text-slate-700">
                    <span>Catalog Match</span>
                    <Layers className="h-4 w-4 text-blue-600" />
                  </div>
                  <div className="mt-3 space-y-2 text-sm">
                    <div className="flex items-center justify-between rounded-lg border border-slate-200/70 bg-white px-3 py-2">
                      <div>
                        <p className="font-medium text-slate-900">Shield Health Plus</p>
                        <p className="text-xs text-slate-400">Zero co-pay · Clause 2.1</p>
                      </div>
                      <Badge variant="accent">94% Match</Badge>
                    </div>
                  </div>
                </div>
                <h3 className="mt-5 text-lg font-semibold text-slate-900">
                  Compare &amp; Recommend Plans
                </h3>
                <p className="mt-1 text-sm leading-relaxed text-slate-500">
                  Filter and rank plans from verified organisations based on what
                  you actually need.
                </p>
              </Card>

              {/* Card 4: Independent Evaluator */}
              <Card className="p-6">
                <div className="rounded-xl border border-slate-200/80 bg-[#FAFAFA] p-4">
                  <div className="flex items-center justify-between text-sm font-medium text-slate-700">
                    <span>Evaluator Check</span>
                    <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                  </div>
                  <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
                    <span className="rounded-md border border-slate-200 bg-white px-2.5 py-1 text-slate-600">
                      Draft Answer
                    </span>
                    <ArrowRight className="h-3.5 w-3.5 text-slate-400" />
                    <span className="rounded-md border border-blue-200 bg-blue-50 px-2.5 py-1 font-medium text-blue-700">
                      Citation Check
                    </span>
                    <ArrowRight className="h-3.5 w-3.5 text-slate-400" />
                    <span className="rounded-md border border-emerald-200 bg-emerald-50 px-2.5 py-1 font-medium text-emerald-700">
                      PASS
                    </span>
                  </div>
                  <div className="mt-2.5 flex items-center gap-1.5 text-xs text-slate-500">
                    <ShieldAlert className="h-3.5 w-3.5 text-slate-400" />
                    <span>Retries up to 2x if the evidence is weak</span>
                  </div>
                </div>
                <h3 className="mt-5 text-lg font-semibold text-slate-900">
                  Double-Checked Answers
                </h3>
                <p className="mt-1 text-sm leading-relaxed text-slate-500">
                  A second check verifies every claim against the cited text. If
                  the proof is not there, we say so.
                </p>
              </Card>
            </div>
          </div>
        </section>

        {/* 3-Step Process */}
        <section
          id="process"
          className="border-b border-dashed border-slate-200 bg-white"
        >
          <div className="mx-auto max-w-5xl px-4 py-20 sm:px-6">
            <div className="text-center">
              <h2 className="text-2xl font-semibold tracking-tight text-slate-900 sm:text-3xl">
                From PDF Upload to Verified Answer
              </h2>
              <p className="mx-auto mt-2 max-w-lg text-base text-slate-500">
                Upload once, then ask questions or compare plans without losing
                track of where anything came from.
              </p>
            </div>

            <div className="mt-12 grid gap-8 md:grid-cols-3">
              {PROCESS_STEPS.map((step) => {
                const Icon = step.icon;
                return (
                  <div key={step.title} className="flex flex-col items-start gap-3">
                    <span className="flex h-10 w-10 items-center justify-center rounded-xl border border-blue-100 bg-blue-50/80 text-blue-600">
                      <Icon className="h-5 w-5" />
                    </span>
                    <h3 className="text-base font-semibold text-slate-900">
                      {step.title}
                    </h3>
                    <p className="text-sm leading-relaxed text-slate-500">
                      {step.description}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        {/* RBAC Roles Section */}
        <section
          id="roles"
          className="border-b border-dashed border-slate-200 bg-[#FAFAFA]"
        >
          <div className="mx-auto max-w-5xl px-4 py-20 sm:px-6">
            <div className="text-center">
              <h2 className="text-2xl font-semibold tracking-tight text-slate-900 sm:text-3xl">
                Who Uses CoverageIQ
              </h2>
              <p className="mx-auto mt-2 max-w-lg text-base text-slate-500">
                Separate workspaces and permissions for policyholders, insurance
                organisations, and admins.
              </p>
            </div>

            <div className="mt-12 grid gap-6 md:grid-cols-3">
              {RBAC_ROLES.map((item) => {
                const Icon = item.icon;
                return (
                  <Card key={item.role} className="flex flex-col justify-between p-6">
                    <div>
                      <div className="flex items-center justify-between">
                        <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-slate-100 text-slate-900">
                          <Icon className="h-4 w-4 text-blue-600" />
                        </span>
                        <Badge variant="default" className="capitalize">
                          {item.role}
                        </Badge>
                      </div>
                      <h3 className="mt-4 text-lg font-semibold text-slate-900">
                        {item.title}
                      </h3>
                      <p className="mt-1.5 text-sm leading-relaxed text-slate-500">
                        {item.description}
                      </p>
                    </div>
                    <Link to={item.to} className="mt-6">
                      <Button variant="secondary" size="sm" className="w-full">
                        {item.cta}
                        <ArrowRight className="h-4 w-4" />
                      </Button>
                    </Link>
                  </Card>
                );
              })}
            </div>
          </div>
        </section>

        {/* FAQ Section */}
        <section
          id="faq"
          className="border-b border-dashed border-slate-200 bg-white"
        >
          <div className="mx-auto max-w-2xl px-4 py-20 sm:px-6">
            <div className="text-center">
              <Badge variant="accent">FAQ</Badge>
              <h2 className="mt-3 text-2xl font-semibold tracking-tight text-slate-900 sm:text-3xl">
                Common Questions
              </h2>
            </div>

            <div className="mt-10 space-y-3">
              {FAQS.map((item, idx) => {
                const isOpen = openFaq === idx;
                return (
                  <div
                    key={item.q}
                    className="rounded-xl border border-slate-200 bg-[#FAFAFA] transition-colors"
                  >
                    <button
                      type="button"
                      onClick={() => setOpenFaq(isOpen ? -1 : idx)}
                      className="flex w-full items-center justify-between px-5 py-4 text-left text-base font-medium text-slate-900 cursor-pointer"
                    >
                      <span>{item.q}</span>
                      <ChevronDown
                        className={`h-4 w-4 shrink-0 text-slate-400 transition-transform ${isOpen ? 'rotate-180' : ''
                          }`}
                      />
                    </button>
                    {isOpen && (
                      <div className="px-5 pb-4 text-sm leading-relaxed text-slate-500">
                        {item.a}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        {/* Dark High-Contrast CTA Banner */}
        <section className="bg-[#FAFAFA] py-16">
          <div className="mx-auto max-w-5xl px-4 sm:px-6">
            <div className="rounded-3xl bg-slate-950 bg-dark-grid px-6 py-16 text-center text-white shadow-sm sm:px-12">
              <h2 className="mx-auto max-w-xl text-2xl font-semibold tracking-tight sm:text-3xl">
                Know What Your Policy Actually Covers
              </h2>
              <p className="mx-auto mt-3 max-w-md text-sm leading-relaxed text-slate-400">
                Create a User or Organisation account in under a minute and
                start checking real clauses.
              </p>
              <div className="mt-7 flex flex-wrap items-center justify-center gap-3">
                <Link to="/register">
                  <Button variant="light" size="lg">
                    Create Your Account
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </Link>
              </div>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}
