import { useState } from 'react';
import {
  AlertCircle,
  AlertTriangle,
  Award,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Filter,
  Loader2,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  TrendingUp,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { recommendationsApi } from '@/lib/api';

const POLICY_TYPE_OPTIONS = [
  { value: 'health', label: 'Health Insurance' },
  { value: 'car', label: 'Car / Motor Insurance' },
  { value: 'life', label: 'Life Insurance' },
  { value: 'term', label: 'Term Life Insurance' },
];

const RISK_OPTIONS = [
  { value: '', label: 'Default (No preference specified)' },
  { value: 'low_premium', label: 'Low Premium Focus' },
  { value: 'balanced', label: 'Balanced Coverage & Cost' },
  { value: 'comprehensive_coverage', label: 'Comprehensive Coverage' },
];

const ROOM_RENT_OPTIONS = [
  { value: 'any', label: 'Any Room Type' },
  { value: 'shared', label: 'Shared Room' },
  { value: 'single_private', label: 'Single Private Room' },
  { value: 'no_limit', label: 'No Room Rent Limit' },
];

const MODEL_OPTIONS = [
  { value: 'auto', label: 'Auto (LightGBM + Weighted Fallback)' },
  { value: 'lightgbm', label: 'LightGBM Ranker' },
  { value: 'weighted', label: 'Deterministic Weighted Ranker' },
];

const INITIAL_FORM = {
  policy_type: 'health',
  age: '32',
  budget_min: '5000',
  budget_max: '25000',
  min_sum_insured: '500000',
  max_sum_insured: '',
  applicant_type: 'salaried',
  coverage_mode: 'individual',
  family_size: '1',
  risk_preference: 'balanced',
  annual_income_inr: '',
  required_benefits: '',
  pre_existing_conditions: '',
  max_copay_pct: '',
  max_deductible: '',
  max_waiting_period_days: '',
  max_ped_waiting_months: '',
  room_rent_preference: 'any',
  require_cashless: false,
  min_network_hospitals: '',
  location: '',
  tenure_years: '1',
  require_lifelong_renewal: false,
  smoking_status: '',
  vehicle_type: '',
  vehicle_age_years: '',
  require_own_damage: false,
  require_third_party: false,
  require_zero_depreciation: false,
  require_roadside_assistance: false,
  require_engine_protection: false,
  require_maturity_benefit: false,
  require_surrender_value: false,
  max_premium_payment_term_years: '',
  require_critical_illness_rider: false,
  require_accidental_death_rider: false,
  top_n: '5',
  ranking_model: 'auto',
};

function parseTagList(raw) {
  return String(raw || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
}

function formatInr(amount) {
  const num = Number(amount);
  if (!Number.isFinite(num)) return 'N/A';
  return `₹${Math.round(num).toLocaleString('en-IN')}`;
}

function validateAndBuildPayload(form) {
  const age = Number(form.age);
  if (!Number.isInteger(age) || age < 0 || age > 120) {
    return { error: 'Age must be an integer between 0 and 120.' };
  }

  const budgetMin = form.budget_min === '' ? 0 : Number(form.budget_min);
  const budgetMax = Number(form.budget_max);
  if (!Number.isFinite(budgetMax) || budgetMax <= 0) {
    return { error: 'Maximum annual budget must be greater than 0.' };
  }
  if (!Number.isFinite(budgetMin) || budgetMin < 0 || budgetMin > budgetMax) {
    return { error: 'Minimum annual budget must be >= 0 and <= maximum budget.' };
  }

  const minSumInsured = Number(form.min_sum_insured);
  if (!Number.isFinite(minSumInsured) || minSumInsured <= 0) {
    return { error: 'Minimum sum insured must be greater than 0.' };
  }

  const familySize = form.family_size === '' ? 1 : Number(form.family_size);
  if (!Number.isInteger(familySize) || familySize < 1 || familySize > 15) {
    return { error: 'Family size must be an integer between 1 and 15.' };
  }
  if (form.coverage_mode === 'individual' && familySize > 1) {
    return { error: 'Family size greater than 1 requires Family Floater coverage mode.' };
  }

  const topN = Number(form.top_n || 5);
  if (!Number.isInteger(topN) || topN < 1 || topN > 20) {
    return { error: 'Results limit (Top N) must be an integer between 1 and 20.' };
  }

  const profile = {
    policy_type: form.policy_type,
    age,
    budget_min: budgetMin,
    budget_max: budgetMax,
    applicant_type: (form.applicant_type || 'salaried').trim() || 'salaried',
    coverage_mode: form.coverage_mode,
    family_size: familySize,
    min_sum_insured: minSumInsured,
    required_benefits: parseTagList(form.required_benefits),
    pre_existing_conditions: parseTagList(form.pre_existing_conditions),
    room_rent_preference: form.room_rent_preference || 'any',
    require_cashless: Boolean(form.require_cashless),
    require_lifelong_renewal: Boolean(form.require_lifelong_renewal),
    tenure_years: Number(form.tenure_years || 1) >= 1 ? Number(form.tenure_years || 1) : 1,
  };

  if (form.max_sum_insured !== '') {
    const maxSi = Number(form.max_sum_insured);
    if (!Number.isFinite(maxSi) || maxSi < minSumInsured) {
      return { error: 'Maximum sum insured must be >= minimum sum insured.' };
    }
    profile.max_sum_insured = maxSi;
  }
  if (form.risk_preference) {
    profile.risk_preference = form.risk_preference;
  }
  if (form.annual_income_inr !== '') {
    const inc = Number(form.annual_income_inr);
    if (!Number.isFinite(inc) || inc <= 0) {
      return { error: 'Annual income must be greater than 0 if specified.' };
    }
    profile.annual_income_inr = inc;
  }
  if (form.max_copay_pct !== '') {
    profile.max_copay_pct = Number(form.max_copay_pct);
  }
  if (form.max_deductible !== '') {
    profile.max_deductible = Number(form.max_deductible);
  }
  if (form.max_waiting_period_days !== '') {
    profile.max_waiting_period_days = Number(form.max_waiting_period_days);
  }
  if (form.max_ped_waiting_months !== '') {
    profile.max_ped_waiting_months = Number(form.max_ped_waiting_months);
  }
  if (form.min_network_hospitals !== '') {
    profile.min_network_hospitals = Number(form.min_network_hospitals);
  }
  if (form.location.trim()) {
    profile.location = form.location.trim();
  }

  if (form.policy_type === 'car') {
    if (form.vehicle_type.trim()) profile.vehicle_type = form.vehicle_type.trim();
    if (form.vehicle_age_years !== '') profile.vehicle_age_years = Number(form.vehicle_age_years);
    profile.require_own_damage = Boolean(form.require_own_damage);
    profile.require_third_party = Boolean(form.require_third_party);
    profile.require_zero_depreciation = Boolean(form.require_zero_depreciation);
    profile.require_roadside_assistance = Boolean(form.require_roadside_assistance);
    profile.require_engine_protection = Boolean(form.require_engine_protection);
  }

  if (form.policy_type === 'life' || form.policy_type === 'term') {
    if (form.smoking_status) profile.smoking_status = form.smoking_status;
    profile.require_maturity_benefit = Boolean(form.require_maturity_benefit);
    profile.require_surrender_value = Boolean(form.require_surrender_value);
    if (form.max_premium_payment_term_years !== '') {
      profile.max_premium_payment_term_years = Number(form.max_premium_payment_term_years);
    }
    profile.require_critical_illness_rider = Boolean(form.require_critical_illness_rider);
    profile.require_accidental_death_rider = Boolean(form.require_accidental_death_rider);
  }

  return {
    payload: {
      profile,
      top_n: topN,
      ranking_model: form.ranking_model || 'auto',
    },
  };
}

export function RecommendationsPanel({ onGoPolicies }) {
  const [form, setForm] = useState(INITIAL_FORM);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [validationError, setValidationError] = useState('');
  const [requestError, setRequestError] = useState(null);
  const [lastPayload, setLastPayload] = useState(null);
  const [result, setResult] = useState(null);
  const [showEliminations, setShowEliminations] = useState(false);
  const [expandedCitations, setExpandedCitations] = useState({});
  const [expandedFeatures, setExpandedFeatures] = useState({});

  function updateField(name, value) {
    setForm((prev) => {
      const next = { ...prev, [name]: value };
      if (name === 'coverage_mode' && value === 'individual' && Number(prev.family_size) > 1) {
        next.family_size = '1';
      }
      return next;
    });
  }

  async function executeRecommend(payload) {
    setSubmitting(true);
    setRequestError(null);
    try {
      const res = await recommendationsApi.recommend(payload);
      setResult(res.data || null);
      setShowEliminations(res.data?.status === 'NO_MATCHING_POLICIES');
    } catch (err) {
      setRequestError({
        title: `Recommendation search failed${err.statusCode ? ` (HTTP ${err.statusCode})` : ''}`,
        message: err.message || 'Could not retrieve recommendations from the server.',
      });
    } finally {
      setSubmitting(false);
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setValidationError('');
    const built = validateAndBuildPayload(form);
    if (built.error) {
      setValidationError(built.error);
      return;
    }
    setLastPayload(built.payload);
    await executeRecommend(built.payload);
  }

  function toggleCitationBox(policyId) {
    setExpandedCitations((prev) => ({ ...prev, [policyId]: !prev[policyId] }));
  }

  function toggleFeatureBox(policyId) {
    setExpandedFeatures((prev) => ({ ...prev, [policyId]: !prev[policyId] }));
  }

  const filterReport = result?.filter_report;
  const recommendations = result?.recommendations || [];
  const eliminations = filterReport?.eliminations || [];

  return (
    <div className="space-y-6">
      <Card className="p-6">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="text-base font-semibold text-slate-900">
              Evidence-Grounded Plan Recommendations
            </h3>
            <p className="mt-1 text-sm text-slate-500">
              Filters only published organisation catalog plans strictly by extracted PDF attributes, ranks matches with CoverageIQ M5, and cites verified policy chunks.
            </p>
          </div>
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600">
            <TrendingUp className="h-5 w-5" />
          </span>
        </div>

        <form onSubmit={handleSubmit} className="mt-5 space-y-4">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div className="space-y-1.5">
              <label htmlFor="rec-policy-type" className="text-xs font-medium text-slate-700">
                Insurance Category
              </label>
              <select
                id="rec-policy-type"
                value={form.policy_type}
                onChange={(e) => updateField('policy_type', e.target.value)}
                className="flex h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm text-slate-900"
              >
                {POLICY_TYPE_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-1.5">
              <label htmlFor="rec-age" className="text-xs font-medium text-slate-700">
                Applicant Age (Years)
              </label>
              <Input
                id="rec-age"
                type="number"
                min="0"
                max="120"
                required
                value={form.age}
                onChange={(e) => updateField('age', e.target.value)}
              />
            </div>

            <div className="space-y-1.5">
              <label htmlFor="rec-budget-min" className="text-xs font-medium text-slate-700">
                Min Annual Budget (INR)
              </label>
              <Input
                id="rec-budget-min"
                type="number"
                min="0"
                value={form.budget_min}
                onChange={(e) => updateField('budget_min', e.target.value)}
              />
            </div>

            <div className="space-y-1.5">
              <label htmlFor="rec-budget-max" className="text-xs font-medium text-slate-700">
                Max Annual Budget (INR)
              </label>
              <Input
                id="rec-budget-max"
                type="number"
                min="1"
                required
                value={form.budget_max}
                onChange={(e) => updateField('budget_max', e.target.value)}
              />
            </div>

            <div className="space-y-1.5">
              <label htmlFor="rec-sum-insured" className="text-xs font-medium text-slate-700">
                Min Sum Insured (INR)
              </label>
              <Input
                id="rec-sum-insured"
                type="number"
                min="1"
                required
                value={form.min_sum_insured}
                onChange={(e) => updateField('min_sum_insured', e.target.value)}
              />
            </div>

            <div className="space-y-1.5">
              <label htmlFor="rec-coverage-mode" className="text-xs font-medium text-slate-700">
                Coverage Mode
              </label>
              <select
                id="rec-coverage-mode"
                value={form.coverage_mode}
                onChange={(e) => updateField('coverage_mode', e.target.value)}
                className="flex h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm text-slate-900"
              >
                <option value="individual">Individual</option>
                <option value="family_floater">Family Floater</option>
              </select>
            </div>

            <div className="space-y-1.5">
              <label htmlFor="rec-family-size" className="text-xs font-medium text-slate-700">
                Family Size
              </label>
              <Input
                id="rec-family-size"
                type="number"
                min="1"
                max="15"
                disabled={form.coverage_mode === 'individual'}
                value={form.family_size}
                onChange={(e) => updateField('family_size', e.target.value)}
              />
            </div>

            <div className="space-y-1.5">
              <label htmlFor="rec-risk-pref" className="text-xs font-medium text-slate-700">
                Risk Preference
              </label>
              <select
                id="rec-risk-pref"
                value={form.risk_preference}
                onChange={(e) => updateField('risk_preference', e.target.value)}
                className="flex h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm text-slate-900"
              >
                {RISK_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <div className="space-y-1.5">
              <label htmlFor="rec-benefits" className="text-xs font-medium text-slate-700">
                Required Benefits (comma-separated)
              </label>
              <Input
                id="rec-benefits"
                placeholder="e.g. cashless_hospitalization, daycare"
                value={form.required_benefits}
                onChange={(e) => updateField('required_benefits', e.target.value)}
              />
            </div>

            <div className="space-y-1.5">
              <label htmlFor="rec-conditions" className="text-xs font-medium text-slate-700">
                Pre-Existing Conditions (comma-separated)
              </label>
              <Input
                id="rec-conditions"
                placeholder="e.g. diabetes, hypertension"
                value={form.pre_existing_conditions}
                onChange={(e) => updateField('pre_existing_conditions', e.target.value)}
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <label htmlFor="rec-ranking-model" className="text-xs font-medium text-slate-700">
                  Ranking Model
                </label>
                <select
                  id="rec-ranking-model"
                  value={form.ranking_model}
                  onChange={(e) => updateField('ranking_model', e.target.value)}
                  className="flex h-11 w-full rounded-xl border border-slate-200 bg-white px-2.5 text-sm text-slate-900"
                >
                  {MODEL_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="space-y-1.5">
                <label htmlFor="rec-top-n" className="text-xs font-medium text-slate-700">
                  Top N Plans
                </label>
                <Input
                  id="rec-top-n"
                  type="number"
                  min="1"
                  max="20"
                  value={form.top_n}
                  onChange={(e) => updateField('top_n', e.target.value)}
                />
              </div>
            </div>
          </div>

          <div className="pt-1">
            <button
              type="button"
              onClick={() => setShowAdvanced((v) => !v)}
              className="inline-flex items-center gap-1.5 text-xs font-medium text-blue-600 hover:text-blue-700"
            >
              <Filter className="h-3.5 w-3.5" />
              {showAdvanced ? 'Hide Additional Filter Constraints' : 'Show Additional Filter Constraints'}
              {showAdvanced ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
            </button>
          </div>

          {showAdvanced && (
            <div className="rounded-xl border border-slate-200 bg-[#FAFAFA] p-4 space-y-4">
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <div className="space-y-1">
                  <label htmlFor="rec-copay" className="text-xs font-medium text-slate-700">
                    Max Co-pay (%)
                  </label>
                  <Input
                    id="rec-copay"
                    type="number"
                    min="0"
                    max="100"
                    value={form.max_copay_pct}
                    onChange={(e) => updateField('max_copay_pct', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <label htmlFor="rec-deductible" className="text-xs font-medium text-slate-700">
                    Max Deductible (INR)
                  </label>
                  <Input
                    id="rec-deductible"
                    type="number"
                    min="0"
                    value={form.max_deductible}
                    onChange={(e) => updateField('max_deductible', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <label htmlFor="rec-waiting" className="text-xs font-medium text-slate-700">
                    Max Initial Waiting (Days)
                  </label>
                  <Input
                    id="rec-waiting"
                    type="number"
                    min="0"
                    max="365"
                    value={form.max_waiting_period_days}
                    onChange={(e) => updateField('max_waiting_period_days', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <label htmlFor="rec-ped-waiting" className="text-xs font-medium text-slate-700">
                    Max PED Waiting (Months)
                  </label>
                  <Input
                    id="rec-ped-waiting"
                    type="number"
                    min="0"
                    max="120"
                    value={form.max_ped_waiting_months}
                    onChange={(e) => updateField('max_ped_waiting_months', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <label htmlFor="rec-room-rent" className="text-xs font-medium text-slate-700">
                    Room Rent Preference
                  </label>
                  <select
                    id="rec-room-rent"
                    value={form.room_rent_preference}
                    onChange={(e) => updateField('room_rent_preference', e.target.value)}
                    className="flex h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm text-slate-900"
                  >
                    {ROOM_RENT_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="space-y-1">
                  <label htmlFor="rec-network" className="text-xs font-medium text-slate-700">
                    Min Cashless Network
                  </label>
                  <Input
                    id="rec-network"
                    type="number"
                    min="0"
                    value={form.min_network_hospitals}
                    onChange={(e) => updateField('min_network_hospitals', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <label htmlFor="rec-income" className="text-xs font-medium text-slate-700">
                    Annual Income (INR)
                  </label>
                  <Input
                    id="rec-income"
                    type="number"
                    min="1"
                    value={form.annual_income_inr}
                    onChange={(e) => updateField('annual_income_inr', e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <label htmlFor="rec-location" className="text-xs font-medium text-slate-700">
                    City / Location
                  </label>
                  <Input
                    id="rec-location"
                    placeholder="e.g. mumbai"
                    value={form.location}
                    onChange={(e) => updateField('location', e.target.value)}
                  />
                </div>
              </div>

              <div className="flex flex-wrap gap-5 text-xs text-slate-700">
                <label className="inline-flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={form.require_cashless}
                    onChange={(e) => updateField('require_cashless', e.target.checked)}
                    className="h-4 w-4 rounded border-slate-300 text-blue-600"
                  />
                  Require Cashless Facility
                </label>
                <label className="inline-flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={form.require_lifelong_renewal}
                    onChange={(e) => updateField('require_lifelong_renewal', e.target.checked)}
                    className="h-4 w-4 rounded border-slate-300 text-blue-600"
                  />
                  Require Lifelong Renewal
                </label>
              </div>

              {form.policy_type === 'car' && (
                <div className="border-t border-slate-200 pt-3 space-y-3">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">
                    Car / Motor Options
                  </p>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <Input
                      aria-label="Vehicle Type"
                      placeholder="Vehicle type (e.g. four_wheeler)"
                      value={form.vehicle_type}
                      onChange={(e) => updateField('vehicle_type', e.target.value)}
                    />
                    <Input
                      aria-label="Vehicle Age (Years)"
                      type="number"
                      min="0"
                      max="50"
                      placeholder="Vehicle age in years"
                      value={form.vehicle_age_years}
                      onChange={(e) => updateField('vehicle_age_years', e.target.value)}
                    />
                  </div>
                  <div className="flex flex-wrap gap-4 text-xs text-slate-700">
                    {[
                      ['require_own_damage', 'Require Own Damage'],
                      ['require_third_party', 'Require Third Party'],
                      ['require_zero_depreciation', 'Require Zero Depreciation'],
                      ['require_roadside_assistance', 'Require Roadside Assistance'],
                      ['require_engine_protection', 'Require Engine Protection'],
                    ].map(([key, label]) => (
                      <label key={key} className="inline-flex items-center gap-1.5 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={Boolean(form[key])}
                          onChange={(e) => updateField(key, e.target.checked)}
                          className="h-4 w-4 rounded border-slate-300 text-blue-600"
                        />
                        {label}
                      </label>
                    ))}
                  </div>
                </div>
              )}

              {(form.policy_type === 'life' || form.policy_type === 'term') && (
                <div className="border-t border-slate-200 pt-3 space-y-3">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">
                    Life / Term Options
                  </p>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <select
                      aria-label="Smoking Status"
                      value={form.smoking_status}
                      onChange={(e) => updateField('smoking_status', e.target.value)}
                      className="flex h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm text-slate-900"
                    >
                      <option value="">Smoking Status (Optional)</option>
                      <option value="non_smoker">Non-Smoker</option>
                      <option value="smoker">Smoker</option>
                    </select>
                    <Input
                      aria-label="Max Premium Payment Term (Years)"
                      type="number"
                      min="1"
                      max="60"
                      placeholder="Max payment term (years)"
                      value={form.max_premium_payment_term_years}
                      onChange={(e) => updateField('max_premium_payment_term_years', e.target.value)}
                    />
                  </div>
                  <div className="flex flex-wrap gap-4 text-xs text-slate-700">
                    {[
                      ['require_maturity_benefit', 'Require Maturity Benefit'],
                      ['require_surrender_value', 'Require Surrender Value'],
                      ['require_critical_illness_rider', 'Critical Illness Rider'],
                      ['require_accidental_death_rider', 'Accidental Death Rider'],
                    ].map(([key, label]) => (
                      <label key={key} className="inline-flex items-center gap-1.5 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={Boolean(form[key])}
                          onChange={(e) => updateField(key, e.target.checked)}
                          className="h-4 w-4 rounded border-slate-300 text-blue-600"
                        />
                        {label}
                      </label>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {validationError && (
            <div
              role="alert"
              className="flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50 px-3.5 py-2.5 text-sm text-amber-900"
            >
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
              <span>{validationError}</span>
            </div>
          )}

          <div className="flex flex-wrap items-center justify-between gap-3">
            <span className="text-xs text-slate-500">
              Private user uploads remain excluded; only published catalog policies are evaluated.
            </span>
            <Button type="submit" disabled={submitting}>
              {submitting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Evaluating Plans...
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4" />
                  Find Matching Plans
                </>
              )}
            </Button>
          </div>

          {submitting && (
            <div
              role="status"
              aria-live="polite"
              className="flex items-center gap-2.5 rounded-xl border border-blue-200 bg-blue-50 px-3.5 py-2.5 text-sm text-blue-900"
            >
              <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
              <span>
                Running deterministic eligibility filters, ranking matches, and verifying PDF citations...
              </span>
            </div>
          )}

          {requestError && (
            <div
              role="alert"
              className="flex flex-wrap items-start justify-between gap-3 rounded-xl border border-red-200 bg-red-50 px-3.5 py-3 text-sm text-red-800"
            >
              <div className="flex items-start gap-2.5">
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-600" />
                <div>
                  <p className="font-semibold">{requestError.title}</p>
                  <p>{requestError.message}</p>
                </div>
              </div>
              {lastPayload && (
                <Button
                  type="button"
                  size="sm"
                  variant="secondary"
                  disabled={submitting}
                  onClick={() => executeRecommend(lastPayload)}
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                  Retry
                </Button>
              )}
            </div>
          )}
        </form>
      </Card>

      {result && (
        <>
          {/* Filter Audit Summary */}
          <Card className="p-6 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap items-center gap-2">
                <ShieldCheck className="h-5 w-5 text-blue-600" />
                <h4 className="text-base font-semibold text-slate-900">
                  Catalog Eligibility Audit
                </h4>
                <Badge
                  variant={result.status === 'RECOMMENDATIONS_READY' ? 'success' : 'warning'}
                >
                  {result.status}
                </Badge>
              </div>
              <div className="flex flex-wrap items-center gap-2 text-xs text-slate-600">
                <span>Model: <strong className="text-slate-900">{result.model_used}</strong></span>
                {result.feature_schema_version && (
                  <span>· Schema v{result.feature_schema_version}</span>
                )}
              </div>
            </div>

            {filterReport && (
              <div className="grid grid-cols-3 gap-3">
                <div className="rounded-xl border border-slate-200 bg-[#FAFAFA] p-3">
                  <p className="text-xs text-slate-500">Catalog Candidates</p>
                  <p className="mt-1 text-lg font-semibold text-slate-900">
                    {filterReport.total_candidates}
                  </p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-[#FAFAFA] p-3">
                  <p className="text-xs text-slate-500">Passed Hard Filters</p>
                  <p className="mt-1 text-lg font-semibold text-emerald-700">
                    {filterReport.passed_count}
                  </p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-[#FAFAFA] p-3">
                  <p className="text-xs text-slate-500">Eliminated</p>
                  <p className="mt-1 text-lg font-semibold text-amber-700">
                    {filterReport.eliminated_count}
                  </p>
                </div>
              </div>
            )}

            {Array.isArray(result.warnings) && result.warnings.length > 0 && (
              <div className="rounded-xl border border-amber-200 bg-amber-50/70 px-4 py-3 text-xs text-amber-900 space-y-1">
                {result.warnings.map((w, idx) => (
                  <p key={`warn-${idx}`}>{w}</p>
                ))}
              </div>
            )}

            {eliminations.length > 0 && (
              <div>
                <button
                  type="button"
                  onClick={() => setShowEliminations((v) => !v)}
                  className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-700 hover:text-slate-900"
                >
                  {showEliminations ? 'Hide' : 'Show'} Eliminated Plans ({eliminations.length})
                  {showEliminations ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                </button>
                {showEliminations && (
                  <div className="mt-3 space-y-2">
                    {eliminations.map((elim, idx) => (
                      <div
                        key={`elim-${elim.policy_id}-${idx}`}
                        className="rounded-xl border border-slate-200 bg-slate-50 px-3.5 py-2.5 text-xs"
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <span className="font-semibold text-slate-900">{elim.policy_name}</span>
                          <Badge variant="danger">{elim.rule_id || elim.rule}</Badge>
                        </div>
                        <p className="mt-1 text-slate-700">{elim.reason}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </Card>

          {result.status === 'NO_MATCHING_POLICIES' ? (
            <Card className="p-8 text-center space-y-3">
              <ShieldAlert className="mx-auto h-10 w-10 text-amber-500" />
              <h4 className="text-base font-semibold text-slate-900">
                No Matching Catalog Policies Found
              </h4>
              <p className="mx-auto max-w-lg text-sm text-slate-600">
                {filterReport?.total_candidates === 0
                  ? 'No active, published catalog policies exist in this category with verified attributes yet.'
                  : 'Every catalog candidate in this category failed one or more strict eligibility rules or lacked mandatory verified attributes in the uploaded PDF (NO EVIDENCE = NO CLAIM).'}
              </p>
              {typeof onGoPolicies === 'function' && (
                <div>
                  <Button type="button" variant="outline" size="sm" onClick={onGoPolicies}>
                    Inspect My Policies
                  </Button>
                </div>
              )}
            </Card>
          ) : (
            <div className="space-y-4">
              {recommendations.map((rec) => {
                const citationsOpen = Boolean(expandedCitations[rec.policy_id] ?? true);
                const featuresOpen = Boolean(expandedFeatures[rec.policy_id]);
                const featureEntries = Object.entries(rec.feature_breakdown || {});
                const citations = Array.isArray(rec.citations) ? rec.citations : [];
                const missingAttrs = Array.isArray(rec.missing_attributes) ? rec.missing_attributes : [];
                const exclusions = Array.isArray(rec.exclusions) ? rec.exclusions : [];

                return (
                  <Card key={rec.policy_id} className="p-6 space-y-4">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div>
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="inline-flex items-center gap-1 rounded-lg bg-emerald-100 px-2.5 py-0.5 text-xs font-bold text-emerald-800">
                            <Award className="h-3.5 w-3.5" />
                            Rank #{rec.rank}
                          </span>
                          <h4 className="text-base font-semibold text-slate-900">
                            {rec.policy_name}
                          </h4>
                          <Badge variant="default" className="capitalize">
                            {rec.policy_type}
                          </Badge>
                          {rec.extraction_status && (
                            <Badge variant={rec.extraction_status === 'completed' ? 'success' : 'warning'}>
                              extraction: {rec.extraction_status}
                            </Badge>
                          )}
                        </div>
                        <p className="mt-1 text-xs text-slate-500">
                          {rec.insurer_name || 'Verified Catalog Plan'}
                          {rec.document_version ? ` · Document v${rec.document_version}` : ''}
                          {` · Score: ${Number(rec.score).toFixed(3)}`}
                        </p>
                      </div>

                      <div className="text-right">
                        <p className="text-sm font-semibold text-slate-900">
                          {formatInr(rec.annual_premium_min)} – {formatInr(rec.annual_premium_max)} / yr
                        </p>
                        <p className="text-xs text-slate-500">
                          Max Cover: {formatInr(rec.sum_insured_max)}
                        </p>
                      </div>
                    </div>

                    {rec.explanation && (
                      <p className="text-sm leading-relaxed text-slate-700">{rec.explanation}</p>
                    )}

                    {missingAttrs.length > 0 && (
                      <div className="flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50/70 px-3.5 py-2.5 text-xs text-amber-900">
                        <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-600" />
                        <span>
                          <strong>Unverified / Missing PDF Attributes:</strong> {missingAttrs.join(', ')} (not stated with evidence in extracted policy chunks).
                        </span>
                      </div>
                    )}

                    <div className="grid gap-3 md:grid-cols-2">
                      {Array.isArray(rec.key_highlights) && rec.key_highlights.length > 0 && (
                        <div className="rounded-xl border border-slate-200/80 bg-[#FAFAFA] p-3">
                          <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">
                            Verified Coverage Highlights
                          </p>
                          <ul className="mt-1.5 list-disc space-y-1 pl-4 text-xs text-slate-700">
                            {rec.key_highlights.map((item, idx) => (
                              <li key={`${rec.policy_id}-hl-${idx}`}>{item}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {exclusions.length > 0 && (
                        <div className="rounded-xl border border-red-100 bg-red-50/40 p-3">
                          <p className="text-xs font-semibold uppercase tracking-wide text-red-700">
                            Verified Exclusions &amp; Waiting Terms
                          </p>
                          <ul className="mt-1.5 list-disc space-y-1 pl-4 text-xs text-red-900">
                            {exclusions.map((item, idx) => (
                              <li key={`${rec.policy_id}-ex-${idx}`}>{item}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>

                    {featureEntries.length > 0 && (
                      <div>
                        <button
                          type="button"
                          onClick={() => toggleFeatureBox(rec.policy_id)}
                          className="text-xs font-medium text-slate-600 hover:text-slate-900"
                        >
                          {featuresOpen ? 'Hide' : 'Show'} Ranking Score Breakdown ({featureEntries.length})
                        </button>
                        {featuresOpen && (
                          <div className="mt-2 flex flex-wrap gap-2">
                            {featureEntries.map(([k, v]) => (
                              <span
                                key={`${rec.policy_id}-feat-${k}`}
                                className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs text-slate-700"
                              >
                                {k}: <strong>{Number(v).toFixed(2)}</strong>
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    <div className="border-t border-slate-100 pt-3 space-y-2">
                      <button
                        type="button"
                        onClick={() => toggleCitationBox(rec.policy_id)}
                        className="inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-600 hover:text-slate-900"
                      >
                        Verified Policy Evidence Citations ({citations.length})
                        {citationsOpen ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                      </button>

                      {citationsOpen && (
                        citations.length === 0 ? (
                          <p className="text-xs text-slate-500">
                            No chunk-level evidence citations attached for this plan.
                          </p>
                        ) : (
                          <div className="space-y-2">
                            {citations.map((cit, idx) => (
                              <div
                                key={`${rec.policy_id}-cit-${cit.chunk_id}-${idx}`}
                                className="rounded-lg border border-blue-100 bg-blue-50/40 p-3 text-xs"
                              >
                                <div className="flex flex-wrap items-center gap-2 font-medium text-slate-800">
                                  <span className="inline-flex items-center gap-1 rounded bg-blue-100 px-2 py-0.5 text-blue-800">
                                    <CheckCircle2 className="h-3 w-3" />
                                    Page {cit.page_range || cit.page}
                                  </span>
                                  <span>Clause: {cit.clause}</span>
                                  {cit.category && (
                                    <Badge variant="default" className="py-0 text-[11px]">
                                      {cit.category}
                                    </Badge>
                                  )}
                                  <span className="font-mono text-[11px] text-slate-400">
                                    {cit.chunk_id}
                                  </span>
                                </div>
                                {cit.snippet && (
                                  <p className="mt-1.5 whitespace-pre-line text-slate-600">
                                    {cit.snippet}
                                  </p>
                                )}
                              </div>
                            ))}
                          </div>
                        )
                      )}
                    </div>
                  </Card>
                );
              })}
            </div>
          )}
        </>
      )}
    </div>
  );
}
