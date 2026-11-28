import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { RecommendationsPanel } from '@/components/RecommendationsPanel';
import { PolicyUploadPanel } from '@/components/PolicyUploadPanel';
import { policiesApi, recommendationsApi } from '@/lib/api';

describe('CoverageIQ Checkpoint 5.0 Recommendations & Catalog Visibility UI', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('recommendationsApi.recommend and policiesApi.updateVisibility send cookie credentials and handle 401 refresh retry', async () => {
    const calls = [];
    globalThis.fetch = vi.fn(async (url, init = {}) => {
      calls.push({ url, init });
      if (url.endsWith('/recommendations') && calls.filter((c) => c.url.endsWith('/recommendations')).length === 1) {
        return {
          ok: false,
          status: 401,
          json: async () => ({ message: 'Access token expired' }),
        };
      }
      if (url.endsWith('/users/refresh-token')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({ success: true }),
        };
      }
      if (url.endsWith('/recommendations')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            success: true,
            data: { status: 'NO_MATCHING_POLICIES', recommendations: [] },
          }),
        };
      }
      if (url.includes('/visibility')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            success: true,
            data: { policy: { id: 'pol-1', visibility: 'catalog' } },
          }),
        };
      }
      throw new Error(`Unexpected fetch URL: ${url}`);
    });

    const recRes = await recommendationsApi.recommend({
      profile: { policy_type: 'health', age: 30, budget_max: 20000, min_sum_insured: 500000 },
    });
    expect(recRes.data.status).toBe('NO_MATCHING_POLICIES');
    expect(calls[0].init.method).toBe('POST');
    expect(calls[0].init.credentials).toBe('include');
    expect(calls[1].url).toMatch(/\/users\/refresh-token$/);

    const visRes = await policiesApi.updateVisibility('pol-1', 'catalog');
    expect(visRes.data.policy.visibility).toBe('catalog');
    expect(calls[3].init.method).toBe('PATCH');
    expect(JSON.parse(calls[3].init.body)).toEqual({ visibility: 'catalog' });
  });

  it('PolicyUploadPanel hides catalog publish toggle for regular users and keeps it unchecked by default for organisations', async () => {
    const { rerender } = render(
      <PolicyUploadPanel policies={[]} userRole="user" />
    );

    fireEvent.click(screen.getByRole('button', { name: /New Policy/i }));
    expect(screen.queryByText(/Publish to Recommendations Catalog/i)).not.toBeInTheDocument();

    const createSpy = vi.spyOn(policiesApi, 'create').mockResolvedValue({
      data: {
        policy: {
          id: '11111111-1111-4111-8111-111111111111',
          title: 'Org Catalog Health Plan',
          policy_type: 'health',
          status: 'draft',
          visibility: 'catalog',
          external_policy_key: 'pol_org_cat',
        },
      },
    });

    rerender(<PolicyUploadPanel policies={[]} userRole="organisation" />);
    const checkbox = screen.getByRole('checkbox', { name: /Publish to Recommendations Catalog/i });
    expect(checkbox).not.toBeChecked();

    fireEvent.change(screen.getByPlaceholderText(/Policy title/i), {
      target: { value: 'Org Catalog Health Plan' },
    });
    fireEvent.click(checkbox);
    expect(checkbox).toBeChecked();

    fireEvent.submit(checkbox.closest('form'));
    await waitFor(() => {
      expect(createSpy).toHaveBeenCalledWith({
        title: 'Org Catalog Health Plan',
        policy_type: 'health',
        insurer_name: undefined,
        visibility: 'catalog',
      });
    });
  });

  it('RecommendationsPanel validates conflicting budget bounds before calling API', () => {
    const recommendSpy = vi.spyOn(recommendationsApi, 'recommend');
    render(<RecommendationsPanel />);

    fireEvent.change(screen.getByLabelText(/Min Annual Budget/i), {
      target: { value: '30000' },
    });
    const submitBtn = screen.getByRole('button', { name: /Find Matching Plans/i });
    fireEvent.submit(submitBtn.closest('form'));

    expect(screen.getByRole('alert')).toHaveTextContent(/Minimum annual budget must be >= 0 and <= maximum budget/i);
    expect(recommendSpy).not.toHaveBeenCalled();
  });

  it('RecommendationsPanel renders ranked policies, missing attribute notices, exclusions, citations, and empty elimination audit', async () => {
    const recommendSpy = vi
      .spyOn(recommendationsApi, 'recommend')
      .mockResolvedValueOnce({
        data: {
          request_id: 'req-rec-1',
          status: 'RECOMMENDATIONS_READY',
          model_used: 'coverageiq_lightgbm_ranker',
          feature_schema_version: '1.0.0',
          warnings: [],
          recommendations: [
            {
              policy_id: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
              policy_name: 'Shield Prime Health',
              insurer_name: 'Shield General Insurance',
              policy_type: 'health',
              document_version: 2,
              extraction_status: 'partial',
              rank: 1,
              score: 2.345,
              model_used: 'coverageiq_lightgbm_ranker',
              annual_premium_min: 11000,
              annual_premium_max: 16500,
              sum_insured_max: 1000000,
              explanation: 'Eligible under age and budget filters with verified cashless coverage.',
              key_highlights: ['Cashless network: 7,500 hospitals/garages'],
              exclusions: ['cosmetic_surgery'],
              missing_attributes: ['copay_pct'],
              feature_breakdown: { affordability_score: 0.61 },
              citations: [
                {
                  chunk_id: 'chunk-health-1',
                  page: 2,
                  page_range: '2',
                  clause: 'Section 1.2 Eligibility & Premium',
                  category: 'coverage',
                  snippet: 'Entry Age 18 to 65 years. Annual Premium INR 11,000 to 16,500.',
                },
              ],
            },
          ],
          filter_report: {
            status: 'ELIGIBLE_CANDIDATES_FOUND',
            total_candidates: 2,
            passed_count: 1,
            eliminated_count: 1,
            eliminations: [
              {
                policy_id: 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb',
                policy_name: 'Ultra Expensive Plan',
                rule: 'budget_affordability',
                rule_id: 'RULE_BUDGET_EXCEEDED',
                reason: 'Minimum annual premium exceeds budget_max.',
              },
            ],
            policy_evaluations: [],
          },
        },
      })
      .mockResolvedValueOnce({
        data: {
          request_id: 'req-rec-2',
          status: 'NO_MATCHING_POLICIES',
          model_used: 'none',
          feature_schema_version: '1.0.0',
          warnings: [],
          recommendations: [],
          filter_report: {
            status: 'NO_MATCHING_POLICIES',
            total_candidates: 1,
            passed_count: 0,
            eliminated_count: 1,
            eliminations: [
              {
                policy_id: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
                policy_name: 'Shield Prime Health',
                rule: 'budget_affordability',
                rule_id: 'RULE_BUDGET_EXCEEDED',
                reason: 'Minimum annual premium INR 11,000 exceeds max budget.',
              },
            ],
            policy_evaluations: [],
          },
        },
      });

    render(<RecommendationsPanel onGoPolicies={vi.fn()} />);

    fireEvent.click(screen.getByRole('button', { name: /Find Matching Plans/i }));

    await waitFor(() => {
      expect(screen.getByText('Shield Prime Health')).toBeInTheDocument();
    });

    expect(recommendSpy).toHaveBeenCalledTimes(1);
    expect(screen.getByText('Rank #1')).toBeInTheDocument();
    expect(screen.getByText(/Unverified \/ Missing PDF Attributes:/i)).toBeInTheDocument();
    expect(screen.getByText(/copay_pct/)).toBeInTheDocument();
    expect(screen.getByText('cosmetic_surgery')).toBeInTheDocument();
    expect(screen.getByText(/Section 1.2 Eligibility & Premium/)).toBeInTheDocument();
    expect(screen.getByText(/chunk-health-1/)).toBeInTheDocument();

    // Expand eliminated policy breakdown
    fireEvent.click(screen.getByRole('button', { name: /Show Eliminated Plans/i }));
    expect(screen.getByText('Ultra Expensive Plan')).toBeInTheDocument();

    // Now test strict NO_MATCHING_POLICIES state
    fireEvent.click(screen.getByRole('button', { name: /Find Matching Plans/i }));
    await waitFor(() => {
      expect(screen.getByText(/No Matching Catalog Policies Found/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/Minimum annual premium INR 11,000 exceeds max budget/i)).toBeInTheDocument();
  });

  it('RecommendationsPanel surfaces errors and retries cleanly', async () => {
    const recommendSpy = vi
      .spyOn(recommendationsApi, 'recommend')
      .mockRejectedValueOnce(Object.assign(new Error('AI gateway timed out'), { statusCode: 504 }))
      .mockResolvedValueOnce({
        data: {
          status: 'NO_MATCHING_POLICIES',
          model_used: 'none',
          recommendations: [],
          filter_report: {
            status: 'NO_MATCHING_POLICIES',
            total_candidates: 0,
            passed_count: 0,
            eliminated_count: 0,
            eliminations: [],
            policy_evaluations: [],
          },
        },
      });

    render(<RecommendationsPanel />);
    fireEvent.click(screen.getByRole('button', { name: /Find Matching Plans/i }));

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(/AI gateway timed out/i);
    });

    fireEvent.click(screen.getByRole('button', { name: /Retry/i }));
    await waitFor(() => {
      expect(recommendSpy).toHaveBeenCalledTimes(2);
      expect(screen.getByText(/No Matching Catalog Policies Found/i)).toBeInTheDocument();
    });
  });
});
