import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { PolicyQAPanel } from '@/components/PolicyQAPanel';
import { qaApi } from '@/lib/api';

const ACTIVE_POLICY_1 = {
  id: '11111111-1111-4111-8111-111111111111',
  external_policy_key: 'pol_health_gold',
  title: 'Gold Health Shield',
  policy_type: 'health',
  status: 'active',
};

const ACTIVE_POLICY_2 = {
  id: '22222222-2222-4222-8222-222222222222',
  external_policy_key: 'pol_auto_prime',
  title: 'Prime Auto Protect',
  policy_type: 'motor',
  status: 'active',
};

const DRAFT_POLICY = {
  id: '33333333-3333-4333-8333-333333333333',
  external_policy_key: 'pol_draft_only',
  title: 'Unindexed Draft Plan',
  policy_type: 'life',
  status: 'draft',
};

describe('CoverageIQ Checkpoint 4.9 Q&A API & Interactive Panel', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('qaApi.ask and qaApi.getHistory use cookie auth, handle 401 refresh retry, and send expected payloads', async () => {
    const calls = [];
    globalThis.fetch = vi.fn(async (url, init = {}) => {
      calls.push({ url, init });
      if (url.endsWith('/ask') && calls.filter((c) => c.url.endsWith('/ask')).length === 1) {
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
      if (url.endsWith('/ask')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            success: true,
            data: {
              conversation_id: '44444444-4444-4444-8444-444444444444',
              answer: { decision: 'COVERED' },
            },
          }),
        };
      }
      if (url.includes('/qa-history?')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            success: true,
            data: { items: [], total_questions: 0 },
          }),
        };
      }
      throw new Error(`Unexpected fetch URL: ${url}`);
    });

    const askRes = await qaApi.ask(ACTIVE_POLICY_1.id, {
      question: 'Is day care surgery covered?',
      conversationId: '55555555-5555-4555-8555-555555555555',
    });

    expect(askRes.data.conversation_id).toBe('44444444-4444-4444-8444-444444444444');
    expect(calls[0].init.method).toBe('POST');
    expect(calls[0].init.credentials).toBe('include');
    expect(JSON.parse(calls[0].init.body)).toEqual({
      question: 'Is day care surgery covered?',
      conversation_id: '55555555-5555-4555-8555-555555555555',
    });
    expect(calls[1].url).toMatch(/\/users\/refresh-token$/);

    await qaApi.getHistory(ACTIVE_POLICY_1.id, {
      page: 2,
      limit: 25,
      conversationId: '44444444-4444-4444-8444-444444444444',
    });
    expect(calls[3].url).toContain(
      `/policies/${ACTIVE_POLICY_1.id}/qa-history?page=2&limit=25&conversationId=44444444-4444-4444-8444-444444444444`
    );
  });

  it('shows draft guidance when no active policies exist and navigates to policies tab', () => {
    const onGoUploadPolicies = vi.fn();
    render(
      <PolicyQAPanel
        policies={[DRAFT_POLICY]}
        policiesLoading={false}
        onGoUploadPolicies={onGoUploadPolicies}
      />
    );

    expect(
      screen.getByText(/No ingested active policies available for Q&A yet/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/still in draft or processing/i)
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /Go to My Policies/i }));
    expect(onGoUploadPolicies).toHaveBeenCalledTimes(1);
  });

  it('loads persisted history, renders verified decision + citations, supports New Thread and Retry on transient failure', async () => {
    const existingConvId = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
    const newConvId = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';

    const getHistorySpy = vi.spyOn(qaApi, 'getHistory').mockImplementation(async (policyId) => {
      if (policyId === ACTIVE_POLICY_1.id) {
        return {
          data: {
            items: [
              {
                id: 'msg-u-1',
                conversation_id: existingConvId,
                role: 'user',
                content: 'Is in-patient hospitalization covered?',
                created_at: '2026-11-27T18:00:00Z',
              },
              {
                id: 'msg-a-1',
                conversation_id: existingConvId,
                role: 'assistant',
                content:
                  'In-patient hospitalization for at least 24 hours is covered up to Sum Insured.',
                decision: 'COVERED',
                confidence_score: 0.92,
                conditions: ['Requires minimum 24 consecutive hours admission'],
                citations: [
                  {
                    chunk_id: 'chunk_hp_001',
                    page: 1,
                    page_range: '1',
                    clause: 'Section 2.1 In-Patient Care',
                    category: 'coverage',
                    snippet: 'Hospitalization expenses for minimum 24 hours are covered.',
                  },
                ],
                evaluator_status: 'PASSED',
                created_at: '2026-11-27T18:00:02Z',
              },
            ],
            total_questions: 1,
          },
        };
      }
      return { data: { items: [], total_questions: 0 } };
    });

    let askCallCount = 0;
    const askSpy = vi.spyOn(qaApi, 'ask').mockImplementation(async (policyId, { question, conversationId }) => {
      askCallCount += 1;
      if (askCallCount === 1) {
        const err = new Error('AI service timed out while analyzing your policy');
        err.statusCode = 504;
        throw err;
      }
      if (askCallCount === 2) {
        return {
          data: {
            conversation_id: conversationId || existingConvId,
            user_message: {
              id: 'msg-u-2',
              conversation_id: conversationId || existingConvId,
              role: 'user',
              content: question,
              created_at: '2026-11-27T18:05:00Z',
            },
            assistant_message: {
              id: 'msg-a-2',
              conversation_id: conversationId || existingConvId,
              role: 'assistant',
              content: 'No evidence in the uploaded policy discusses dental cosmetic implants.',
              decision: 'INSUFFICIENT EVIDENCE',
              confidence_score: null,
              conditions: [],
              citations: [],
              evaluator_status: 'ABSTAINED',
              created_at: '2026-11-27T18:05:02Z',
            },
          },
        };
      }
      return {
        data: {
          conversation_id: newConvId,
          user_message: {
            id: 'msg-u-3',
            conversation_id: newConvId,
            role: 'user',
            content: question,
            created_at: '2026-11-27T18:06:00Z',
          },
          assistant_message: {
            id: 'msg-a-3',
            conversation_id: newConvId,
            role: 'assistant',
            content: 'Pre-existing conditions have a 36-month waiting period.',
            decision: 'CONDITIONAL',
            confidence_score: 0.88,
            conditions: ['Covered only after 36 consecutive policy months'],
            citations: [
              {
                chunk_id: 'chunk_hp_009',
                page: 2,
                page_range: '2',
                clause: 'Section 4.2 Waiting Periods',
                category: 'condition',
                snippet: '36 months continuous coverage required.',
              },
            ],
            evaluator_status: 'PASSED',
            created_at: '2026-11-27T18:06:02Z',
          },
        },
      };
    });

    const onQaCompleted = vi.fn();

    render(
      <PolicyQAPanel
        policies={[ACTIVE_POLICY_1, ACTIVE_POLICY_2, DRAFT_POLICY]}
        policiesLoading={false}
        onQaCompleted={onQaCompleted}
      />
    );

    // Only active policies appear in selector
    const selector = screen.getByLabelText(/Indexed Policy/i);
    const optionTexts = Array.from(selector.querySelectorAll('option')).map((o) => o.textContent);
    expect(optionTexts).toHaveLength(2);
    expect(optionTexts.join(' ')).not.toContain('Unindexed Draft Plan');

    // History message + decision + confidence + citations render cleanly
    await waitFor(() => {
      expect(screen.getByText('COVERED')).toBeInTheDocument();
      expect(screen.getByText('Confidence: 92%')).toBeInTheDocument();
      expect(screen.getByText('PASSED')).toBeInTheDocument();
      expect(screen.getByText('Page 1')).toBeInTheDocument();
      expect(screen.getByText(/Clause: Section 2\.1 In-Patient Care/i)).toBeInTheDocument();
      expect(screen.getByText('chunk_hp_001')).toBeInTheDocument();
    });

    // Initial mount fetched history once (not twice due to conversationId update)
    expect(getHistorySpy).toHaveBeenCalledTimes(1);

    // Submit follow-up question that initially fails with 504 -> shows Retry button
    const input = screen.getByLabelText(/Your Question/i);
    fireEvent.change(input, { target: { value: 'Are cosmetic dental implants covered?' } });
    fireEvent.click(screen.getByRole('button', { name: /^Ask$/i }));

    const alertBox = await screen.findByRole('alert');
    expect(alertBox).toHaveTextContent(/Q&A request failed \(HTTP 504\)/i);
    expect(input).toHaveValue('Are cosmetic dental implants covered?');

    // Retry reuses existingConvId and renders INSUFFICIENT EVIDENCE without fake Confidence badge
    fireEvent.click(screen.getByRole('button', { name: /^Retry$/i }));

    await waitFor(() => {
      expect(screen.getByText('INSUFFICIENT EVIDENCE')).toBeInTheDocument();
      expect(
        screen.getByText(/No conclusive policy evidence was verified/i)
      ).toBeInTheDocument();
    });
    expect(askSpy).toHaveBeenNthCalledWith(1, ACTIVE_POLICY_1.id, {
      question: 'Are cosmetic dental implants covered?',
      conversationId: existingConvId,
    });
    expect(askSpy).toHaveBeenNthCalledWith(2, ACTIVE_POLICY_1.id, {
      question: 'Are cosmetic dental implants covered?',
      conversationId: existingConvId,
    });
    expect(onQaCompleted).toHaveBeenCalledTimes(1);

    // Clicking "New Thread" starts a fresh thread (conversationId omitted on next ask)
    fireEvent.click(screen.getByRole('button', { name: /New Thread/i }));
    fireEvent.change(screen.getByLabelText(/Your Question/i), {
      target: { value: 'What waiting period applies to pre-existing disease?' },
    });
    fireEvent.click(screen.getByRole('button', { name: /^Ask$/i }));

    await waitFor(() => {
      expect(screen.getByText('CONDITIONAL')).toBeInTheDocument();
      expect(screen.getByText('Page 2')).toBeInTheDocument();
    });
    expect(askSpy).toHaveBeenNthCalledWith(3, ACTIVE_POLICY_1.id, {
      question: 'What waiting period applies to pre-existing disease?',
      conversationId: '',
    });
    // Clicking New Thread did NOT unnecessarily re-trigger history load and overwrite empty conversationId
    expect(getHistorySpy).toHaveBeenCalledTimes(1);
  });
});
