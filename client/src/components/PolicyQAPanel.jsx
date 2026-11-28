import { useEffect, useRef, useState } from 'react';
import {
  AlertCircle,
  BookOpenCheck,
  CheckCircle2,
  Loader2,
  MessageSquareText,
  RefreshCw,
  RotateCcw,
  Send,
  ShieldAlert,
  ShieldCheck,
  Upload,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { qaApi } from '@/lib/api';

function getDecisionBadgeVariant(decision) {
  switch (decision) {
    case 'COVERED':
      return 'success';
    case 'NOT COVERED':
      return 'danger';
    case 'CONDITIONAL':
      return 'warning';
    case 'INSUFFICIENT EVIDENCE':
    default:
      return 'default';
  }
}

function formatMessageTime(ts) {
  if (!ts) return '';
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleString();
}

export function PolicyQAPanel({
  policies = [],
  policiesLoading = false,
  onGoUploadPolicies,
  onQaCompleted,
}) {
  const activePolicies = policies.filter((p) => p.status === 'active');

  const [selectedPolicyId, setSelectedPolicyId] = useState('');
  const [question, setQuestion] = useState('');
  const [conversationId, setConversationId] = useState('');

  const [messages, setMessages] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState('');

  const [submitting, setSubmitting] = useState(false);
  const [askError, setAskError] = useState(null);
  const [lastFailedQuestion, setLastFailedQuestion] = useState('');
  const [reloadNonce, setReloadNonce] = useState(0);
  const autoResumeThreadRef = useRef(true);

  const effectivePolicyId =
    selectedPolicyId && activePolicies.some((p) => p.id === selectedPolicyId)
      ? selectedPolicyId
      : activePolicies[0]?.id || '';

  useEffect(() => {
    let active = true;
    if (!effectivePolicyId) {
      Promise.resolve().then(() => {
        if (!active) return;
        setMessages([]);
        setHistoryError('');
        setHistoryLoading(false);
      });
      return () => {
        active = false;
      };
    }

    Promise.resolve().then(() => {
      if (!active) return;
      setHistoryLoading(true);
      setHistoryError('');
    });

    qaApi
      .getHistory(effectivePolicyId, { page: 1, limit: 50 })
      .then((res) => {
        if (!active) return;
        const items = res.data?.items || [];
        setMessages(items);
        if (autoResumeThreadRef.current && items.length > 0) {
          const lastItem = items[items.length - 1];
          if (lastItem?.conversation_id) {
            setConversationId((prev) => prev || lastItem.conversation_id);
          }
        }
      })
      .catch((err) => {
        if (!active) return;
        setHistoryError(err.message || 'Could not load conversation history for this policy.');
      })
      .finally(() => {
        if (active) setHistoryLoading(false);
      });

    return () => {
      active = false;
    };
  }, [effectivePolicyId, reloadNonce]);

  async function submitQuestion(rawText) {
    const clean = rawText.trim();
    if (!effectivePolicyId) return;
    if (clean.length < 3) {
      setAskError({
        title: 'Question too short',
        message: 'Enter at least 3 characters describing what you want to check in the policy.',
        retryable: false,
      });
      return;
    }

    setAskError(null);
    setLastFailedQuestion('');
    setSubmitting(true);

    try {
      const res = await qaApi.ask(effectivePolicyId, {
        question: clean,
        conversationId,
      });
      const data = res.data || {};
      if (data.conversation_id) {
        setConversationId(data.conversation_id);
      }

      const appended = [];
      if (data.user_message) appended.push(data.user_message);
      if (data.assistant_message) appended.push(data.assistant_message);
      if (appended.length > 0) {
        setMessages((prev) => [...prev, ...appended]);
      } else {
        setReloadNonce((n) => n + 1);
      }

      setQuestion('');
      if (typeof onQaCompleted === 'function') {
        onQaCompleted(data);
      }
    } catch (err) {
      const statusCode = Number(err.statusCode || 0);
      const retryable =
        statusCode === 0 || statusCode === 429 || statusCode === 502 || statusCode === 503 || statusCode === 504;
      setLastFailedQuestion(clean);
      setAskError({
        title: `Q&A request failed${statusCode ? ` (HTTP ${statusCode})` : ''}`,
        message: err.message || 'Unable to verify answer with the AI service.',
        retryable,
      });
    } finally {
      setSubmitting(false);
    }
  }

  function handleFormSubmit(event) {
    event.preventDefault();
    submitQuestion(question);
  }

  function handleRetryLast() {
    if (!lastFailedQuestion || submitting) return;
    submitQuestion(lastFailedQuestion);
  }

  function handlePolicySelectionChange(nextPolicyId) {
    autoResumeThreadRef.current = true;
    setSelectedPolicyId(nextPolicyId);
    setConversationId('');
    setAskError(null);
    setLastFailedQuestion('');
  }

  function handleNewConversation() {
    autoResumeThreadRef.current = false;
    setConversationId('');
    setAskError(null);
    setLastFailedQuestion('');
  }

  return (
    <div className="space-y-6">
      <Card className="p-6">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="text-base font-semibold text-slate-900">
              Ask a Coverage Question
            </h3>
            <p className="mt-1 text-sm text-slate-500">
              Answers are retrieved strictly from your indexed policy PDF chunks, audited by an independent verifier, and returned with exact page and clause citations.
            </p>
          </div>
          {effectivePolicyId && (
            <div className="flex items-center gap-2">
              <Button
                type="button"
                variant="secondary"
                size="sm"
                onClick={handleNewConversation}
                disabled={submitting}
              >
                <RotateCcw className="h-3.5 w-3.5" />
                New Thread
              </Button>
              <Button
                type="button"
                variant="secondary"
                size="sm"
                onClick={() => setReloadNonce((n) => n + 1)}
                disabled={historyLoading || submitting}
              >
                <RefreshCw className={`h-3.5 w-3.5 ${historyLoading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
            </div>
          )}
        </div>

        {policiesLoading ? (
          <div className="mt-5 flex items-center gap-2 text-sm text-slate-500">
            <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
            <span>Loading authorized policies...</span>
          </div>
        ) : activePolicies.length === 0 ? (
          <div className="mt-5 rounded-xl border border-dashed border-slate-300 bg-[#FAFAFA] p-6 text-center">
            <BookOpenCheck className="mx-auto h-8 w-8 text-slate-400" />
            <p className="mt-2 text-sm font-medium text-slate-800">
              No ingested active policies available for Q&amp;A yet
            </p>
            <p className="mt-1 text-xs text-slate-500">
              {policies.length > 0
                ? 'Your existing policy is still in draft or processing. Complete a PDF upload in My Policies first.'
                : 'Create a policy and upload an insurance policy PDF in My Policies to enable grounded Q&A.'}
            </p>
            {typeof onGoUploadPolicies === 'function' && (
              <Button size="sm" className="mt-4" onClick={onGoUploadPolicies}>
                <Upload className="h-4 w-4" />
                Go to My Policies
              </Button>
            )}
          </div>
        ) : (
          <form onSubmit={handleFormSubmit} className="mt-5 space-y-4">
            <div className="space-y-1.5">
              <label
                htmlFor="qa-policy-selector"
                className="text-sm font-medium text-slate-700"
              >
                Indexed Policy
              </label>
              <select
                id="qa-policy-selector"
                aria-label="Indexed Policy"
                value={effectivePolicyId}
                onChange={(e) => handlePolicySelectionChange(e.target.value)}
                disabled={submitting}
                className="flex h-11 w-full rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-sm text-slate-900 focus:border-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-600/15 disabled:opacity-50"
              >
                {activePolicies.map((pol) => (
                  <option key={pol.id} value={pol.id}>
                    {pol.title} ({pol.policy_type} · {pol.external_policy_key})
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-1.5">
              <label
                htmlFor="qa-question-input"
                className="text-sm font-medium text-slate-700"
              >
                Your Question
              </label>
              <div className="flex flex-col gap-2.5 sm:flex-row">
                <input
                  id="qa-question-input"
                  type="text"
                  minLength={3}
                  maxLength={2000}
                  required
                  disabled={submitting}
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="e.g. Is in-patient hospitalization covered, or what exclusions apply?"
                  className="flex h-11 flex-1 rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:border-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-600/15 disabled:opacity-50"
                />
                <Button type="submit" disabled={submitting || !question.trim()}>
                  {submitting ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Verifying...
                    </>
                  ) : (
                    <>
                      <Send className="h-4 w-4" />
                      Ask
                    </>
                  )}
                </Button>
              </div>
            </div>

            {submitting && (
              <div
                role="status"
                aria-live="polite"
                className="flex items-center gap-2.5 rounded-xl border border-blue-200 bg-blue-50 px-3.5 py-2.5 text-sm text-blue-900"
              >
                <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
                <span>
                  Searching vector &amp; BM25 clauses, verifying evidence, and evaluating decision gate...
                </span>
              </div>
            )}

            {askError && (
              <div
                role="alert"
                className="flex flex-wrap items-start justify-between gap-3 rounded-xl border border-red-200 bg-red-50 px-3.5 py-3 text-sm text-red-800"
              >
                <div className="flex items-start gap-2.5">
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-600" />
                  <div>
                    <p className="font-semibold">{askError.title}</p>
                    <p>{askError.message}</p>
                  </div>
                </div>
                {askError.retryable && lastFailedQuestion && (
                  <Button
                    type="button"
                    size="sm"
                    variant="secondary"
                    onClick={handleRetryLast}
                    disabled={submitting}
                  >
                    <RefreshCw className="h-3.5 w-3.5" />
                    Retry
                  </Button>
                )}
              </div>
            )}
          </form>
        )}
      </Card>

      {effectivePolicyId && (
        <Card className="p-6">
          <div className="flex items-center justify-between gap-2">
            <h3 className="text-base font-semibold text-slate-900">
              Conversation History &amp; Verified Evidence
            </h3>
            <span className="text-xs text-slate-500">
              {messages.filter((m) => m.role === 'user').length} question(s)
            </span>
          </div>

          {historyError && (
            <div
              role="alert"
              className="mt-4 rounded-xl border border-red-200 bg-red-50 px-3.5 py-2.5 text-sm text-red-700"
            >
              {historyError}
            </div>
          )}

          {historyLoading ? (
            <div className="mt-6 flex items-center justify-center gap-2 py-8 text-sm text-slate-500">
              <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
              <span>Loading conversation history...</span>
            </div>
          ) : messages.length === 0 ? (
            <div className="mt-4 rounded-xl border border-slate-200/80 p-8 text-center text-sm text-slate-400">
              <MessageSquareText className="mx-auto h-8 w-8 text-slate-300" />
              <p className="mt-2">
                No questions asked on this policy yet. Submit a question above to verify coverage.
              </p>
            </div>
          ) : (
            <div className="mt-4 space-y-4">
              {messages.map((msg) =>
                msg.role === 'user' ? (
                  <div
                    key={msg.id}
                    className="rounded-xl border border-slate-200 bg-[#FAFAFA] px-4 py-3"
                  >
                    <div className="flex items-center justify-between gap-2 text-xs text-slate-500">
                      <span className="font-semibold uppercase tracking-wide text-slate-700">
                        Your Question
                      </span>
                      <span>{formatMessageTime(msg.created_at)}</span>
                    </div>
                    <p className="mt-1.5 text-sm font-medium text-slate-900">
                      {msg.content}
                    </p>
                  </div>
                ) : (
                  <div
                    key={msg.id}
                    className="rounded-xl border border-slate-200 bg-white p-4 shadow-xs space-y-3"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge variant={getDecisionBadgeVariant(msg.decision)}>
                          {msg.decision}
                        </Badge>
                        {typeof msg.confidence_score === 'number' && (
                          <span className="rounded-md border border-slate-200 bg-slate-50 px-2 py-0.5 text-xs font-medium text-slate-700">
                            Confidence: {Math.round(msg.confidence_score * 100)}%
                          </span>
                        )}
                        {msg.evaluator_status && (
                          <span className="inline-flex items-center gap-1 text-xs font-medium text-slate-600">
                            <ShieldCheck className="h-3.5 w-3.5 text-blue-600" />
                            {msg.evaluator_status}
                          </span>
                        )}
                      </div>
                      <span className="text-xs text-slate-400">
                        {formatMessageTime(msg.created_at)}
                      </span>
                    </div>

                    {msg.decision === 'INSUFFICIENT EVIDENCE' && (
                      <div className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50/70 px-3 py-2 text-xs text-amber-900">
                        <ShieldAlert className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-600" />
                        <span>
                          No conclusive policy evidence was verified for this claim (NO EVIDENCE = NO CLAIM).
                        </span>
                      </div>
                    )}

                    <p className="text-sm leading-relaxed text-slate-800">
                      {msg.content}
                    </p>

                    {Array.isArray(msg.conditions) && msg.conditions.length > 0 && (
                      <div className="rounded-lg border border-slate-200/80 bg-[#FAFAFA] p-3">
                        <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">
                          Conditions &amp; Limits
                        </p>
                        <ul className="mt-1.5 list-disc space-y-1 pl-4 text-xs text-slate-700">
                          {msg.conditions.map((cond, idx) => (
                            <li key={`${msg.id}-cond-${idx}`}>{cond}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {Array.isArray(msg.citations) && msg.citations.length > 0 && (
                      <div className="space-y-2">
                        <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">
                          Verified Policy Citations ({msg.citations.length})
                        </p>
                        <div className="space-y-2">
                          {msg.citations.map((cit, idx) => (
                            <div
                              key={`${msg.id}-cit-${cit.chunk_id}-${idx}`}
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
                      </div>
                    )}
                  </div>
                )
              )}
            </div>
          )}
        </Card>
      )}
    </div>
  );
}
