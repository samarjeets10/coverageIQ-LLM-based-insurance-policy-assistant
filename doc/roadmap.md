# CoverageIQ — Roadmap

> **Assumption:** 16 weeks starting the week of **Oct 12, 2026**, ending **~Feb 1, 2027**. This is a placeholder. Replace it with the real submission and review dates, then shift phases accordingly.
> Strategy: **thin vertical slice first**. Get one policy → ingest → ask → cited answer working end to end, then widen.

---

## Overview

| Phase | Weeks | Goal | Exit gate |
|---|---|---|---|
| 0 | 1 | Foundations and decisions | Repo, stack decision, gold-set plan |
| 1 | 2–4 | Ingestion + Knowledge Base + gold set | 10 policies parse correctly |
| 2 | 5–7 | Q&A vertical slice (6C + Grounded LLM) | Cited answers on gold questions |
| 3 | 8–9 | Evaluator + Retry Loop + Structured Output | Hallucination ≤ 5% on adversarial set |
| 4 | 10–11 | Summarization (6B) + Catalog (M4) | Cited summaries; org CRUD works |
| 5 | 12–13 | Recommendation (6A) | Top-N with explanations |
| 6 | 14 | Frontend polish, PDF viewer, dashboards | Full demo flow clickable |
| 7 | 15 | Deployment, hardening, evaluation report | One-command deploy; metrics table |
| 8 | 16 | Buffer, docs, viva/demo prep | Final submission |

The frontend and Auth run **in parallel** from Phase 1 (see below) so UI work is never blocked on ML.

---

## Phase 0 — Foundations (Week 1)

- [ ] Confirm deadlines and review dates with guide; update this roadmap
- [x] Stack locked: React + Vite (JS) + Tailwind; Node.js + Express REST; Python (FastAPI + Celery) AI service; PostgreSQL + pgvector
- [ ] **Freeze Node↔Python contract v0** (OpenAPI for `/ai/ingest`, `/ai/ask`, `/ai/summarize`, `/ai/recommend`, and the structured output shape)
- [ ] **Build a mock AI service** returning fixed valid JSON so Node and frontend are never blocked
- [ ] Decide DB access on Node (Prisma vs `pg` + query builder) and the single migration tool
- [ ] Set up lint/test per service: ESLint + Vitest/Jest (Node, React), Ruff + pytest (Python)
- [ ] Create GitHub repo, branching rules (`main`, `dev`, feature branches, PR review required)
- [ ] Folder structure: `backend-node/ (src: auth, policies, jobs, chat, ai-client, common)`, `ai-service/ (ingestion, retrieval, recommend, summarize, qa, llm, evaluator, common)`, `frontend/` (Vite), `docs/contracts/`, `infra/`, `eval/`
- [ ] docker-compose skeleton: postgres+pgvector, redis, node-api, ai-service (mock), ai-worker, nginx
- [ ] Assign module owners (see ModulesandFeatures.md)
- [ ] Pick LLM provider, get API keys, set budget cap
- [ ] **Collect 10+ real policy PDFs** (mix of types, include 2 scanned)
- [ ] Write remaining docs: `projectflow.md`, `databaseDesign.md`, `schemaDesign.md`, `apiContract.md`, `decisionLog.md`

**Exit:** everyone can run the skeleton locally.

---

## Phase 1 — Ingestion + Knowledge Base (Weeks 2–4)

| Track | Tasks |
|---|---|
| Ingestion (M2) | **Node:** upload endpoint (multer), file storage, job rows, `GET /jobs/:id`. **Python:** PyMuPDF extraction; page-number preservation; section/clause/table detection; semantic chunker with 4 labels; Celery job updating job status |
| Knowledge Base (M3) | DB schema + migrations (Alembic); pgvector HNSW; `tsvector` GIN; `hybrid_search`, RRF, `get_chunk` |
| Auth (M1) — parallel (Node) | Register/login/JWT, `requireRole` middleware, forbidden-access tests, trusted-scope + service-token plumbing to the AI service |
| Frontend — parallel | Vite + React + Tailwind scaffold, login, upload page with job status |
| Eval (M12) | Write **100 Q&A gold pairs** across the 10 policies (everyone helps) |

**Exit gate:** ≥ 95% of sampled chunks cite the correct page; hybrid retrieval recall@5 measured and recorded.

**Risk watch:** table extraction and scanned PDFs. If parsing is bad, stop and fix here. Do not build on it.

---

## Phase 2 — Q&A Vertical Slice (Weeks 5–7)

- [ ] Query analysis (intent + entities), simple first version
- [ ] Hybrid retrieval + cross-encoder rerank (bge-reranker)
- [ ] Grounded generator: atomic claims JSON with chunk IDs; prompt enforcing NO EVIDENCE = NO CLAIM
- [ ] `/policies/{id}/ask` endpoint (no evaluator yet)
- [ ] Frontend Q&A chat with evidence panel (mock → real API)
- [ ] OCR (PaddleOCR) for scanned pages — **P1, timebox to 3 days**
- [ ] Run gold set; record baseline metrics in `eval/results/`

**Exit gate:** end-to-end demo: upload → ask → answer with citations. Decision accuracy baseline recorded (target ≥ 70% at this stage).

---

## Phase 3 — Evaluator, Retry Loop, Structured Output (Weeks 8–9)

- [ ] Evaluator agent: evidence support, contradiction, citation accuracy, completeness, confidence
- [ ] Retry controller (max 2; widen k → rewrite query → re-rank → re-answer with feedback)
- [ ] INSUFFICIENT_EVIDENCE path
- [ ] Structured output layer with decision precedence rules and OpenAPI schema freeze
- [ ] Adversarial test set (unanswerable, contradictory, misleading questions)
- [ ] Audit log tables (claims, verdicts, retries, cost)
- [ ] Frontend: decision badges, confidence, evidence expansion

**Exit gate:** hallucination ≤ 5%, citation accuracy ≥ 90%, p95 latency ≤ 15 s on gold set. If cost or latency fails, tune the evaluator model and caching now.

---

## Phase 4 — Summarization + Catalog (Weeks 10–11)

- [ ] 6B: section-wise targeted retrieval, grounded summary, "Not found in document" handling, caching per version
- [ ] M4: organization catalog CRUD, versioning, soft delete, ownership enforcement
- [ ] Attribute extraction (premium, sum insured, waiting period, co-pay, room rent, etc.) into structured columns — **needed by 6A, start early**
- [ ] Organization dashboard
- [ ] Admin: organization approval flow

**Exit gate:** all gold policies have summaries with zero uncited statements; org can only touch its own policies.

---

## Phase 5 — Recommendation (Weeks 12–13)

- [ ] Profile intake form and schema
- [ ] Hard-filter rule engine with elimination reasons (unit test every rule)
- [ ] Ranking v1: transparent weighted score
- [ ] Ranking v2 (**stretch**): LightGBM/XGBoost trained on synthetic + team-labeled data, compared against v1 on held-out profiles
- [ ] Explanation generation through M8 (grounded + evaluated)
- [ ] "Select policy" hand-off to Summary / Q&A
- [ ] Recommendations UI

**Exit gate:** 10 test profiles produce reasonable Top-3 lists with verified explanations.

> If ML ranking has no usable labels by Week 12, ship v1 and document honestly. A transparent scorer you can defend beats an untrained model you cannot.

---

## Phase 6 — Frontend Completion (Week 14)

- [ ] PDF viewer jumping to cited page with clause highlight (FR-UI-07)
- [ ] Role-based dashboards finalized (User / Organization / Admin)
- [ ] Error, empty and loading states everywhere
- [ ] Responsive layout check
- [ ] Usability test with 3 outside users; fix top issues

---

## Phase 7 — Deployment, Hardening, Evaluation (Week 15)

- [ ] Nginx, TLS, env secrets, Docker images for each service, cloud VM deploy; confirm the Python service is not publicly exposed
- [ ] CI: lint + tests for Node, React and Python on every PR, plus contract tests (Node↔Python)
- [ ] Security pass: RBAC test matrix, upload validation, rate limiting, no PII in logs
- [ ] Final evaluation run; write results table (retrieval recall, decision accuracy, citation accuracy, hallucination rate, latency, cost/query)
- [ ] Ablations for the report: vector-only vs hybrid vs hybrid+rerank; with vs without evaluator

---

## Phase 8 — Buffer and Submission (Week 16)

- [ ] Final report, architecture diagram, user manual
- [ ] Demo script with a rehearsed happy path and one "Insufficient Evidence" example
- [ ] Backup demo video in case of live failure
- [ ] Viva Q&A rehearsal: why hybrid retrieval, why an independent evaluator, why RRF, what the failure modes are

---

## Milestones

| # | Milestone | Target week |
|---|---|---|
| M-A | Repo + skeleton running for everyone | 1 |
| M-B | Gold set (10 policies, 100 Q&A) complete | 4 |
| M-C | First cited answer end to end | 6 |
| M-D | Evaluator + retry loop live | 9 |
| M-E | Summaries + catalog CRUD | 11 |
| M-F | Recommendations working | 13 |
| M-G | Deployed demo | 15 |
| M-H | Submission | 16 |

---

## Scope Control

**Cut order if time runs short (cut from the top):**
1. LightGBM/XGBoost model (keep weighted scorer)
2. Policy comparison and follow-up resolution
3. PDF clause highlighting (keep page jump)
4. Admin dashboard extras
5. OCR (keep digital PDFs only)

**Never cut:** grounding rule, evaluator, citations, RBAC, the gold-set evaluation, the frozen Node↔Python contract.

---

## Working Agreements

- Weekly 30-minute sync (same slot every week); each person reports done / next / blocked.
- Every PR references a requirement ID (e.g., `FR-QA-03`) and is reviewed by one other member.
- No module changes an interface (API schema, DB schema) without updating the docs in the same PR.
- Decisions go into `decisionLog.md` with date and reason.
- Contract changes (`/docs/contracts/`) need one reviewer from the Node side and one from the Python side.
- Gold-set metrics are re-run before merging any change to retrieval, prompts or the evaluator.

---

## Weekly Tracker (copy each week)

```
Week N — dates
Goal:
Done:
Blocked:
Metrics this week:
Next week:
```
