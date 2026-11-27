<div align="center">

# CoverageIQ

**An evidence-grounded insurance policy intelligence system**

Recommend policies. Summarize them. Answer coverage questions, with the exact page and clause as proof.

`NO EVIDENCE = NO CLAIM`

</div>

---

## Table of Contents

- [Overview](#overview)
- [Core Principle](#core-principle)
- [Features](#features)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Structured Output](#structured-output)
- [User Roles](#user-roles)
- [API Overview](#api-overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Testing and Evaluation](#testing-and-evaluation)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Team](#team)

---

## Overview

Insurance policy documents run to dozens or hundreds of pages of dense wording, nested exclusions, waiting periods, and sub-limits. Aggregator portals filter on metadata only and never read the policy text. Generic LLM chatbots read it, but they hallucinate coverage, miss exclusions, and cite pages that don't support the claim.

CoverageIQ unifies three capabilities on one shared knowledge base, and routes every answer through an independent verifier before it reaches the user.

| Capability | What it does |
|---|---|
| **Recommendation (6A)** | Filters and ranks policies from an organization-managed catalog based on the user's profile and needs |
| **Summarization (6B)** | Produces a plain-language summary of a policy, with every statement cited |
| **Query Retrieval / Q&A (6C)** | Answers a coverage question with a decision and page/clause citations |

## Core Principle

> **NO EVIDENCE = NO CLAIM.**

Every statement the system outputs is tied to a retrieved chunk. An independent **Evaluator Agent** checks each claim against its cited evidence. If verification fails, the system retries (max 2). If evidence is still insufficient, it answers `INSUFFICIENT_EVIDENCE` instead of guessing.

## Features

- **Policy ingestion** for Car, Health, Life, and Term PDFs, with OCR for scanned pages
- **Structure-aware parsing**: sections, clauses, tables, and page numbers preserved
- **Semantic chunking** tagged as Coverage / Exclusions / Conditions / Definitions
- **Hybrid retrieval**: vector search + BM25, fused with Reciprocal Rank Fusion, then cross-encoder reranking
- **Hard filtering + ranking** for recommendations (transparent weighted scorer first, LightGBM/XGBoost later)
- **Grounded generation**: atomic JSON claims, each with chunk-ID citations
- **Independent Evaluator Agent**: evidence support, contradiction detection, citation accuracy, completeness, confidence score
- **Closed retry loop** with a safe `INSUFFICIENT_EVIDENCE` fallback
- **Role-based access control** (User / Organization / Admin) with per-user data isolation
- **Citation viewer**: expandable evidence and a PDF viewer that jumps to the cited page

## How It Works

```
Actors (User / Organization / Admin)
        │
Authentication + RBAC Gateway
        │
Ingestion Pipeline ──► Knowledge Base (PostgreSQL + pgvector)
                              │
        ┌─────────────────────┼─────────────────────┐
   6A Recommendation     6B Summarization      6C Query Retrieval
        └─────────────────────┼─────────────────────┘
                              │
              Grounded LLM (JSON atomic claims + chunk IDs)
                              │
                 Evaluator Agent (independent verifier)
                    │ FAIL                    │ PASS
              Retry Loop (max 2)       Structured Output
                                              │
                                          Frontend
```

**Ingestion:** PDF upload → text extraction (PyMuPDF) / OCR (PaddleOCR) → structuring → semantic chunking → BGE-M3 embeddings + `tsvector` → PostgreSQL + pgvector.

**Q&A (6C):** question → query analysis (intent + entities) → hybrid retrieval over one policy → cross-encoder rerank → Grounded LLM → Evaluator → structured verdict.

Retrieval for summarization and Q&A is scoped to a **single policy ID** to prevent cross-policy contamination.

## Architecture

Two backend services plus a frontend, sharing one PostgreSQL database under table-ownership rules.

```
Browser ──HTTPS──► Nginx ──► Node API ──internal HTTP──► Python AI Service
                              │                               │
                              └────────► PostgreSQL ◄─────────┘
                                         (pgvector)
                              Redis (Celery broker, cache)
```

| Service | Stack | Owns |
|---|---|---|
| **Frontend** | React + Vite, JavaScript, Tailwind CSS | UI, dashboards, PDF viewer, evidence panels |
| **Node API** (`server/`) | Node.js, Express | Auth + RBAC, CRUD, uploads, job records, chat history, orchestration. The only service the browser talks to |
| **Python AI Service** (`ai_service/`) | FastAPI, Celery | Extraction, OCR, chunking, embeddings, retrieval, reranking, 6A/6B/6C, Grounded LLM, Evaluator, retry loop |
| **Database** | PostgreSQL 16 + pgvector | Relational data, vectors, BM25 index |

**Rule of thumb:** if it needs a model, an embedding, or an LLM call, it lives in Python. If it is identity, permissions, CRUD, files, or HTTP orchestration, it lives in Node.

- The browser never calls the Python service. Nginx does not expose it.
- Node → Python calls carry an `X-Service-Token` and a request ID.
- Node verifies the user JWT and passes a trusted scope (`user_id`, `role`, `owner_scope`) to Python.
- Ingestion is asynchronous: Node creates a job row, Python queues a Celery task, the frontend polls `GET /jobs/:id`.

Full details: [`doc/`](./doc) and [`diagrams/`](./diagrams).

## Tech Stack

| Area | Technology |
|---|---|
| Frontend | React, Vite, Tailwind CSS, React Router, TanStack Query, react-pdf |
| Public API | Node.js (LTS), Express, Zod, JWT, bcrypt/argon2, multer |
| AI service | Python 3.11, FastAPI, Pydantic, Celery |
| Document processing | PyMuPDF, PaddleOCR |
| Embeddings / reranking | BGE-M3, bge-reranker |
| Ranking | Weighted scorer, then LightGBM / XGBoost |
| Database | PostgreSQL 16, pgvector (HNSW), `tsvector` (BM25) |
| Queue / cache | Redis |
| LLM | API-hosted (provider-agnostic client) |
| Deployment | Docker Compose, Nginx, cloud VM |
| Quality | ESLint + Vitest/Jest (JS), Ruff + pytest (Python), GitHub Actions |

## Structured Output

Every answer has the same shape:

```json
{
  "decision": "COVERED | NOT_COVERED | CONDITIONAL | INSUFFICIENT_EVIDENCE",
  "confidence": 0.0,
  "conditions": ["..."],
  "reason": "...",
  "citations": [{ "chunk_id": "c_101", "page": 12, "clause": "4.2" }]
}
```

Recommendation responses add a ranked list of **Recommended Policies (verified)**, each with citations.

| Decision | UI badge |
|---|---|
| Covered | Green |
| Not Covered | Red |
| Conditional | Amber |
| Insufficient Evidence | Grey |

## User Roles

| Role | Permissions |
|---|---|
| **User** | Upload private policy PDFs, get recommendations, summaries, and Q&A |
| **Organization** | Create, read, update, delete policies in the shared recommendation catalog |
| **Admin** | All user permissions, plus approve/suspend organizations and manage roles |

User-uploaded PDFs stay private to that user. Organization policies feed the shared catalog.

## API Overview

**Public (Node), selected endpoints**

| Endpoint | Purpose |
|---|---|
| `POST /auth/register`, `/auth/login`, `/auth/refresh` | Authentication |
| `POST /policies/upload` | Upload a PDF, returns `job_id` |
| `GET /jobs/:id` | Ingestion job status |
| `POST /policies/:id/ask` | Q&A (6C) |
| `POST /policies/:id/summary` | Summary (6B) |
| `POST /recommendations` | Recommendation (6A) |
| `POST/GET/PUT/DELETE /org/policies` | Catalog management |

**Internal (Python AI service)**

| Endpoint | Purpose | Mode |
|---|---|---|
| `POST /ai/ingest` | Start ingestion | async, `202` |
| `POST /ai/ask` | Q&A | sync |
| `POST /ai/summarize` | Summary | sync (cached) |
| `POST /ai/recommend` | Recommendation | sync |
| `GET /ai/health` | Liveness + model readiness | sync |

The OpenAPI contract is the single source of truth for both services.

## Repository Structure

```
.
├── server/                    # Node.js + Express public API
├── ai_service/                # Python FastAPI + Celery AI service
├── frontend/                  # React + Vite app (planned layout, add when pushed)
├── diagrams/                  # System architecture diagrams
├── doc/                       # Architecture, modules, requirements, roadmap
├── LOCAL_EXECUTION_GUIDE.md   # Step-by-step local setup
├── team-instructions.md       # Team workflow and conventions
└── README.md
```

## Getting Started

### Prerequisites

- Node.js (LTS)
- Python 3.11
- Docker and Docker Compose
- PostgreSQL 16 with pgvector, and Redis (or run both via Docker)
- An API key for the chosen LLM provider

### Setup

```bash
git clone https://github.com/samarjeets10/coverageIQ-LLM-based-insurance-policy-assistant.git
cd coverageIQ-LLM-based-insurance-policy-assistant
```

Then follow **[LOCAL_EXECUTION_GUIDE.md](./LOCAL_EXECUTION_GUIDE.md)** to configure environment variables and start each service.

## Testing and Evaluation

Quality is measured against a **gold set**: 10+ real policies (Car, Health, Life, Term) with 100 hand-written Q&A pairs, plus an adversarial set (unanswerable, contradictory, misleading questions).

| Metric | Target |
|---|---|
| Citation accuracy | ≥ 90% |
| Hallucinated claims reaching the user | ≤ 5% |
| Q&A decision accuracy | ≥ 80% |
| Q&A latency (p95, no retry) | ≤ 15 s |
| Ingestion, 50-page digital PDF | ≤ 2 min |

Gold-set metrics are re-run before merging any change to retrieval, prompts, or the evaluator.

## Roadmap

Strategy: **thin vertical slice first.** One policy → ingest → ask → cited answer, then widen.

| Phase | Goal |
|---|---|
| 0 | Foundations: repo, contracts, mock AI service, gold-set plan |
| 1 | Ingestion + Knowledge Base + gold set |
| 2 | Q&A vertical slice (6C + Grounded LLM) |
| 3 | Evaluator, retry loop, structured output |
| 4 | Summarization (6B) + catalog management |
| 5 | Recommendation (6A) |
| 6 | Frontend completion, PDF viewer, dashboards |
| 7 | Deployment, hardening, evaluation report |
| 8 | Buffer, documentation, demo |

**Out of scope (v1):** claims processing, payments, live insurer integrations, mobile app, multilingual support.
**Future scope:** multilingual and voice interface, side-by-side policy comparison, marketplace/broker API integration.

## Contributing

- Branches: `main` (stable), `develop` (integration), `feature/<module>` for work
- Every PR references a requirement ID (e.g., `FR-QA-03`) and needs one peer review
- Contract changes (OpenAPI) need one reviewer from the Node side and one from the Python side
- Interface or schema changes update the docs in the same PR

See [`team-instructions.md`](./team-instructions.md) for the full workflow.

## Team

**Project Group B2**, Department of Computer Science and Engineering
SWVSM's Tatyasaheb Kore Institute of Engineering and Technology, Warananagar (An Autonomous Institute)
Academic Year 2026-27

| Name | Focus |
|---|---|
| Prithviraj Shamrao Patil | PDF ingestion and OCR, document structuring, chunking,  |
| Shravan Sambhaji Kapare | Recommendation module (hard filtering, ranking), pgvector schema |
| Yash Shankar Nimbalkar | Hybrid retrieval, reranker, Grounded LLM, Evaluator Agent, retry loop |
| Samarjeet Sambhaji Sabale | Frontend, citation viewer, Node/FastAPI integration, Docker deployment, QA |

**Project Guide:** Prof. Mrs. B. A. Chougule

## License

Not yet specified. Add a `LICENSE` file before allowing reuse.
