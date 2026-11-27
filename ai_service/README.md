# CoverageIQ AI Service (`ai_service`)

Unified Python FastAPI service hosting:
1. **Policy Q&A & Verification Engine (M3 / M7 / M8 — 6C Pipeline):** Hybrid retrieval (Dense + Sparse BM25), RRF fusion, cross-encoder reranking, grounded LLM generation (`NO EVIDENCE = NO CLAIM`), and independent 5-point evaluator retry loop (`POST /ai/ask`).
2. **Policy Recommendation Engine (M5 — 6A Pipeline):** PostgreSQL catalog retrieval, Stage 4 deterministic hard eligibility filtering, Stage 5 leakage-free 59-feature engineering, and LightGBM LambdaRank (`"auto"` / `"lightgbm"`) or explicit Weighted (`"weighted"`) scoring (`POST /api/v1/recommend`, `POST /ai/recommend`).

---

## 1. Policy Q&A & Verification Engine (M3 / M7 / M8)

- **Query Analyzer:** [`ai_service/retrieval/query_analyzer.py`](retrieval/query_analyzer.py)
- **Dense Vector & BM25 Retrieval + RRF Fusion + Cross-Encoder Reranker:** [`ai_service/retrieval/`](retrieval/)
- **Grounded LLM Generator & Independent Evaluator Agent:** [`ai_service/llm/`](llm/), [`ai_service/evaluator/`](evaluator/)
- **Q&A Endpoint:** `POST /ai/ask` ([`ai_service/api/routes_qa.py`](api/routes_qa.py))
- **Upstream Detailed Guide:** [`LOCAL_EXECUTION_GUIDE.md`](LOCAL_EXECUTION_GUIDE.md)

---

## 2. Policy Recommendation Engine (Module M5 / 6A)

- **Stages 2–7 (Implemented):**
  - Synthetic dataset generator ([`recommendation/dataset.py`](recommendation/dataset.py): 80 policies, 60 profiles, 4,800 relevance rows).
  - Deterministic hard filtering engine ([`recommendation/filters.py`](recommendation/filters.py)).
  - Leakage-free 59-feature engineering pipeline ([`recommendation/features.py`](recommendation/features.py), schema `1.0.0`).
  - LightGBM LambdaRank training & schema-validated model persistence ([`recommendation/train_lightgbm.py`](recommendation/train_lightgbm.py)) with tracked baseline artifacts under `artifacts/models/`.
  - Recommendation inference service & router ([`recommendation/service.py`](recommendation/service.py), [`recommendation/router.py`](recommendation/router.py)).
- **Stages 8A / 8B / 9A / Phase 9B-1 (Implemented):**
  - Architecture-aligned DB integration ([`common/tables.py`](common/tables.py), [`recommendation/db.py`](recommendation/db.py)), candidate migration [`../server/migrations/003_recommendation_schema.sql`](../server/migrations/003_recommendation_schema.sql) (unapplied, awaiting Node/DB review), and preserved SQL drafts in [`sql/`](sql/).
  - Configurable weighted fallback backend ([`recommendation/weighted.py`](recommendation/weighted.py), explicit `ranking_model="weighted"` opt-in only), corrupt-artifact & warm-up checks, incomplete-profile `profile_completeness` reporting, structured JSON logging with PII/credential scrubbing ([`common/logging.py`](common/logging.py)), and interface-only citation seam ([`recommendation/citations.py`](recommendation/citations.py)).
  - Non-development fail-closed `X-Service-Token` authentication (`AI_SERVICE_TOKEN` in [`recommendation/settings.py`](recommendation/settings.py)), payload size limit (`RequestBodyLimitMiddleware`), and frozen OpenAPI contract [`../docs/contracts/ai-service.openapi.yaml`](../docs/contracts/ai-service.openapi.yaml) via [`recommendation/openapi_export.py`](recommendation/openapi_export.py).
- **Endpoints Exposed:**
  - `GET /ai/health` — Unified health endpoint returning M3 retrieval/evaluator telemetry and M5 DB/model readiness.
  - `GET /health` — Strict M5 PostgreSQL connection, schema readiness, and LightGBM warm-up status (`HTTP 503` if PostgreSQL is unreachable).
  - `POST /ai/ask` — M3/M7/M8 evidence-grounded policy Q&A and verification loop.
  - `POST /api/v1/recommend` and `POST /ai/recommend` — M5 recommendation endpoints (`HTTP 503` against live DB until `server/migrations/003_recommendation_schema.sql` is reviewed and applied).

---

## 3. Local Setup & Verification (From Repository Root)

```powershell
# Install unified AI Service dependencies
pip install -r ai_service/requirements.txt

# Start unified FastAPI server from repository root
python -m uvicorn ai_service.main:app --host 127.0.0.1 --port 8000 --reload

# Export / verify frozen OpenAPI contract
python -m ai_service.recommendation.openapi_export

# Run Ruff checks (from ai_service/)
cd ai_service
python -m ruff format --check .
python -m ruff check .
cd ..

# Run complete unified test suite (M3/M7/M8 + M5)
python -m pytest ai_service/tests/ -v
```
