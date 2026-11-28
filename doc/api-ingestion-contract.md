# CoverageIQ Checkpoint 4.5 — Policy PDF Upload & Ingestion Job API Contract

Public endpoints are served by the Node.js + Express gateway (`SERVER_PORT=8000`) under both `/api/...` and `/api/v1/...` prefixes (matching Vite frontend `VITE_API_BASE_URL` `/api/v1` with cookie credentials `credentials: 'include'` or `Authorization: Bearer <accessToken>`).

> **Note for Frontend `multipart/form-data` Uploads:** When uploading a PDF with `FormData`, omit the JSON `'Content-Type': 'application/json'` header so the browser sets `multipart/form-data; boundary=...` automatically.

---

## 0. Policy Selector & Creation Endpoints

- `GET /api/policies` / `GET /api/v1/policies` *(Auth + `policy:read`, optional `?writable=true&status=...`)*: Returns caller's authorized non-archived policies (`can_upload` flag derived from ownership/org-admin role).
- `POST /api/policies` / `POST /api/v1/policies` *(Auth + `policy:create`)*: Creates a `'draft'` policy in `user:<id>` or authorized `org:<organization_id>`.

---

## 1. Upload Policy PDF Document

- **Endpoints:**
  - `POST /api/policies/:policyId/documents`
  - `POST /api/v1/policies/:policyId/documents`
- **Auth / RBAC:** Authenticated user (`verifyJWT`), permission `policy:update`, plus policy ownership (`owner_scope = user:<userId>` or org `owner`/`admin` membership or `admin` platform role).
- **Rate Limit:** Configurable (`UPLOAD_RATE_LIMIT_MAX` per `UPLOAD_RATE_LIMIT_WINDOW_MS`); returns `429` with `Retry-After`.
- **Request (`multipart/form-data`):**
  - `file` *(required)*: PDF binary (`extension=.pdf`, `Content-Type=application/pdf`, `<= MAX_PDF_SIZE_BYTES`, default 25 MiB, non-empty, `%PDF-` signature, unencrypted, valid PDF structure).
  - `version_label` *(optional string)*: Human-readable label (e.g., `"2026-endorsement"`).

### 1.1 Success Response — New Version Created & Dispatched (`201 Created`)

```json
{
  "statusCode": 201,
  "success": true,
  "message": "Policy document uploaded and ingestion job queued successfully",
  "data": {
    "httpStatus": 201,
    "duplicate": false,
    "enqueued": true,
    "dispatch": {
      "state": "dispatched",
      "task_id": "ingest-8d8734d2-..."
    },
    "policy": {
      "id": "b6d9f3a8-...",
      "external_policy_key": "pol_health_01",
      "title": "Health Shield Plan",
      "status": "draft"
    },
    "document": {
      "id": "e1d0b7c4-...",
      "policy_id": "b6d9f3a8-...",
      "version_number": 1,
      "version_label": "v1",
      "file_name": "policy.pdf",
      "mime_type": "application/pdf",
      "file_size_bytes": 184320,
      "file_hash_sha256": "a3f1...",
      "page_count": 0,
      "ingestion_status": "queued",
      "created_at": "2026-11-27T15:30:00.000Z",
      "updated_at": "2026-11-27T15:30:00.000Z"
    },
    "job": {
      "id": "8d8734d2-...",
      "policy_id": "b6d9f3a8-...",
      "document_id": "e1d0b7c4-...",
      "external_policy_key": "pol_health_01",
      "document_version_snapshot": 1,
      "status": "queued",
      "current_stage": "QUEUED",
      "progress": {
        "total_chunks": 0,
        "processed_chunks": 0,
        "percent": 0
      },
      "attempt_count": 0,
      "error": null,
      "started_at": null,
      "completed_at": null,
      "created_at": "2026-11-27T15:30:00.000Z",
      "updated_at": "2026-11-27T15:30:00.000Z"
    }
  }
}
```

### 1.2 Queued With Delayed Dispatch Recovery (`202 Accepted`)

Returned when PostgreSQL commit succeeded cleanly, but the internal FastAPI/Redis enqueue call is temporarily unavailable. Automatic reconciliation retries dispatch without losing or silently stalling the job (`current_stage: "ENQUEUE_RETRY_PENDING"`).

### 1.3 Duplicate Document Response (`200 OK`)

Returned when the same `file_hash_sha256` already exists as an active/queued/processing/completed document under `:policyId` (`duplicate: true`, `enqueued: false`, returning existing `document` and latest `job` without creating duplicate files or jobs).

### 1.4 Upload Validation & Authorization Errors

| HTTP Status | Trigger |
|---|---|
| `400 Bad Request` | Invalid `:policyId` UUID, missing `file`, non-`.pdf` extension, empty `0`-byte PDF, missing `%PDF-` header signature |
| `401 Unauthorized` | Missing, expired, or invalid JWT access token |
| `403 Forbidden` | Authenticated user lacks `policy:update` permission or policy ownership |
| `404 Not Found` | Policy does not exist |
| `409 Conflict` | Policy is `archived` |
| `413 Payload Too Large` | File exceeds `MAX_PDF_SIZE_BYTES` |
| `415 Unsupported Media Type` | File MIME type is not `application/pdf` |
| `422 Unprocessable Entity` | Encrypted/password-protected PDF (`PASSWORD_PROTECTED_PDF`) or corrupted/truncated PDF (`CORRUPTED_PDF`) |
| `429 Too Many Requests` | Upload rate limit exceeded (`Retry-After` header set) |

All errors follow:
```json
{
  "success": false,
  "statusCode": 422,
  "message": "Encrypted or password-protected PDF documents are not supported",
  "errors": [{ "code": "PASSWORD_PROTECTED_PDF" }],
  "data": null
}
```

---

## 2. Get Single Ingestion Job Status

- **Endpoints:**
  - `GET /api/ingestion/jobs/:jobId`
  - `GET /api/v1/ingestion/jobs/:jobId`
- **Auth / RBAC:** Authenticated user (`verifyJWT`), permission `policy:read`. Strictly scoped by caller `owner_scope` (`user:<id>` and `organization_members` org scopes). Returns `404` for nonexistent OR cross-tenant job IDs without leaking existence.
- **Response (`200 OK`):** `data.job` contains `id`, `policy_id`, `document_id`, `status` (`queued` | `processing` | `completed` | `failed` | `cancelled`), `current_stage`, `progress` (`total_chunks`, `processed_chunks`, `percent`), `attempt_count`, sanitized `error` (`{ code, message } | null`), timestamps, and summary `policy` & `document` objects (never exposing `storage_path`, DSNs, stack traces, or internal tokens).

---

## 3. List Authorized Ingestion Jobs (Paginated)

- **Endpoints:**
  - `GET /api/ingestion/jobs?page=1&limit=20&status=completed&policyId=<uuid>`
  - `GET /api/v1/ingestion/jobs?...`
- **Query Params:**
  - `page` (int `>= 1`, default `1`)
  - `limit` (int `1..100`, default `20`)
  - `status` (`queued` | `processing` | `completed` | `failed` | `cancelled`)
  - `policyId` (UUID filter)
  - Rejects `owner_scope` / `ownerScope` query params (`400 Bad Request`).
- **Ordering:** Stable descending chronological order (`created_at DESC, id DESC`).
- **Response (`200 OK`):**
```json
{
  "statusCode": 200,
  "success": true,
  "message": "Ingestion jobs listed successfully",
  "data": {
    "items": [ "...same job schema as GET /api/ingestion/jobs/:jobId..." ],
    "pagination": {
      "page": 1,
      "limit": 20,
      "total_items": 42,
      "total_pages": 3,
      "has_next": true,
      "has_prev": false
    }
  }
}
```

---

## 4. Internal FastAPI Enqueue Contract (Not Publicly Exposed)

- **Endpoint:** `POST /internal/ingestion/enqueue`
- **Headers:** `X-Service-Token: <INTERNAL_SERVICE_TOKEN>`, `X-Request-Id: <uuid>`
- **Request Body (`extra="forbid"`):**
  ```json
  {
    "job_id": "...",
    "policy_id": "...",
    "document_id": "...",
    "request_id": "..."
  }
  ```
- Python loads the persisted job, policy, and document from PostgreSQL, verifies ownership/status/storage path, resolves trusted `owner_scope` exclusively from PostgreSQL, dispatches idempotently (`task_id = "ingest-" + job_id`), and records dispatch state atomically.
