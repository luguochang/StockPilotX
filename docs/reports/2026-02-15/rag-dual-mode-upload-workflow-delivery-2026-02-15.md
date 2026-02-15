# RAG Dual-Mode UX and Upload Workflow Delivery (2026-02-15)

## 1. Problem Statement
- Existing `RAG运营台` exposed too many low-level fields, which created high cognitive load for business users.
- There was no visible attachment upload entry in `RAG运营台`, and Docs Center mostly relied on JSON text upload.
- Backend RAG capabilities existed, but lacked a business-oriented upload + dashboard API bundle.

## 2. Goals
- Split frontend into **business mode** and **ops mode**.
- Provide attachment-style ingestion path usable from both `RAG运营台` and `Docs Center`.
- Keep existing governance APIs unchanged for backward compatibility.
- Add persistence for upload assets and dedupe by file hash.

## 3. Backend Changes

### 3.1 New persistence tables
- `rag_upload_asset`
  - records upload metadata (filename/source/hash/status/size/tags/stock codes).
  - supports dedupe lookup and recent upload list.
- `rag_ops_meta`
  - stores operation metadata such as `last_reindex_at`.

### 3.2 New web-service capabilities
- upload asset upsert/list/get-by-hash/status update.
- rag dashboard summary aggregation:
  - `doc_total`
  - `active_chunks`
  - `review_pending`
  - `qa_memory_total`
  - `retrieval_hit_rate_7d`
  - `last_reindex_at`

### 3.3 New service APIs
- `rag_upload_from_payload`
  - supports payload `content` or `content_base64`.
  - attachment text extraction (txt/md/csv/docx/xlsx/pdf fallback).
  - dedupe by sha256.
  - optional auto-index and status sync.
- `rag_workflow_upload_and_index`
  - returns stage timeline for frontend progress display.
- `rag_uploads_list`
- `rag_dashboard`

### 3.4 HTTP API additions
- `GET /v1/rag/dashboard`
- `GET /v1/rag/uploads`
- `POST /v1/rag/uploads`
- `POST /v1/rag/workflow/upload-and-index`

## 4. Frontend Changes

### 4.1 RAG Center redesign
- Added mode switch:
  - `业务模式` (default): dashboard + attachment upload + latest uploads.
  - `运维模式`: source/chunk/memory/trace panels retained.
- Reduced business input burden by removing mandatory low-level IDs from default flow.

### 4.2 Docs Center upload alignment
- Added attachment upload action using `/v1/rag/workflow/upload-and-index`.
- Kept legacy text upload (`/v1/docs/upload` + `/index`) for compatibility.

## 5. Compatibility and Risk Notes
- Existing `/v1/docs/*`, `/v1/rag/*`, `/v1/ops/rag/*` interfaces remain available.
- PDF extraction uses fallback strategy when dedicated parser is unavailable in environment.
- Upload APIs currently use JSON/base64 payloads (no multipart dependency), which avoids runtime dependency issues.

## 6. Self-test Evidence
- Backend:
  - `.\.venv\Scripts\python.exe -m pytest tests/test_service.py tests/test_http_api.py -q`
  - `43 passed`
- Frontend:
  - `cd frontend && npm run -s build`
  - build success, `/rag-center` and `/docs-center` routes generated.

## 7. Business Outcome
- Business users can now complete the end-to-end path from upload to retrievable asset without understanding low-level governance fields.
- Ops users still keep full observability and control in dedicated mode.
- The product now supports a clearer two-layer RAG operating model: business workflow + technical governance.
