# RAG Upload Immediate Activation (2026-02-15)

## Background
- User requirement: uploaded files must be effective immediately.
- Previous behavior still had a review-gate path:
  - `doc_index.needs_review` could be set by parse confidence.
  - RAG chunk activation depended on source policy (`auto_approve` + `enabled`).
- This mismatch caused business confusion: upload completed but retrieval did not always reflect immediate effect.

## Root Cause
- Gate logic existed in three places:
  - `docs_upload`: set `needs_review` by confidence threshold.
  - `docs_index`: set `needs_review` by confidence threshold.
  - `_persist_doc_chunks_to_rag`: set `effective_status="review"` when source policy did not auto-approve.
- As a result, some uploads needed manual review transition before becoming searchable.

## Fix Implemented
- File: `backend/app/service.py`
- Changes:
  - In `docs_upload`, force `needs_review=False`.
  - In `docs_index`, force `needs_review=False`.
  - In `_persist_doc_chunks_to_rag`, force `effective_status="active"` for persisted chunks.
- Added code comments to explain the product decision:
  - upload now follows "immediate activation" policy.
  - source-policy metadata remains for governance visibility, but not as activation gate.

## Regression Tests
- File: `tests/test_service.py`
  - `test_doc_upload_and_index`: verify uploaded doc is not in review queue and chunks are `active`.
  - `test_rag_upload_workflow_and_dashboard`: verify uploaded asset status is `active` (force reupload to avoid dedupe branch).
- File: `tests/test_http_api.py`
  - `test_rag_asset_management_endpoints`: verify workflow upload result asset status is `active`.

## Validation
- Backend:
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_service.py tests/test_http_api.py -q`
  - Result: `43 passed`
- Frontend:
  - `cd frontend && npm run -s build`
  - Result: `Compiled successfully`

## Impact
- Upload UX is now consistent with business expectation: "upload == effective".
- Removes operational friction from manual approval for normal upload flow.
- Keeps observability and governance metadata intact without blocking retrieval.
