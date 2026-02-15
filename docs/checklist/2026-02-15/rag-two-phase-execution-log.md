# RAG Two-Phase Execution Log (2026-02-15)

## Batch-1 (Completed)

### Scope
- Schema expansion for RAG governance and shared memory assets.
- Web/API capability for source-policy, chunk governance, QA memory inspection, retrieval traces.
- Docs index -> chunk persistence bridge.

### Self-test Evidence
- Command:
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_service.py tests/test_http_api.py -q`
- Result:
  - `39 passed in 28.34s`

### Commit
- `1db032a` - `feat(rag): add phase-a governance schema and management apis`

---

## Batch-2 ~ Batch-5 (Completed)

### Scope
- Persist query/query-stream memory into `rag_qa_memory` with quality gate.
- Add retrieval trace persistence (`rag_retrieval_trace`).
- Fuse persisted docs + QA summaries into runtime corpus.
- Add semantic retrieval stack: embedding provider, local vector store, and hybrid rerank.
- Add summary-first -> origin-backfill retrieval path.

### Key Files
- `backend/app/config.py`
- `backend/app/service.py`
- `backend/app/rag/embedding_provider.py`
- `backend/app/rag/vector_store.py`
- `backend/app/rag/hybrid_retriever_v2.py`
- `tests/test_service.py`
- `tests/test_http_api.py`

### Self-test Evidence
- Command:
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_service.py tests/test_http_api.py -q`
- Result:
  - `42 passed in 34.72s`

### Commit
- `6cd8e8c` - `feat(rag): persist qa memory and add semantic summary-origin retrieval`

---

## Batch-6 (Completed)

### Scope
- Final checklist close-out and architecture/business report delivery.
- Reported external references for hierarchical retrieval and FAISS best practices.

### Deliverables
- `docs/checklist/2026-02-15/rag-two-phase-checklist.md`
- `docs/checklist/2026-02-15/rag-two-phase-execution-log.md`
- `docs/reports/2026-02-15/rag-two-phase-architecture-business-plan-2026-02-15.md`
- `docs/reports/2026-02-15/README.md`

### Commit
- `c81f12c` - `docs(rag): finalize two-phase checklist and architecture business report`

---

## Batch-7 (Completed)

### Scope
- Deliver RAG operations frontend page `/rag-center`:
  - source policy governance,
  - doc chunk status operations,
  - QA memory enable/disable operations,
  - retrieval trace viewing + reindex trigger.
- Add homepage entry card and global nav link for `RAG运营台`.
- Fix previous frontend text/encoding issues and keep key logic annotated.

### Self-test Evidence
- Command:
  - `cd frontend && npm run -s build`
- Result:
  - `Compiled successfully`
  - `Route (app) ... /rag-center ...`

---

## Batch-8 (Completed)

### Scope
- DeepThink advanced analysis UX improvements:
  - consume `knowledge_persisted` from `/v1/query/stream`,
  - add shared-knowledge hits card for business users,
  - add engineering card purpose texts for timeline/task/conflict/opinion/budget.
- Keep engineering mode intact while improving business explanation path.

### Self-test Evidence
- Backend test command:
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_service.py tests/test_http_api.py -q`
- Backend test result:
  - `42 passed in 45.30s`
- API smoke command:
  - local `TestClient` script invoking:
    - `/v1/query/stream` (verified `answer_delta`, `knowledge_persisted`, `analysis_brief`)
    - `/v1/deep-think/sessions` + `/v2/deep-think/sessions/{id}/rounds/stream` (verified `round_started` -> `done`)
- API smoke result:
  - `SMOKE_OK`

---

## Batch-9 (Completed)

### Scope
- Implement RAG dual-mode product UX in frontend:
  - `business` mode: upload, dashboard, latest uploads list.
  - `ops` mode: keep source/chunk/memory/trace governance panels.
- Implement backend upload workflow APIs for attachment-style ingestion:
  - `GET /v1/rag/dashboard`
  - `GET /v1/rag/uploads`
  - `POST /v1/rag/uploads`
  - `POST /v1/rag/workflow/upload-and-index`
- Add upload-asset persistence (`rag_upload_asset`) and ops metadata (`rag_ops_meta`), plus dedupe by file hash.
- Add docs-center attachment upload path to reuse the same RAG workflow chain.

### Key Files
- `backend/app/http_api.py`
- `backend/app/service.py`
- `backend/app/web/service.py`
- `backend/app/web/store.py`
- `frontend/app/rag-center/page.tsx`
- `frontend/app/docs-center/page.tsx`
- `tests/test_service.py`
- `tests/test_http_api.py`

### Self-test Evidence
- Backend tests:
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_service.py tests/test_http_api.py -q`
  - Result: `43 passed in 58.09s`
- Frontend build:
  - `cd frontend && npm run -s build`
  - Result: `Compiled successfully`, route `/rag-center` and `/docs-center` generated.

---

## Batch-10 (Completed)

### Scope
- Remove review approval gate for uploaded docs so users can retrieve immediately after upload.
- Ensure doc metadata and RAG chunk persistence are aligned to immediate activation semantics.
- Add regression checks to prevent future rollback to pseudo-review gating behavior.

### Key Files
- `backend/app/service.py`
- `tests/test_service.py`
- `tests/test_http_api.py`

### Self-test Evidence
- Backend tests:
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_service.py tests/test_http_api.py -q`
  - Result: `43 passed in 53.06s`
- Frontend build:
  - `cd frontend && npm run -s build`
  - Result: `Compiled successfully`, route `/rag-center` and `/docs-center` generated.

### Commit
- `3be2e59` - `feat(rag): activate uploads immediately without review gate`
