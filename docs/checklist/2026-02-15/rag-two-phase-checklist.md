# RAG Two-Phase Checklist (2026-02-15)

## Goal
- Build a business-usable RAG knowledge flywheel for StockPilotX:
  - Phase-A: governance + persistence + shared memory loop.
  - Phase-B: semantic retrieval with local FAISS and summary-first then origin-backfill.

## Global Rules
- [x] Full-site shared memory scope (`share_scope=global`) for stock QA.
- [x] Source whitelist policy for document auto-activation.
- [x] Dual-track storage for QA/doc (`raw + redacted/summary`).
- [x] Every batch requires: code comments + self-test + docs update + commit.

## Batch Plan
- [x] Batch-1: Schema and API foundations
  - [x] Add RAG governance tables (`rag_doc_source_policy`, `rag_doc_chunk`, `rag_qa_memory`, `rag_qa_feedback`, `rag_retrieval_trace`).
  - [x] Add WebService methods for source policy/chunk/qa-memory/trace CRUD.
  - [x] Add HTTP APIs for `/v1/rag/*` and `/v1/ops/rag/retrieval-trace`.
  - [x] Add docs chunk persistence hook in `docs_index` + review-status sync.
  - [x] Run tests and capture evidence.
  - [x] Commit Batch-1.

- [ ] Batch-2: QA memory persistence and quality gate
  - [ ] Persist query/query-stream outputs into `rag_qa_memory`.
  - [ ] Add retrieval gate (`retrieval_enabled`) based on citation/risk/quality.
  - [ ] Write retrieval trace rows for observability.
  - [ ] Run tests and capture evidence.
  - [ ] Commit Batch-2.

- [ ] Batch-3: Business retrieval integration (non-vector baseline)
  - [ ] Fuse active doc chunks + QA summaries into runtime corpus.
  - [ ] Add business-facing retrieval references in stock analysis chain.
  - [ ] Run tests and capture evidence.
  - [ ] Commit Batch-3.

- [ ] Batch-4: Embedding abstraction + local FAISS store
  - [ ] Add configurable embedding provider (current transit model; later switchable small model).
  - [ ] Add local FAISS-backed summary index with fallback path.
  - [ ] Add ops reindex endpoint wiring.
  - [ ] Run tests and capture evidence.
  - [ ] Commit Batch-4.

- [ ] Batch-5: Summary-first retrieval then origin-backfill
  - [ ] Implement coarse-to-fine retrieval path.
  - [ ] Merge semantic + lexical ranking outputs.
  - [ ] Add regression tests for semantic recall and fallback behavior.
  - [ ] Commit Batch-5.

- [ ] Batch-6: Final regression and delivery docs
  - [ ] End-to-end chain self-test (`/v1/query/stream`, `/v1/rag/*`, `/v1/ops/rag/*`).
  - [ ] Update report docs and date README indexes.
  - [ ] Final commit and delivery summary.
