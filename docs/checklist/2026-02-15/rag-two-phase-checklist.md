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

- [x] Batch-2: QA memory persistence and quality gate
  - [x] Persist query/query-stream outputs into `rag_qa_memory`.
  - [x] Add retrieval gate (`retrieval_enabled`) based on citation/risk/quality.
  - [x] Write retrieval trace rows for observability.
  - [x] Run tests and capture evidence.
  - [x] Commit Batch-2.

- [x] Batch-3: Business retrieval integration (non-vector baseline)
  - [x] Fuse active doc chunks + QA summaries into runtime corpus.
  - [x] Add business-facing retrieval references in stock analysis chain.
  - [x] Run tests and capture evidence.
  - [x] Commit Batch-3.

- [x] Batch-4: Embedding abstraction + local FAISS store
  - [x] Add configurable embedding provider (current transit model; later switchable small model).
  - [x] Add local FAISS-backed summary index with fallback path.
  - [x] Add ops reindex endpoint wiring.
  - [x] Run tests and capture evidence.
  - [x] Commit Batch-4.

- [x] Batch-5: Summary-first retrieval then origin-backfill
  - [x] Implement coarse-to-fine retrieval path.
  - [x] Merge semantic + lexical ranking outputs.
  - [x] Add regression tests for semantic recall and fallback behavior.
  - [x] Commit Batch-5.

- [x] Batch-6: Final regression and delivery docs
  - [x] End-to-end chain self-test (`/v1/query/stream`, `/v1/rag/*`, `/v1/ops/rag/*`).
  - [x] Update report docs and date README indexes.
  - [x] Final commit and delivery summary.

- [x] Batch-7: Frontend RAG operations console delivery
  - [x] Build and wire `/rag-center` page with 4 tabs (source policy, chunk governance, QA memory, retrieval trace).
  - [x] Add homepage navigation and feature card entry for RAG operations.
  - [x] Add code comments for key data-fetch and governance actions.
  - [x] Run frontend build self-test and capture evidence.

- [x] Batch-8: DeepThink business UX + streaming knowledge feedback
  - [x] Handle `knowledge_persisted` in `/v1/query/stream` consumer and expose persistence feedback.
  - [x] Add “Shared Knowledge Hits” business card to show doc/QA memory reuse in analysis output.
  - [x] Add engineering-card usage descriptions (timeline/task graph/conflict/opinion/budget).
  - [x] Run backend tests + API smoke test (`/v1/query/stream`, `/v2/deep-think/*/rounds/stream`).

- [x] Batch-9: RAG dual-mode UX and upload workflow integration
  - [x] Add business/ops dual mode in `/rag-center` to reduce user input burden.
  - [x] Add attachment upload workflow APIs (`/v1/rag/dashboard`, `/v1/rag/uploads`, `/v1/rag/workflow/upload-and-index`).
  - [x] Add upload asset persistence and dedupe by file hash.
  - [x] Add docs-center attachment upload entry and align with RAG workflow.
  - [x] Run backend + frontend self-test and update docs.
