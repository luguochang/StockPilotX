# RAG Two-Phase Architecture & Business Plan (2026-02-15)

## 1) What Was Implemented

### Phase-A (Business loop)
- Added persistent RAG governance schema:
  - `rag_doc_source_policy`
  - `rag_doc_chunk`
  - `rag_qa_memory`
  - `rag_qa_feedback`
  - `rag_retrieval_trace`
- Added source-whitelist policy seed and management APIs.
- Added docs-index bridge: indexed doc chunks are persisted into `rag_doc_chunk` with whitelist-based activation.
- Added review sync: `approve/reject` now updates chunk effective status for retrieval.
- Added QA memory persistence from both `/v1/query` and `/v1/query/stream`.
- Added retrieval observability trace (retrieved IDs vs selected citation IDs + latency).

### Phase-B (Semantic retrieval)
- Added embedding abstraction (`EmbeddingProvider`) with provider config support and local-hash fallback.
- Added local summary vector store (`LocalSummaryVectorStore`):
  - FAISS backend when dependency exists
  - JSON fallback backend when FAISS is unavailable
- Added hybrid retriever v2 (`HybridRetrieverV2`): lexical + semantic + freshness + reliability rerank.
- Implemented coarse-to-fine retrieval path:
  - summary hit first
  - origin/backfill text injected for richer evidence context.

## 2) Business Impact in Stock Analysis

### How it changes user value
- The system no longer treats each question as isolated.
- Past high-quality stock QA now contributes to future answers via shared memory.
- Uploaded financial docs become governance-controlled retrieval assets instead of one-time parsing artifacts.

### Where users can see real benefits
- Better consistency across users asking similar stock/industry questions.
- Faster grounding when question wording changes but semantics are similar.
- More explainable outputs: users can trace whether the answer relied on realtime data, doc assets, or shared QA memory.

### Output integration points
- Query answers now can include shared-knowledge hit context.
- Ops can inspect retrieval trace to debug why a citation was or wasn’t chosen.
- Reindex endpoint supports operational maintenance of semantic index quality.

## 3) Retrieval Design

### Dual-track memory
- Raw track: keeps original content for audit/replay.
- Redacted/summary track: default online retrieval surface for safer reuse.

### Summary-first then origin-backfill
- Coarse retrieval on concise summary vectors.
- Fine backfill on origin text to improve answer completeness and evidence readability.
- This pattern reduces noisy long-text vector search while preserving detail recovery.

## 4) APIs Added
- `GET /v1/rag/source-policy`
- `POST /v1/rag/source-policy/{source}`
- `GET /v1/rag/docs/chunks`
- `POST /v1/rag/docs/chunks/{chunk_id}/status`
- `GET /v1/rag/qa-memory`
- `POST /v1/rag/qa-memory/{memory_id}/toggle`
- `GET /v1/ops/rag/retrieval-trace`
- `POST /v1/ops/rag/reindex`

## 5) Self-test Summary
- Command:
  - `.\\.venv\\Scripts\\python.exe -m pytest tests/test_service.py tests/test_http_api.py -q`
- Result:
  - `42 passed in 34.72s`

## 6) External Technical References
- RAG original paper (NeurIPS 2020):
  - https://arxiv.org/abs/2005.11401
- FAISS library paper:
  - https://arxiv.org/abs/1702.08734
- RAPTOR (hierarchical summarization retrieval):
  - https://arxiv.org/abs/2401.18059
- LangChain ParentDocumentRetriever:
  - https://python.langchain.com/docs/how_to/parent_document_retriever/
- LlamaIndex Document Summary Index:
  - https://developers.llamaindex.ai/python/examples/index_structs/doc_summary/docsummary/

## 7) Known Follow-ups
- Replace local hash fallback with production embedding provider by config.
- Add feedback-driven weighting (`rag_qa_feedback`) into final rerank score.
- Add periodic index compaction and staleness pruning policy.
