# Optimization Iteration (2026-02-15, Round-C)

## Goal
- Implement three advanced engineering capabilities in a real runnable path:
  - multi-agent debate visualization
  - RAG quality dashboard (Recall/MRR/nDCG + case-level detail)
  - prompt version compare & replay

## Delivered
1. Multi-agent debate API
- New endpoint: `GET /v1/ops/agent/debate?stock_code=...&question=...`
- Service logic:
  - PM agent opinion (trend + daily move)
  - Quant agent opinion (prediction horizon result)
  - Risk agent opinion (drawdown + volatility)
- Output:
  - `opinions[]`
  - `consensus_signal`
  - `disagreement_score`
  - `market_snapshot`

2. RAG quality API
- New endpoint: `GET /v1/ops/rag/quality`
- Output:
  - aggregate metrics: `recall_at_k`, `mrr`, `ndcg_at_k`
  - case-level details:
    - query
    - expected sources
    - predicted sources
    - hit sources
    - per-case recall/mrr/ndcg

3. Prompt compare & replay API
- New endpoint: `POST /v1/ops/prompts/compare`
- Added prompt registry/runtime capabilities:
  - `PromptRegistry.get_prompt()`
  - `PromptRegistry.list_prompt_versions()`
  - `PromptRuntime.build_version()`
  - `PromptRuntime.build_from_prompt()`
- Compare output:
  - base/candidate rendered prompt
  - metadata
  - unified diff preview

4. Ops frontend integration
- Updated page: `frontend/app/ops/evals/page.tsx`
- Added panel sections:
  - capability snapshot
  - multi-agent debate table
  - RAG quality metrics + case table
  - prompt compare raw replay output

5. Stability hardening
- Rebuilt `backend/app/data/sources.py` to clean encoding corruption and keep live+fallback behavior stable.
- Enforced short timeout and no system proxy for external fetch calls, improving API-test stability.

## Validation
- Backend tests:
  - `.\.venv\Scripts\python -m pytest -q`
  - result: `58 passed`
- API contract subset:
  - `.\.venv\Scripts\python -m pytest -q tests/test_http_api.py`
  - result: `12 passed`
- Frontend:
  - `cd frontend && npm run build` passed
  - `cd frontend && npx tsc --noEmit` passed

## Main Changed Files
- `backend/app/service.py`
- `backend/app/http_api.py`
- `backend/app/prompt/registry.py`
- `backend/app/prompt/runtime.py`
- `backend/app/data/sources.py`
- `frontend/app/ops/evals/page.tsx`
- `tests/test_http_api.py`
- `tests/test_prompt_engineering.py`
