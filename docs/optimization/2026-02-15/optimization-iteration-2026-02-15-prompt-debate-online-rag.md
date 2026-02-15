# Optimization Iteration (2026-02-15, Round-D)

## Scope
- Execute requested items only:
  - 3) real prompt multi-version compare path
  - 4) multi-agent debate upgraded toward real model parallel discussion
  - 5) online RAG continuous evaluation dataset and metrics

## Implemented
1. Prompt multi-version compare (real versions)
- Prompt registry now ensures both versions exist:
  - `fact_qa@1.0.0` (stable)
  - `fact_qa@1.1.0` (candidate)
- Added prompt versions API:
  - `GET /v1/ops/prompts/{prompt_id}/versions`
- Compare API now defaults to cross-version replay:
  - `POST /v1/ops/prompts/compare` default `1.0.0` vs `1.1.0`
- Frontend ops page supports selecting base/candidate versions and replaying compare.

2. Multi-agent debate upgraded
- `ops_agent_debate` now supports:
  - `llm_parallel` mode when external LLM is enabled and providers available
  - `rule_fallback` mode otherwise
- LLM path runs PM/Quant/Risk prompts in parallel and parses strict JSON signal output.
- Automatic fallback keeps service available if model output is invalid or provider fails.

3. Online RAG continuous evaluation
- Added persistent table:
  - `rag_eval_case(query_text, positive_source_ids, predicted_source_ids, created_at)`
- Query flow now writes online RAG eval cases from real requests.
- `GET /v1/ops/rag/quality` now returns:
  - combined metrics
  - offline benchmark block
  - online dataset block

4. Stability follow-up
- Rebuilt `backend/app/data/sources.py` into a clean, parse-safe implementation.
- External source requests use short timeout + no system proxy to avoid long blocking in API tests.

## Validation
- `.\.venv\Scripts\python -m pytest -q tests/test_http_api.py tests/test_prompt_engineering.py tests/test_service.py`
  - result: `25 passed`
- `.\.venv\Scripts\python -m pytest -q`
  - result: `58 passed`
- Frontend:
  - `npm run build` passed
  - `npx tsc --noEmit` passed

## Files
- `backend/app/prompt/registry.py`
- `backend/app/web/store.py`
- `backend/app/web/service.py`
- `backend/app/service.py`
- `backend/app/http_api.py`
- `backend/app/data/sources.py`
- `frontend/app/ops/evals/page.tsx`
- `frontend/tsconfig.json`
- `tests/test_http_api.py`
- `tests/test_prompt_engineering.py`
