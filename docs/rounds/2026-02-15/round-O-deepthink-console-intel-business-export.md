# Round-O: DeepThink Console Redesign + LLM WebSearch Intelligence + Business Export

Date: 2026-02-15

## 1. Objectives
- Redesign DeepThink console into analysis/engineering dual-mode UI.
- Add user-visible three-layer feedback during round execution.
- Integrate LLM WebSearch-driven intelligence signals into DeepThink round stream.
- Add business-readable export and fix CSV Excel compatibility.

## 2. Key Changes
- Frontend (`frontend/app/deep-think/page.tsx`)
  - Added `analysis` / `engineering` mode switch using `Segmented`.
  - Analysis mode now focuses on run controls, stage progress, decision summary, business summary, intelligence snapshot, and calendar watchlist.
  - Engineering mode keeps archive filters, replay, pagination, and audit exports.
  - Added stage mapping and progress text updates from SSE events.
  - Added business export action (`导出业务结论CSV`).
- Backend (`backend/app/service.py`)
  - Added LLM-intelligence pipeline helpers:
    - `_deep_build_intel_prompt`
    - `_deep_safe_json_loads`
    - `_deep_validate_intel_payload`
    - `_deep_fetch_intel_via_llm_websearch`
    - `_deep_local_intel_fallback`
    - `_deep_build_business_summary`
  - DeepThink round stream now emits:
    - `intel_snapshot`
    - `calendar_watchlist`
    - `business_summary`
  - Added business export service:
    - `deep_think_export_business(..., format=csv|json)`
  - Event CSV export now prepends UTF-8 BOM for Excel compatibility.
- Backend API (`backend/app/http_api.py`)
  - Added route:
    - `GET /v1/deep-think/sessions/{session_id}/business-export`
- Storage concurrency hardening (`backend/app/web/store.py`)
  - Added connection lock to serialize sqlite operations and avoid intermittent API misuse under async export polling.

## 3. Tests
- Backend targeted regressions:
  - `tests/test_service.py`
  - `tests/test_http_api.py`
- Added/updated assertions for:
  - `business_summary` event presence.
  - CSV BOM-compatible header checks.
  - business export endpoint behavior.

## 4. Commands and Results
- `python -m pytest tests/test_service.py::ServiceTestCase::test_deep_think_session_and_round tests/test_service.py::ServiceTestCase::test_deep_think_v2_stream_round tests/test_http_api.py::HttpApiTestCase::test_deep_think_and_a2a tests/test_http_api.py::HttpApiTestCase::test_deep_think_v2_round_stream -q`
  - Result: `4 passed`
- `python -m pytest tests/test_service.py tests/test_http_api.py -q`
  - Result: `32 passed`
- `npm --prefix frontend run -s build`
  - Result: passed

## 5. Notes
- Intelligence retrieval is prompt-driven through external LLM gateway capabilities, with local fallback to prevent hard failure when web-search is unavailable.
- Business export and audit export are now explicitly separated by purpose.
