# Optimization Iteration (2026-02-15, Round-B)

## Scope
- Strengthen evidence transparency for `/v1/query` and stream flow.
- Upgrade prediction pipeline to use real historical bars first (free source), synthetic only as fallback.
- Expose stronger front-end trust signals for analysis and prediction outputs.

## Implemented
1. Query structured evidence output
- Added `analysis_brief` in `/v1/query` response.
- Added SSE event `analysis_brief` in `/v1/query/stream`.
- `analysis_brief` includes:
  - `confidence_level`, `confidence_reason`
  - per-stock realtime freshness seconds
  - history sample size and trend metrics
  - citation count and average reliability

2. Prediction realism and explainability
- `PredictionService` now uses `HistoryService.fetch_daily_bars()` first.
- Added fallback mode `synthetic_fallback` when real history unavailable.
- Added source metadata:
  - `history_data_mode`
  - `history_source_id`
  - `history_source_url`
  - `history_sample_size`
- Extended factor set:
  - `trend_strength`, `drawdown_60`, `atr_14`, `volume_stability_20`
- Added horizon-level `rationale` text.

3. Frontend trust UI
- Home page (`/`) now renders `analysis_brief` card:
  - confidence tag
  - citation reliability summary
  - per-stock freshness and trend stats
- Predict page (`/predict`) now displays:
  - history data mode/source/sample size
  - rationale column per horizon

4. Reliability hardening
- `AnnouncementService.fetch_announcements()` now catches generic exceptions and falls back to mock data (prevents network timeout from breaking service tests).
- Fixed frontend typecheck stability by changing `frontend/tsconfig.json` include from `.next/types/**/*.ts` to `.next/types/**/*.d.ts`.

## Validation Evidence
- Backend:
  - `.\.venv\Scripts\python -m pytest -q`
  - Result: `57 passed`
- Frontend:
  - `cd frontend && npm run build`
  - Result: `Next.js build success`
  - `cd frontend && npx tsc --noEmit`
  - Result: `passed`
- Runtime smoke:
  - `AShareAgentService().query(...)` returns keys including `analysis_brief`.
  - `AShareAgentService().predict_run(...)` returns `history_data_mode=real_history` in source metadata.

## Touched Files
- `backend/app/models.py`
- `backend/app/service.py`
- `backend/app/predict/service.py`
- `backend/app/data/sources.py`
- `frontend/app/page.tsx`
- `frontend/app/predict/page.tsx`
- `frontend/tsconfig.json`
- `tests/test_http_api.py`
- `tests/test_service.py`
