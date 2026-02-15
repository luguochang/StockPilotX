# Round S - DeepThink Business Console Clarity

## Goal
Address user-facing confusion in DeepThink console by making analysis mode business-oriented, while preserving engineering observability in engineering mode.

## User Problems Addressed
- Users could not understand what `macro_agent`, `pm_agent`, etc. meant.
- Users did not know whether “执行下一轮” required running “高级分析” first.
- DeepThink panel showed too many technical controls/tables in one view.
- Cross-stock switching risked carrying stale DeepThink context.

## Implementation
- Updated `frontend/app/deep-think/page.tsx`:
  - Added semantic label helpers:
    - `getSignalLabel`
    - `getPriorityLabel`
    - `getConflictSourceLabel`
  - Added Chinese business rendering for:
    - task graph agent/priority
    - opinions table
    - opinion diff table
    - conflict drilldown table
    - conflict source tags in summary cards
  - Kept stock-switch reset behavior with explicit notice, ensuring per-stock data isolation.
  - Split DeepThink view responsibilities:
    - analysis mode:
      - business interpretation cards and usage guidance
      - simplified process summary and actionable hints
    - engineering mode:
      - timeline, task graph, budget, conflict chart, diff, drilldown, SSE replay
  - Switched timeline item payload from deprecated `children` to `content`.

## Why This Design
- Business users need recommendation rationale and risk action first, not raw telemetry.
- Engineering users still need full diagnostics for replay, audit, and tuning.
- The mode split keeps both without conflating intent.

## Self-Test Evidence
- `npx tsc --noEmit` (frontend): passed.
- `npm --prefix frontend run -s build`: passed.
- `.\.venv\Scripts\python.exe -m pytest tests/test_service.py tests/test_http_api.py -q`: `37 passed`.
- Real API chain:
  - `/v1/query/stream` returned streaming deltas.
  - `/v2/deep-think/sessions/{session_id}/rounds/stream` emitted full round events and done.
  - `business-export` returned CSV with UTF-8 BOM.

## Result
DeepThink console now provides a business-readable default path and a clean engineering fallback path, reducing cognitive overload while keeping full traceability.
