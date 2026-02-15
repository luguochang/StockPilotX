# Round-J: DeepThink Archive Filter + Retention Governance

Date: 2026-02-15  
Scope: DeepThink event archive queryability and retention controls (backend + frontend)

## 1. Objective
- Upgrade archive replay from "raw list fetch" to "ops-grade query":
  - filter by `round_id`
  - filter by `event_name`
  - configurable query `limit`
- Add archive retention governance to prevent unbounded session event growth.
- Keep compatibility with Round-I contract and existing frontend workflows.

## 2. Design Decisions
- Extend existing archive endpoint instead of adding new path:
  - `GET /v1/deep-think/sessions/{session_id}/events`
  - new query arg: `event_name`
- Keep retention policy session-scoped:
  - after each round snapshot write, keep only latest N events for that session
  - default N=1200, configurable by round request payload `archive_max_events`
- Keep frontend filter-driven load explicit:
  - users select round/event/limit in DeepThink console
  - click "加载会话存档" to fetch deterministic filtered snapshots.

## 3. Implementation Details
- Backend API/service/storage:
  - `backend/app/http_api.py`
    - `/events` route now accepts `event_name`
  - `backend/app/service.py`
    - `deep_think_list_events(...)` now accepts `event_name`
    - `deep_think_run_round(...)` now supports `archive_max_events`
    - pass retention cap into archive write path
  - `backend/app/web/service.py`
    - `deep_think_list_events(...)` now supports SQL filtering by `event_name`
    - `deep_think_replace_round_events(...)` now trims archived rows by session
    - added `deep_think_trim_events(...)`
  - `backend/app/web/store.py`
    - added index `idx_deep_think_event_name`
- Frontend DeepThink UX:
  - `frontend/app/deep-think/page.tsx`
    - added archive filter states:
      - `deepArchiveRoundId`
      - `deepArchiveEventName`
      - `deepArchiveLimit`
    - added archive query controls (`Select + Select + InputNumber`) in control console
    - `loadDeepThinkEventArchive(...)` now supports options object `{roundId,eventName,limit}`
    - replay list now shows active filter context and uses filtered rows (`deepReplayRows`).

## 4. Changed Files
- `backend/app/http_api.py`
- `backend/app/service.py`
- `backend/app/web/service.py`
- `backend/app/web/store.py`
- `frontend/app/deep-think/page.tsx`
- `tests/test_http_api.py`
- `tests/test_service.py`
- `docs/rounds/2026-02-15/round-J-deepthink-archive-filter-retention.md`
- `docs/agent-column/14-Round-J-DeepThink事件过滤与归档保留治理实现记录.md`
- `docs/agent-column/00-总览-多agent-a2a-deepagent.md`
- `docs/implementation-checklist.md`
- `docs/spec-traceability-matrix.md`
- `docs/reports/2026-02-15/implementation-status-matrix-2026-02-15.md`

## 5. Checklist Mapping
- `AGT-014` (new): DeepThink archive filtering and retention governance on backend.
- `FRONT-007` (new): DeepThink archive filter console and filtered replay UX.
- `GOV-004` (existing): per-round implementation documentation and traceability updates.

## 6. Self-test Evidence
- Backend targeted tests:
  - Command: `.\.venv\Scripts\python -m pytest -q tests/test_service.py tests/test_http_api.py`
  - Result: `27 passed in 24.51s`
- Backend regression:
  - Command: `.\.venv\Scripts\python -m pytest -q`
  - Result: `63 passed in 40.11s`
- Frontend production build:
  - Command: `cd frontend && npm run build`
  - Result: passed (`/deep-think` route generated)
- Frontend typecheck:
  - Command: `cd frontend && npx tsc --noEmit`
  - Result: passed

## 7. Risks / Follow-ups
- Current retention is count-based; time-based retention policy is not implemented yet.
- `archive_max_events` is request-level configurable; no tenant-level quota policy yet.
- Next round can add:
  - archive filters by `event_seq` / time range
  - retention policy profile by environment (dev/staging/prod)
  - archive export endpoint for external audit pipelines.
