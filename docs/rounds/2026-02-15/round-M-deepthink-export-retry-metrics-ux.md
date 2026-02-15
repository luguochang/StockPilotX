# Round-M: DeepThink Export Retry Hardening, Audit Percentiles, and Time-Filter UX

Date: 2026-02-15  
Scope: DeepThink archive export reliability + observability + frontend filtering UX

## 1. Objective
- Harden async export tasks with deterministic retry behavior and attempt tracking.
- Upgrade archive audit metrics with latency percentiles and richer aggregations.
- Reduce timestamp input mistakes in frontend by replacing free text with structured datetime controls.

## 2. Design Decisions
- Export task reliability model:
  - task starts in `queued`
  - worker performs atomic claim (`queued -> running`) and increments `attempt_count`
  - transient failures are requeued until `max_attempts` reached
  - terminal states remain `completed` / `failed`
- Retry policy:
  - configurable max attempts and backoff from settings
  - exponential backoff with capped wait to avoid busy looping
- Metrics model:
  - keep existing summary/by_action/by_status for backward compatibility
  - add `p50/p95/p99` latency, `slow_calls_over_1000ms`, `by_action_status`, and `top_sessions`
- Frontend time filter UX:
  - use `datetime-local` inputs with second-level precision (`step=1`)
  - normalize to backend-required `YYYY-MM-DD HH:MM:SS`
  - add quick actions (`最近24小时`, `清空时间过滤`)

## 3. Implementation Details
- Backend config
  - `backend/app/config.py`
    - added `deep_archive_export_task_max_attempts`
    - added `deep_archive_export_retry_backoff_seconds`
- Backend storage
  - `backend/app/web/store.py`
    - export task schema now includes:
      - `attempt_count`
      - `max_attempts`
    - added queue-oriented index for export task status progression
- Backend web service
  - `backend/app/web/service.py`
    - `deep_think_export_task_create(..., max_attempts=...)`
    - new task queue methods:
      - `deep_think_export_task_try_claim(...)`
      - `deep_think_export_task_requeue(...)`
    - metrics expanded with percentiles and action/session dimensions
- Backend app service
  - `backend/app/service.py`
    - export task creation now binds configured `max_attempts`
    - `_run_deep_archive_export_task(...)` now uses claim + retry loop + backoff
    - task snapshot payload now includes `attempt_count` and `max_attempts`
- Frontend
  - `frontend/app/deep-think/page.tsx`
    - task snapshot type includes `attempt_count/max_attempts`
    - added timestamp normalization helpers
    - replaced plain timestamp text input with `datetime-local` controls
    - added quick filter actions and attempt tag display

## 4. Changed Files
- `backend/app/config.py`
- `backend/app/service.py`
- `backend/app/web/service.py`
- `backend/app/web/store.py`
- `frontend/app/deep-think/page.tsx`
- `tests/test_service.py`
- `tests/test_http_api.py`
- `docs/rounds/2026-02-15/round-M-deepthink-export-retry-metrics-ux.md`
- `docs/agent-column/17-Round-M-DeepThink导出重试与审计分位指标实现记录.md`
- `docs/agent-column/00-总览-多agent-a2a-deepagent.md`
- `docs/implementation-checklist.md`
- `docs/spec-traceability-matrix.md`
- `docs/implementation-status-matrix-2026-02-15.md`

## 5. Checklist Mapping
- `AGT-017` (new): export task claim/retry attempts + percentile audit metrics.
- `FRONT-010` (new): structured archive time filters + quick windows + attempt visibility.
- `GOV-004` (existing): per-round documentation and traceability.

## 6. Self-test Evidence
- Backend targeted tests:
  - Command: `.\.venv\Scripts\python -m pytest -q tests/test_service.py tests/test_http_api.py`
  - Result: `29 passed in 34.52s`
- Backend regression:
  - Command: `.\.venv\Scripts\python -m pytest -q`
  - Result: `65 passed in 39.41s`
- Frontend production build:
  - Command: `cd frontend && npm run build`
  - Result: passed
- Frontend typecheck:
  - Command: `cd frontend && npx tsc --noEmit`
  - Result: passed

## 7. Risks / Follow-ups
- Current retry orchestration is process-local; for cross-instance at-least-once guarantees, external queue infra is still recommended.
- Percentile metrics are computed from in-window rows in app memory; long windows may require pre-aggregation strategy later.
- Windows file-lock (`.next/trace`) can intermittently affect build in CI/dev; retry strategy or dedicated clean/build step may be needed.
