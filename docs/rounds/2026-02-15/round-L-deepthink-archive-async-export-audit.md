# Round-L: DeepThink Archive Async Export Tasks and Audit Metrics

Date: 2026-02-15  
Scope: DeepThink archive governance hardening (backend + frontend)

## 1. Objective
- Move archive export from sync-only mode to async task mode for large payload safety.
- Add strict time-filter validation to prevent ambiguous timestamp parsing.
- Add archive audit trail + ops metrics endpoint for observability.
- Keep existing Round-I/J/K archive APIs backward compatible.

## 2. Design Decisions
- Export task model:
  - create task (`queued`) -> worker executes (`running`) -> terminal state (`completed`/`failed`)
  - task snapshot endpoint for polling
  - download endpoint only allows completed tasks
- Audit model:
  - record archive query/export/task actions to a dedicated audit table
  - expose aggregated metrics by window + action + status for ops
- Time filter policy:
  - only accept `YYYY-MM-DD HH:MM:SS`
  - reject ISO-8601 `T` style strings to keep client/server behavior deterministic
- Error contract hardening:
  - API now checks `error` value instead of key existence
  - export task snapshot returns `failure_reason` on failure and avoids false `error` key conflicts on success

## 3. Implementation Details
- Backend config and storage
  - `backend/app/config.py`
    - added deep archive retention/env policy knobs (`deep_archive_max_events_*`, tenant policy JSON)
  - `backend/app/web/store.py`
    - added `deep_think_export_task`
    - added `deep_think_archive_audit`
    - added supporting indexes
- Backend service/API
  - `backend/app/web/service.py`
    - added export task CRUD helpers
    - added archive audit write + metrics aggregation
  - `backend/app/service.py`
    - added strict timestamp parser and archive filter normalization
    - added env+tenant retention policy resolver for round event trimming
    - added async export task workflow:
      - `deep_think_create_export_task`
      - `_run_deep_archive_export_task`
      - `deep_think_get_export_task`
      - `deep_think_download_export_task`
    - added archive audit emitter and ops metrics adapter
  - `backend/app/http_api.py`
    - added export task APIs:
      - `POST /v1/deep-think/sessions/{session_id}/events/export-tasks`
      - `GET /v1/deep-think/sessions/{session_id}/events/export-tasks/{task_id}`
      - `GET /v1/deep-think/sessions/{session_id}/events/export-tasks/{task_id}/download`
    - added ops metrics API:
      - `GET /v1/ops/deep-think/archive-metrics`
    - normalized error-code checks to avoid false 404 on successful task snapshots
- Frontend
  - `frontend/app/deep-think/page.tsx`
    - archive cursor history navigation (`上一页存档` / `回到第一页` / `下一页存档`)
    - async export task create + polling + download flow
    - added export-task status tags and task ID visibility
    - supports `failure_reason` fallback for failed-task messaging

## 4. Changed Files
- `backend/app/config.py`
- `backend/app/http_api.py`
- `backend/app/service.py`
- `backend/app/web/service.py`
- `backend/app/web/store.py`
- `frontend/app/deep-think/page.tsx`
- `tests/test_http_api.py`
- `tests/test_service.py`
- `docs/rounds/2026-02-15/round-L-deepthink-archive-async-export-audit.md`
- `docs/agent-column/16-Round-L-DeepThink归档异步导出任务与审计指标实现记录.md`
- `docs/agent-column/00-总览-多agent-a2a-deepagent.md`
- `docs/implementation-checklist.md`
- `docs/spec-traceability-matrix.md`
- `docs/implementation-status-matrix-2026-02-15.md`

## 5. Checklist Mapping
- `AGT-016` (new): archive async export task + strict filter validation + audit metrics.
- `FRONT-009` (new): deep-think archive async export UX + history navigation.
- `GOV-004` (existing): per-round documentation and traceability updates.

## 6. Self-test Evidence
- Backend targeted tests:
  - Command: `.\.venv\Scripts\python -m pytest -q tests/test_service.py tests/test_http_api.py`
  - Result: `28 passed in 25.67s`
- Backend regression:
  - Command: `.\.venv\Scripts\python -m pytest -q`
  - Result: `64 passed in 35.36s`
- Frontend production build:
  - Command: `cd frontend && npm run build`
  - Result: passed (all app routes generated)
- Frontend typecheck:
  - Command: `cd frontend && npx tsc --noEmit`
  - Result: passed

## 7. Risks / Follow-ups
- Export worker uses in-process thread pool; multi-instance deployment may require external queue for HA.
- Audit metrics are window aggregates; percentile latency and per-tenant breakdown can be added later.
- Strict timestamp contract is intentionally narrow; frontend helper formatter can be added to reduce input mistakes.
