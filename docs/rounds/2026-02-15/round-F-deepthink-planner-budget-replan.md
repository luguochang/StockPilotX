# Round-F: DeepThink Planner + Budget + Replan

Date: 2026-02-15  
Scope: DeepThink reasoning quality and governance hardening

## 1. Objective
- Upgrade DeepThink from fixed-role round execution to:
  - task-graph planning
  - budget snapshot and stop reason
  - replan trigger signaling
- Keep existing APIs stable while enriching per-round payload.

## 2. Design Decisions
- Plan-first execution:
  - each round now generates `task_graph` before opinion synthesis.
- Budget governance:
  - each round computes `budget_usage` with limit/used/remaining/warn/exceeded.
  - hard stop uses `stop_reason=DEEP_BUDGET_EXCEEDED`.
- Replan strategy:
  - when disagreement exceeds threshold and rounds remain, append replan task.
  - stream emits `replan_triggered` event.
- Backward compatibility:
  - old endpoints remain unchanged; payload fields are additive.

## 3. Implementation Details
- Schema upgrades (`deep_think_round`):
  - `task_graph`
  - `replan_triggered`
  - `stop_reason`
  - `budget_usage`
- Service layer:
  - `_deep_plan_tasks(...)`
  - `_deep_budget_snapshot(...)`
  - enhanced `deep_think_run_round(...)`
  - enhanced `deep_think_stream_events(...)` with `budget_warning`/`replan_triggered`
- Web domain:
  - `deep_think_append_round(...)` persists new fields
  - `deep_think_get_session(...)` parses new round fields

## 4. Changed Files
- `backend/app/service.py`
- `backend/app/web/service.py`
- `backend/app/web/store.py`
- `tests/test_service.py`
- `tests/test_http_api.py`
- `docs/implementation-checklist.md`
- `docs/spec-traceability-matrix.md`
- `docs/reports/2026-02-15/implementation-status-matrix-2026-02-15.md`
- `docs/rounds/2026-02-15/round-F-deepthink-planner-budget-replan.md`
- `docs/agent-column/10-Round-F-DeepThink-Planner-Budget-Replan实现记录.md`
- `docs/agent-column/00-总览-多agent-a2a-deepagent.md`

## 5. Self-test Evidence
- Backend targeted:
  - Command: `.\.venv\Scripts\python -m pytest -q tests/test_service.py tests/test_http_api.py`
  - Result: `27 passed`
- Backend full:
  - Command: `.\.venv\Scripts\python -m pytest -q`
  - Result: `63 passed`
- Frontend build:
  - Command: `cd frontend && npm run build`
  - Result: `build passed`
- Frontend typecheck:
  - Command: `cd frontend && npx tsc --noEmit`
  - Result: failed due existing `frontend/tsconfig.json` include pattern `.next/types/**/*.ts` referencing missing generated files.

## 6. Next Round Suggestions
- Persist round-level budget history and cumulative consumption at session level.
- Replace synthetic budget usage with real token/tool counters from middleware.
- Add deep-think ops dashboard for replay and conflict analytics.
