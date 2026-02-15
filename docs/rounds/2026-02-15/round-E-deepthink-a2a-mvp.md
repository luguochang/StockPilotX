# Round-E: DeepThink + Internal A2A MVP

Date: 2026-02-15  
Scope: Backend capability upgrade + API exposure + tests + governance records

## 1. Objective
- Deliver a runnable first version of deep thinking interfaces:
  - session creation
  - round execution
  - round-level stream events
- Introduce internal A2A adapter primitives:
  - agent card registry
  - task lifecycle endpoint
- Keep existing `/v1/query` path stable and independent.

## 2. Design Decisions
- DeepThink is isolated from light query path:
  - `/v1/deep-think/*` handles deep multi-agent rounds.
  - `/v1/query` and `/v1/query/stream` remain unchanged in semantics.
- A2A starts as internal adapter, not external network integration:
  - registry + lifecycle are implemented in current service.
  - external peer interoperability is deferred to later rounds.
- Multi-agent set uses 8 roles for round-level reasoning:
  - `supervisor_agent`, `pm_agent`, `quant_agent`, `risk_agent`, `critic_agent`, `macro_agent`, `execution_agent`, `compliance_agent`.
- Group memory uses strict quality gate:
  - only writes anonymous shared card when quality score and evidence count pass thresholds.

## 3. Implementation Details
- Persistence layer (`WebStore`) adds tables:
  - `deep_think_session`
  - `deep_think_round`
  - `deep_think_opinion`
  - `agent_card_registry`
  - `a2a_task`
  - `group_knowledge_card`
- Web domain service (`WebAppService`) adds:
  - deep-think CRUD-style operations
  - round append and opinion persistence
  - agent card register/list
  - A2A task create/update/get
  - group knowledge card write API
- Application service (`AShareAgentService`) adds:
  - default agent card bootstrap
  - deep-think session create / round run / session get / stream events
  - arbitration and counter-view extraction
  - internal A2A task lifecycle orchestration
- HTTP layer adds endpoints:
  - `POST /v1/deep-think/sessions`
  - `POST /v1/deep-think/sessions/{session_id}/rounds`
  - `GET /v1/deep-think/sessions/{session_id}`
  - `GET /v1/deep-think/sessions/{session_id}/stream`
  - `GET /v1/a2a/agent-cards`
  - `POST /v1/a2a/tasks`
  - `GET /v1/a2a/tasks/{task_id}`

## 4. Changed Files
- `backend/app/web/store.py`
- `backend/app/web/service.py`
- `backend/app/service.py`
- `backend/app/http_api.py`
- `tests/test_service.py`
- `tests/test_http_api.py`
- `docs/rounds/README.md`
- `docs/implementation-checklist.md`
- `docs/spec-traceability-matrix.md`

## 5. Self-test Evidence
- Backend targeted:
  - Command: `.\.venv\Scripts\python -m pytest -q tests/test_service.py tests/test_http_api.py`
  - Result: `25 passed`
- Backend full:
  - Command: `.\.venv\Scripts\python -m pytest -q`
  - Result: `61 passed`
- Frontend build:
  - Command: `cd frontend && npm run build`
  - Result: `build passed`
- Frontend typecheck:
  - Command: `cd frontend && npx tsc --noEmit`
  - Result: `typecheck passed`

## 6. Notes for Next Round
- Add explicit deep-think budget decrement and stop reason fields per round.
- Move from rule-first extra opinions to planner-driven dynamic subtask graph.
- Extend A2A task status model with intermediate event records instead of single row updates.
- Add ops page for deep-think session replay and round-level metrics.
