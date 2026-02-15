# Round-I: DeepThink Opinion Diff + Conflict Drill-down + Event Archive

Date: 2026-02-15  
Scope: DeepThink observability and replay persistence (backend + frontend)

## 1. Objective
- Close the three follow-ups from Round-H:
  - round-to-round opinion diff panel
  - conflict source drill-down by agent evidence mapping
  - persisted stream snapshots for replay/audit
- Keep existing `/v1/deep-think/*` and `/v1/a2a/*` contracts stable for previous clients.

## 2. Design Decisions
- Add a dedicated event archive endpoint instead of overloading stream endpoint:
  - new API: `GET /v1/deep-think/sessions/{session_id}/events`
  - supports optional `round_id` and `limit`
- Persist round stream events in SQLite table `deep_think_event`:
  - deterministic replay order: `round_no -> event_seq`
  - regenerated from round snapshot if archive is empty (backward compatibility path)
- Keep frontend diff and drill-down pure client-side derived views:
  - no new backend schema needed for opinion diff
  - conflict drill-down derives candidate rows from `consensus_signal`, `conflict_sources`, `opinions`.

## 3. Implementation Details
- Backend API and service:
  - `backend/app/http_api.py`
    - added `GET /v1/deep-think/sessions/{session_id}/events`
  - `backend/app/service.py`
    - added `_build_deep_think_round_events(...)`
    - deep round completion now writes archive snapshots via `deep_think_replace_round_events(...)`
    - stream path now prefers archived events; auto-recovers by regenerating if missing
    - added `deep_think_list_events(...)` service API
- Backend storage:
  - `backend/app/web/store.py`
    - added table `deep_think_event`
    - added index `idx_deep_think_event_session`
  - `backend/app/web/service.py`
    - added `deep_think_replace_round_events(...)`
    - added `deep_think_list_events(...)`
- Frontend DeepThink page:
  - `frontend/app/deep-think/page.tsx`
    - added archive model `DeepThinkEventArchiveSnapshot`
    - added archive load action `loadDeepThinkEventArchive(...)`
    - added archive state `deepArchiveLoading`, `deepArchiveCount`
    - added cross-round diff panel (`deepOpinionDiffRows`)
    - added conflict drill-down panel (`deepConflictDrillRows`)
    - added “加载会话存档” control and archive count tag

## 4. Changed Files
- `backend/app/http_api.py`
- `backend/app/service.py`
- `backend/app/web/service.py`
- `backend/app/web/store.py`
- `frontend/app/deep-think/page.tsx`
- `tests/test_http_api.py`
- `tests/test_service.py`
- `docs/rounds/2026-02-15/round-I-deepthink-diff-drilldown-archive.md`
- `docs/agent-column/13-Round-I-DeepThink跨轮差分与事件存档实现记录.md`
- `docs/agent-column/00-总览-多agent-a2a-deepagent.md`
- `docs/governance/implementation-checklist.md`
- `docs/governance/spec-traceability-matrix.md`
- `docs/reports/2026-02-15/implementation-status-matrix-2026-02-15.md`

## 5. Checklist Mapping
- `AGT-013` (new): DeepThink persisted event archive and replay API.
- `FRONT-006` (new): DeepThink cross-round diff + conflict evidence drill-down + archive UX.
- `GOV-004` (existing): per-round implementation documentation and traceability updates.

## 6. Self-test Evidence
- Backend targeted tests:
  - Command: `.\.venv\Scripts\python -m pytest -q tests/test_service.py tests/test_http_api.py`
  - Result: `27 passed in 26.05s`
- Backend regression:
  - Command: `.\.venv\Scripts\python -m pytest -q`
  - Result: `63 passed in 36.52s`
- Frontend production build:
  - Command: `cd frontend && npm run build`
  - Result: passed (`/deep-think` route generated)
- Frontend typecheck:
  - Command: `cd frontend && npx tsc --noEmit`
  - Result: passed

## 7. Risks / Follow-ups
- Event archive currently stores generated replay payloads, not raw model token stream chunks.
- Archive read API currently supports list/query only; no retention policy or purge endpoint yet.
- Next round can add:
  - session-level archive timeline filters (date/time/round/event type)
  - configurable retention and snapshot compaction strategy
  - UI link from conflict drill-down row to full citation context preview.
