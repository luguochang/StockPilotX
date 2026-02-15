# Round-H: DeepThink Round Visualization + Governance Panel

Date: 2026-02-15  
Scope: Frontend DeepThink workspace visualization and operator controls

## 1. Objective
- Turn backend DeepThink governance fields into explicit UI signals in `/deep-think`.
- Let users run and inspect deep rounds from one console:
  - round timeline
  - conflict source visualization
  - budget usage and remaining
  - replan/budget warning/stop reason states
- Expose internal A2A dispatch path for “next round” execution from frontend.

## 2. Design Decisions
- Reuse existing backend contracts without introducing new API schema:
  - `/v1/deep-think/sessions*`
  - `/v1/deep-think/sessions/{session_id}/stream`
  - `/v1/a2a/tasks`
- Keep “light query analysis” and “deep round governance” in one page but split by cards:
  - existing analysis cards stay unchanged
  - new DeepThink governance cards are inserted as an independent section.
- Treat stream events as a replay log (latest 80 events capped in memory), not a persistent datastore.

## 3. Implementation Details
- Added DeepThink domain types in `frontend/app/deep-think/page.tsx`:
  - `DeepThinkSession`, `DeepThinkRound`, `DeepThinkBudgetUsage`, `DeepThinkOpinion`, `DeepThinkTask`
- Added session/action state:
  - `deepSession`, `deepLoading`, `deepStreaming`, `deepError`, `deepStreamEvents`, `deepLastA2ATask`
- Added DeepThink control actions:
  - `startDeepThinkSession()`
  - `runDeepThinkRound()`
  - `runDeepThinkRoundViaA2A()`
  - `refreshDeepThinkSession()`
  - `replayDeepThinkStream()`
- Added SSE parser utility for reusable event consumption:
  - `readSSEAndConsume(...)`
  - event append helper with bounded queue (`appendDeepEvent`)
- Added governance visualization blocks:
  - round timeline card (`Timeline`)
  - latest round task graph table
  - budget progress bars (`Progress`) with warn/exceeded state
  - conflict chart (`ReactECharts`): disagreement score + conflict count per round
  - latest opinion table
  - stream event replay list

## 4. Changed Files
- `frontend/app/deep-think/page.tsx`
- `docs/rounds/2026-02-15/round-H-deepthink-round-visualization.md`
- `docs/agent-column/12-Round-H-DeepThink轮次可视化与治理看板实现记录.md`
- `docs/agent-column/00-总览-多agent-a2a-deepagent.md`
- `docs/governance/implementation-checklist.md`
- `docs/governance/spec-traceability-matrix.md`
- `docs/reports/2026-02-15/implementation-status-matrix-2026-02-15.md`

## 5. Checklist Mapping
- `FRONT-005` (new): DeepThink governance visualization on frontend.
- `GOV-004` (existing): per-round implementation documentation and traceability updates.

## 6. Self-test Evidence
- Backend regression:
  - Command: `.\.venv\Scripts\python -m pytest -q`
  - Result: `63 passed in 28.77s`
- Frontend production build:
  - Command: `cd frontend && npm run build`
  - Result: passed (`/deep-think` generated, compile success)
- Frontend typecheck:
  - Command: `cd frontend && npx tsc --noEmit`
  - Result: passed

## 7. Risks / Follow-ups
- Current stream replay list is in-memory only; page refresh clears history.
- A2A path currently exposes supervisor dispatch only; no custom card selection UI yet.
- Next round can add:
  - round-to-round diff panel (opinion changes)
  - conflict source drill-down by agent evidence mapping
  - persisted stream snapshots for audit replay.
