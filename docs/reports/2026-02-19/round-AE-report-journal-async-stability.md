# Round-AE Technical Summary: Report/Journal UX and Runtime Stabilization

Date: 2026-02-19

## Problem statement

After introducing DeepAgent/DeepThink and richer report flows, three practical issues showed up together:

1. Report generation felt like a long spinner with poor feedback.
2. Journal module asked too many fields up front for normal users.
3. Service runtime became unstable due inherited mojibake-corrupted literals in a large Python service file.

## Architecture decisions

1. Asynchronous report lifecycle (task-based)
- Switch report generation from single blocking request to task lifecycle.
- Expose status/result/cancel endpoints to support polling and progressive UI.
- Introduce `partial_ready` state with minimum viable output before full report finishes.

2. Journal input strategy (template-first)
- Keep user path simple by default (template + symbol + core viewpoint).
- Move optional parameters behind expert-mode, reducing cognitive load.

3. Stability-first repair policy
- For corrupted literals/docstrings, prefer minimal semantic changes:
  - fix syntax and keep behavior,
  - keep interfaces and tests stable,
  - avoid broad refactors during recovery.

## Key implementation points

1. Backend
- Added report task state machine and worker executor.
- Added sanitizers (`_sanitize_report_text`, `_sanitize_report_payload`) for response robustness.
- Added `_build_report_task_partial_result` for early usable output.
- Repaired service syntax corruption and restored DeepThink round/stream path behavior.

2. Frontend
- Reports page now orchestrates task creation, polling, partial/full rendering, and cancellation.
- Journal page now uses two-step flow and expert-mode collapse.

3. Quality and regression
- Restored compatibility with existing deep-think routes and tests.
- Updated lifecycle test expectation to accept `partial_ready` as a valid intermediate usable state.

## Verification

- `python -m py_compile backend/app/service.py backend/app/http_api.py`: pass
- `.venv\Scripts\python.exe -m pytest tests/test_service.py tests/test_http_api.py -q`: `80 passed`
- `frontend npm run build`: pass

## Outcome

- User waits now have actionable progress and early output in report center.
- Journal first-use friction reduced significantly.
- Backend returned to compilable/tested state with no known syntax regressions in changed paths.
