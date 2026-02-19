# Round-AE Checklist: Report Async + Journal UX + Backend Stability

Date: 2026-02-19
Scope: Stabilize report async lifecycle, simplify journal UX, and repair backend runtime syntax regressions introduced by mojibake fragments.

## Execution Requirements

- [x] Add clear code comments for non-trivial logic paths.
- [x] Run self-tests after implementation.
- [x] Record key technical decisions and verification evidence.
- [x] Submit commit after checks pass.

## Backend

- [x] Add async report task APIs and service lifecycle (`queued/running/partial_ready/completed/failed/cancelled`).
- [x] Add partial-result fallback builder to improve long-running report UX.
- [x] Add report payload sanitization to remove common mojibake fragments.
- [x] Repair `backend/app/service.py` syntax regressions caused by malformed strings/docstrings.
- [x] Keep `/v1/deep-think/*` round APIs and `/v2` streaming path working after repairs.

## Frontend

- [x] Rework `reports` page to task-based async flow (create task -> poll -> fetch partial/full -> cancel).
- [x] Rework `journal` page to template-first two-step flow with expert-mode collapse.
- [x] Keep error messages readable and map backend failure codes to user-facing hints.

## Tests

- [x] `python -m py_compile backend/app/service.py backend/app/http_api.py`
- [x] `.venv\Scripts\python.exe -m pytest tests/test_service.py tests/test_http_api.py -q`
- [x] `frontend: npm run build`

## Commit Record

- [ ] Commit hash: (to fill after commit)
