# DeepThink Business Explanation Checklist (2026-02-15)

## Scope
- Make DeepThink console business-first in analysis mode.
- Keep engineering diagnostics complete but isolated in engineering mode.
- Clarify agent roles and signal/conflict semantics in Chinese.
- Improve interaction clarity: no advanced-analysis prerequisite, cross-stock state isolation.
- Run full self-test chain and record verifiable evidence.

## Delivery Rules
- [x] Every non-obvious code block includes comments for maintainability.
- [x] Each completed item is immediately marked as done.
- [x] Self-test output is recorded with concrete command evidence.
- [x] Round and report docs are updated for traceability.
- [x] Final commit includes both code and docs.

## Batch S1 - Console Semantics and Interaction
- [x] Add unified signal/priority/conflict label mapping helpers for business text.
- [x] Show agent Chinese display name + role in task/opinion/diff/drill tables.
- [x] Add analysis-mode usage guidance card (what to click and how to read outcomes).
- [x] Explicitly state that running next round auto-creates session (no advanced-analysis dependency).
- [x] Keep stock-switch auto-reset notice for DeepThink context isolation.

## Batch S2 - Mode Separation and Cognitive Load Reduction
- [x] In analysis mode, prioritize decision/risk/action and hide engineering noise.
- [x] In engineering mode, preserve full timeline/task graph/conflict/budget/replay capabilities.
- [x] Add business-oriented process summary card in analysis mode.
- [x] Keep conflict tags and decision labels human-readable (Chinese semantics).

## Batch S3 - End-to-End Validation
- [x] Frontend type check: `npx tsc --noEmit`
- [x] Frontend production build: `npm --prefix frontend run -s build`
- [x] Backend regression tests: `.\.venv\Scripts\python.exe -m pytest tests/test_service.py tests/test_http_api.py -q`
- [x] Real interface chain test:
  - `/v1/query/stream`
  - `/v1/deep-think/sessions`
  - `/v2/deep-think/sessions/{session_id}/rounds/stream`
  - `/v1/deep-think/sessions/{session_id}/business-export`

## Batch S4 - Documentation and Commit
- [x] Add round record for this implementation batch.
- [x] Add report with business/architecture rationale and self-test evidence.
- [x] Update checklist/round/report index files.
- [x] Commit with clear summary.

## Batch T - Agent Role Area Compression
- [x] Convert “Agent 角色说明” to default-collapsed block with animation.
- [x] Keep full role details available on demand (no information loss).
- [x] Validate with frontend production build.
- [x] Add Round-T doc and commit.

## Batch U - Agent Role Popover (Option 1)
- [x] Replace collapse card with compact `?` trigger + click popover.
- [x] Keep full role details in popover content.
- [x] Validate with frontend production build.
- [x] Add Round-U doc and commit.
