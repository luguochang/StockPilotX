# DeepThink Full Support Checklist (2026-02-15)

## Scope
- DeepThink console redesign (analysis/engineering mode, three-layer feedback)
- LLM WebSearch-based real-time intelligence integration
- Business-meaningful summary and decision fusion
- CSV export usability fix (UTF-8 BOM) and business export route
- Documentation, self-test, and commit discipline

## Global Rules
- [x] Create checklist and execution-log docs under `docs/checklist/2026-02-15/`
- [x] Every code change includes clear comments for readability
- [x] Each finished implementation item is marked done in this checklist
- [x] Run self-test for each implementation batch and record results
- [x] Commit after each completed batch with clear commit message

## Batch A - Console UX Redesign
- [x] Add `analysis` / `engineering` view mode in DeepThink page
- [x] Keep engineering controls in engineering mode only
- [x] Add three-layer feedback in analysis mode: action state, stage progress, result summary
- [x] Add clear result cards for business users (decision/risk/next steps)

## Batch B - LLM WebSearch Intelligence
- [x] Add backend prompt builder for intelligence retrieval via LLM WebSearch
- [x] Add backend parser/validator for strict JSON intelligence payload
- [x] Inject intelligence events into round SSE pipeline (`intel_snapshot`, `calendar_watchlist`, `business_summary`)
- [x] Add fallback path when intelligence payload is unavailable or invalid

## Batch C - Decision Fusion
- [x] Merge intelligence outputs into final deep-think business summary
- [x] Emit explicit decision fields: signal/confidence/trigger/invalidation/review_time
- [x] Show intelligence + fusion summary in frontend analysis mode

## Batch D - Export Usability
- [x] Fix existing event CSV encoding with UTF-8 BOM (Excel-compatible)
- [x] Add business export endpoint and data shape for readable outputs
- [x] Add frontend business export action and keep audit export path

## Batch E - QA, Docs, Commit
- [x] Extend or update backend tests for new intelligence + export behavior
- [x] Extend or update API tests for new routes/events
- [x] Update round docs and report docs with implementation notes
- [x] Final self-test run and result capture
- [x] Final commit(s) with referenced checklist/log

## Batch F - Intel Observability and Self-Test
- [x] Add stable fallback reason codes for intelligence downgrade path
- [x] Add DeepThink intel self-test API for one-shot diagnostics
- [x] Add trace event query API for intelligence chain debugging
- [x] Expose intel status/reason/trace fields in stream events and frontend cards
- [x] Verify with backend tests + frontend build + manual self-test output
