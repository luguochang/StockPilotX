# Round Delivery Logs

## Purpose
- Record every implementation round with design intent, key decisions, code changes, and verification evidence.
- Keep history understandable for future maintainers by version and date.

## Required Structure Per Round
- Create one markdown file per round under a date folder:
  - `docs/rounds/YYYY-MM-DD/round-<tag>-<topic>.md`
- Every round log must include:
  - Objective and scope
  - Design decisions and tradeoffs
  - Changed files list
  - API/data model updates
  - Self-test commands and key results
  - Checklist/task mapping and completion status
  - Risks and next-step recommendations

## Commit Rule
- Each completed round must be committed once with:
  - round log file
  - code changes
  - checklist and traceability updates
