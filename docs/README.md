# Docs Structure

## Purpose
This directory keeps project documentation organized by **artifact type** and **date**.

## Top-Level Rules
- Keep **core specs and governance docs** in `docs/` root.
- Put **date-based reports** under `docs/reports/YYYY-MM-DD/`.
- Put **optimization iteration notes** under `docs/optimization/YYYY-MM-DD/`.
- Put **round execution records** under `docs/rounds/YYYY-MM-DD/`.
- Put **topic summaries** under `docs/summaries/<topic>/YYYY-MM-DD/`.
- Put reusable references under `docs/references/`.

## Current Layout
- Core specs and governance (root):
  - `docs/a-share-agent-system-executable-spec.md`
  - `docs/a-share-agent-system-tech-solution.md`
  - `docs/free-data-source-implementation.md`
  - `docs/prompt-engineering-spec.md`
  - `docs/prompt-test-cases-template.md`
  - `docs/implementation-checklist.md`
  - `docs/spec-traceability-matrix.md`
  - `docs/global-constraints.md`
  - `docs/ops-runbook.md`
- Reports:
  - `docs/reports/2026-02-14/`
  - `docs/reports/2026-02-15/`
- Optimization:
  - `docs/optimization/2026-02-15/`
- Round records:
  - `docs/rounds/2026-02-15/`
- Summaries:
  - `docs/summaries/deepagent/2026-02-15/`
- References:
  - `docs/references/`

## Naming Conventions
- Report files: `<topic>-report-YYYY-MM-DD.md` or existing project-specific names.
- Date folders: always `YYYY-MM-DD`.
- Topic folders: lowercase kebab-case, for example `deepagent`.
