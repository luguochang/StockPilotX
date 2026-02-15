# Docs Structure

## Purpose
This directory keeps project documentation organized by **artifact type** and **date**.

## Top-Level Rules
- Keep `docs/` root minimal and use category subdirectories.
- Put specs under `docs/specs/`.
- Put governance and gate docs under `docs/governance/`.
- Put operations runbooks under `docs/operations/`.
- Put templates under `docs/templates/`.
- Put **date-based reports** under `docs/reports/YYYY-MM-DD/`.
- Put **optimization iteration notes** under `docs/optimization/YYYY-MM-DD/`.
- Put **round execution records** under `docs/rounds/YYYY-MM-DD/`.
- Put **execution checklists** under `docs/checklist/YYYY-MM-DD/`.
- Put **topic summaries** under `docs/summaries/<topic>/YYYY-MM-DD/`.
- Put reusable references under `docs/references/`.

## Current Layout
- Specs:
  - `docs/specs/README.md`
  - `docs/specs/a-share-agent-system-executable-spec.md`
  - `docs/specs/a-share-agent-system-tech-solution.md`
  - `docs/specs/free-data-source-implementation.md`
  - `docs/specs/prompt-engineering-spec.md`
- Templates:
  - `docs/templates/README.md`
  - `docs/templates/prompt-test-cases-template.md`
- Governance:
  - `docs/governance/README.md`
  - `docs/governance/implementation-checklist.md`
  - `docs/governance/spec-traceability-matrix.md`
  - `docs/governance/global-constraints.md`
- Operations:
  - `docs/operations/README.md`
  - `docs/operations/ops-runbook.md`
- Reports:
  - `docs/reports/README.md`
  - `docs/reports/2026-02-14/`
  - `docs/reports/2026-02-15/`
- Optimization:
  - `docs/optimization/README.md`
  - `docs/optimization/2026-02-15/`
- Round records:
  - `docs/rounds/README.md`
  - `docs/rounds/2026-02-15/`
- Checklist:
  - `docs/checklist/README.md`
  - `docs/checklist/2026-02-15/`
- Summaries:
  - `docs/summaries/README.md`
  - `docs/summaries/deepagent/README.md`
  - `docs/summaries/deepagent/2026-02-15/`
- Agent column:
  - `docs/agent-column/README.md`
- Java backend guide:
  - `docs/java-backend-guide/README.md`
- References:
  - `docs/references/README.md`
  - `docs/references/`

## Naming Conventions
- Report files: `<topic>-report-YYYY-MM-DD.md` or existing project-specific names.
- Date folders: always `YYYY-MM-DD`.
- Topic folders: lowercase kebab-case, for example `deepagent`.
- Each date folder should include a local `README.md` index.
