# AGENTS.md

## Default Execution Policy (Project Root Scope)
Scope: `D:\fileSum\studyFile\openproject\designproject`

### 1. Auto-Approved (No Per-Command Confirmation Needed)
- Read/write/create/update/delete files under project scope.
- Create and maintain docs, specs, and architecture files.
- Run local checks and tests:
  - `python`, `pytest`
  - `npm`, `pnpm`, `yarn` install/build/test/lint/dev
  - `uv`, `pip` install project dependencies
- Run local services/dev servers for verification.
- Use git read-only commands:
  - `git status`, `git diff`, `git log`, `git show`, `git fetch`

### 2. Ask First (Explicit Confirmation Required)
- Potentially destructive actions:
  - `rm -rf` outside generated temp/cache cleanup
  - `git reset --hard`
  - `git checkout -- <file>`
  - history rewrites / force-push
- Commands affecting global/system configuration:
  - global package installs
  - system env vars
  - registry/system proxy changes
- Any action outside project scope.

### 3. Git Safety
- Do not amend commits unless explicitly requested.
- Do not revert user's existing unrelated changes.
- Prefer non-interactive git commands.

### 4. Delivery Mode
- Default mode: execute-first.
- Unless user explicitly requests planning-only, implement directly and report results.

### 5. High-Risk Exception
- If an action can cause irreversible data loss, always ask before execution.


