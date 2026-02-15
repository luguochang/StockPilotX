# Round U - DeepThink Agent Role Popover (Option 1)

## Goal
Apply the lightweight interaction requested by user: replace role block with a compact `?` entry.

## Implementation
- File: `frontend/app/deep-think/page.tsx`
- Replaced analysis-mode role explanation UI:
  - from: default-collapsed card (`Collapse`)
  - to: right-aligned `? 角色说明` trigger (`Popover`, click-to-open)
- Kept complete role details in the popover content.
- Added inline comment for maintainers:
  - role explanation is now on-demand to avoid occupying analysis-mode area.

## UX Effect
- Default viewport becomes cleaner.
- Business users stay focused on decision/risk/action cards.
- Role glossary is still one-click accessible.

## Self-Test
- `npm run -s build` (workdir: `frontend`) passed.
