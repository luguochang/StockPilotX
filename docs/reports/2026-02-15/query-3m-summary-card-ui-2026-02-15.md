# Query 3M Summary Card UI Report (2026-02-15)

## Change
- Added fixed “最近三个月连续样本” card to the advanced analysis page (`/deep-think`).
- The card is independent from LLM text output and computed directly from `overview.history`.

## Why
- Users need immediate visibility on sample sufficiency and 3-month interval performance.
- Avoids relying on long-form answer parsing and prevents “looks sparse” UX confusion.

## Validation
- `npm --prefix frontend run -s build` passed.

## Notes
- This UI card complements (not replaces) the backend-injected 3-month context used in model prompts.
