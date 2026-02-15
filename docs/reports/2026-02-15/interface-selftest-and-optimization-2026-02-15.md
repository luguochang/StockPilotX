# Interface Self-Test And Optimization Report (2026-02-15)

## Scope
- Interface-driven optimization for:
  - `/v1/query/stream` (高级分析)
  - `/v2/deep-think/sessions/{session_id}/rounds/stream` (执行下一轮)

## Method
- Start local API with `LLM_EXTERNAL_ENABLED=true`.
- Run real HTTP calls (not mock) and capture SSE events.
- Iterate code changes based on returned payloads and trace diagnostics.

## Findings
- Query stream had enough backend history (`history_sample_size=260`) but model output still tended to treat evidence as sparse points.
- DeepThink intel path had recoverable provider/tool issue and then parse fragility:
  - tool call returned `Unsupported tool type: web_search_preview`.
  - fallback call succeeded, but parser failed on `confidence_adjustment='down'`.

## Fixes
- Gateway now preserves upstream HTTP error body for diagnostics.
- Intel parser now accepts text confidence hints (`down/up/neutral`) and numeric extraction from strings.
- Query model input now injects a 3-month continuous K-line summary automatically.
- Runtime corpus now includes `eastmoney_history_3m_window` evidence item.
- History refresh now also checks minimum sample coverage.

## Post-Fix Verification
- `pytest tests/test_service.py tests/test_http_api.py -q` => `37 passed`.
- Query stream:
  - answer includes 3-month window interpretation with `eastmoney_history_3m_window` citation.
  - no sparse-sample misclassification text in sampled run.
- DeepThink stream:
  - `intel_status=external_ok`
  - `citations_count=5`
  - no fallback reason.
