# Phase 0 Baseline Report Template

- generated_at: `{{generated_at}}`
- owner: `{{owner}}`
- env: `{{env}}`

## Core Metrics

- intent_accuracy: `{{intent_accuracy}}`
- retrieval_recall_at_k: `{{retrieval_recall_at_k}}`
- query_latency_p95_ms: `{{query_latency_p95_ms}}`
- query_failure_rate: `{{query_failure_rate}}`
- model_call_count_avg: `{{model_call_count_avg}}`
- cost_estimate_per_query: `{{cost_estimate_per_query}}`

## Observability Fields

- `intent_confidence`: `{{intent_confidence_coverage}}`
- `retrieval_track`: `{{retrieval_track_coverage}}`
- `model_call_count`: `{{model_call_count_coverage}}`
- `timeout_reason`: `{{timeout_reason_coverage}}`

## Mode Breakdown

| mode | sample_count | avg_latency_ms | fail_rate |
| --- | ---: | ---: | ---: |
| fact | {{fact_count}} | {{fact_latency}} | {{fact_fail_rate}} |
| deep | {{deep_count}} | {{deep_latency}} | {{deep_fail_rate}} |
| doc_qa | {{doc_count}} | {{doc_latency}} | {{doc_fail_rate}} |
| compare | {{compare_count}} | {{compare_latency}} | {{compare_fail_rate}} |

## Gate Recommendation

- Gate A (`2026-03-16`): `{{gate_a_recommendation}}`
- Gate B (`2026-04-13`): `{{gate_b_recommendation}}`

## Notes

- data source limitations: `{{limitations}}`
- regression result link: `{{regression_result_link}}`
- rollback plan: `{{rollback_plan}}`
