# Round Z - DeepThink Auto Journal Link Backend

日期：2026-02-19  
目标：让 DeepThink 每轮分析结果自动沉淀为 Journal 资产，并通过流事件可感知。

## Implementation

1. DeepThink -> Journal 自动落库
- `backend/app/service.py`
  - 新增 `_deep_journal_related_research_id`，以 `session_id + round_id` 生成幂等键。
  - 新增 `_deep_build_journal_from_business_summary`，将业务摘要映射为 Journal 结构化内容。
  - 新增 `_deep_auto_link_journal_entry`，自动创建或复用 Journal，避免重复写入。

2. 流事件扩展
- `backend/app/service.py`
  - 在 `deep_think_run_round_stream_events` 中新增 `journal_linked` 事件。
  - 事件数据包含：`action(created/reused/failed)`、`journal_id`、`related_research_id`。

3. 事件重放兜底
- `backend/app/service.py`
  - 在 `_build_deep_think_round_events` 中补充 `journal_linked` 重建逻辑，保证事件回放一致性。

4. 测试更新
- `tests/test_service.py`
  - `test_deep_think_session_and_round`：断言 `journal_linked` 事件与落库结果。
  - `test_deep_think_v2_stream_round`：断言 `journal_linked` 事件与幂等复用。
- `tests/test_http_api.py`
  - `test_deep_think_and_a2a`：断言 SSE 回放包含 `journal_linked`。
  - `test_deep_think_v2_round_stream`：断言 V2 流式接口包含 `journal_linked`。

## Verification

1. `.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "deep_think_session_and_round or deep_think_v2_stream_round"`
- 结果：`3 passed, 34 deselected`

2. `.\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "deep_think_and_a2a or deep_think_v2_round_stream"`
- 结果：`2 passed, 25 deselected`

3. `.\.venv\Scripts\python.exe -m pytest -q tests -k "deep_think or journal or api or web"`
- 结果：`39 passed, 61 deselected`
