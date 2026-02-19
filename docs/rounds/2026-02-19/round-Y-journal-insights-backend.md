# Round Y - Journal Insights Backend

日期：2026-02-19  
目标：为 Investment Journal 提供聚合洞察接口，支撑“分布/活跃度/覆盖率/关键词”可视化。

## Implementation

1. 数据聚合层
- `backend/app/web/service.py`
  - 新增 `journal_insights_rows`：返回 Journal 主记录 + `reflection_count` + `has_ai_reflection` 标记。
  - 新增 `journal_insights_timeline`：按日聚合 `journal/reflection/ai_reflection` 三类活动计数。

2. 应用服务层
- `backend/app/service.py`
  - 新增 `journal_insights`：统一输出洞察数据结构，前端可直接消费。
  - 新增 `_journal_counter_breakdown`：输出 `count/ratio` 分布结构。
  - 新增 `_journal_extract_keywords`：基于标题/正文/tag 的轻量关键词统计。

3. API
- `backend/app/http_api.py`
  - 新增 `GET /v1/journal/insights`
  - 支持参数：`window_days`、`limit`、`timeline_days`

4. 测试
- `tests/test_service.py`
  - `test_journal_insights`
- `tests/test_http_api.py`
  - `test_journal_insights_endpoint`

## Verification

1. `.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "journal_lifecycle or journal_ai_reflection_generate_and_get or journal_insights"`
- 结果：`3 passed, 34 deselected`

2. `.\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "journal_endpoints or journal_ai_reflection_endpoints or journal_insights_endpoint"`
- 结果：`3 passed, 24 deselected`

3. `.\.venv\Scripts\python.exe -m pytest -q tests -k "journal or api or web"`
- 结果：`31 passed, 69 deselected`
