# Round W - Phase3 Investment Journal MVP

日期：2026-02-19  
范围：启动 Phase3，第 1 批落地 Investment Journal 最小闭环

## Objective

- 按阶段规划继续推进，从 Phase2 进入 Phase3。
- 打通“记录决策 + 复盘沉淀”的后端可用能力。
- 保持现有 Phase1/2 已上线模块行为稳定。

## Implementation

1. 数据层扩展
- 在 `backend/app/web/store.py` 新增：
  - `investment_journal`
  - `journal_reflection`
- 补充查询索引，覆盖用户、股票、复盘维度。

2. 领域服务
- 在 `backend/app/web/service.py` 新增：
  - `journal_create`
  - `journal_list`
  - `journal_reflection_add`
  - `journal_reflection_list`
- 增加关键注释说明：
  - 权限边界（租户隔离）
  - 标签归一化策略
  - 类型校验与约束

3. 应用服务与 API
- `backend/app/service.py` 增加 Journal 封装方法。
- `backend/app/http_api.py` 增加 `/v1/journal*` 四个接口。

4. 测试
- `tests/test_service.py` 新增 `test_journal_lifecycle`。
- `tests/test_http_api.py` 新增 `test_journal_endpoints`。

## Verification

1. `.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "journal_lifecycle or portfolio_lifecycle or alert_rule_lifecycle_and_check or backtest_run_and_get"`
- 结果：`4 passed, 31 deselected`

2. `.\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "journal_endpoints or portfolio_endpoints or alert_rule_endpoints or backtest_endpoints"`
- 结果：`4 passed, 21 deselected`

3. `.\.venv\Scripts\python.exe -m pytest -q tests -k "web or api or query or docs or portfolio or alert or backtest or journal"`
- 结果：`41 passed, 55 deselected`

## Deliverables

- Checklist: `docs/checklist/2026-02-19/phase3-investment-journal-checklist.md`
- Execution Log: `docs/checklist/2026-02-19/phase3-investment-journal-execution-log.md`
- Report: `docs/reports/2026-02-19/phase3-investment-journal-mvp-2026-02-19.md`
