# Round-AC 技术记录：报告质量可视化与日志模板化输入

日期: 2026-02-19

## 目标

围绕“页面输入过多、用户不知道如何使用、结果缺少业务解释”三个痛点，完成以下落地:

1. 报告模块把质量门禁与数据充分性直接展示给用户，而非只显示原始 JSON。
2. 报告模块补充业务级数据健康快照，明确当前结论是否建立在可靠数据基础上。
3. 投资日志模块改造为模板优先模式，减少首屏输入负担并保留专家模式扩展能力。

## 核心改动

### 1) Reports 页面业务化

文件: `frontend/app/reports/page.tsx`

- 新增质量门禁卡片:
  - `generation_mode`
  - `quality_gate.status/score/reasons`
  - 质量得分进度条
- 新增数据包摘要卡片:
  - `history_sample_size`
  - `predict_quality`
  - `intel_signal/intel_confidence`
  - `news_count/research_count/macro_count`
- 新增业务数据健康卡片:
  - 调用 `GET /v1/business/data-health`
  - 展示全局状态、reports 模块覆盖率、标的快照(quote/history/financial)
- 交互优化:
  - 主流程输入仅保留“股票 + 报告类型”
  - 高级参数移入折叠区（`template_id/run_id/pool_snapshot_id`）

### 2) Journal 页面输入重构

文件: `frontend/app/journal/page.tsx`

- 模板优先输入:
  - `decision / risk / review` 三类模板
  - 自动给出默认 `journal_type/decision_type/sentiment/tags`
  - 自动生成标题和内容骨架（触发条件/失效条件/执行计划）
- 高级项折叠:
  - `custom_title/custom_tags/journal_type/decision_type/sentiment` 放入可选区
- 能力保持:
  - 列表筛选、手工复盘、AI 复盘、洞察看板仍可完整使用

## 业务收益

1. 用户不再需要理解内部字段即可开始使用报告和日志能力。
2. 报告结论会同时展示“质量与覆盖”上下文，降低误读风险。
3. 日志结构更统一，便于后续做复盘统计与策略行为分析。

## 自测结果

1. `.\.venv\Scripts\python.exe -m pytest tests/test_service.py -k "report_generate_and_get or prediction_run_and_eval or datasource_ops_catalog_health_fetch_logs"`  
结果: `3 passed, 39 deselected`

2. `.\.venv\Scripts\python.exe -m pytest tests/test_http_api.py -k "test_report_generate_and_get or test_predict_endpoints or test_datasource_management_endpoints"`  
结果: `3 passed, 33 deselected`

3. `cd frontend && npm run build`  
结果: `Compiled successfully`

4. `cd frontend && npx tsc --noEmit`  
结果: 通过（顺序执行无类型错误）

