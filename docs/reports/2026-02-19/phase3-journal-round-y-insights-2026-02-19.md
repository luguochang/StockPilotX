# Phase3 Journal Round-Y Report: Insights Backend

## 1. 目标

构建 Journal 聚合洞察能力，让用户可以直接看到最近窗口内的行为画像，而不是只看单条日志。

## 2. 交付范围

1. 聚合查询能力
- Journal 明细（含手工复盘覆盖、AI复盘覆盖标记）
- 日度时间线（journal/reflection/ai_reflection）

2. 业务洞察输出
- 类型分布（`journal_type`）
- 决策分布（`decision_type`）
- 标的活跃度（`stock_code`）
- 复盘覆盖率（手工 + AI）
- 关键词画像（标题/正文/tag）

3. API
- `GET /v1/journal/insights`

## 3. 关键实现点

- 统一输出结构：`count + ratio`，前端无需二次计算比例。
- 关键词策略：使用轻量正则分词 + 停用词过滤，兼顾中文与英文 token。
- 时间线合并：将三条事件流按日期聚合到同一结构，便于图表直接渲染。
- 参数约束：对 `window_days/limit/timeline_days` 做边界裁剪，避免误用导致慢查询。

## 4. 自测结论

- `tests/test_service.py -k "journal_lifecycle or journal_ai_reflection_generate_and_get or journal_insights"`  
  结果：`3 passed, 34 deselected`
- `tests/test_http_api.py -k "journal_endpoints or journal_ai_reflection_endpoints or journal_insights_endpoint"`  
  结果：`3 passed, 24 deselected`
- `tests -k "journal or api or web"`  
  结果：`31 passed, 69 deselected`

## 5. 下一轮衔接

Round-Z 将在 DeepThink 轮次结束后自动落库 Journal（幂等）并写入 `journal_linked` 流事件，让“分析结果 -> 复盘资产”形成闭环。
