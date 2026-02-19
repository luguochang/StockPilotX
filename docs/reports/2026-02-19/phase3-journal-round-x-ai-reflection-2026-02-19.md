# Phase3 Journal Round-X Report: AI Reflection Backend

## 1. 目标

为 Investment Journal 增加可复用的 AI 复盘能力，避免“只有手工复盘、无结构化洞察”的状态。

## 2. 交付范围

1. 持久化层
- 新增 `journal_ai_reflection` 表，保存每条 Journal 的最新 AI 复盘结果。

2. 业务服务
- 支持按 Journal 生成 AI 复盘（优先外部模型，失败自动 fallback）。
- 支持查询最新 AI 复盘结果。
- 输出统一结构：`summary / insights[] / lessons[] / confidence`。

3. API
- `POST /v1/journal/{journal_id}/ai-reflection/generate`
- `GET /v1/journal/{journal_id}/ai-reflection`

## 3. 关键实现点

- 严格 JSON 输出约束：Prompt 强制模型返回单对象 JSON，降低解析不确定性。
- 防御性解析：沿用 `service._deep_safe_json_loads`，兼容 fenced code 与杂文本。
- 失败降级：模型失败时返回本地 fallback 复盘，保证接口始终可用。
- 可诊断字段：存储 `provider/model/trace_id/error_code/error_message`，便于排查。

## 4. 自测结论

- `tests/test_service.py -k "journal_lifecycle or journal_ai_reflection_generate_and_get"`  
  结果：`2 passed, 34 deselected`
- `tests/test_http_api.py -k "journal_endpoints or journal_ai_reflection_endpoints"`  
  结果：`2 passed, 24 deselected`
- `tests -k "journal or api or web"`  
  结果：`29 passed, 69 deselected`

## 5. 下一轮衔接

Round-Y 将基于 Journal + Reflection + AI Reflection 构建聚合洞察接口，输出复盘覆盖率、决策分布、活跃度和关键词画像。
