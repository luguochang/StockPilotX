# Round-M 实现记录：DeepThink 导出重试与审计分位指标

日期：2026-02-15  
对应轮次：Round-M  
关联范围：DeepThink archive 导出可靠性 + 审计观测 + 前端时间过滤体验

## 1. 本轮目标
- 让导出任务具备稳定的重试能力与尝试次数可追踪。
- 提升归档审计指标的信息密度（分位延迟、会话维度）。
- 降低时间过滤输入错误率，提高归档检索可用性。

## 2. 核心改动
- 导出任务可靠性：
  - 任务模型新增 `attempt_count/max_attempts`
  - 执行流程改为 `queued -> running` 原子 claim
  - 失败可重入队重试，达上限后标记 `failed`
- 审计指标增强：
  - 新增 `p50/p95/p99`、`slow_calls_over_1000ms`
  - 新增 `by_action_status`、`top_sessions`
- 前端 DeepThink 页优化：
  - 时间过滤改为 `datetime-local` 输入并自动规范化为后端格式
  - 增加 `最近24小时`、`清空时间过滤`
  - 导出任务状态标签新增 `attempt_count/max_attempts` 显示

## 3. 自测结果
- `\.venv\Scripts\python -m pytest -q tests/test_service.py tests/test_http_api.py` -> `29 passed`
- `\.venv\Scripts\python -m pytest -q` -> `65 passed`
- `cd frontend && npm run build` -> passed
- `cd frontend && npx tsc --noEmit` -> passed

## 4. 价值与后续
- 价值：导出链路从“可用”升级为“可恢复、可量化、可追踪”。
- 后续建议：
1. 将任务执行升级到外部队列，支撑多实例一致性。
2. 审计指标增加租户维度与长期窗口聚合。
3. 在前端增加时间过滤模板（最近7天/30天）和错误提示引导。
