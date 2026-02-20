# ROUND-R4：DeepThink 稳定性与超时治理

## 1. 目标
1. 为 DeepThink 单轮执行增加运行时守卫，避免长时间无反馈。
2. 提供可观测事件与结构化字段，明确超时原因与阶段。
3. 让超时场景输出“最小可用结果”并保持会话可继续。

## 2. 关键改动
- `backend/app/config.py`
  - 新增配置：
    - `deep_round_timeout_seconds`
    - `deep_round_stage_soft_timeout_seconds`
  - 支持环境变量：
    - `DEEP_ROUND_TIMEOUT_SECONDS`
    - `DEEP_ROUND_STAGE_SOFT_TIMEOUT_SECONDS`

- `backend/app/service.py`
  - `deep_think_run_round_stream_events(...)` 增加 runtime guard：
    - 运行时快照：`runtime_guard_snapshot(stage)`
    - 事件：`runtime_guard`（armed/warning）、`runtime_timeout`
    - 超时 stop reason：`DEEP_ROUND_TIMEOUT`
  - 在 `budget_usage.runtime_guard` 中落库治理信息：
    - `warn_emitted`
    - `timed_out`
    - `timeout_stage`
    - `elapsed_ms`
    - `round_timeout_seconds`
    - `stage_soft_timeout_seconds`
  - 在业务摘要中透出：
    - `runtime_guard`
    - `runtime_timeout`
  - 会话状态策略调整：
    - `DEEP_BUDGET_EXCEEDED` 视为终止轮次（session completed）
    - `DEEP_ROUND_TIMEOUT` 允许会话保持 `in_progress`（可继续下一轮）

## 3. 自测
执行命令：
- `.venv/Scripts/python.exe -m pytest -q tests/test_service.py -k "deep_think_runtime_timeout_guard or deep_think_budget_exceeded_stop"`
- `.venv/Scripts/python.exe -m pytest -q tests/test_http_api.py -k "deep_think_runtime_timeout_guard or deep_think_budget_exceeded"`
- `.venv/Scripts/python.exe -m pytest -q tests/test_service.py -k "predict or report or deep_think"`
- `.venv/Scripts/python.exe -m pytest -q tests/test_http_api.py -k "predict or report or deep_think"`

手工 TestClient 冒烟关键结果：
- 正常轮次：
  - `multi_role_pre.enabled = true`
  - 事件流包含 `pre_arbitration`
- 超时轮次（`round_timeout_seconds=0.1`）：
  - `stop_reason = DEEP_ROUND_TIMEOUT`
  - `status = in_progress`
  - 事件流包含 `runtime_timeout`
  - `budget_usage.runtime_guard.timed_out = true`

## 4. 影响
- 前端可直接基于 `runtime_guard` 渲染“执行中/超时降级”状态。
- 后续 R5 前端收口可以把 timeout 解释和下一步动作统一到一个提示条组件。
