# A-Share Regime Rollout Checklist (2026-02-15)

## Scope
- Goal: 落地 A 股“牛短熊长”策略处理，覆盖 `query` 与 `deep_think` 两条链路。
- Rule: 保持信号方向不变，只做置信度折扣与上限约束。

## Checklist
- [x] 在 `backend/app/config.py` 增加 A-share regime 开关与阈值参数（支持环境变量）。
- [x] 在 `backend/app/state.py` 增加 `market_regime_context`，支持链路共享上下文。
- [x] 在 `backend/app/agents/workflow.py` 注入 regime 到 analysis 与 prompt 渲染变量。
- [x] 在 `backend/app/service.py` 实现 `_build_a_share_regime_context`（1-20 日特征/标签/约束）。
- [x] 在 `backend/app/service.py` 实现 `_apply_a_share_signal_guard`（方向不变，仅调置信度）。
- [x] `query` 链路接入 regime context，并写入 `analysis_brief` 的 regime/guard 字段。
- [x] `query_stream_events` 接入 `market_regime` 事件并输出增强 `analysis_brief`。
- [x] `deep_think_run_round_stream_events` 接入 `market_regime` 事件并复评刷新后状态。
- [x] `deep_think` 业务摘要接入 guard 后字段：`market_regime` / `signal_guard_applied` 等。
- [x] 更新测试：`tests/test_service.py`、`tests/test_http_api.py`。
- [x] 自测通过：`pytest` 关键用例 + 前端 `npm run -s build`。
- [x] 补充接口级自测：query/deepthink 返回包含 regime/guard 字段。

## Verification
- `./.venv/Scripts/python.exe -m pytest tests/test_service.py tests/test_http_api.py -q`
- `cd frontend && npm run -s build`

## Status
- Completed.
