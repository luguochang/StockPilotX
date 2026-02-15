# Round V - A-share Regime Guard (Query + DeepThink)

## Objective
实现 A 股“牛短熊长”市场状态处理，统一接入 `query` 与 `deep_think`，让结论在不同市场阶段下具备稳定、可解释的置信度约束。

## Implementation
- 新增配置与环境变量，支持策略阈值可调。
- 引入 `market_regime_context` 到 `AgentState`，支持跨组件传递。
- Workflow prompt 接入 `[a_share_regime]` 行与结构化 JSON。
- Service 新增：
  - `_build_a_share_regime_context`
  - `_apply_a_share_signal_guard`
- Query 输出新增字段：
  - `market_regime`
  - `regime_confidence`
  - `risk_bias`
  - `signal_guard_applied`
  - `signal_guard_detail`
- DeepThink 输出新增：
  - 流事件 `market_regime`
  - `business_summary` 的 regime/guard 字段。

## Verification
- `./.venv/Scripts/python.exe -m pytest tests/test_service.py tests/test_http_api.py -q` -> `43 passed`
- `cd frontend && npm run -s build` -> success
- 额外快速接口链路自测：query/deepthink 均返回 regime + guard 结果。

## Result
- 完成并通过测试。
- 具备后续前端“市场状态解释卡片”与“置信度变化提示”的数据基础。
