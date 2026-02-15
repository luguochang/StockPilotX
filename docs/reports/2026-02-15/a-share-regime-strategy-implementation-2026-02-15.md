# A股“牛短熊长”策略实现记录（2026-02-15）

## 背景
当前 DeepAgent/DeepThink 在市场环境切换时，容易出现“方向合理但置信度过激”的问题。A 股典型特征是“牛短熊长、波动快、情绪切换频繁”，需要在不改变主方向判断的前提下，对信号置信度做制度化约束。

## 目标
- 在 `query` 与 `deep_think` 两条链路统一引入市场状态（regime）上下文。
- 将“牛短熊长”特征显式化，变成可追踪字段与可解释事件。
- 落地“方向不变、置信度折扣”的 guard 机制，避免结论过激。

## 技术实现
1. 配置层
- `backend/app/config.py` 新增：
  - `a_share_regime_enabled`
  - `a_share_regime_vol_threshold`
  - `a_share_regime_conf_discount_bear`
  - `a_share_regime_conf_discount_range`
  - `a_share_regime_conf_discount_bull_high_vol`

2. 上下文层
- `backend/app/state.py` 新增 `market_regime_context`，在 workflow/service/deepthink 之间共享。
- `backend/app/agents/workflow.py`：
  - `_prepare_state` 保留外部注入的 retrieval_plan 元数据。
  - `_analyze` 输出 `market_regime` 摘要。
  - `_build_prompt` 将 `[a_share_regime]` 注入 evidence，并传递结构化 `market_regime` JSON 给 prompt renderer。

3. 规则引擎层
- `backend/app/service.py` 新增 `_build_a_share_regime_context`：
  - 输入：股票代码及历史行情。
  - 特征：`trend_5d/trend_20d/vol_20d/drawdown_20d/up_day_ratio_20d/gap_risk_flag`。
  - 输出：`regime_label`（`bull_burst`/`bear_grind`/`range_chop`/`rebound_probe`）、`risk_bias`、`regime_confidence`、`action_constraints`。
- 新增 `_apply_a_share_signal_guard`：
  - 保持 `signal` 原方向。
  - 根据 regime 对 `confidence` 打折并应用上限 cap。

4. 链路接入
- `query`：注入 regime context，增强 `analysis_brief`（regime 与 guard 字段）。
- `query_stream_events`：新增 `market_regime` 流事件，结尾 `analysis_brief` 含 guard 信息。
- `deep_think_run_round_stream_events`：
  - 开始阶段与数据刷新后都会发送 `market_regime` 事件。
  - `business_summary` 接入 guard 后字段：`market_regime/regime_confidence/signal_guard_applied/confidence_adjustment_detail`。

## 业务价值
- 从“黑盒结论”升级为“可解释结论”：用户可看到当前市场标签、风险偏置、置信度如何被约束。
- 避免熊市/震荡期信号过度自信，降低错误追涨杀跌风险。
- 为前端策略解释与风控提示提供结构化字段，支持后续 UI 设计（卡片、提示、对比）。

## 自测结果
- 单元与接口测试：`43 passed`。
- 前端构建：`next build` 成功。
- 代码内快速自测：
  - query 返回 `analysis_brief.market_regime` 与 `signal_guard_applied`。
  - deep-think 流事件包含 `market_regime`。
  - business_summary 返回 guard 字段。

## 影响文件
- `backend/app/config.py`
- `backend/app/state.py`
- `backend/app/agents/workflow.py`
- `backend/app/service.py`
- `tests/test_service.py`
- `tests/test_http_api.py`
