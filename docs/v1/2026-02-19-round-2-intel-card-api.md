# ROUND-2 技术记录：Intel Card 业务聚合接口

## 1. 本轮目标

在已有多数据源接入基础上，提供一个“业务可消费”的统一接口，避免前端逐个拼接技术字段。  
本轮交付 `GET /v1/analysis/intel-card`，输出：

1. 结论（signal/confidence/risk_level/position_hint）  
2. 证据（evidence）  
3. 风险与催化（risk_watch/key_catalysts）  
4. 事件日历（event_calendar）  
5. 情景矩阵（scenario_matrix）  

---

## 2. 改动清单

## 2.1 服务层

文件：`backend/app/service.py`

新增 `analysis_intel_card(stock_code, horizon, risk_profile)`，核心逻辑：

1. 基于 `market_overview` 统一拉取实时行情、趋势、公告、新闻、研报、宏观、资金。  
2. 将新闻/研报/宏观/资金/公告转换为标准证据结构（含 source/time/url/reliability/retrieval_track）。  
3. 用“趋势 + 基本面 + 资金 + 证据极性”进行打分，映射 `buy/hold/reduce`。  
4. 输出可执行字段：`position_hint`、`trigger_conditions`、`invalidation_conditions`、`next_review_time`。  
5. 输出 `data_freshness` 以显示每类数据最近更新时延。

> 说明：本轮采用规则融合，后续 ROUND 将把该结构对接更强的 RAG 重排和 DeepThink 轮次解释。

## 2.2 API 层

文件：`backend/app/http_api.py`

新增接口：

- `GET /v1/analysis/intel-card?stock_code=SH600000&horizon=30d&risk_profile=neutral`

参数校验：
- `horizon` 仅允许 `7d/30d/90d`
- `risk_profile` 仅允许 `conservative/neutral/aggressive`
- 非法参数返回 `400`

## 2.3 测试

文件：
- `tests/test_service.py`
- `tests/test_http_api.py`

新增用例：
1. service 级 contract 测试：字段完整性、范围校验。  
2. API 级契约测试：正常响应 + 非法 horizon 的 400 分支。

---

## 3. 接口示例（缩略）

```json
{
  "stock_code": "SH600000",
  "time_horizon": "30d",
  "risk_profile": "neutral",
  "overall_signal": "hold",
  "confidence": 0.67,
  "risk_level": "medium",
  "position_hint": "20-35%",
  "key_catalysts": [],
  "risk_watch": [],
  "event_calendar": [],
  "scenario_matrix": [],
  "evidence": [],
  "trigger_conditions": [],
  "invalidation_conditions": [],
  "next_review_time": "2026-02-20Txx:xx:xx+00:00",
  "data_freshness": {
    "quote_minutes": 2,
    "financial_minutes": 1440
  }
}
```

---

## 4. 自测记录

执行时间：2026-02-19

1. service 回归  
命令：
```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "datasource_ops_catalog_health_fetch_logs or analysis_intel_card_contract"
```
结果：`2 passed, 38 deselected`

2. API 回归  
命令：
```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "datasource_management_endpoints or analysis_intel_card"
```
结果：`2 passed, 33 deselected`

---

## 5. 风险与后续

1. 当前打分仍是规则融合，需在后续 ROUND 接入更强的“检索重排 + 轮次解释”以提高稳定性。  
2. 目前还未接入前端 Intel Card 专属展示模块，下一步应在 DeepThink 分析模式中优先消费此接口。  
3. `used_in_ui_modules` 仍为静态映射，后续可升级为真实调用埋点驱动。
