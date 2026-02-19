# ROUND-1 技术记录：Ops Health 一致性修复与数据源可见性增强

## 1. 本轮目标

围绕 `docs/v1/2026-02-19-DeepThink-数据源业务化增强计划.md` 的 ROUND-1，完成三件事：

1. 修复前端 `ops/health` 仍调用旧接口的问题。  
2. 将数据源补抓入口统一到 `datasources/fetch`，避免前端分散调用旧 ingest 接口。  
3. 输出“数据源 -> 页面模块”的可见性信息，解释每个数据源对业务页面的价值。

---

## 2. 关键改动

## 2.1 后端改动

文件：`backend/app/service.py`

1. 在数据源目录中补充 `used_in_ui_modules` 字段。  
2. 新增 `_datasource_ui_modules(category)` 映射函数。  
3. `datasource_health` 返回中新增：
   - `last_used_at`
   - `staleness_minutes`
   - `used_in_ui_modules`
4. scheduler 健康行同样增加上述可见性字段，保持返回结构一致。

## 2.2 前端改动

文件：`frontend/app/ops/health/page.tsx`

1. 页面请求改为新接口：
   - `/v1/datasources/sources`
   - `/v1/datasources/health`
   - `/v1/datasources/fetch`
2. 增加“按类别补抓”交互（行情/公告/财务/新闻/研报/宏观/资金/历史K线）。  
3. 增加“服务模块”列，直接显示 `used_in_ui_modules`，让运维知道该源影响哪些页面。  
4. 增加“新鲜度（分钟）”列，便于识别长期未调用的数据源。  
5. 保留原始接口返回区，便于排障。

## 2.3 测试改动

文件：
- `tests/test_service.py`
- `tests/test_http_api.py`

新增回归断言：
1. `datasources/sources` 的 `used_in_ui_modules` 字段类型校验。  
2. `datasources/health` 的 `last_used_at`、`staleness_minutes`、`used_in_ui_modules` 字段校验。

---

## 3. 数据源覆盖矩阵（当前基线）

| 数据源类别 | 主要后端入口 | 检索/语料轨道 | 主要页面消费 | 当前状态 |
|---|---|---|---|---|
| quote | `/v1/ingest/market-daily`、`/v1/datasources/fetch(category=quote)` | 行情快照/趋势特征 | `/deep-think` `/analysis-studio` `/predict` `/watchlist` | 已接入并可见 |
| announcement | `/v1/ingest/announcements`、`datasources/fetch` | 公告证据 | `/deep-think` `/reports` | 后端已接入，前端业务化待增强 |
| financial | `/v1/ingest/financials`、`datasources/fetch` | 财务快照因子 | `/deep-think` `/analysis-studio` `/predict` | 已接入并可见 |
| news | `/v1/ingest/news`、`datasources/fetch` | `news_event` | `/deep-think` `/analysis-studio` `/reports` | 已接入，结论映射待增强 |
| research | `/v1/ingest/research`、`datasources/fetch` | `research_report` | `/deep-think` `/analysis-studio` `/reports` | 已接入，结论映射待增强 |
| macro | `/v1/ingest/macro`、`datasources/fetch` | `macro_indicator` | `/deep-think` `/analysis-studio` | 已接入，事件日历待增强 |
| fund | `/v1/ingest/fund`、`datasources/fetch` | `fund_flow` | `/deep-think` `/analysis-studio` | 已接入，执行建议待增强 |
| history | `HistoryService`、`datasources/fetch(category=history)` | 历史 K 线特征 | `/deep-think` `/analysis-studio` `/predict` | 已接入并可见 |
| scheduler | `scheduler_status` | 运维状态 | `/ops/scheduler` `/ops/health` | 已接入并可见 |

> 注：表中“待增强”项对应 ROUND-2/ROUND-4 的业务化内容，不属于 ROUND-1 实施范围。

---

## 4. 自测记录

执行时间：2026-02-19

1. 后端 service 回归  
命令：
```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "datasource_ops_catalog_health_fetch_logs"
```
结果：`1 passed, 38 deselected`

2. HTTP API 契约回归  
命令：
```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "datasource_management_endpoints"
```
结果：`1 passed, 33 deselected`

3. 前端类型检查  
命令：
```powershell
cd frontend
npx tsc --noEmit
```
结果：通过（exit code 0）

---

## 5. 风险与后续

1. 当前 `used_in_ui_modules` 为静态映射，后续可引入“真实调用追踪”做自动化修正。  
2. 目前仅实现了“数据源可见性”，尚未形成“业务结论可执行卡片”（ROUND-2）。  
3. 建议下一轮优先落地 `intel-card` 聚合接口，避免页面继续以技术字段直出。
