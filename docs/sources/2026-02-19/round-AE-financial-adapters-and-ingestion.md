# ROUND-AE 技术记录：财务适配器迁移与摄取链路接入

> 日期：2026-02-19  
> 轮次：ROUND-AE  
> 关联计划：`docs/sources/2026-02-19/datasource-integration-master-plan.md`

## 1. 目标

完成财务数据源的首版迁移，并让 Query/DeepThink 可以消费财务证据：

1. 迁移 Tushare 和 Eastmoney 财务适配器。
2. 建立财务快照服务与回退策略。
3. 将财务快照纳入 `IngestionService` 标准化落库。
4. 在 Query/DeepThink 刷新流程中接入财务摄取。

## 2. 核心实现

### 2.1 新增财务模块

新增文件：

1. `backend/app/datasources/financial/common.py`
2. `backend/app/datasources/financial/tushare.py`
3. `backend/app/datasources/financial/eastmoney.py`
4. `backend/app/datasources/financial/service.py`

能力说明：

1. `TushareFinancialAdapter`：基于 token 拉取 `fina_indicator` 快照。
2. `EastmoneyFinancialAdapter`：通过公开接口补充估值与同比字段。
3. `FinancialService`：按 `tushare -> eastmoney -> mock` 回退链路执行。

### 2.2 基础 HTTP 客户端增强

`backend/app/datasources/base/http_client.py` 新增：

1. `post_json_bytes(...)`  
2. 统一 `_request_bytes(...)` 重试入口（GET/POST 复用）

用于支持 Tushare 的 POST JSON 协议调用。

### 2.3 Ingestion 扩展

`backend/app/data/ingestion.py` 新增：

1. `IngestionStore.financial_snapshots`
2. `financial_service` 注入
3. `ingest_financials(...)` 方法
4. 财务 payload 标准化与质量校验

质量规则包含：

1. 必填字段校验（`stock_code/ts/source_id/source_url`）
2. 可靠性阈值校验
3. 估值字段缺失标记（`valuation_missing`）

### 2.4 Service / API 接入

改动：

1. `backend/app/datasources/factory.py` 增加 `build_default_financial_service(...)`
2. `backend/app/service.py` 的 `IngestionService` 初始化注入 financial service
3. 新增 `AShareAgentService.ingest_financials(...)`
4. `backend/app/http_api.py` 新增 `POST /v1/ingest/financials`

### 2.5 Query / DeepThink 证据接入

改动点：

1. Query 与 stream refresh 阶段新增 `fin_refresh`。
2. `_build_runtime_corpus(...)` 新增财务快照文本证据条目。
3. `market_overview(...)` 返回 `financial` 字段。
4. DeepThink 相关路径（包括自检与 debate）增加财务刷新与快照透出。

## 3. 新增测试

新增文件：

1. `tests/test_datasource_financial_adapters.py`
2. `tests/test_ingestion_financial.py`

更新文件：

1. `tests/test_datasources_factory.py`
2. `tests/test_service.py`
3. `tests/test_http_api.py`

覆盖重点：

1. Tushare/Eastmoney 字段解析正确性。
2. FinancialService 回退可用性。
3. Ingestion 财务落库与未配置路径。
4. `service/http_api` 财务摄取入口。

## 4. 自测记录

执行命令：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_datasource_financial_adapters.py tests/test_ingestion_financial.py tests/test_datasources_factory.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "ingest_endpoints"
.\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "ingest_market_daily or ingest_announcements or ingest_financials"
.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "query_basic"
```

结果摘要：

1. `9 passed`
2. `1 passed, 37 deselected`
3. `3 passed, 26 deselected`
4. `1 passed, 37 deselected`

## 5. 风险与下一步

当前风险：

1. 财务字段口径还未完全统一（不同数据源定义差异大）。
2. 某些财务字段在公开接口中覆盖不稳定，仍需长期监控 source 健康度。

下一轮（ROUND-AF）重点：

1. 新闻/研报/宏观/基金多源迁移。
2. 长文本清洗后进入 RAG 的结构化流程。
3. 为后续 datasource 管理 API 提供数据基础。

