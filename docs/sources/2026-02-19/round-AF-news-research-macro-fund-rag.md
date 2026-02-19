# ROUND-AF：新闻/研报/宏观/基金数据源迁移与 RAG 对接

日期：2026-02-19

## 1. 目标与范围

本轮目标：完成 `news/research/macro/fund` 四类数据源接入，并把新闻/研报文本自动接入本地 RAG 索引，打通 `ingest -> query/query_stream -> market_overview` 的可消费链路。

## 2. 核心实现

### 2.1 数据源与工厂接入

- 扩展工厂构建器：
  - `backend/app/datasources/factory.py`
  - 新增 `build_default_news_service` / `build_default_research_service` / `build_default_macro_service` / `build_default_fund_service`
- 扩展导出入口：
  - `backend/app/datasources/__init__.py`

### 2.2 Ingestion 扩展

- 扩展存储与协议：
  - `backend/app/data/ingestion.py`
  - 新增 `news_items` / `research_reports` / `macro_indicators` / `fund_snapshots`
  - 新增 `ingest_news` / `ingest_research_reports` / `ingest_macro_indicators` / `ingest_fund_snapshots`
  - 增加对应规范化/质量校验函数

### 2.3 Service 业务链路接入

- 注入新增 datasource service：
  - `backend/app/service.py`
- 在 `query` / `query_stream` 的刷新阶段加入 news/research/fund/macro 刷新判断
- 新增新源刷新判断：
  - `_needs_news_refresh`
  - `_needs_research_refresh`
  - `_needs_macro_refresh`
  - `_needs_fund_refresh`
- 扩展运行时证据语料：
  - `_build_runtime_corpus` 增加 news/research/macro/fund 文本事实
- 扩展业务视图：
  - `market_overview` 返回 `news/research/fund/macro`
- 新增服务入口：
  - `ingest_news`
  - `ingest_research_reports`
  - `ingest_macro_indicators`
  - `ingest_fund_snapshots`
- 新增 `_index_text_rows_to_rag`：
  - 将新闻/研报结构化行按稳定签名映射为文档并执行 `docs_upload + docs_index`
  - 设计为 best-effort，不阻断摄取主流程

### 2.4 API 接口扩展

- 文件：`backend/app/http_api.py`
- 新增：
  - `POST /v1/ingest/news`
  - `POST /v1/ingest/research`
  - `POST /v1/ingest/macro`
  - `POST /v1/ingest/fund`

### 2.5 编码与可读性修复

- 文件：
  - `backend/app/datasources/news/service.py`
  - `backend/app/datasources/research/service.py`
- 修复 mock 文案乱码，并补充关键注释，明确兜底语义。

## 3. 测试与自测

### 3.1 新增/更新测试

- 新增：
  - `tests/test_datasource_intel_adapters.py`
  - `tests/test_ingestion_extended_sources.py`
- 更新：
  - `tests/test_datasources_factory.py`
  - `tests/test_service.py`
  - `tests/test_http_api.py`

### 3.2 执行命令

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_datasource_intel_adapters.py tests/test_ingestion_extended_sources.py tests/test_datasources_factory.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "ingest_endpoints or market_overview_contains_realtime_and_history"
.\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "ingest_market_daily or ingest_announcements or ingest_financials or ingest_news or ingest_research or ingest_macro or ingest_fund"
```

### 3.3 结果摘要

- `21 passed`（datasource + ingestion + factory）
- `2 passed, 36 deselected`（service 子集）
- `7 passed, 26 deselected`（http_api ingest 子集）

## 4. 设计取舍与风险

- 当前 RAG 入索引采用“按 stable signature 去重 + best-effort 入库”策略，优先保障 ingest 不被单条脏数据阻断。
- `news/research` 自动转文档后，文档数量会增长；下一轮建议配合 datasource 观测与清理策略（TTL/保留期）。
- `macro` 当前为全局证据，未与个股强绑定；后续可在重排阶段增加行业/主题映射权重。

## 5. 下一轮输入（ROUND-AG）

- 建立 `/v1/datasources/*` 运维 API：sources/health/fetch/logs
- 建立 source 级健康聚合、失败率统计、熔断/降级可观测输出
- 增加权限与 API 回归测试
