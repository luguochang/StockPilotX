# ROUND-AH：回归与交付收口

日期：2026-02-19

## 1. 交付范围（AC ~ AG）

本次 datasource 集成已完成以下能力闭环：

- 统一数据源模块骨架与配置注入（AC）
- 行情源迁移与回退链（AD）
- 财务源迁移与摄取落库（AE）
- 新闻/研报/宏观/基金迁移，新闻与研报自动 RAG 入索引（AF）
- 数据源管理与可观测接口（sources/health/fetch/logs），告警策略接入（AG）

## 2. 最终架构说明

### 2.1 数据源层

- 目录：`backend/app/datasources/`
- 模块：
  - `quote/`：Sina/Tencent/Netease/Xueqiu
  - `financial/`：Tushare/Eastmoney
  - `news/`：CLS/TradingView/Xueqiu News
  - `research/`：Eastmoney Research
  - `macro/`：Eastmoney Macro
  - `fund/`：TTJJ Fund Flow
- 工厂：`backend/app/datasources/factory.py`
  - 统一读取 runtime 配置（timeout/retry/proxy/token/cookie）

### 2.2 Ingestion 与证据构建

- `backend/app/data/ingestion.py`
  - 新增 `news_items/research_reports/macro_indicators/fund_snapshots`
  - 新增对应 ingest 入口与 payload 规范化/质量校验
- `backend/app/service.py`
  - query/query_stream 刷新逻辑接入新源
  - 运行时语料 `_build_runtime_corpus` 增加 news/research/macro/fund track
  - market_overview 返回扩展证据字段
  - 新闻/研报自动通过 `_index_text_rows_to_rag` 落入 docs 索引

### 2.3 运维 API 与观测

- `backend/app/http_api.py`
  - 新增：
    - `GET /v1/datasources/sources`
    - `GET /v1/datasources/health`
    - `POST /v1/datasources/fetch`
    - `GET /v1/datasources/logs`
- `backend/app/service.py`
  - catalog/health/fetch/logs 服务实现
  - source 健康聚合与 scheduler circuit 状态合并
  - 告警策略：
    - `datasource_failure_rate_high`
    - `datasource_fetch_failed`
    - `scheduler_circuit_open`

## 3. 端到端回归

### 3.1 执行命令

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "query_basic or query_stream or deep_think_v2_stream_round or rag_upload_workflow_and_dashboard or ingest_endpoints or datasource_ops_catalog_health_fetch_logs"
.\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "query_stream or deep_think_v2_round_stream or rag_asset_management_endpoints or datasource_management_endpoints or ingest_market_daily"
.\.venv\Scripts\python.exe -m pytest -q tests/test_datasource_intel_adapters.py tests/test_ingestion_extended_sources.py tests/test_datasources_factory.py
```

### 3.2 回归结果

- `6 passed`（service 关键链路子集）
- `5 passed`（http_api 关键链路子集）
- `21 passed`（datasource/ingestion/factory）

## 4. 兼容性与风险清单

- 数据源日志目前是进程内内存缓存（上限 5000），重启后清空。
- `datasource_fetch` 按 category 调度 ingest 链路，不是“强制指定单 adapter 直连”。
- 外部数据源稳定性受网络环境与第三方接口变更影响，仍依赖 fallback 与 mock 兜底。
- RAG 自动入索引会增加文档规模，后续建议补充 TTL/归档策略。

## 5. 运维使用说明

- 查看数据源目录：`GET /v1/datasources/sources`
- 查看健康状态：`GET /v1/datasources/health`
- 手动触发抓取：`POST /v1/datasources/fetch`
- 查询抓取日志：`GET /v1/datasources/logs`

推荐流程：

1. 先看 `sources` 确认目标 source_id/category
2. 用 `fetch` 手动触发一次验证
3. 再看 `logs` 与 `health` 判断失败率与告警信号

## 6. 本轮结论

本阶段数据源集成计划已按 AC~AH 目标收口，具备可用的数据接入、证据消费、运维观测与回归验证基础。后续可进入“单源精确调度 + 日志持久化 + 成本治理”优化阶段。
