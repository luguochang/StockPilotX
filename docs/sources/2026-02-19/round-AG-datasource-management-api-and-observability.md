# ROUND-AG：数据源管理 API 与可观测性

日期：2026-02-19

## 1. 目标与范围

本轮目标：落地 `/v1/datasources/*` 运维接口，提供数据源目录、健康状态、手动触发与日志追踪能力，并接入基础告警策略。

## 2. 核心实现

### 2.1 Service 侧能力

文件：`backend/app/service.py`

新增能力：

- 数据源目录构建：`_build_datasource_catalog`
  - 从当前注入的 adapters 生成 source metadata
  - 输出 category / enabled / reliability_score / source_url / proxy_enabled
  - 补充 `history` 的 synthetic source
- 运维日志：`_append_datasource_log`
  - 记录 `source_id/category/action/status/latency/error/detail`
  - 采用内存 ring buffer（上限 5000）
- 对外接口：
  - `datasource_sources`
  - `datasource_health`
  - `datasource_fetch`
  - `datasource_logs`
- 分类推断：`_infer_datasource_category`
  - 支持从 `source_id` 自动推断 quote/news/research/macro/fund/history 等类别

### 2.2 健康聚合与告警策略

- `datasource_health` 聚合逻辑：
  - 基于 datasource logs 统计 attempts/success_rate/failure_rate/last_error/last_latency
  - 同步 upsert 到 `source_health`（复用现有表）
  - 合并 scheduler circuit 状态
- 告警触发策略：
  - 当单源 `attempts >= 3` 且 `failure_rate >= 0.6`，触发 `datasource_failure_rate_high`
  - 当 scheduler circuit 打开时，触发 `scheduler_circuit_open`
  - 当手动 fetch 失败时，触发 `datasource_fetch_failed`

### 2.3 API 路由

文件：`backend/app/http_api.py`

新增接口：

- `GET /v1/datasources/sources`
- `GET /v1/datasources/health`
- `POST /v1/datasources/fetch`
- `GET /v1/datasources/logs`

接口均复用 token 解析与统一鉴权错误处理，兼容当前 auth_bypass 开发模式。

### 2.4 测试

更新文件：

- `tests/test_service.py`
  - 新增 `test_datasource_ops_catalog_health_fetch_logs`
- `tests/test_http_api.py`
  - 新增 `test_datasource_management_endpoints`

## 3. 自测记录

### 3.1 执行命令

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "datasource_ops_catalog_health_fetch_logs"
.\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "datasource_management_endpoints"
.\.venv\Scripts\python.exe -m pytest -q tests/test_datasource_intel_adapters.py tests/test_ingestion_extended_sources.py tests/test_datasources_factory.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "ingest_endpoints or market_overview_contains_realtime_and_history or datasource_ops_catalog_health_fetch_logs"
.\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "ingest_market_daily or ingest_announcements or ingest_financials or ingest_news or ingest_research or ingest_macro or ingest_fund or datasource_management_endpoints"
```

### 3.2 结果摘要

- `1 passed`（service datasource 管理）
- `1 passed`（http_api datasource 管理）
- `21 passed`（adapter/ingestion/factory）
- `9 passed`（service/http_api ingest + datasource 子集）

## 4. 风险与后续

- 当前 datasource logs 为内存级，进程重启会丢失；如需审计追踪需下轮持久化。
- 目前 `datasource_fetch` 按 category 调用 ingest 链路，未做“强制指定单 adapter”执行；如需 source 级精确重放需进一步拆分 adapter 调度器。

## 5. 下一轮输入（ROUND-AH）

- 执行端到端回归（ingest -> query/deepthink -> rag）
- 完成交付收口文档（能力清单、风险清单、运维说明）
- 收敛 checklist 全量状态并完成最终提交
