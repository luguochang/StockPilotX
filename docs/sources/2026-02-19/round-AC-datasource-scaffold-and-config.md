# ROUND-AC 技术记录：datasource 骨架与配置接入

> 日期：2026-02-19  
> 轮次：ROUND-AC  
> 关联计划：`docs/sources/2026-02-19/datasource-integration-master-plan.md`

## 1. 目标

本轮目标是完成数据源模块的第一阶段基础设施建设，不改动现有业务接口协议：

1. 建立 `backend/app/datasources/` 模块骨架。
2. 增加基础能力：协议定义、HTTP 客户端、工具函数、服务工厂。
3. 在配置层补齐 datasource 运行参数。
4. 在 `AShareAgentService` 构造路径中接入 datasource factory。

## 2. 关键实现

### 2.1 新增目录与文件

新增目录：`backend/app/datasources/` 及其子目录 `base/quote/financial/news/research/macro/fund/tests`。

新增核心文件：

1. `backend/app/datasources/factory.py`
2. `backend/app/datasources/base/adapter.py`
3. `backend/app/datasources/base/http_client.py`
4. `backend/app/datasources/base/utils.py`
5. 各子目录 `README.md` 与 `__init__.py`

### 2.2 工厂接入策略

在 ROUND-AC 中采用“稳定优先”的迁移策略：

1. 工厂层 `build_default_*_service` 已建立。
2. 当前工厂内部仍复用 `backend.app.data.sources` 的成熟实现，避免一次性替换导致回归。
3. 后续轮次再将具体 adapter 逐步迁入 `backend/app/datasources/*`。

### 2.3 配置扩展

`backend/app/config.py` 新增 datasource 相关配置项：

1. `datasource_request_timeout_seconds`
2. `datasource_retry_count`
3. `datasource_retry_backoff_seconds`
4. `datasource_proxy_url`
5. `datasource_xueqiu_cookie`
6. `datasource_tushare_token`
7. `datasource_tradingview_proxy_url`

并在 `Settings.from_env()` 中增加对应环境变量加载逻辑。

### 2.4 Service 注入改造

`backend/app/service.py` 中数据摄取服务初始化改为通过 factory 注入：

1. `build_default_quote_service(self.settings)`
2. `build_default_announcement_service(self.settings)`
3. `build_default_history_service(self.settings)`

该改造保持现有 API 行为不变，但为后续迁移建立了清晰的组合根入口。

## 3. 新增测试

### 3.1 新增测试文件

1. `tests/test_datasources_factory.py`
2. `tests/test_datasource_utils.py`

### 3.2 覆盖点

1. 工厂读取 `datasource_xueqiu_cookie` 并传递给 quote service。
2. announcement/history 工厂构造契约有效。
3. `normalize_stock_code` 标准化逻辑正确。
4. `decode_response` 的 UTF-8/GBK 回退逻辑正确。

## 4. 自测记录

执行命令：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_datasources_factory.py tests/test_datasource_utils.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_data_sources_live.py tests/test_announcements.py tests/test_ingestion_quality.py tests/test_doc_pipeline.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "ingest_endpoints"
.\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "ingest_market_daily or ingest_announcements"
```

结果摘要：

1. `7 passed`
2. `10 passed`
3. `1 passed, 37 deselected`
4. `2 passed, 26 deselected`

## 5. 风险与后续

当前风险：

1. 工厂层虽然接入完成，但具体 source adapter 仍在旧模块，尚未真正完成 provider 迁移。
2. `data.sources` 仍为大文件，后续需要在 ROUND-AD/AE/AF 分拆迁移时防止字段行为漂移。

下一轮输入（ROUND-AD）：

1. 开始将行情适配器逐个迁移到 `backend/app/datasources/quote/`。
2. 用工厂层切换到新适配器实现，并保留回退兜底。

