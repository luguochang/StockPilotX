# ROUND-AD 技术记录：行情适配器迁移与回退链

> 日期：2026-02-19  
> 轮次：ROUND-AD  
> 关联计划：`docs/sources/2026-02-19/datasource-integration-master-plan.md`

## 1. 目标

在不破坏现有 `ingest/query` 链路的前提下，将行情数据源能力迁移到新模块：

1. 落地 `tencent/netease/sina/xueqiu` 四个 adapter。
2. 新增统一 `QuoteService`（source 级 fallback + mock 兜底）。
3. 切换 `datasources.factory` 到新 `QuoteService`。

## 2. 关键改动

### 2.1 新增 quote 子模块实现

新增文件：

1. `backend/app/datasources/quote/models.py`
2. `backend/app/datasources/quote/common.py`
3. `backend/app/datasources/quote/tencent.py`
4. `backend/app/datasources/quote/netease.py`
5. `backend/app/datasources/quote/sina.py`
6. `backend/app/datasources/quote/xueqiu.py`
7. `backend/app/datasources/quote/service.py`

实现说明：

1. 统一输出 `Quote` dataclass，兼容 ingestion `asdict(quote)` 处理路径。
2. 每个 adapter 使用 `base/http_client.py` 与 `base/utils.py`。
3. 引入编码回退（UTF-8 -> GBK -> GB18030），减少 provider 编码差异造成的解析失败。
4. `xueqiu` 明确 cookie 依赖，无 cookie 时主动抛错并交给 fallback 链处理。

### 2.2 回退链与离线可用性

`QuoteService` 回退顺序：

1. Tencent
2. Netease
3. Sina
4. Xueqiu
5. 三个 deterministic mock adapter

这样可以保证在外网不可用时，本地开发与测试链路仍可持续运行。

### 2.3 工厂切换

`backend/app/datasources/factory.py` 已改为调用新 `quote/service.py`：

1. 使用 `Settings` 的 datasource 配置传入 timeout/retry/backoff/proxy/cookie。
2. 保持 `build_default_quote_service` 的对外签名不变，降低上层耦合。

## 3. 测试与回归

### 3.1 新增测试

新增：`tests/test_datasource_quote_adapters.py`

覆盖：

1. Tencent 解析
2. Netease 解析
3. Sina 解析（GBK）
4. Xueqiu cookie 缺失错误路径
5. QuoteService fallback 路径

### 3.2 回归命令

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_datasource_quote_adapters.py tests/test_datasources_factory.py tests/test_datasource_utils.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "ingest_endpoints"
.\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "ingest_market_daily or ingest_announcements"
.\.venv\Scripts\python.exe -m pytest -q tests/test_data_sources_live.py tests/test_announcements.py tests/test_ingestion_quality.py tests/test_doc_pipeline.py
```

结果摘要：

1. `12 passed`
2. `1 passed, 37 deselected`
3. `2 passed, 26 deselected`
4. `10 passed`

## 4. 风险与后续

当前已知风险：

1. 目前只迁移 quote，announcement/history 仍在旧模块。
2. 真实 provider 线上协议可能变化，后续需要 source 级健康监控与重试指标。

下一轮（ROUND-AE）输入：

1. 开始迁移财务源（Tushare + Eastmoney financial）。
2. 设计财务规范化对象与存储映射，确保 DeepThink 可消费。

