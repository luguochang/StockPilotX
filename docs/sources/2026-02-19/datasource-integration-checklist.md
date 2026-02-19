# 数据源集成执行 Checklist（ROUND-AC ~ ROUND-AH）

> 日期：2026-02-19  
> 关联计划：`docs/sources/2026-02-19/datasource-integration-master-plan.md`  
> 执行原则：每轮必须完成「实现 + 自测 + 技术文档 + checklist 更新 + commit」

---

## A. 全局执行约束（每轮都必须满足）

- [ ] 完成当轮代码实现（含必要注释，解释关键逻辑与用途）
- [ ] 完成当轮自测（记录命令与结果）
- [ ] 完成当轮技术文档（记录设计、改动、风险）
- [ ] 更新本 checklist 的状态与证据字段
- [ ] 完成独立 commit（一轮一个 commit）

---

## B. 轮次任务拆解

## ROUND-AC：基础骨架与配置接入

- [x] 创建 `backend/app/datasources/` 模块结构与基类
- [x] 新增 `base/adapter.py`、`base/http_client.py`、`base/utils.py`
- [x] 在 `backend/app/config.py` 增加数据源配置项
- [x] 接入服务构造工厂，替换直接实例化路径（最小侵入）
- [x] 添加基础单测（配置与基础工具）
- [x] 记录技术文档（round-AC）
- [x] 提交 commit（round-AC）

证据记录：

- 自测命令：
  - `.\.venv\Scripts\python.exe -m pytest -q tests/test_datasources_factory.py tests/test_datasource_utils.py`
  - `.\.venv\Scripts\python.exe -m pytest -q tests/test_data_sources_live.py tests/test_announcements.py tests/test_ingestion_quality.py tests/test_doc_pipeline.py`
  - `.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "ingest_endpoints"`
  - `.\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "ingest_market_daily or ingest_announcements"`
- 自测结果：`20 passed`（含 deselected 子集）
- 技术文档路径：`docs/sources/2026-02-19/round-AC-datasource-scaffold-and-config.md`
- Commit Hash：`89c8f01`

## ROUND-AD：行情数据源迁移

- [x] 迁移并实现 `sina.py`
- [x] 迁移并实现 `tencent.py`
- [x] 迁移并实现 `netease.py`
- [x] 迁移并实现 `xueqiu.py`
- [x] 完成行情回退链和失败降级策略
- [x] 接入 `ingest_market_daily` 与相关调用点
- [x] 添加行情相关单测（解析、编码、回退）
- [x] 记录技术文档（round-AD）
- [x] 提交 commit（round-AD）

证据记录：

- 自测命令：
  - `.\.venv\Scripts\python.exe -m pytest -q tests/test_datasource_quote_adapters.py tests/test_datasources_factory.py tests/test_datasource_utils.py`
  - `.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "ingest_endpoints"`
  - `.\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "ingest_market_daily or ingest_announcements"`
  - `.\.venv\Scripts\python.exe -m pytest -q tests/test_data_sources_live.py tests/test_announcements.py tests/test_ingestion_quality.py tests/test_doc_pipeline.py`
- 自测结果：`25 passed`（含 deselected 子集）
- 技术文档路径：`docs/sources/2026-02-19/round-AD-quote-adapters-migration.md`
- Commit Hash：`7b5feb0`

## ROUND-AE：财务数据源迁移

- [x] 迁移并实现 `financial/tushare.py`
- [x] 迁移并实现 `financial/eastmoney.py`
- [x] 补齐财务数据规范化与落库逻辑
- [x] 接入 DeepThink/Query 可消费的财务证据路径
- [x] 添加财务相关单测（字段映射、异常数据、空响应）
- [x] 记录技术文档（round-AE）
- [x] 提交 commit（round-AE）

证据记录：

- 自测命令：
  - `.\.venv\Scripts\python.exe -m pytest -q tests/test_datasource_financial_adapters.py tests/test_ingestion_financial.py tests/test_datasources_factory.py`
  - `.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "ingest_endpoints"`
  - `.\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "ingest_market_daily or ingest_announcements or ingest_financials"`
  - `.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "query_basic"`
- 自测结果：`14 passed`（含 deselected 子集）
- 技术文档路径：`docs/sources/2026-02-19/round-AE-financial-adapters-and-ingestion.md`
- Commit Hash：`见本轮 round-AE commit`

## ROUND-AF：新闻/研报/宏观/基金迁移

- [x] 迁移并实现 `news/cls.py`
- [x] 迁移并实现 `news/tradingview.py`
- [x] 迁移并实现 `news/xueqiu_news.py`
- [x] 迁移并实现 `research/eastmoney_research.py`
- [x] 迁移并实现 `macro/eastmoney_macro.py`
- [x] 迁移并实现 `fund/ttjj.py`
- [x] 打通长文本入 RAG（摘要 + 原文切块）
- [x] 添加相关单测与入库校验
- [x] 记录技术文档（round-AF）
- [x] 提交 commit（round-AF）

证据记录：

- 自测命令：
  - `.\.venv\Scripts\python.exe -m pytest -q tests/test_datasource_intel_adapters.py tests/test_ingestion_extended_sources.py tests/test_datasources_factory.py`
  - `.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "ingest_endpoints or market_overview_contains_realtime_and_history"`
  - `.\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "ingest_market_daily or ingest_announcements or ingest_financials or ingest_news or ingest_research or ingest_macro or ingest_fund"`
- 自测结果：`21 passed` + `2 passed` + `7 passed`（其余为 deselected 子集）
- 技术文档路径：`docs/sources/2026-02-19/round-AF-news-research-macro-fund-rag.md`
- Commit Hash：见本轮 round-AF commit

## ROUND-AG：数据源管理 API 与可观测性

- [x] 新增 `/v1/datasources/sources`
- [x] 新增 `/v1/datasources/health`
- [x] 新增 `/v1/datasources/fetch`
- [x] 新增 `/v1/datasources/logs`
- [x] 新增 source 级日志/健康聚合逻辑
- [x] 对接告警策略（失败率、熔断状态）
- [x] 添加 API 与权限相关测试
- [x] 记录技术文档（round-AG）
- [x] 提交 commit（round-AG）

证据记录：

- 自测命令：
  - `.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "datasource_ops_catalog_health_fetch_logs"`
  - `.\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "datasource_management_endpoints"`
  - `.\.venv\Scripts\python.exe -m pytest -q tests/test_datasource_intel_adapters.py tests/test_ingestion_extended_sources.py tests/test_datasources_factory.py`
  - `.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "ingest_endpoints or market_overview_contains_realtime_and_history or datasource_ops_catalog_health_fetch_logs"`
  - `.\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "ingest_market_daily or ingest_announcements or ingest_financials or ingest_news or ingest_research or ingest_macro or ingest_fund or datasource_management_endpoints"`
- 自测结果：`1 passed` + `1 passed` + `21 passed` + `9 passed`（其余为 deselected 子集）
- 技术文档路径：`docs/sources/2026-02-19/round-AG-datasource-management-api-and-observability.md`
- Commit Hash：见本轮 round-AG commit

## ROUND-AH：回归与交付收口

- [ ] 全链路回归（ingest -> query/deepthink -> rag）
- [ ] 整理兼容性与风险清单
- [ ] 输出最终交付文档与运维说明
- [ ] 确认 checklist 全部完成并补全证据
- [ ] 提交最终收口 commit（round-AH）

证据记录：

- 自测命令：
- 自测结果：
- 技术文档路径：
- Commit Hash：

---

## C. 回归建议命令清单（按影响面选择）

后端相关：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests -k "datasource or ingestion or service or http_api"
```

前端相关（若涉及）：

```powershell
cd frontend
npm run build
npx tsc --noEmit
```

端到端链路（建议在关键轮执行）：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests -k "deep_think or rag or api or web"
```

---

## D. 执行日志（滚动追加）

> 说明：每完成一轮，在这里追加一条记录（日期、轮次、摘要、hash、文档路径）。

1. `2026-02-19 | ROUND-AC | 完成 datasources 骨架、factory、config 接入与基础回归测试 | hash: 89c8f01 | doc: docs/sources/2026-02-19/round-AC-datasource-scaffold-and-config.md`
2. `2026-02-19 | ROUND-AD | 完成 quote 四源迁移、新 QuoteService 回退链及回归验证 | hash: 7b5feb0 | doc: docs/sources/2026-02-19/round-AD-quote-adapters-migration.md`
3. `2026-02-19 | ROUND-AE | 完成 financial 双源迁移、摄取落库、Query/DeepThink 财务证据接入 | hash: 见本轮 round-AE commit | doc: docs/sources/2026-02-19/round-AE-financial-adapters-and-ingestion.md`
4. `2026-02-19 | ROUND-AF | 完成 news/research/macro/fund 迁移、自动 RAG 入索引、ingest API 与回归测试 | hash: 见本轮 round-AF commit | doc: docs/sources/2026-02-19/round-AF-news-research-macro-fund-rag.md`
5. `2026-02-19 | ROUND-AG | 完成 datasources 管理 API（sources/health/fetch/logs）、健康聚合与告警策略接入 | hash: 见本轮 round-AG commit | doc: docs/sources/2026-02-19/round-AG-datasource-management-api-and-observability.md`
