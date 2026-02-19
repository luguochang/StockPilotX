# 数据源集成总体实施计划（StockPilotX）

> 日期：2026-02-19  
> 目标目录：`docs/sources/2026-02-19/`  
> 适用范围：`StockPilotX` 数据源能力迁移与集成（`stock-api` + `go-stock`）

## 1. 目标与背景

当前 `StockPilotX` 的数据源信息密度不足，影响高级分析与业务结论质量。  
本计划用于在项目内新增统一的数据源模块 `backend/app/datasources/`，迁移并整合 `stock-api` 与 `go-stock` 的相关抓取能力，形成可维护、可观测、可扩展的数据层。

本计划强调以下原则：

1. 不运行外部项目作为依赖，只迁移能力代码。
2. 保持 `StockPilotX` 现有架构风格与接口兼容。
3. 每一轮必须有自测、文档记录、checklist 勾选、独立 commit。
4. 关键代码需添加必要注释，便于后续维护。

## 2. 实施范围

### 2.1 数据源类型

1. 行情：Sina、Tencent、Netease、Xueqiu
2. 财务：Tushare、Eastmoney Financial
3. 新闻：CLS、TradingView、Xueqiu News
4. 研报：Eastmoney Research
5. 宏观：Eastmoney Macro
6. 基金：TTJJ

### 2.2 接入方式

1. 内部接入：替换现有 `backend/app/data/sources.py` 的直接依赖路径，统一经 `datasources` 模块提供服务。
2. 运维接入：新增 `/v1/datasources/*` 管理与诊断 API。

### 2.3 存储策略

1. 热数据（结构化）：行情/财务/宏观/新闻元数据进入现有结构化存储。
2. 冷数据（长文本）：新闻正文/研报正文通过现有 RAG 流程（切块、索引、检索）入库。

## 3. 目标目录结构

```text
backend/app/datasources/
  README.md
  __init__.py
  base/
    __init__.py
    adapter.py
    http_client.py
    utils.py
  quote/
    __init__.py
    README.md
    sina.py
    tencent.py
    netease.py
    xueqiu.py
  financial/
    __init__.py
    README.md
    tushare.py
    eastmoney.py
  news/
    __init__.py
    README.md
    cls.py
    tradingview.py
    xueqiu_news.py
  research/
    __init__.py
    README.md
    eastmoney_research.py
  macro/
    __init__.py
    README.md
    eastmoney_macro.py
  fund/
    __init__.py
    README.md
    ttjj.py
  tests/
    test_quote.py
    test_financial.py
    test_news.py
    test_research.py
```

## 4. 轮次计划（ROUND-AC ~ ROUND-AH）

1. ROUND-AC：基础骨架与配置接入
2. ROUND-AD：行情源迁移与回退链路
3. ROUND-AE：财务源迁移与结构化入库
4. ROUND-AF：新闻/研报/宏观/基金迁移与 RAG 对接
5. ROUND-AG：数据源运维 API 与健康监控
6. ROUND-AH：全链路回归、文档收口、交付总结

## 5. 执行要求（强制）

每一轮（Round）完成时，必须同时满足以下要求：

1. 完成当轮代码实现，关键逻辑增加注释说明。
2. 完成当轮自测，并记录执行命令与结果摘要。
3. 产出当轮技术文档（实现内容、设计取舍、风险与后续项）。
4. 更新本目录 checklist 的状态勾选与证据字段。
5. 提交独立 commit（不得把多轮工作合并成一个 commit）。

## 6. 自测基线要求

按改动影响面执行，不得跳过：

1. 仅后端改动：至少执行相关后端单测（datasources/ingestion/service/api 相关）。
2. 包含 API 协议变更：增加接口级自测（请求示例 + 返回结构）。
3. 包含前端改动：至少执行 `npm run build` 与 `npx tsc --noEmit`。
4. 重大链路变更：执行端到端链路自测（ingest -> query/deepthink -> rag）。

## 7. 文档产出规范

每轮文档至少包含：

1. 轮次目标与完成范围
2. 关键代码变更点（文件路径）
3. 自测命令与结果
4. 已知问题与风险
5. 对下一轮的输入条件

推荐命名：

`docs/rounds/2026-02-19/round-XX-datasource-*.md`

## 8. Commit 规范

1. 每轮一个 commit，信息格式建议：
   - `feat(datasource): round-AC scaffold and config integration`
   - `feat(datasource): round-AD quote adapters and fallback chain`
2. commit 前必须确保 checklist 与当轮文档已更新。
3. commit 信息需可定位轮次、模块和核心变更。

## 9. 验收标准

1. 数据源模块具备统一接口与可扩展结构。
2. 关键数据源迁移完成并可在现有业务链路被调用。
3. 数据源可观测性可用于定位失败源与延迟瓶颈。
4. 文档、checklist、commit 与自测证据完整可追溯。

