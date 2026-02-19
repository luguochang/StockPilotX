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

- [ ] 创建 `backend/app/datasources/` 模块结构与基类
- [ ] 新增 `base/adapter.py`、`base/http_client.py`、`base/utils.py`
- [ ] 在 `backend/app/config.py` 增加数据源配置项
- [ ] 接入服务构造工厂，替换直接实例化路径（最小侵入）
- [ ] 添加基础单测（配置与基础工具）
- [ ] 记录技术文档（round-AC）
- [ ] 提交 commit（round-AC）

证据记录：

- 自测命令：
- 自测结果：
- 技术文档路径：
- Commit Hash：

## ROUND-AD：行情数据源迁移

- [ ] 迁移并实现 `sina.py`
- [ ] 迁移并实现 `tencent.py`
- [ ] 迁移并实现 `netease.py`
- [ ] 迁移并实现 `xueqiu.py`
- [ ] 完成行情回退链和失败降级策略
- [ ] 接入 `ingest_market_daily` 与相关调用点
- [ ] 添加行情相关单测（解析、编码、回退）
- [ ] 记录技术文档（round-AD）
- [ ] 提交 commit（round-AD）

证据记录：

- 自测命令：
- 自测结果：
- 技术文档路径：
- Commit Hash：

## ROUND-AE：财务数据源迁移

- [ ] 迁移并实现 `financial/tushare.py`
- [ ] 迁移并实现 `financial/eastmoney.py`
- [ ] 补齐财务数据规范化与落库逻辑
- [ ] 接入 DeepThink/Query 可消费的财务证据路径
- [ ] 添加财务相关单测（字段映射、异常数据、空响应）
- [ ] 记录技术文档（round-AE）
- [ ] 提交 commit（round-AE）

证据记录：

- 自测命令：
- 自测结果：
- 技术文档路径：
- Commit Hash：

## ROUND-AF：新闻/研报/宏观/基金迁移

- [ ] 迁移并实现 `news/cls.py`
- [ ] 迁移并实现 `news/tradingview.py`
- [ ] 迁移并实现 `news/xueqiu_news.py`
- [ ] 迁移并实现 `research/eastmoney_research.py`
- [ ] 迁移并实现 `macro/eastmoney_macro.py`
- [ ] 迁移并实现 `fund/ttjj.py`
- [ ] 打通长文本入 RAG（摘要 + 原文切块）
- [ ] 添加相关单测与入库校验
- [ ] 记录技术文档（round-AF）
- [ ] 提交 commit（round-AF）

证据记录：

- 自测命令：
- 自测结果：
- 技术文档路径：
- Commit Hash：

## ROUND-AG：数据源管理 API 与可观测性

- [ ] 新增 `/v1/datasources/sources`
- [ ] 新增 `/v1/datasources/health`
- [ ] 新增 `/v1/datasources/fetch`
- [ ] 新增 `/v1/datasources/logs`
- [ ] 新增 source 级日志/健康聚合逻辑
- [ ] 对接告警策略（失败率、熔断状态）
- [ ] 添加 API 与权限相关测试
- [ ] 记录技术文档（round-AG）
- [ ] 提交 commit（round-AG）

证据记录：

- 自测命令：
- 自测结果：
- 技术文档路径：
- Commit Hash：

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

1. 待开始

