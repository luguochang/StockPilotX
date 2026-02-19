# DeepThink 数据源业务化执行 Checklist（v1）

> 日期：2026-02-19  
> 主计划：`docs/v1/2026-02-19-DeepThink-数据源业务化增强计划.md`  
> 执行要求：每轮必须完成「代码实现（含注释）+ 自测 + 技术文档 + Checklist 更新 + Commit」

---

## ROUND 状态

- [x] ROUND-1：可见性与一致性修复
  - [x] `ops/health` 改为 `datasources` 新接口族（`/v1/datasources/sources|health|fetch|logs`）
  - [x] 前端补抓逻辑由旧 ingest 直调改为 `datasources/fetch`（按类别补抓）
  - [x] 后端 `datasource_health` 增强 `used_in_ui_modules`、`last_used_at`、`staleness_minutes`
  - [x] 产出“数据源 -> 页面模块”覆盖矩阵文档
  - [x] 完成回归自测并记录

- [x] ROUND-2：后端业务映射层（Intel Card）
  - [x] 新增 `GET /v1/analysis/intel-card`
  - [x] 统一新闻/研报/资金/宏观到业务字段层
  - [x] 补齐接口测试与类型注释

- [x] ROUND-3：RAG 两阶段检索
  - [x] 粗排（BM25+向量）
  - [x] 精排（rerank）
  - [x] 引用归因一致性校验

- [x] ROUND-4：DeepThink 业务视图重构
  - [x] 结论卡/证据卡/事件卡/情景矩阵
  - [x] 工程视图默认折叠
  - [x] 交互链路回归

- [x] ROUND-5：风控与执行建议
  - [x] 仓位区间与节奏建议
  - [x] 失效条件模板与触发阈值
  - [x] 降级策略可视化

- [ ] ROUND-6：复盘闭环
  - [ ] 用户采纳记录
  - [ ] T+1/T+5/T+20 偏差统计
  - [ ] 复盘指标落库与可视化

- [ ] ROUND-7：回归与收口
  - [ ] 全链路回归
  - [ ] 风险清单更新
  - [ ] 最终交付文档与运维说明

---

## 执行日志

1. `2026-02-19 | ROUND-1 | 完成 ops/health 新接口切换、可见性字段增强与覆盖矩阵文档 | commit: 63f672e | doc: docs/v1/2026-02-19-round-1-ops-health-and-visibility-matrix.md`
2. `2026-02-19 | ROUND-2 | 新增 intel-card 聚合接口，输出业务结论/证据/事件日历/情景矩阵并补充回归测试 | commit: e857ed2 | doc: docs/v1/2026-02-19-round-2-intel-card-api.md`
3. `2026-02-19 | ROUND-3 | 完成 RAG 两阶段检索（粗排+精排）与 citation 归因一致性校验，补充检索元数据与回归测试 | commit: b7fda8c | doc: docs/v1/2026-02-19-round-3-rag-two-stage-rerank-and-attribution.md`
4. `2026-02-19 | ROUND-4 | DeepThink 分析模式接入业务情报卡片（结论/证据/事件/情景），并完成前端交互与编译回归 | commit: 29745e7 | doc: docs/v1/2026-02-19-round-4-deepthink-business-view-restructure.md`
5. `2026-02-19 | ROUND-5 | intel-card 增加执行节奏、风控阈值与降级状态，并在前端业务卡片中可视化展示 | commit: 29a171f | doc: docs/v1/2026-02-19-round-5-risk-and-execution-guardrails.md`
