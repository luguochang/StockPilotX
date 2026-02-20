# Multi-Role 产品执行 Checklist

## 执行要求
- 每轮必须完成：代码注释 + 接口自测 + 轮次文档 + checklist 勾选 + commit。
- 接口实现后必须有可复现的自测结果（请求与关键响应摘要）。
- 不允许仅改文档不验证链路。

## ROUND 总览
- [x] R0-01：沉淀 DeepThink 与 MultiRole 分层架构决策文档
- [x] R0-02：建立总 checklist 与轮次执行规范
- [x] R1-01：Predict 接入多角色裁决输出（role_opinions / judge_summary / conflict_sources）
- [x] R1-02：质量门禁改造：`research_insufficient` -> `watch`
- [x] R1-03：新增 `GET /v1/predict/self-test`
- [x] R1-04：新增 `GET /v1/multi-role/traces/{trace_id}`
- [x] R1-05：R1 接口自测与文档归档
- [x] R2-01：Report 接入多角色内核
- [x] R2-02：新增 `GET /v1/report/self-test`，覆盖报告同步/异步链路冒烟
- [ ] R3-01：DeepThink 接入多角色预裁决
- [ ] R4-01：稳定性与超时治理
- [ ] R5-01：前端收口与最终验收

## 本轮记录
- 2026-02-20：启动 R1 开发，先完成后端核心与诊断接口。
- 2026-02-20：R1 完成并补齐可复现自测（predict + trace）。
- 2026-02-20：R2 完成，Report 接入多角色裁决并新增报告自测接口。
