# Round-AK Checklist（2026-02-20）

## 执行要求
1. 每轮按顺序执行：实现 -> 自测 -> 文档记录 -> checklist 勾选 -> commit。
2. 代码需增加可读注释，避免不可解释的大块逻辑。
3. 仅提交本轮相关文件，不回滚用户无关改动。

## Phase B（完成）
- [x] 报告研究汇总器节点化（research_summarizer）
- [x] 报告风险仲裁器节点化（risk_arbiter）
- [x] DeepThink 跨轮读取 report node 摘要（research_summary/risk_summary）

## Phase C（完成）
- [x] 报告导出支持 `markdown/module_markdown/json_bundle`
- [x] 报告质量看板（覆盖率/证据密度/一致性）落地
- [x] 任务轮询返回 `report_quality_dashboard`
- [x] 前端展示节点卡片、质量看板与导出格式切换

## 自测（完成）
- [x] `pytest tests/test_service.py tests/test_http_api.py -q` 通过（82 passed）
- [x] `frontend npm run build` 通过
- [x] `frontend npx tsc --noEmit` 通过
- [x] 轻量 smoke 通过（129 routes, failed=0）
- [x] heavy-critical smoke 通过（129 routes, failed=0）
