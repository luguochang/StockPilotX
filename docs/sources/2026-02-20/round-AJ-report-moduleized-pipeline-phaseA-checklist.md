# Round-AJ Checklist（2026-02-20）

## 执行要求
1. 每轮必须：实现 -> 自测 -> 文档记录 -> checklist 勾选 -> commit。
2. 代码改动需保留可读注释，不做无说明的大块黑盒逻辑。
3. 不回滚用户已有无关改动，仅提交本轮相关文件。

## Phase A（本轮）
- [x] 后端报告结构升级为模块化输出
- [x] 报告接口新增 `report_modules/final_decision/committee/metric_snapshot`
- [x] 报告任务 partial 阶段返回最小可用模块
- [x] 前端报告中心增加模块化展示区
- [x] 服务与 API 测试断言补齐
- [x] 前端 `tsc` 与 `build` 通过
- [x] 全接口轻量 smoke 自测通过
- [x] 技术文档记录完成

## Phase B（已完成）
- [x] 研究汇总器 + 风险仲裁器节点化
- [x] DeepThink 读取报告委员会结论并参与跨轮推演
- [x] 模块级质量分与降级码输出

## Phase C（已完成）
- [x] 报告导出格式增强（模块化导出模板）
- [x] 报告质量评估看板（覆盖率/证据/一致性）
- [x] 重型链路 smoke（query/deepthink/predict/report 专项）
