# Round-AD Checklist: Predict 模块去 Mock 与业务化评测

日期: 2026-02-19  
范围: 预测评测来源去 mock、前端评测来源透明化、预测模块卡片业务说明增强

## 执行要求

- [x] 代码必须加注释，说明关键逻辑目的
- [x] 每个改动点完成后执行自测
- [x] 输出技术记录文档，方便后续维护
- [x] 完成后提交 commit，并回填 hash

## 后端改造

- [x] 预测评测从“纯模拟标签”升级为“优先真实历史回测代理（walk-forward）”
- [x] 在回测样本不足时自动降级为 simulated，并输出 `fallback_reason`
- [x] 增加 `evaluated_stocks/skipped_stocks/history_modes` 评测来源元数据
- [x] `predict_run` 返回 `eval_provenance`，用于前端解释指标可信度
- [x] 指标按来源拆分：`metrics_live/metrics_backtest/metrics_simulated`

## 前端改造

- [x] 预测页按 `metric_mode` 严格选取指标来源，避免混用
- [x] 评测门禁卡片新增覆盖行数、评测股票数、跳过原因、回退原因
- [x] 模拟模式下新增风险提示卡片，明确不可用于绝对收益承诺

## 测试与验证

- [x] `tests/test_service.py::test_prediction_run_and_eval`
- [x] `tests/test_http_api.py::test_predict_endpoints`
- [x] 关联回归：report + predict 组合用例
- [x] 前端 `npm run build`
- [x] 前端 `npx tsc --noEmit`（顺序执行）
- [x] 本地服务实测 `AShareAgentService().predict_run(...)` 返回评测来源元数据

## 提交记录

- [ ] Commit Hash（待回填）

