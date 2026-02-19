# Round-AD 技术记录：Predict 模块去 Mock 与评测来源透明化

日期: 2026-02-19

## 问题背景

预测页面此前存在一个核心问题:  
指标虽然“看起来实时”，但评测来源在数据不足时会落到 simulated，用户难以判断结果可信度。

这会造成两类业务风险:

1. 用户把 simulated 指标误读为真实回测结果。
2. 预测结论无法解释“为何当前是降级结论”。

## 本轮方案

### 1) 评测链路去 mock（后端）

文件: `backend/app/predict/service.py`

- 评测策略升级:
  - 优先使用真实历史数据做 walk-forward 回测代理（`backtest_proxy`）
  - 若回测样本不足，再回退 simulated
- 新增评测来源元数据:
  - `evaluated_stocks`
  - `skipped_stocks`（含 reason）
  - `history_modes`
  - `fallback_reason`
- 指标计算统一:
  - 抽象 `_build_metrics` 聚合 `ic/hit_rate/top_bottom_spread/max_drawdown/coverage`
  - 去除歧义命名，避免继续以 “mock” 命名核心指标逻辑

### 2) 预测运行结果业务化（服务层）

文件: `backend/app/service.py`

- `predict_run` 新增:
  - `eval_provenance`
- 指标按来源拆分:
  - `metrics_live`
  - `metrics_backtest`
  - `metrics_simulated`
- 保留:
  - `metric_mode`
  - `metrics_note`

这使前端可以精确展示“当前指标来自哪里”。

### 3) 前端门禁卡片增强（展示层）

文件: `frontend/app/hooks/usePredict.ts`  
文件: `frontend/app/predict/page.tsx`

- 类型增加 `metrics_backtest` 与 `eval_provenance`
- 指标选择策略:
  - `metric_mode=live` -> 只读 `metrics_live`
  - `metric_mode=backtest_proxy` -> 只读 `metrics_backtest`
  - `metric_mode=simulated` -> 只读 `metrics_simulated`
- 评测门禁新增字段展示:
  - `coverage_rows`
  - `evaluated_stocks`
  - `skipped_stocks`（Top3）
  - `fallback_reason`
- simulated 模式增加风险提示，避免误用为“可直接交易”的绝对结论

## 业务价值

1. 评测来源可追踪，避免“假实时”误读。
2. 用户可直接看到为何降级、为何回退，不再黑盒。
3. 后续可基于 `skipped_stocks.reason` 反向驱动数据补齐策略。

## 验证结果

1. 后端用例通过:
   - `test_prediction_run_and_eval`
   - `test_predict_endpoints`
2. 关联回归通过:
   - predict + report 目标用例
3. 前端构建与类型通过:
   - `npm run build`
   - `npx tsc --noEmit`（顺序执行）
4. 本地链路实测:
   - `predict_run` 返回 `metric_mode=backtest_proxy`
   - 覆盖行数与评测股票数正常返回

