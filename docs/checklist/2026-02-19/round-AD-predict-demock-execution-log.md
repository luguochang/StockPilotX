# Round-AD Execution Log

日期: 2026-02-19

## 变更摘要

1. 后端预测评测去 mock
   - 文件: `backend/app/predict/service.py`
   - 关键改动:
     - 评测逻辑改为“优先 walk-forward 历史回测代理”
     - 回测样本不足时自动回退到 simulated，并给出 `fallback_reason`
     - 新增评测来源字段:
       - `evaluated_stocks`
       - `skipped_stocks`
       - `history_modes`
     - 指标聚合统一走 `_build_metrics`，避免重复计算与命名歧义

2. 预测运行出参业务化
   - 文件: `backend/app/service.py`
   - 关键改动:
     - `predict_run` 增加 `eval_provenance`
     - 指标分源输出:
       - `metrics_live`
       - `metrics_backtest`
       - `metrics_simulated`
     - 保留 `metric_mode/metrics_note` 供前端解释

3. 前端预测卡片去“假统一指标”
   - 文件: `frontend/app/hooks/usePredict.ts`
     - 扩展 `PredictRunResponse` 类型: `metrics_backtest` + `eval_provenance`
   - 文件: `frontend/app/predict/page.tsx`
     - 按 `metric_mode` 严格选择指标源，不再混读
     - 评测门禁新增 coverage/evaluated/skipped/fallback 可视化
     - simulated 模式新增提示，防止误读为真实回测结论

4. 测试更新
   - 文件:
     - `tests/test_service.py`
     - `tests/test_http_api.py`
   - 调整点:
     - `metric_mode` 断言从固定 `simulated` 放宽为 `{simulated, backtest_proxy}`
     - 新增 `eval_provenance/metrics_backtest` 出参断言

## 自测记录

1. Python 语法检查

```bash
python -m py_compile backend/app/predict/service.py backend/app/service.py
```

结果: 通过

2. 后端目标用例

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_service.py -k "prediction_run_and_eval"
```

结果: `1 passed, 41 deselected`

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_http_api.py -k "test_predict_endpoints"
```

结果: `1 passed, 35 deselected`

3. 关联回归

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_service.py -k "prediction_run_and_eval or report_generate_and_get"
```

结果: `2 passed, 40 deselected`

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_http_api.py -k "test_predict_endpoints or test_report_generate_and_get"
```

结果: `2 passed, 34 deselected`

4. 前端构建/类型

```bash
cd frontend
npm run build
```

结果: `Compiled successfully`

```bash
cd frontend
npx tsc --noEmit
```

结果: 通过（需与 build 顺序执行，避免 `.next/types` 生成时序问题）

5. 本地链路实测

```python
from backend.app.service import AShareAgentService
s = AShareAgentService()
r = s.predict_run({"stock_codes": ["SH600000", "SZ000001"], "horizons": ["5d", "20d"]})
```

关键输出:
- `metric_mode = backtest_proxy`
- `eval_provenance.coverage_rows = 364`
- `eval_provenance.evaluated_stocks = 2`

## 提交记录

- Commit: `145fa07`
