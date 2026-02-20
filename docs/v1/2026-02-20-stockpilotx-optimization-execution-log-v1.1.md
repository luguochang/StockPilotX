# StockPilotX 优化执行日志（v1.1）

## Batch 1（2026-02-20）

### 对应计划项

- Phase 0 / 3.1-3.2：基线指标采集脚本 + 报表模板 + 可观测字段落地
- Phase 1 / 4.1：意图路由增强（规则优先 + 置信度）
- Phase 1 / 4.2：MemoryStore 相似检索 + TTL + 清理统计
- Phase 1 / 4.3：`_deep_retrieve` 任务级超时隔离
- 执行入口 / 第 1-5 项：完成首轮回归并输出 Gate A 预判输入数据

### 本批代码改动

1. `backend/app/agents/workflow.py`
- 新增 `route_intent_with_confidence`，保留 `route_intent` 兼容调用。
- 路由阶段记录 `intent_confidence/冲突标记/命中关键词` 到 trace。
- `_deep_retrieve` 新增子任务超时与异常隔离，输出 `timeout_subtasks`。
- 修复分析阶段覆盖问题，保留路由阶段遥测字段。

2. `backend/app/memory/store.py`
- 新增 schema 字段：`ttl_seconds`、`expires_at`，带自动迁移。
- 新增 `similarity_search`（n-gram/Jaccard）用于记忆相似召回。
- 新增 `cleanup_expired` 与 `stats`，提供命中率与清理统计日志。

3. `backend/app/config.py`
- 新增 `deep_subtask_timeout_seconds` 配置与环境变量读取。

4. 新增脚本与文档
- `scripts/collect_phase0_baseline.py`
- `docs/v1/phase0-baseline-report-template.md`
- 基线产物：
  - `docs/v1/baseline/phase0-baseline-20260220-131645.json`
  - `docs/v1/baseline/phase0-baseline-20260220-131645.md`

5. 新增测试
- `tests/test_phase0_optimization_v11.py`

### 自测记录

1. 命令：
```powershell
.\.venv\Scripts\python -m pytest -q tests\test_phase0_optimization_v11.py tests\test_service.py::ServiceTestCase::test_query_basic tests\test_http_api.py::HttpApiTestCase::test_query
```
结果：`6 passed`

2. 命令：
```powershell
$env:PYTHONPATH='.'; .\.venv\Scripts\python scripts\collect_phase0_baseline.py
```
结果：成功产出基线 JSON + Markdown 报表。

### Gate A 预判（首轮输入）

- 当前首轮离线样本 `intent_accuracy=1.0`（样本量较小，后续需扩样）。
- 已具备 `intent_confidence` 观测能力，可用于后续灰度评估。
- 预判结论：暂不引入意图小模型，先继续扩充规则样本与回归集，再评估是否达到 Gate A 阈值（`F1 >= 0.85`）。

### Checklist

- [x] 基线指标采集脚本
- [x] 基线报表模板
- [x] `deep_retrieve` 超时控制
- [x] memory 相似检索 + TTL
- [x] 意图路由规则增强（含置信度）
- [x] 针对性回归自测
- [x] 执行日志记录

---

## Batch 2（2026-02-20）

### 对应计划项

- Phase 0 / 3.1：将 Gate A / Gate B 决策条件代码化，避免主观判断偏差
- Gate A/B：形成可自动产出的评审报告模板化流程

### 本批代码改动

1. `backend/app/evals/service.py`
- `run_eval` 新增 `gate_assessment` 输出。
- 新增 `assess_gate_a(intent_f1)`：
  - `<0.85` => `hold`（建议引入轻量意图模型）
  - `>=0.85` => `go`（继续规则优先）
- 新增 `assess_gate_b(sql_pass_rate, high_risk_count, observability_ready)`：
  - 三条件全部满足才 `go`，否则 `hold`
- 新增 `assess_gate_readiness(...)` 统一输出。

2. 新增脚本
- `scripts/generate_gate_decision_report.py`
- 输入 Phase0 baseline JSON，输出 Gate A/B 决策报告 Markdown。

3. 新增测试
- `tests/test_eval_gate_decision.py`

4. 产物
- `docs/v1/baseline/gate-decision-20260220-133350.md`

### 自测记录

1. 命令：
```powershell
.\.venv\Scripts\python -m pytest -q tests\test_eval_gate_decision.py tests\test_service.py::ServiceTestCase::test_eval_gate tests\test_http_api.py::HttpApiTestCase::test_evals_run_and_get
```
结果：`6 passed`

2. 命令：
```powershell
$env:PYTHONPATH='.'; .\.venv\Scripts\python scripts\generate_gate_decision_report.py --baseline-json docs\v1\baseline\phase0-baseline-20260220-131645.json
```
结果：成功产出 Gate 决策报告。

### Checklist

- [x] Gate A 规则代码化
- [x] Gate B 规则代码化
- [x] Gate 决策报告脚本化
- [x] 回归自测
- [x] 执行日志更新
