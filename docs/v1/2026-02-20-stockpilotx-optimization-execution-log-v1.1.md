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

---

## Batch 3（2026-02-20）

### 对应计划项

- Phase 2 / 6.2：Corrective RAG 闭环（低相关检索触发 query rewrite，二次检索并合并证据）
- Phase 2 灰度要求：新增能力可开关、可回退

### 本批代码改动

1. `backend/app/config.py`
- 新增 `corrective_rag_enabled`（默认 `true`）。
- 新增 `corrective_rag_rewrite_threshold`（默认 `0.42`）。

2. `backend/app/agents/workflow.py`
- `_deep_retrieve` 增加低分触发二次检索逻辑：
  - 当首轮 top score < 阈值时，执行 `_rewrite_query_for_corrective_rag`。
  - 二次召回后与首轮结果去重合并、重排。
- 新增追踪字段：
  - `corrective_rag_applied`
  - `corrective_rag_rewritten_query`
- trace 事件补充 `rewrite_threshold` 与是否触发闭环。

3. `tests/test_phase0_optimization_v11.py`
- 新增 `test_corrective_rag_rewrite_applies_on_low_score`，覆盖低相关触发闭环路径。

### 自测记录

```powershell
.\.venv\Scripts\python -m pytest -q tests\test_phase0_optimization_v11.py tests\test_eval_gate_decision.py tests\test_service.py::ServiceTestCase::test_eval_gate tests\test_http_api.py::HttpApiTestCase::test_evals_run_and_get
```

结果：`11 passed`

### Checklist

- [x] Corrective RAG 开关与阈值配置
- [x] 低相关触发 rewrite
- [x] 二次检索合并证据
- [x] 路径级测试覆盖
- [x] 回归自测与记录

---

## Batch 4（2026-02-20）

### 对应计划项

- Phase 2 / 6.1：ReAct（deep 模式灰度）
  - 限制 `max_iterations`
  - 仅在 deep 场景启用
  - 保持异常/超时不阻塞主流程

### 本批代码改动

1. `backend/app/config.py`
- 新增 `react_deep_enabled`（默认 `false`，灰度开关）。
- 新增 `react_max_iterations`（默认 `2`，上限 `4`）。

2. `backend/app/agents/workflow.py`
- `_deep_retrieve` 增加迭代检索控制：
  - `react_deep_enabled=true` 且 `intent=deep` 时按 `react_max_iterations` 执行多轮子任务。
  - 每轮仍保留子任务超时隔离。
  - 当检索质量达到阈值时提前结束迭代。
- 新增 follow-up 子任务规划 `_plan_react_followup_subtasks`。
- 新增分析字段：
  - `react_iterations_planned`
  - `react_iterations_executed`

3. `tests/test_phase0_optimization_v11.py`
- 新增 `test_react_deep_mode_respects_max_iterations`，验证 deep 灰度迭代次数约束。

### 自测记录

```powershell
.\.venv\Scripts\python -m pytest -q tests\test_phase0_optimization_v11.py tests\test_eval_gate_decision.py
```

结果：`10 passed`

### Checklist

- [x] ReAct deep 灰度开关
- [x] max_iterations 约束
- [x] 迭代追踪字段
- [x] 路径测试覆盖
- [x] 回归自测与记录

---

## Batch 5（2026-02-20）

### 对应计划项

- Phase 2 / 6.3：SQL Agent PoC 安全约束（只读 + 白名单 + 安全校验 + 行数上限）

### 本批代码改动

1. 新增 `backend/app/query/sql_guard.py`
- `SQLSafetyValidator.validate_select_sql(...)`：
  - 仅允许 `SELECT`
  - 禁止写操作和危险函数/注释注入模式
  - 校验 `FROM/JOIN` 表白名单
  - 校验 `SELECT` 字段白名单（PoC 级近似解析）
  - 强制 `LIMIT` 且限制上限

2. 新增 `tests/test_sql_guard.py`
- 覆盖正常查询、非只读、越权表、超限、危险语句五类场景。

### 自测记录

```powershell
.\.venv\Scripts\python -m pytest -q tests\test_sql_guard.py tests\test_eval_gate_decision.py
```

结果：`9 passed`

### Checklist

- [x] SQL 只读约束
- [x] 表白名单约束
- [x] 字段白名单约束
- [x] LIMIT 上限约束
- [x] 安全回归用例

---

## Batch 6（2026-02-20）

### 对应计划项

- Phase 2 / 6.3：SQL Agent PoC 落地到可调用 API（默认只读、可审计）

### 本批代码改动

1. `backend/app/service_modules/runtime_core_mixin.py`
- 新增 `_sql_agent_poc_allowed_schema`：
  - 从 `web.db` 中动态发现存在的白名单表和字段。
- 新增 `sql_agent_poc_query(payload)`：
  - 使用 `SQLSafetyValidator` 进行只读安全校验；
  - 校验通过后执行查询并限制返回行数；
  - 统一返回 `ok/error/reason/validation/rows/trace_id`；
  - 通过 trace 记录审计事件。

2. `backend/app/http_api.py`
- 新增接口：`POST /v1/sql-agent/poc/query`

3. `backend/app/service_modules/shared.py`
- 引入 `SQLSafetyValidator` 供 mixin 使用。

4. 测试补充
- `tests/test_service.py`：新增 `test_sql_agent_poc_query`
- `tests/test_http_api.py`：新增 `test_sql_agent_poc_query`

### 自测记录

```powershell
.\.venv\Scripts\python -m pytest -q tests\test_sql_guard.py tests\test_service.py::ServiceTestCase::test_sql_agent_poc_query tests\test_http_api.py::HttpApiTestCase::test_sql_agent_poc_query
```

结果：`7 passed`

### Checklist

- [x] SQL PoC 服务方法
- [x] SQL PoC HTTP 接口
- [x] 只读校验接入执行链路
- [x] 接口级自测通过

---

## Batch 7（2026-02-20）

### 对应计划项

- Phase 3 / 8.2：中间件扩展（RateLimit / Cache / PII）

### 本批代码改动

1. `backend/app/middleware/hooks.py`
- 新增 `RateLimitMiddleware`：按 `user_id` 做窗口限流，超过阈值抛错。
- 新增 `CacheMiddleware`：按 `user_id+prompt` 做模型输出 TTL 缓存，命中减少重复调用。
- 新增 `PIIMiddleware`：对 prompt/output 做邮箱/手机号/证件号脱敏。

2. 新增 `tests/test_middleware_extensions.py`
- 覆盖限流、缓存命中、PII 脱敏三类行为。

### 自测记录

```powershell
.\.venv\Scripts\python -m pytest -q tests\test_middleware_extensions.py tests\test_phase0_optimization_v11.py::Phase0OptimizationV11Tests::test_workflow_prepare_state_preserves_intent_confidence
```

结果：`4 passed`

### Checklist

- [x] RateLimitMiddleware
- [x] CacheMiddleware
- [x] PIIMiddleware
- [x] 中间件行为单测
