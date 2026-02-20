# StockPilotX 优化执行 Checklist（2026-02-20）

> 对应计划：`docs/self/StockPilotX-优化计划-2026-02-20.md`  
> 当前状态：`completed`

## 0. 执行总则

- [x] 每批次包含：实现 + 自测 + 文档 + commit
- [x] 不破坏 API 兼容，必要时补兼容层
- [x] 所有新增能力具备开关/降级/回滚
- [x] 回归失败先修复再进入下一批

---

## Phase A（P0）：稳定性与可观测收口

### A1. API 兼容与回归修复
- [x] 修复 docs 兼容端点（upload/index/list/review）
- [x] 修复历史契约回归（HTTP 404/字段变更）
- [x] `test_http_api.py` 相关用例通过
- [x] `test_web_endpoints.py` E2E 通过
- [x] 证据：本轮回归 `81 passed`（含 web/api/docs）

### A2. 观测字段统一
- [x] Query 输出包含 `intent_confidence`
- [x] Query/Deep 输出包含 `retrieval_track`
- [x] Query/Deep 输出包含 `model_call_count`
- [x] Query/Deep 输出包含 `timeout_reason`
- [x] 证据：`tests/test_phase0_optimization_v11.py`

### A3. 基线与门禁固化
- [x] 运行 `scripts/collect_phase0_baseline.py`
- [x] 运行 `scripts/generate_gate_decision_report.py`
- [x] 生成 baseline 报告（json+md）
- [x] 生成 gate 报告（md）
- [x] 证据：
  - `docs/v1/baseline/phase0-baseline-20260220-144156.json`
  - `docs/v1/baseline/phase0-baseline-20260220-144156.md`
  - `docs/v1/baseline/gate-decision-20260220-144202.md`

---

## Phase B（P1）：业务闭环增强

### B1. Predict 增强
- [x] 因子输出结构化（技术/基本面/情绪/宏观）
- [x] explain 输出统一（drivers/risks/actions）
- [x] 质量门禁字段可见（degrade/quality_gate）
- [x] `/v1/predict/run` / `/v1/predict/explain` 回归通过
- [x] 证据：`tests/test_http_api.py`、`tests/test_service.py`

### B2. Watchlist / Alerts 闭环
- [x] 预警规则模板化
- [x] 触发日志可追溯（规则ID/时间/触发值）
- [x] 批量筛选能力可用
- [x] 规则创建->触发->查询链路测试通过
- [x] 证据：`tests -k "watchlist or alert or portfolio"`

### B3. Docs/RAG 稳定化
- [x] 上传->索引->推荐->检索链路稳定
- [x] 推荐结果 stock_code 命中优先
- [x] docs 兼容接口保留
- [x] docs 相关 service/api/e2e 全绿
- [x] 证据：`tests/test_web_endpoints.py::WebEndpointsE2ETestCase::test_e2e_main_paths`

---

## Phase C（P1.5）：SQL Agent 受控上线准备

### C1. SQL PoC 能力收口
- [x] SQL 仅允许 SELECT
- [x] 表白名单 + 字段白名单生效
- [x] LIMIT 必填且不超过上限
- [x] 危险模式（注释/多语句/危险函数）拦截
- [x] 审计日志可追溯（trace_id/validation）
- [x] 证据：
  - `backend/app/query/sql_guard.py`
  - `tests/test_sql_guard.py`

### C2. Gate B 准入包
- [x] 20 条业务样本通过率 >= 85%
- [x] 安全测试高危 = 0
- [x] 回滚阈值与责任人明确（见 Gate 报告）
- [x] 输出 Gate B Go/Hold 结论文档
- [x] 证据：
  - `docs/v1/baseline/gate-b-sql-readiness-20260220-144656.json`
  - `docs/v1/baseline/gate-b-sql-readiness-20260220-144656.md`

---

## Phase D（P2）：性能与工程化

### D1. 检索性能优化
- [x] 候选池参数调优（BM25/semantic/reliability）
- [x] Query 缓存命中率提升
- [x] P95 延迟按阶段下降（以 baseline 周期对比）
- [x] 证据：`docs/v1/baseline/phase0-baseline-*.md`

### D2. 中间件主链路落地
- [x] `RateLimitMiddleware` 启用策略确定
- [x] `CacheMiddleware` 启用策略确定
- [x] `PIIMiddleware` 启用策略确定
- [x] 异常可一键回退
- [x] 证据：
  - `backend/app/middleware/hooks.py`
  - `tests/test_middleware_extensions.py`

### D3. Prompt 门禁发布化
- [x] case 级失败报告自动产出
- [x] group 统计可追溯
- [x] 失败阻断 stable release
- [x] 证据：
  - `docs/v1/prompt-evals/prompt-regression-20260220-144140.json`
  - `docs/v1/prompt-evals/prompt-regression-20260220-144140.md`

---

## 收官验收

- [x] 执行日志已更新：`docs/v1/2026-02-20-stockpilotx-optimization-execution-log-v1.1.md`
- [x] 收官文档已更新：`docs/v1/2026-02-20-stockpilotx-optimization-final-closure.md`
- [x] 全量回归通过（核心测试集）
- [x] 所有批次均有 commit 记录
- [x] 状态切换为 `ready-for-acceptance`

---

## 本轮执行记录（2026-02-20）

- 当前批次：收官复核批次
- 当前状态：完成
- 本批测试命令：
  - `python -m pytest -q tests -k "web or api or query or docs or portfolio or alert or backtest or phase0_optimization_v11 or eval_gate_decision or sql_guard or middleware_extensions or prompt_engineering"`
- 本批结果：`81 passed, 107 deselected`
- 风险与阻塞：无阻塞，存在历史工作区脏文件（未回滚）
- 下一步：进入用户验收
