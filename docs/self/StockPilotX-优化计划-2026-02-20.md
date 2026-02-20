# StockPilotX 优化计划（可执行版）

> 文档日期：2026-02-20  
> 文档状态：`ready-for-execution`  
> 适用范围：`StockPilotX`（后端为主，前端联动）

## 0. 计划目标（对齐当前项目）

在**不大规模重构**的前提下，基于现有 `FastAPI + LangGraph/LangChain + SQLite + Next.js` 架构，完成：

1. 稳定性提升：查询降级、超时隔离、安全护栏、回滚可用。
2. 检索与分析提升：Agentic RAG、Corrective RAG、多轮深度分析稳定运行。
3. 业务模块可用化：Predict / Watchlist / Report / Docs / Journal 核心链路闭环。
4. 工程化与验收：可观测、门禁、回归、文档、提交全流程标准化。

---

## 1. 当前系统基线（以代码现状为准）

### 1.1 技术与架构

- 前端：Next.js + TypeScript（多页面业务模块已落地）
- 后端：FastAPI（`backend/app/http_api.py`）
- 核心服务：`AShareAgentService` + `service_modules/*`
- 检索链路：HybridRetriever / GraphRAG / Prompt Runtime / Eval Service
- 数据存储：SQLite（`memory.db`、`prompt.db`、`web.db`）

### 1.2 数据源能力（已接入）

- 行情：Tencent / Netease / Sina / Xueqiu
- 财务：Tushare / Eastmoney financial
- 新闻：CLS / TradingView / Xueqiu
- 研报：Eastmoney research
- 宏观：Eastmoney macro
- 资金/基金：TTJJ 等

### 1.3 已完成能力（本轮执行后）

- 意图路由置信度 + Deep 子任务超时隔离
- Memory TTL / 相似检索 / 清理统计
- Corrective RAG（低相关触发 rewrite）
- ReAct deep 灰度骨架（max_iterations）
- SQL PoC 安全护栏（只读、白名单、limit）
- 中间件扩展（RateLimit / Cache / PII）
- Gate A/B 自动评估与报告脚本
- Prompt 评测扩展（case 级失败定位、分组统计）

---

## 2. 执行约束（必须遵守）

1. 不引入高风险重依赖作为默认路径（`deepagents`、外部强依赖服务）。
2. 所有新能力必须有开关、降级与回滚路径。
3. 每个功能批次必须包含：
- 代码实现
- 自测（pytest）
- 文档记录（执行日志）
- Git 提交（独立 commit）
4. 不破坏现有 API 兼容（已有端点需保留或提供兼容层）。

---

## 3. 分阶段可执行计划

## Phase A（P0）：稳定性与可观测收口（1 周）

### A1. API 兼容与回归修复

- 范围：`backend/app/http_api.py`、`tests/test_http_api.py`、`tests/test_web_endpoints.py`
- 目标：修复兼容回归（历史接口 404、契约变更导致测试失败）。
- 验收：关键 E2E、HTTP 契约测试全绿。

### A2. 观测字段全链路统一

- 范围：`workflow/query/deep/evals` 输出结构
- 字段：`intent_confidence`、`retrieval_track`、`model_call_count`、`timeout_reason`
- 验收：Query/Deep/Report 都可查询到一致字段。

### A3. 基线与门禁固化

- 脚本：
- `scripts/collect_phase0_baseline.py`
- `scripts/generate_gate_decision_report.py`
- 验收：每次回归均生成 baseline + gate 报告。

---

## Phase B（P1）：业务闭环增强（2-3 周）

### B1. Predict 模块增强（可解释 + 质量门禁）

- 范围：`backend/app/service_modules/predict_mixin.py` + 前端 `predict`
- 任务：
- 因子分解结果结构化（技术/基本面/情绪/宏观）
- 解释输出统一（drivers/risks/actions）
- 质量门禁与降级原因可见
- 验收：`/v1/predict/run`、`/v1/predict/explain` 契约稳定，回归通过。

### B2. Watchlist / Alerts 使用闭环

- 范围：`portfolio_watchlist_mixin.py` + `web/service.py` + 前端 `watchlist`
- 任务：
- 规则模板化（价格、波动、事件）
- 触发日志可追溯（规则 ID、触发时间、触发值）
- 批量管理与筛选体验优化
- 验收：创建规则 -> 触发 -> 查询日志全链路可测。

### B3. Docs/RAG 流程稳定化

- 范围：`rag_mixin.py`、`web/store.py`、RAG 页面
- 任务：
- 上传->索引->推荐->检索结果一致性
- 推荐稳定性（stock_code 命中优先）
- docs 兼容接口保留
- 验收：docs 相关 service + api + e2e 通过。

---

## Phase C（P1.5）：SQL Agent 可控上线准备（1-2 周）

### C1. SQL PoC -> 受控能力

- 范围：`query/sql_guard.py`、`runtime_core_mixin.py`、`http_api.py`
- 任务：
- SQL 安全校验完善（函数白名单/黑名单、limit 强约束）
- 审计日志（trace_id、sql 摘要、validation 结果）
- 白名单表字段配置化
- 验收：
- 安全测试高危=0
- 20 条业务样本通过率 >= 85%

### C2. Gate B 准入包

- 输出：`Go/Hold` 结论 + 风险项 + 回滚阈值 + 负责人
- 验收：具备灰度上线路径。

---

## Phase D（P2）：性能与工程化（2 周）

### D1. 检索性能优化

- 范围：HybridRetriever / Vector / Query cache
- 任务：
- 候选池参数调优（BM25 + semantic + reliability）
- query 缓存命中率提升
- 验收：P95 延迟、召回率达到阶段阈值。

### D2. 中间件落地到主链路

- 范围：MiddlewareStack 组装
- 任务：
- `RateLimitMiddleware`、`CacheMiddleware`、`PIIMiddleware` 按环境启用策略
- 告警与日志归因
- 验收：生产开关可控，异常时可一键回退。

### D3. Prompt 评测纳入发布门禁

- 范围：PromptRegistry + EvalService + 发布流程
- 任务：
- case 级失败报告自动化
- 回归失败阻断 stable release
- 验收：每次发布都有可追溯评测包。

---

## 4. 里程碑与验收指标

### 里程碑

1. M1（Phase A 结束）：稳定性收口，回归全绿。
2. M2（Phase B 结束）：核心业务闭环可演示、可自测。
3. M3（Phase C 结束）：SQL 能力达到 Gate B 准入条件。
4. M4（Phase D 结束）：工程化门禁与性能指标达标。

### 指标（执行期）

- Query 失败率：持续下降，异常有明确降级原因。
- Query P95：逐步降低（按阶段周报跟踪）。
- RAG 召回：相对基线提升（目标 >=15%）。
- Prompt 门禁：持续可跑，失败可定位到 case。
- 回归结果：核心测试集必须全绿。

---

## 5. 每批执行模板（必须）

1. 实现：仅改本批必要文件。
2. 自测：至少包含新增测试 + 受影响回归。
3. 文档：更新执行日志（改动、命令、结果、风险）。
4. 提交：独立 commit，message 使用 `phaseX(scope): ...`。
5. 产物：基线/门禁/回归报告写入 `docs/v1/`。

---

## 6. 风险与回滚

- 风险：外部数据源波动、接口兼容破坏、检索策略回归、SQL 误放行。
- 回滚策略：
- 所有新增能力通过配置开关控制；
- API 兼容层保留；
- 新策略异常时回退到单轮/本地 fallback；
- SQL 功能默认 PoC，仅白名单可用。

---

## 7. 交付清单（验收时应存在）

- 执行日志：`docs/v1/2026-02-20-stockpilotx-optimization-execution-log-v1.1.md`
- 收官报告：`docs/v1/2026-02-20-stockpilotx-optimization-final-closure.md`
- 基线报告：`docs/v1/baseline/*.md|json`
- Gate 报告：`docs/v1/baseline/gate-decision-*.md`
- Prompt 评测报告：`docs/v1/prompt-evals/*.md|json`

---

结论：本计划是面向当前 StockPilotX 代码现状的可执行版本，按批次推进即可持续交付、持续回归、可控上线。
