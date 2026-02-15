# DeepAgent 实现总结（技术 / 架构 / 业务）

日期：2026-02-15  
适用范围：StockPilotX 中 DeepThink + Internal A2A + Archive Governance（Round-E ~ Round-M）

## 1. 背景与目标
- 背景：在原有轻量问答链路（`/v1/query`）之外，建立可持续多轮推理的 DeepAgent 能力。
- 目标：
  - 支持多角色深度协作推理（会话化、轮次化、可回放）。
  - 支持治理能力（预算、重规划、保留策略、审计、指标）。
  - 支持可交付能力（存档查询、导出、异步任务、失败重试）。
  - 保持原有主链路兼容，不破坏存量接口。

## 2. 架构总结
### 2.1 分层结构
- 接口层（API）：
  - `backend/app/http_api.py`
  - 暴露 DeepThink、A2A、Archive、Ops 指标等 REST 契约。
- 编排层（Application Service）：
  - `backend/app/service.py`
  - 负责 DeepThink 轮次执行、任务规划、预算治理、导出任务执行与审计埋点。
- 域服务层（Web Domain Service）：
  - `backend/app/web/service.py`
  - 负责会话/轮次/事件/任务/审计等存储模型的读写与聚合。
- 存储层（SQLite）：
  - `backend/app/web/store.py`
  - 持久化 DeepThink 会话、轮次观点、事件归档、导出任务、审计指标基础数据。
- 前端交互层（DeepThink 工作台）：
  - `frontend/app/deep-think/page.tsx`
  - 提供轮次控制、事件回放、差分对比、过滤分页、导出任务追踪。

### 2.2 关键链路
1. 会话与轮次：
   - 创建会话 -> 执行轮次 -> 产出观点与裁决 -> 写入 round/opinion/event。
2. 治理链路：
   - 每轮预算快照 -> 超限停止（`DEEP_BUDGET_EXCEEDED`）-> 冲突触发重规划。
3. 存档链路：
   - 事件按 `session_id/round_id/event_seq` 持久化，支持过滤/分页/导出。
4. 导出任务链路：
   - `queued -> running -> completed/failed`，支持重试与尝试次数追踪。
5. 审计链路：
   - query/export/task 全链路记录，按窗口聚合运营指标。

## 3. 技术实现总结
### 3.1 DeepAgent 主能力（Round-E/F）
- 多角色协作集合（8 角色）与会话化执行落地。
- 轮次结构增强：`task_graph`、`budget_usage`、`replan_triggered`、`stop_reason`。
- 内部 A2A 适配器落地：
  - `agent card` 注册与发现
  - `task lifecycle` 创建/推进/完成

### 3.2 可观测与可回放（Round-I/J/K）
- 事件存档模型落地：`deep_think_event`。
- 支持按 `round/event/time/cursor` 查询回放。
- 支持 JSONL/CSV 导出，形成可审计交付件。
- 前端增加差分面板、冲突下钻、分页回放、导出入口。

### 3.3 稳定性与治理强化（Round-L/M）
- 异步导出任务模型：
  - `deep_think_export_task` 支持创建、轮询、下载。
- 审计模型：
  - `deep_think_archive_audit` 记录动作、状态、时延、结果量、导出字节。
- 严格时间格式校验：
  - `YYYY-MM-DD HH:MM:SS`，避免跨端时间解析歧义。
- 重试机制（Round-M）：
  - 增加 `attempt_count/max_attempts`
  - 原子 claim + requeue + backoff retry
- 指标增强（Round-M）：
  - `p50/p95/p99`、`slow_calls_over_1000ms`
  - `by_action_status`、`top_sessions`

### 3.4 工程质量与交付方式
- 回归测试持续扩展：
  - 服务层与 API 层共同覆盖 DeepThink / A2A / Archive / Export Task。
- 每轮强制文档化：
  - round 文档 + 专栏记录 + checklist + traceability + status matrix 同步更新。
- 按轮提交，保证实现与证据可追溯。

## 4. 业务价值总结
### 4.1 对研究与决策
- 从“单次回答”升级为“多轮研判”：
  - 能看到不同角色观点、冲突来源、裁决路径与轮次演进。
- 从“结果导向”升级为“过程可解释”：
  - 差分、下钻、回放、导出可直接用于复盘与汇报。

### 4.2 对风控与合规
- 引入预算治理与停止机制，降低无约束推理风险。
- 引入审计日志与结构化导出，提高审计可达性。
- 严格时间契约与保留策略减少数据解释分歧。

### 4.3 对运营与交付
- 导出任务异步化 + 重试，提高大数据量场景稳定性。
- 指标分位数与会话维度聚合，提升排障效率与容量评估能力。
- 前端工作台能力完整，便于非研发角色直接操作与验证。

## 5. 当前边界与待优化项
- A2A 目前为内部适配层，尚未扩展到跨进程/跨服务标准互联。
- 导出任务执行仍为进程内线程池，跨实例一致性依赖后续外部队列化。
- 指标聚合为应用侧窗口计算，超长窗口性能可继续优化为预聚合。

## 6. 结论
- 本次 DeepAgent 实现已从 MVP 进入“可运行 + 可治理 + 可审计 + 可交付”的工程阶段。
- 技术上完成了从推理能力到治理能力、再到稳定性能力的连续闭环。
- 架构上形成了接口、编排、域服务、存储、前端的清晰分层。
- 业务上显著提升了多轮研判的可解释性、可追溯性与可运营性。
