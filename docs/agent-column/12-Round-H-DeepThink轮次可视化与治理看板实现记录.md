# Round-H 实现记录：DeepThink 轮次可视化与治理看板

日期：2026-02-15  
对应轮次：Round-H  
关联路径：`frontend/app/deep-think/page.tsx`

## 1. 本轮目标
- 把 Round-E / Round-F 已落地的后端治理字段真正“可视化”到前端：
  - `task_graph`
  - `conflict_sources`
  - `budget_usage`
  - `replan_triggered`
  - `stop_reason`
- 让开发和运营在同一页面完成：
  - 会话创建
  - 下一轮执行（直连）
  - 下一轮执行（A2A 派发）
  - 轮次事件流回放

## 2. 实现思路
- **协议不扩展**：优先复用既有 `/v1/deep-think/*` 与 `/v1/a2a/tasks`，避免前后端协议漂移。
- **页面分区**：保持原有“行情/报告”区域不动，新增独立的 DeepThink 治理区块。
- **数据结构前置声明**：先在前端补齐 DeepThink 类型定义，再挂状态和动作，降低后续字段改动成本。
- **统一 SSE 读取器**：新增 `readSSEAndConsume`，避免 query-stream 与 deep-think-stream 两套解析逻辑继续分叉。

## 3. 关键落地点
- 新增状态：
  - `deepSession`, `deepLoading`, `deepStreaming`, `deepError`
  - `deepStreamEvents`（最多保留 80 条）
  - `deepLastA2ATask`
- 新增动作：
  - `startDeepThinkSession`
  - `runDeepThinkRound`
  - `runDeepThinkRoundViaA2A`
  - `refreshDeepThinkSession`
  - `replayDeepThinkStream`
- 新增可视化组件：
  - **Round Timeline**：展示每轮共识、分歧、重规划与停止原因
  - **Task Graph Table**：展示最新轮次任务图
  - **Budget Panel**：token/time/tool 三维预算进度条
  - **Conflict Chart**：分歧得分 + 冲突源数量双序列
  - **Opinion Table**：Agent 信号与置信度
  - **SSE Replay List**：事件名 + payload 摘要

## 4. A2A 与 DeepAgent 的工程意义
- 本轮把“深度推理治理”从后端字段提升为前端操作闭环：
  - 这使 A2A 派发（`/v1/a2a/tasks`）不再是黑盒触发，而是可追踪执行过程。
- 深度接口的价值不是“多轮本身”，而是“多轮治理可观测”：
  - 用户可看到每轮的预算约束、冲突来源与重规划触发条件。
- 对专栏写作的帮助：
  - 可以基于真实界面解释 A2A（任务派发协议）和 DeepAgent（多轮治理执行体）的职责边界。

## 5. 变更文件
- `frontend/app/deep-think/page.tsx`
- `docs/rounds/2026-02-15/round-H-deepthink-round-visualization.md`
- `docs/agent-column/12-Round-H-DeepThink轮次可视化与治理看板实现记录.md`
- `docs/agent-column/00-总览-多agent-a2a-deepagent.md`
- `docs/implementation-checklist.md`
- `docs/spec-traceability-matrix.md`
- `docs/reports/2026-02-15/implementation-status-matrix-2026-02-15.md`

## 6. 本轮自测
- `.\.venv\Scripts\python -m pytest -q` -> `63 passed in 28.77s`
- `cd frontend && npm run build` -> passed
- `cd frontend && npx tsc --noEmit` -> passed

## 7. 下一轮建议
- 增加“跨轮次观点差分（opinion diff）”。
- 增加“冲突源 -> 证据ID -> 原始引用”的下钻面板。
- 增加“会话回放存档”能力，支持跨天审计与复盘。
