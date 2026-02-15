# Round-F 实现记录：DeepThink Planner、Budget、Replan

日期：2026-02-15

## 1. 本轮目标
- 把 DeepThink 从“固定轮次执行”升级为“有任务规划和预算约束的轮次执行”。
- 补齐可观测事件：预算预警与重规划触发。

## 2. 落地内容
- 每轮新增 `task_graph`：
  - 基于问题语义动态生成任务（量化、风险、叙事、合规，按需加宏观/执行任务）。
- 每轮新增 `budget_usage`：
  - `limit/used/remaining/warn/exceeded`。
- 每轮新增 `stop_reason`：
  - 超预算时标记 `DEEP_BUDGET_EXCEEDED` 并提前收敛输出。
- 每轮新增 `replan_triggered`：
  - 分歧高于阈值时触发补证任务，SSE 输出 `replan_triggered` 事件。

## 3. 数据结构扩展
- `deep_think_round` 增量字段：
  - `task_graph`
  - `replan_triggered`
  - `stop_reason`
  - `budget_usage`

## 4. 接口行为变化（兼容）
- 原接口不变，仅返回字段增强：
  - `/v1/deep-think/sessions/{session_id}`
  - `/v1/deep-think/sessions/{session_id}/rounds`
  - `/v1/deep-think/sessions/{session_id}/stream`

## 5. 自测结果
- 后端定向：27 passed
- 后端全量：63 passed
- 前端 build：通过
- 前端 tsc：受现有 tsconfig include 规则影响失败（与本轮后端改造无关）

## 6. 工程意义
- DeepThink 从“多角色并行输出”迈向“可治理的决策循环”：
  - 可解释：每轮先有计划再有结论
  - 可治理：预算触顶有明确停止机制
  - 可演进：replan 机制可平滑升级到真实 planner/todo graph
