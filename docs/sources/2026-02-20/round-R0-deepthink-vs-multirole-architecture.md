# ROUND-R0 技术决策：DeepThink 多Agent vs 多角色 StateGraph

## 1. 背景
当前产品存在“预测研究台”和“DeepThink”在用户感知上相似的问题：
- 两者都输出结论与风险提示，边界不清晰。
- 用户无法理解何时使用快速模式，何时进入深度仲裁模式。

## 2. 核心区分（技术与业务）

### 2.1 DeepThink 多Agent（外层编排）
- 作用：轮次治理、任务图、冲突追踪、重规划、预算管理。
- 输出：round timeline、task graph、conflict sources、replan signals。
- 强项：流程可追踪与仲裁治理。

### 2.2 多角色 StateGraph（内层决策引擎）
- 作用：角色化分析、辩论、裁决，生成结构化决策。
- 输出：role_opinions、consensus_signal、conflict_sources、execution_plan。
- 强项：同一问题内部推理质量与决策一致性。

## 3. 结合方案（分层架构）
采用“外层编排 + 内层决策”的分层：
1. DeepThink 保留外层治理能力，不替代。
2. 多角色引擎作为内层决策核心，被 Predict/Report/DeepThink 复用。
3. 统一决策契约 `MultiRoleDecision`，避免三套逻辑漂移。

## 4. 迁移策略
1. Shadow：并跑对比，不改用户结果。
2. Partial takeover：先接管 Predict，再接 Report。
3. Full takeover：DeepThink 的核心观点来源切换为多角色裁决结果。

## 5. 风险与回滚
- 风险：时延上升、结果漂移、接口字段扩展影响前端。
- 回滚：通过 `multi_role_enabled` 开关退回原有预测输出路径。

## 6. 验收标准
- Predict 可输出多角色裁决，且研报不足不再硬阻断。
- Report/DeepThink 可消费统一决策契约。
- 全链路有 trace_id 可回放诊断。
