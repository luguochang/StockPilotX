# Round-E 实现记录：DeepThink + Internal A2A MVP

日期：2026-02-15

## 1. 本轮交付
- DeepThink 接口落地：
  - `POST /v1/deep-think/sessions`
  - `POST /v1/deep-think/sessions/{session_id}/rounds`
  - `GET /v1/deep-think/sessions/{session_id}`
  - `GET /v1/deep-think/sessions/{session_id}/stream`
- 内部 A2A 适配层落地：
  - `GET /v1/a2a/agent-cards`
  - `POST /v1/a2a/tasks`
  - `GET /v1/a2a/tasks/{task_id}`

## 2. 架构要点
- DeepThink 与普通 query 分流，避免轻路径被复杂逻辑污染。
- 8 角色方案：
  - supervisor / pm / quant / risk / critic / macro / execution / compliance
- 轮次输出具备：
  - `consensus_signal`
  - `disagreement_score`
  - `conflict_sources`
  - `counter_view`

## 3. 数据模型新增
- `deep_think_session`
- `deep_think_round`
- `deep_think_opinion`
- `agent_card_registry`
- `a2a_task`
- `group_knowledge_card`

## 4. 关键取舍
- A2A 先做内部适配层，不引入外部依赖，先验证生命周期和可追溯。
- 深推理轮次先使用“规则+可选LLM并行”，后续再切 planner 动态任务树。
- 共享知识卡采用质量门禁，不做全量跨用户直接复用。

## 5. 质量验证
- 后端定向测试：`25 passed`
- 后端全量测试：`61 passed`
- 前端 build/typecheck：通过

## 6. 下一步
- 增加显式 budget 消耗与 stop reason。
- 增加 re-plan 触发条件与轨迹评测指标。
- 给 DeepThink 增加独立前端页面与回放面板。
