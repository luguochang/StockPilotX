# Deep Think 接口规范：会话、轮次、流式裁决

日期：2026-02-15
状态：接口规范稿（用于下一阶段编码）

## 1. 目标
- 提供“深度思考”专用接口，不与普通问答混用。
- 支持多 agent 多轮协商、流式中间态、最终仲裁输出。

## 2. API 列表
- `POST /v1/deep-think/sessions`
- `POST /v1/deep-think/sessions/{session_id}/rounds`
- `GET /v1/deep-think/sessions/{session_id}`
- `GET /v1/deep-think/sessions/{session_id}/stream`

## 3. 关键请求结构
- 创建会话（示例字段）：
  - `question`
  - `stock_codes`
  - `agent_profile`（pm/quant/risk/critic/supervisor）
  - `max_rounds`
  - `budget`（tokens_ms/tool_calls）
  - `mode`（internal_orchestration|a2a_hybrid）

## 4. 关键响应结构
- `DeepThinkSession`
  - `session_id`
  - `status`
  - `current_round`
  - `created_at`
  - `trace_id`
- `DecisionRound`
  - `round_id`
  - `opinions[]`
  - `consensus`
  - `disagreement_score`
  - `evidence_ids[]`

## 5. SSE 事件定义
- `round_started`
- `agent_opinion_delta`
- `agent_opinion_final`
- `critic_feedback`
- `arbitration_final`
- `done`

## 6. 错误码建议
- `DEEP_BUDGET_EXCEEDED`
- `DEEP_AGENT_TIMEOUT`
- `DEEP_EVIDENCE_INSUFFICIENT`
- `DEEP_POLICY_BLOCKED`

## 7. 与现有接口关系
- 普通问答仍走 `/v1/query` 与 `/v1/query/stream`。
- 深推理显式走 `/v1/deep-think/*`，避免污染轻查询路径。

## 8. 验收标准
- 同一会话可追溯每轮意见。
- 可看到每个 agent 的中间输出与最终输出。
- 分歧高时必须输出“冲突解释”。

