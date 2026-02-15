# Round-N DeepThink 真流式改造记录（V2 SSE）

## 背景
之前 DeepThink 的交互路径是：
1. `POST /v1/deep-think/.../rounds` 同步执行整轮。
2. 执行完成后再通过 `/stream` 做回放。

这导致前端在执行阶段缺少持续反馈，用户体感为“长时间等待”。

## 本轮决策
- 保留 v1 路径，新增 v2 真流式路径。
- 服务层统一执行核心，避免 v1/v2 逻辑漂移。

## 实现摘要

### Service 层
- 增加 session 级互斥，避免同会话并发 round 污染状态。
- 增加统一事件包装器，强制补齐元字段（session_id/round_id/round_no/event_seq/emitted_at）。
- 引入 `deep_think_run_round_stream_events`：
  - 执行中实时 `yield` 事件。
  - 完成后落库事件并返回 snapshot。
  - 异常路径统一 `error + done(ok=false)`。
- `deep_think_run_round` 改为消费流式核心，保持旧返回契约。

### API 层
- 新增：`POST /v2/deep-think/sessions/{session_id}/rounds/stream`
- 返回 `StreamingResponse` (`text/event-stream`)。

### 前端层
- 默认走 v2 流式执行；保留 v1 回退开关。
- 流式过程中将事件直接写入回放面板。
- 流结束后强制刷新 session + archive，保证一致性。

## 验证
- service/http_api 新增并通过 v2 流式用例。
- 既有 deep-think 与 a2a 回归用例通过。
- Next.js build 通过。

## 结果
- DeepThink 从“执行后回放”升级为“执行即可见”。
- 事件存档与回放治理能力保留。
- v1 兼容保持，支持灰度切换。
