# Round-N 实现记录：DeepThink V2 真流式执行（SSE）

## 1. 目标
- 将 DeepThink 从“先执行完成再回放”升级为“执行中实时推送事件”。
- 保持 v1 接口兼容，避免影响既有调用方。

## 2. 关键改动

### 2.1 后端 Service（`backend/app/service.py`）
- 新增会话级互斥控制：
  - `_deep_round_try_acquire(session_id)`
  - `_deep_round_release(session_id)`
- 新增统一事件封装：
  - `_deep_stream_event(...)`
  - 统一补齐 `session_id/round_id/round_no/event_seq/emitted_at`。
- 新增真流式执行主链路：
  - `deep_think_run_round_stream_events(session_id, payload)`
  - 在执行过程中按阶段 `yield` 事件：
    - `round_started`
    - `progress`
    - `budget_warning`（条件触发）
    - `agent_opinion_delta`
    - `agent_opinion_final`
    - `critic_feedback`（条件触发）
    - `arbitration_final`
    - `replan_triggered`（条件触发）
    - `round_persisted`
    - `done`
- `deep_think_run_round(...)` 改为复用上述流式核心并消费事件，保持 v1 行为兼容。

### 2.2 后端 HTTP API（`backend/app/http_api.py`）
- 新增接口：
  - `POST /v2/deep-think/sessions/{session_id}/rounds/stream`
- 该接口直接返回 `text/event-stream`，执行与推送同请求完成。

### 2.3 前端 DeepThink 页面（`frontend/app/deep-think/page.tsx`）
- 新增开关：
  - `ENABLE_DEEPTHINK_V2_STREAM`（默认启用，可通过 `NEXT_PUBLIC_DEEPTHINK_V2_STREAM=0` 回退 v1）。
- 新增函数：
  - `runDeepThinkRoundStreamV2(sessionId, requestPayload)`
- `runDeepThinkRound()` 改造：
  - 默认走 v2 真流式。
  - 失败时回退错误提示。
  - 结束后统一刷新 session 与事件归档，确保前后端一致。

## 3. 兼容性策略
- v1 接口保留不变：
  - `POST /v1/deep-think/sessions/{session_id}/rounds`
  - `GET /v1/deep-think/sessions/{session_id}/stream`
- v2 为新增能力，不破坏旧客户端。

## 4. 自测与结果
- 通过测试：
  - `tests/test_service.py::ServiceTestCase::test_deep_think_v2_stream_round`
  - `tests/test_service.py::ServiceTestCase::test_deep_think_v2_stream_round_mutex_conflict`
  - `tests/test_service.py::ServiceTestCase::test_deep_think_session_and_round`
  - `tests/test_http_api.py::HttpApiTestCase::test_deep_think_v2_round_stream`
  - `tests/test_http_api.py::HttpApiTestCase::test_deep_think_and_a2a`
- 前端构建通过：
  - `npm --prefix frontend run build`

## 5. 风险与后续
- 当前仍是单请求长连接推送，后续可补充：
  - `Last-Event-ID` 断线续传。
  - 更细粒度吞吐与时延指标（首事件 p95 / 完成时延 p95 / 错误率）。
  - 灰度开关与自动回退策略。

## 6. 追加修复（流式可见性增强）
- 前端 SSE 解析器增强：兼容 `\n\n` 与 `\r\n\r\n` 分隔，尾包无空行时也可解析。
- v2 流式失败自动回退 v1 路径，避免用户卡在空白等待。
- 执行前自动清空事件过滤条件，避免筛选条件导致误判“无流式输出”。

## 7. 追加修复（/v1/query/stream 首包时延与前端解析统一）
- 后端 `query_stream_events` 调整为先发送 `start`，再执行数据刷新，避免首包被前置刷新耗时阻塞。
- 新增 `progress` 阶段事件（data_refresh/retriever/model），用于前端显示阶段进度。
- 前端 `runAnalysis` 改为复用统一 SSE 读取器 `readSSEAndConsume`，统一处理 CRLF 分隔与尾包。
- 当 `query/stream` 连接成功但未收到任何事件时，前端将直接报错，避免“静默等待”。
