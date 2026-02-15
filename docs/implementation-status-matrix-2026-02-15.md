# StockPilotX 技术实现现状矩阵（2026-02-15）

> 基准文档：`docs/a-share-agent-system-executable-spec.md`

## 1. 总体判定
- 核心系统属于“可运行工程骨架 + 关键能力可验证”状态。
- 多数能力已具备最小可用实现，但部分为轻量版或可选启用，不等同于生产增强版。

## 2. 技术点落地矩阵

| 技术点 | 实现状态 | 代码锚点 | 说明 |
|---|---|---|---|
| 状态管理 | 已实现（请求态） | `backend/app/state.py`, `backend/app/agents/workflow.py` | 采用 `AgentState` 贯穿路由、检索、分析、引用构建。 |
| LangSmith | 已实现（可选启用） | `backend/app/observability/tracing.py` | 配置 `LANGSMITH_API_KEY` 后上报；无 key 自动本地降级。 |
| 多 Agent 协作 | 已实现（同进程阶段式） | `backend/app/agents/workflow.py` | Router/Data/RAG/Analysis/Report/Critic 在同一工作流类中编排。 |
| Long-term Memory | 已实现 | `backend/app/memory/store.py`, `backend/app/service.py` | SQLite 持久化记忆，查询前注入 memory hint，完成后回写。 |
| LangGraph 集成 | 已实现（主链路可切换） | `backend/app/agents/langgraph_runtime.py`, `backend/app/service.py` | `query` 主链路支持 LangGraph runtime，支持 direct 回退。 |
| Middleware 工程化 | 已实现 | `backend/app/middleware/hooks.py` | 含 `before/after`、`wrap_model_call`、`wrap_tool_call` 洋葱模型。 |
| Deep Agents | 已实现（轻量并行） | `backend/app/agents/workflow.py` | 通过子任务拆分 + `ThreadPoolExecutor` 并行检索。 |
| 文档处理工程 | 已实现 | `backend/app/docs/pipeline.py`, `backend/app/service.py` | 上传、索引、质量门禁、复核队列已接通。 |
| RAG（AgenticRAG） | 已实现（轻量版） | `backend/app/rag/retriever.py` | BM25 + ngram 向量近似 + rerank，不是外部向量库方案。 |
| RAG（GraphRAG） | 已实现（可选Neo4j） | `backend/app/rag/graphrag.py` | 默认 InMemory 图；配置 `NEO4J_*` 才切真实图数据库。 |
| RAG 测评 | 已实现 | `backend/app/rag/evaluation.py`, `tests/test_rag_retrieval.py` | 提供检索指标基线测试。 |
| Prompt 工程化管理 | 已实现 | `backend/app/prompt/registry.py`, `backend/app/prompt/runtime.py`, `backend/app/prompt/evaluator.py` | 三层模板、版本、评测、发布门禁已具备。 |
| LangChain 1.0 | 部分实现（已接入工具绑定与模板） | `backend/app/agents/tools.py`, `backend/app/prompt/runtime.py` | 已接入 `StructuredTool` + `ChatPromptTemplate`，主编排仍以 LangGraph + 自有 workflow 为核心。 |
| 外部 LLM 多 Provider | 已实现 | `backend/app/llm/gateway.py`, `backend/config/llm_providers.local.json` | 支持 openai-responses/openai-chat/anthropic-messages，并含回退策略。 |
| 流式输出（SSE） | 已实现 | `backend/app/http_api.py`, `backend/app/agents/workflow.py`, `frontend/app/page.tsx` | `/v1/query/stream` 端到端可用，前端可显示 provider/model/api。 |

## 3. API 与前端覆盖结论
- 后端：`/v1/*` 接口已覆盖核心域与运维域。
- 前端：主站、预测、报告、文档中心、运维页均已存在，且 2026-02-15 新增了文档上传/复核、调度状态与停启、报告生成与版本查询、`auth me/refresh` 操作入口，以及技术点能力快照展示（`ops/evals`）。

## 4. 自测证据
- 命令：`\.venv\Scripts\python -m pytest -q`
- 结果：`57 passed in 16.97s`

## 5. 当前主要差距
1. LangChain 的 structured output / agent executor 尚未进入主执行链（目前仅工具绑定已接入）。
2. RAG 检索层尚未接入 embedding + 向量数据库（当前为轻量实现）。
3. 多 Agent 仍是单进程内编排，未引入独立 Agent Runtime 或跨进程协作。

## 6. 建议下一阶段
1. 将 `query_stream` 迁移到 LangGraph 事件流节点。
2. 引入可替换向量后端（Milvus/Qdrant/PGVector 其一）并保留当前轻量检索作回退。
3. 形成线上门禁：RAG 指标阈值 + Prompt Gate + 模型回退演练。

## Round-B Update (2026-02-15)
- Query evidence surface: `analysis_brief` response field + stream event implemented.
- Prediction source quality: real history first, synthetic fallback explicitly tagged.
- Frontend trust UX: confidence/freshness/source-mode/rationale visible in `/` and `/predict`.
- Validation: backend `57 passed`, frontend build + tsc passed.

## Round-C Update (2026-02-15)
- Added ops endpoints: `/v1/ops/agent/debate`, `/v1/ops/rag/quality`, `/v1/ops/prompts/compare`.
- Added prompt runtime versioned replay and registry version listing.
- Integrated new ops panels in frontend `/ops/evals`.
- Full validation: backend 58 passed; http_api 12 passed; frontend build + tsc passed.

## Round-D Update (2026-02-15)
- Prompt compare upgraded from same-version replay to real multi-version replay (1.0.0 vs 1.1.0).
- Agent debate upgraded to optional LLM-parallel mode with fallback.
- RAG quality upgraded with online continuous dataset persistence and merged metrics.
- Validation kept green: backend 58 passed; frontend build + typecheck passed.

## Round-E Update (2026-02-15)
- Added DeepThink APIs: `/v1/deep-think/sessions`, `/v1/deep-think/sessions/{session_id}/rounds`, `/v1/deep-think/sessions/{session_id}`, `/v1/deep-think/sessions/{session_id}/stream`.
- Added internal A2A adapter APIs: `/v1/a2a/agent-cards`, `/v1/a2a/tasks`, `/v1/a2a/tasks/{task_id}`.
- Added persistence tables for deep-think sessions/rounds/opinions, agent cards, a2a tasks, and shared knowledge cards.
- Validation kept green: backend 61 passed; frontend build + typecheck passed.

## Round-F Update (2026-02-15)
- DeepThink round payload upgraded with planner and governance fields: `task_graph`, `budget_usage`, `replan_triggered`, `stop_reason`.
- Added planner/replan control path: disagreement threshold can trigger replan task insertion.
- Added budget guard path: exceeded budget yields deterministic stop reason `DEEP_BUDGET_EXCEEDED`.
- Validation: backend 63 passed; frontend build passed; frontend typecheck currently blocked by existing `tsconfig` include mismatch.

## Round-G Update (2026-02-15)
- Frontend IA refactor: `/` now serves as product-style navigation homepage, while deep analysis is moved to `/deep-think`.
- Added dedicated route `frontend/app/deep-think/page.tsx` and updated top nav in `frontend/app/layout.tsx`.
- Kept existing analysis capability intact by migrating previous workspace page to the new route.
- Validation: frontend build passed; frontend typecheck passed; backend regression remained green (63 passed).

## Round-H Update (2026-02-15)
- Added DeepThink governance panel in `/deep-think` for round-level execution and inspection.
- Visualized core governance signals: `task_graph`, `conflict_sources`, `budget_usage`, `replan_triggered`, `stop_reason`.
- Added frontend A2A dispatch action (`/v1/a2a/tasks`) to trigger supervisor-led next round from UI.
- Added DeepThink stream replay panel for SSE event review and latest round trace visibility.
- Validation kept green: backend `63 passed`; frontend build passed; frontend typecheck passed.

## Round-I Update (2026-02-15)
- Added persisted DeepThink event archive with query API: `/v1/deep-think/sessions/{session_id}/events`.
- Added backend event snapshot storage/replay path (`deep_think_event`) with fallback regeneration for historical rounds.
- Added frontend archive controls (`加载会话存档` + `archive_events` counter) for audit-friendly replay.
- Added frontend round-to-round opinion diff panel and conflict drill-down table (including `evidence_ids`).
- Validation kept green: targeted backend tests `27 passed`; full backend regression `63 passed`; frontend build + typecheck passed.

## Round-J Update (2026-02-15)
- Extended DeepThink archive API with event-level filter: `/v1/deep-think/sessions/{session_id}/events?event_name=...`.
- Added session-scoped archive retention trimming to control event growth after each round snapshot write.
- Added frontend archive filter console (`round/event/limit`) and filtered replay rendering in `/deep-think`.
- Added regression assertions for event filter contract and retention cap behavior.
- Validation kept green: targeted backend tests `27 passed`; full backend regression `63 passed`; frontend build + typecheck passed.

## Round-K Update (2026-02-15)
- Extended DeepThink archive API with cursor and time-window filters: `/v1/deep-think/sessions/{session_id}/events?cursor=&created_from=&created_to=`.
- Added DeepThink archive export endpoint: `/v1/deep-think/sessions/{session_id}/events/export?format=jsonl|csv`.
- Upgraded frontend DeepThink console with archive time filters, next-page replay loading, and one-click JSONL/CSV export.
- Added regression assertions for pagination metadata (`has_more`, `next_cursor`) and export content-type/body contracts.
- Validation kept green: targeted backend tests `27 passed`; full backend regression `63 passed`; frontend build + typecheck passed.
