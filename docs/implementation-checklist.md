# StockPilotX 实施执行清单（SSOT）

## 1. 使用规则（强约束）
- 本文件是唯一执行总表（Single Source of Truth）。
- 每个任务必须绑定 `Spec Ref`（主文档章节编号）与 `Task ID`。
- 状态规则：
  - `[ ]` 待执行
  - `[-]` 执行中
  - `[x]` 已完成（必须附 Evidence）
  - `[!]` 阻塞（必须附阻塞原因与解除条件）
- 任何任务在无 Evidence 的情况下禁止标记为 `[x]`。

## 2. 当前基线
- 主规范：`docs/a-share-agent-system-executable-spec.md`
- 追踪矩阵：`docs/spec-traceability-matrix.md`
- 全局门禁：`docs/global-constraints.md`
- 最近更新时间：2026-02-15

## 3. 里程碑总览
- `M0` 文档治理与基线稳定（当前）
- `M1` FastAPI 真联调 + API 契约测试
- `M2` 免费数据源真实接入 + 标准化 + 容错
- `M3` RAG/GraphRAG 工程化
- `M4` Prompt/LangSmith 门禁化
- `M5` 前端接入与端到端联调
- `M6` 上线前工程化治理（SLO/安全/回滚）

## 4. 任务总表
| Task ID | Spec Ref | Scope | Status | DoD（完成标准） | Evidence |
|---|---|---|---|---|---|
| GOV-001 | S1,S16 | 建立单一执行清单与流程规则 | [x] | 清单创建并定义状态机 | 创建 `docs/implementation-checklist.md`（2026-02-13） |
| GOV-002 | S1,S16 | 建立需求追踪矩阵 | [x] | 主文档 2~17 节均可映射到任务 | 创建 `docs/spec-traceability-matrix.md`（2026-02-13） |
| GOV-003 | S14,S15 | 建立全局强约束门禁文档 | [x] | 硬门禁、软门禁、阻塞规则可执行 | 创建 `docs/global-constraints.md`（2026-02-13） |
| BASE-001 | S8,S12,S17.3 | 多 Agent MVP 主流程（Router->Data->RAG->Analysis->Report->Critic） | [x] | 问答路径可运行并返回 citations | `python -m unittest ...` 7/7 通过（2026-02-13） |
| BASE-002 | S17.6 | Middleware 生命周期与 wrap 链路 | [x] | before/after + wrap 已接入 | 同上测试通过（2026-02-13） |
| BASE-003 | S17.4 | Long-term Memory 基线 | [x] | Memory Store 可写入/读取 | 同上测试通过（2026-02-13） |
| BASE-004 | S12 | API 方法层契约实现 | [x] | Query/Report/Ingest/Docs/Evals 方法可调用 | 同上测试通过（2026-02-13） |
| API-001 | S4,S12 | FastAPI 真启动联调 | [x] | 本地 uvicorn 启动 + 路由 smoke test 全通过 | `python -m uvicorn backend.app.http_api:create_app --factory --host 127.0.0.1 --port 8010` + `Invoke-WebRequest /docs` 返回 `200`（2026-02-13） |
| API-002 | S10,S12 | API 契约自动化测试 | [x] | 覆盖全部 `/v1/*` 首批契约 | 新增 `tests/test_http_api.py`，`python -m unittest discover -s tests -p "test_*.py" -v`，13/13 通过（2026-02-13） |
| DATA-001 | S5,S6,S11 | 二级行情真实源接入（腾讯->网易->新浪->雪球） | [x] | 真数据回退链生效，失败自动降级 | 新增真实适配器 `TencentLive/NeteaseLive/SinaLive/XueqiuLive` + `QuoteService.build_default()`（真实源失败自动回退 mock）；`python -m unittest ...` 17/17 通过（2026-02-13） |
| DATA-002 | S5,S9,S11 | 一级公告源接入（巨潮/交易所） | [x] | 结构化事件入库 + 引用可追溯 | 新增公告真实源适配器 `Cninfo/SSE/SZSE`，不可达时自动回退 mock；新增 `tests/test_announcements.py`，`python -m unittest ...` 19/19 通过（2026-02-13） |
| DATA-003 | S5,S14,S15 | 标准化、去重、冲突标记与可靠性评分 | [x] | `conflict_flag` 与 score 输出可用 | 在 `ingestion.py` 增加 canonical 标准化、质量校验、`conflict_flag` 规则；新增 `tests/test_ingestion_quality.py`，`python -m unittest ...` 21/21 通过（2026-02-13） |
| DATA-004 | S13 | 定时调度（Temporal/Airflow）与重试熔断 | [x] | 日级与周级任务可调度可观测 | 新增 `LocalJobScheduler`（任务注册、重试、熔断、状态查询）并在 `AShareAgentService` 注册 `intraday/daily/weekly` 任务；新增 `tests/test_scheduler.py`，`python -m unittest ...` 24/24 通过（2026-02-13） |
| RAG-001 | S6,S17.9 | Hybrid RAG 正式化（BM25+Vector+Reranker） | [x] | Recall@k/MRR/nDCG 有基线结果 | 重写 `backend/app/rag/retriever.py`（BM25+Vector+Rerank）；新增 `backend/app/rag/evaluation.py` 与 `tests/test_rag_retrieval.py`，指标基线通过（2026-02-13） |
| RAG-002 | S6,S17.9 | GraphRAG 接入 Neo4j | [x] | 关系问题走图检索，结果可引用 | 新增 `Neo4jGraphStore` + `InMemoryGraphStore` 回退，更新 `workflow` 图证据拼装；`tests/test_graphrag.py` 通过（2026-02-13） |
| RAG-003 | S9,S17.8 | 文档处理工程（PDF/DOCX/HTML + 表格 + OCR门禁） | [x] | 文档解析、分块、索引、复核队列完整 | 新增 `backend/app/docs/pipeline.py`，接入 `ingestion` 的 parse/table/chunk/review_queue；`tests/test_doc_pipeline.py` 通过（2026-02-13） |
| AGT-001 | S8,S17.7 | Deep Agents（planner + sub-agents 并行） | [x] | 复杂问题并行子任务可追溯 | `workflow` 增加 planner + ThreadPool 并行子任务；`tests/test_agents_deep_acl.py` 通过（2026-02-13） |
| AGT-002 | S8,S17.6 | Agent 工具白名单与权限控制 | [x] | 角色级工具 ACL 强制生效 | 新增 `backend/app/agents/tools.py`，按角色工具白名单强制；`tests/test_agents_deep_acl.py` 含拒绝越权用例（2026-02-13） |
| AGT-003 | S17.5,S17.10 | LangGraph 运行时接入主查询链路 | [x] | `query` 主链可走 LangGraph 且可 direct 回退 | 新增 `backend/app/agents/langgraph_runtime.py`；`service.query` 接入 runtime 选择；新增 `tests/test_langgraph_runtime.py`；`pytest` 51/51 通过（2026-02-15） |
| AGT-004 | S17.5,S17.10 | LangGraph 运行时接入流式查询链路 | [x] | `/v1/query/stream` 可输出 runtime 事件并走 runtime 实现 | `workflow_runtime.run_stream` 接入 `service.query_stream_events`；新增 `stream_runtime` SSE 事件；`tests/test_http_api.py` 与 `tests/test_langgraph_runtime.py` 增强；`pytest` 53/53 通过（2026-02-15） |
| AGT-005 | S17.10,S12 | Query 显式返回运行时 + LangChain 工具绑定落地 | [x] | `/v1/query` 返回 `workflow_runtime`；工具调用走结构化绑定且保留 ACL 回退 | `models.QueryResponse` 新增字段；新增 `LangChainToolRunner` 并接入 `workflow`；`tests/test_http_api.py` 与 `tests/test_agents_deep_acl.py` 增强；`pytest` 55/55 通过（2026-02-15） |
| AGT-006 | S17.2,S17.10,S12 | 技术点能力核查接口 + LangChain Prompt 模板渲染 | [x] | `/v1/ops/capabilities` 可返回技术点快照；PromptRuntime 优先走 LangChain 渲染并可回退 | 新增 `backend/app/capabilities.py`、`/v1/ops/capabilities`、`ops/evals` 前端展示；`PromptRuntime` 接入 `ChatPromptTemplate`；`tests/test_http_api.py` 与 `tests/test_prompt_engineering.py` 增强；`pytest` 56/56 通过（2026-02-15） |
| AGT-007 | S17.10,S12 | 细粒度 Tool Schema + 请求级 runtime 覆盖 | [x] | Tool binding 支持按工具参数 schema；`query/query_stream` 可按请求指定 `workflow_runtime` | `tools.py` 增加 `Quote/Retrieve/Graph` schema；`workflow` 注册 schema；`service` 增加 `_select_runtime`；`tests/test_http_api.py` 与 `tests/test_agents_deep_acl.py` 增强；`pytest` 57/57 通过（2026-02-15） |
| AGT-009 | S8,S12,S17.7 | DeepThink 会话接口与轮次流式裁决 MVP | [x] | `/v1/deep-think/*` 可创建会话、执行轮次、查询会话、SSE 回放轮次事件 | 新增 deep_think session/round/opinion 存储与服务编排；新增 `tests/test_service.py::test_deep_think_session_and_round` 与 `tests/test_http_api.py::test_deep_think_and_a2a`；`pytest` 61/61 通过（2026-02-15） |
| AGT-010 | S8,S12,S17.11 | 内部 A2A 适配层（agent card + task lifecycle） | [x] | `/v1/a2a/agent-cards` 与 `/v1/a2a/tasks` 可完成卡片发现与任务状态流转 | 新增 `agent_card_registry/a2a_task` 表与应用服务流程；`tests/test_service.py::test_a2a_task_lifecycle` 与 `tests/test_http_api.py::test_deep_think_and_a2a` 覆盖；`pytest` 61/61 通过（2026-02-15） |
| AGT-011 | S8,S17.7,S17.11 | DeepThink 任务规划（task graph）与重规划触发 | [x] | 每轮输出 `task_graph`，分歧超阈值可触发 `replan_triggered` | 新增 `_deep_plan_tasks` 与 `replan_triggered` 事件流；新增 `tests/test_service.py::test_deep_think_session_and_round` 与 `tests/test_http_api.py::test_deep_think_and_a2a` 增强；`pytest` 63/63 通过（2026-02-15） |
| AGT-012 | S8,S15,S17.6 | DeepThink 预算治理（budget usage + stop reason） | [x] | 每轮输出 `budget_usage`，超预算时 `stop_reason=DEEP_BUDGET_EXCEEDED` | 新增 `_deep_budget_snapshot` 与 round 字段 `budget_usage/stop_reason`；新增 `tests/test_service.py::test_deep_think_budget_exceeded_stop` 与 `tests/test_http_api.py::test_deep_think_budget_exceeded`；`pytest` 63/63 通过（2026-02-15） |
| AGT-013 | S8,S12,S17.7 | DeepThink 事件存档与可回放接口（Round-I） | [x] | 轮次事件可持久化并通过 API 查询回放 | 新增 `deep_think_event` 表与 `deep_think_replace_round_events/deep_think_list_events`；新增 `/v1/deep-think/sessions/{session_id}/events`；`tests/test_service.py` 与 `tests/test_http_api.py` 增强；`pytest` 27/27 通过（2026-02-15） |
| AGT-014 | S8,S12,S17.6 | DeepThink 事件过滤与归档保留治理（Round-J） | [x] | `/events` 支持 `event_name` 过滤，且会话事件归档受上限约束 | 增强 `deep_think_list_events(event_name)` 与 `deep_think_trim_events`；`deep_think_run_round` 支持 `archive_max_events`；`tests/test_service.py` 与 `tests/test_http_api.py` 增强；`pytest` 27/27 通过（2026-02-15） |
| AGT-015 | S8,S12,S17.6 | DeepThink 归档游标分页、时间过滤与导出接口（Round-K） | [x] | `/events` 支持 `cursor/created_from/created_to`，并提供 `/events/export`（jsonl/csv） | 新增 `deep_think_list_events_page` 与 `deep_think_export_events`，扩展 `/v1/deep-think/sessions/{session_id}/events*` 契约；`tests/test_service.py` 与 `tests/test_http_api.py` 增强；`pytest` 27/27 通过（2026-02-15） |
| AGT-016 | S8,S12,S17.6 | DeepThink 归档异步导出任务、审计日志与指标（Round-L） | [x] | 新增导出任务创建/查询/下载接口，严格时间过滤校验，提供 `/v1/ops/deep-think/archive-metrics` | 新增 `deep_think_export_task/deep_think_archive_audit` 存储与服务；扩展 `deep_think_create_export_task` 等流程；增强 `tests/test_service.py` 与 `tests/test_http_api.py`；`pytest` 28/28 通过（2026-02-15） |
| AGT-017 | S8,S12,S17.6 | DeepThink 导出任务重试强化与分位审计指标（Round-M） | [x] | 导出任务支持 `attempt_count/max_attempts` 重试链路，审计指标支持 P95/P99 与会话维度聚合 | 增强 `deep_think_export_task` 队列模型（claim/requeue/retry）与 `archive-metrics` 聚合字段；增强 `tests/test_service.py` 与 `tests/test_http_api.py`；`pytest` 29/29 通过（2026-02-15） |
| GOV-004 | S1,S16 | 每轮交付文档化与可追溯规范 | [x] | 每轮必须有独立 md 记录设计、改动、验证、风险与后续建议 | 新增 `docs/rounds/README.md` 与 `docs/rounds/2026-02-15/round-E-deepthink-a2a-mvp.md`（2026-02-15） |
| PROMPT-001 | S7 | Prompt 三层模板运行时装配 | [x] | system/policy/task 全链路启用 | 新增 `backend/app/prompt/runtime.py` 并在 `service/workflow` 接入；`tests/test_prompt_engineering.py` 通过（2026-02-13） |
| PROMPT-002 | S7,S10 | `prompt_eval_result` 回写与发布门禁 | [x] | 不达阈值禁止进入 stable | `prompt_registry` 增加 `prompt_eval_result` 表、release gate；`evals_run` 自动回写；`tests/test_prompt_persistence.py` 通过（2026-02-13） |
| PROMPT-003 | S10 | 30 条回归样本自动执行 | [x] | Golden/Boundary/RedTeam/Freshness 全跑通 | 新增 `backend/app/prompt/evaluator.py`（30 样本分组执行），评测指标并入 `/v1/evals/run`（2026-02-13） |
| OBS-001 | S4,S10,S17.2 | LangSmith 真接入（trace/eval/experiment） | [x] | trace 与 eval 可关联到 report/prompt/source | 重写 `backend/app/observability/tracing.py`，增加 `LangSmithAdapter`（有 key 真上报，无 key 本地降级）；`tests/test_observability.py` 通过（2026-02-13） |
| FRONT-001 | S4 | Next.js 前端最小闭环 | [x] | 支持 query/report/citations 展示 | 新建 `frontend/` Next.js 最小项目（query + answer + citations）；`tests/test_project_assets.py` 资产检查通过（2026-02-13） |
| FRONT-003 | S4,S12 | 首页导航化与 DeepThink 独立页面拆分 | [x] | 首页仅展示系统介绍与模块跳转，深度分析迁移到 `/deep-think` | 新增 `frontend/app/deep-think/page.tsx`，重构 `frontend/app/page.tsx` 为导航首页；更新 `frontend/app/layout.tsx` 导航；`npm run build` 通过（2026-02-15） |
| FRONT-004 | S14,S15 | 前端构建与类型检查稳定化（Round-G） | [x] | 本轮改造后 `build` 与 `tsc` 均通过 | 更新 `frontend/tsconfig.json` 并完成 `npm run build`、`npx tsc --noEmit`（2026-02-15） |
| FRONT-005 | S4,S12,S15,S17.7 | DeepThink 轮次可视化与治理看板（Round-H） | [x] | `/deep-think` 可展示 round timeline、conflict_sources 可视化、budget usage、replan/stop reason 与 SSE 回放 | 增强 `frontend/app/deep-think/page.tsx` 并接入 `/v1/deep-think/*` + `/v1/a2a/tasks`；`pytest` 63/63、`npm run build`、`npx tsc --noEmit` 通过（2026-02-15） |
| FRONT-006 | S4,S12,S15,S17.7 | DeepThink 跨轮差分与冲突下钻（Round-I） | [x] | `/deep-think` 可展示跨轮观点差分、冲突证据下钻与会话事件存档加载 | 增强 `frontend/app/deep-think/page.tsx`，新增 `deepOpinionDiffRows/deepConflictDrillRows/loadDeepThinkEventArchive` 与归档状态；`npm run build`、`npx tsc --noEmit` 通过（2026-02-15） |
| FRONT-007 | S4,S12,S15,S17.7 | DeepThink 存档筛选控制台（Round-J） | [x] | `/deep-think` 可按 round/event/limit 过滤加载归档并显示过滤回放 | 增强 `frontend/app/deep-think/page.tsx`，新增筛选控件与 `deepReplayRows`；`npm run build`、`npx tsc --noEmit` 通过（2026-02-15） |
| FRONT-008 | S4,S12,S15,S17.7 | DeepThink 存档分页与导出控制台（Round-K） | [x] | `/deep-think` 支持按时间过滤、游标翻页并导出 JSONL/CSV | 增强 `frontend/app/deep-think/page.tsx`，新增 `created_from/created_to`、`next_cursor` 翻页与导出按钮；`npm run build`、`npx tsc --noEmit` 通过（2026-02-15） |
| FRONT-009 | S4,S12,S15,S17.7 | DeepThink 存档异步导出任务与回放导航（Round-L） | [x] | `/deep-think` 支持导出任务轮询下载与上一页/回到第一页导航 | 增强 `frontend/app/deep-think/page.tsx`，新增 `deepArchiveCursorHistory` 与导出任务状态流；`npm run build`、`npx tsc --noEmit` 通过（2026-02-15） |
| FRONT-010 | S4,S12,S15,S17.7 | DeepThink 时间过滤输入规范化与导出尝试可视化（Round-M） | [x] | `/deep-think` 使用结构化时间输入并可快速设置窗口，导出任务展示尝试次数 | 增强 `frontend/app/deep-think/page.tsx`（datetime-local 规范化、最近24小时、attempt tag）；`npm run build`、`npx tsc --noEmit` 通过（2026-02-15） |
| OPS-001 | S14,S15 | 上线前工程化检查（SLO/Runbook/回滚） | [x] | 检查项完成并可审计 | 新增 `docs/ops-runbook.md`（SLO、告警、回滚、发布前检查）；`tests/test_project_assets.py` 校验存在（2026-02-13） |

## 4.1 完整 Web 应用扩展清单（新增）
> 说明：以下为“从 MVP 到完整 Web 应用”必须补齐的产品功能，避免范围遗漏。

| Task ID | Scope | Status | DoD（完成标准） | Evidence |
|---|---|---|---|---|
| WEB-001 | 用户体系与鉴权（注册/登录/租户隔离/RBAC） | [x] | 登录态、权限模型、审计日志可用 | 新增 `backend/app/web/security.py` + `backend/app/web/store.py` + `backend/app/web/service.py`，开放 `/v1/auth/*` 与 RBAC 校验（2026-02-13） |
| WEB-002 | 关注股管理与个性化看板 | [x] | watchlist CRUD + 仪表盘卡片可配置 | 新增 `/v1/watchlist*` 与 `/v1/dashboard/overview`，并在 E2E 测试覆盖（2026-02-13） |
| WEB-003 | 报告中心（列表/筛选/导出PDF/版本历史） | [x] | 报告可检索、可导出、可回溯版本 | 新增 `report_index/report_version` 持久化与 `/v1/reports*` 接口（2026-02-13） |
| WEB-004 | 文档知识库管理台（上传/索引状态/复核队列） | [x] | 文档处理状态可视化、复核闭环可操作 | 新增 `/v1/docs`、`/v1/docs/review-queue`、`/review/approve|reject`（2026-02-13） |
| WEB-005 | 数据源健康与抓取监控页面 | [x] | 每个源成功率、熔断状态、延迟可视化 | 新增 `source_health` 聚合与 `/v1/ops/data-sources/health`（2026-02-13） |
| WEB-006 | 评测与发布门禁控制台 | [x] | prompt/model 发布前门禁结果可视化 | 新增 `/v1/ops/evals/history`、`/v1/ops/prompts/releases`（2026-02-13） |
| WEB-007 | 任务调度管理页（手动触发/重试/停启） | [x] | scheduler 任务可视化操作与历史记录 | 扩展 scheduler pause/resume/history；新增 `/v1/scheduler/pause|resume`（2026-02-13） |
| WEB-008 | 告警中心与值班信息 | [x] | 告警聚合、确认、升级流程可用 | 新增 `alert_event/alert_ack` 与 `/v1/alerts`、`/v1/alerts/{id}/ack`（2026-02-13） |
| WEB-009 | 前后端 E2E 自动化（关键用户路径） | [x] | 关键路径自动化回归可跑并出报告 | 新增 `tests/test_web_endpoints.py`（登录->watchlist->reports->docs->ops）通过（2026-02-13） |
| WEB-010 | 部署交付（dev/staging/prod + CI/CD） | [x] | 一键部署、环境隔离、回滚可执行 | 新增 `docker-compose.yml`、`deploy/*.Dockerfile`、`.github/workflows/ci.yml`（2026-02-13） |
| PRED-001 | S2,S12 | 预测域 API（run/get/factors/evals） | [x] | 提供双周期预测接口并可查询详情与评测摘要 | 新增 `/v1/predict/run`、`/v1/predict/{run_id}`、`/v1/factors/{stock_code}`、`/v1/predict/evals/latest`（2026-02-13） |
| PRED-002 | S5,S11,S17.1 | 因子工程（技术+风险）与状态落盘 | [x] | 输出 MA/Momentum/Vol/Drawdown/RSI/Liquidity/Risk 因子 | 新增 `backend/app/predict/service.py` 因子计算与运行存储（2026-02-13） |
| PRED-003 | S10,S17.2,S17.9 | 预测评测摘要（IC/命中率/分层收益/回撤） | [x] | 每次预测后自动刷新 latest eval | 新增 `PredictionService._build_eval_summary()` 与 `/v1/predict/evals/latest`（2026-02-13） |
| PRED-004 | S4,S17.10 | 前端预测驾驶舱页面 | [x] | 页面展示双周期信号、因子快照与评测摘要 | 新增 `frontend/app/predict/page.tsx`，并在首页增加导航入口（2026-02-13） |
| PRED-005 | S12,S15 | 预测链路自动化测试与真实源烟测 | [x] | 单测/E2E/HTTP 契约覆盖预测链路，真实源可抓取 | `python -m unittest ...` 43/43 通过；真实烟测命中 `source=tencent`（2026-02-13） |

## 5. 每日执行日志（模板）
```md
### YYYY-MM-DD
- Completed:
  - [Task ID] ...
- In Progress:
  - [Task ID] ...
- Blocked:
  - [Task ID] ...（reason + unblock condition）
- Evidence:
  - command:
  - key output:
```

### 2026-02-13
- Completed:
  - [API-001] FastAPI 真启动联调完成
  - [API-002] API 契约自动化测试完成
  - [DATA-001] 行情真实源接入与回退链完成
  - [DATA-002] 一级公告源接入完成
  - [DATA-003] 标准化与冲突标记完成
  - [DATA-004] 定时调度与重试熔断完成
  - [RAG-001] Hybrid RAG 与检索指标基线完成
  - [RAG-002] GraphRAG Neo4j 接入与回退完成
  - [RAG-003] 文档处理工程与复核队列完成
  - [AGT-001] Deep Agents 并行子任务完成
  - [AGT-002] 工具白名单 ACL 完成
  - [PROMPT-001] Prompt 三层模板运行时装配完成
  - [PROMPT-002] Prompt 评测回写与发布门禁完成
  - [PROMPT-003] 30 条回归样本自动执行完成
  - [OBS-001] LangSmith 适配接入完成
  - [FRONT-001] Next.js 前端最小闭环完成
  - [OPS-001] 运维 Runbook 与发布检查完成
  - [WEB-001] 用户体系与鉴权完成
  - [WEB-002] 关注股与个性化看板完成
  - [WEB-003] 报告中心完成
  - [WEB-004] 文档知识库管理台完成
  - [WEB-005] 数据源健康监控完成
  - [WEB-006] 评测与门禁控制台完成
  - [WEB-007] 调度管理能力完成
  - [WEB-008] 告警中心完成
  - [WEB-009] 前后端 E2E 自动化完成
  - [WEB-010] 部署交付基础资产完成
  - [PRED-001] 预测域 API 完成
  - [PRED-002] 因子工程与预测状态存储完成
  - [PRED-003] 预测评测摘要完成
  - [PRED-004] 预测驾驶舱页面完成
  - [PRED-005] 预测链路测试与真实源烟测完成
- In Progress:
  - 无（WEB 扩展清单已完成）
- Blocked:
  - 无
- Evidence:
  - command: `python -m uvicorn backend.app.http_api:create_app --factory --host 127.0.0.1 --port 8010`
  - key output: `uvicorn_smoke_status=200`
  - command: `python -m unittest discover -s tests -p "test_*.py" -v`
  - key output: `Ran 13 tests ... OK`
  - command: `python -m unittest discover -s tests -p "test_*.py" -v`
  - key output: `Ran 17 tests ... OK`
  - command: `python -m unittest discover -s tests -p "test_*.py" -v`
  - key output: `Ran 19 tests ... OK`
  - command: `python -m unittest discover -s tests -p "test_*.py" -v`
  - key output: `Ran 21 tests ... OK`
  - command: `python -m unittest discover -s tests -p "test_*.py" -v`
  - key output: `Ran 24 tests ... OK`
  - command: `python -m unittest discover -s tests -p "test_*.py" -v`
  - key output: `Ran 39 tests ... OK`
  - command: `uvicorn + urllib 全接口端到端烟测`
  - key output: `/v1/* 全部 200（query/report/ingest/docs/evals/scheduler）`
  - command: `AShareAgentService().ingest_market_daily(['SH600000'])`
  - key output: `source=tencent, source_url=https://qt.gtimg.cn/q=sh600000（真实源命中）`
  - command: `python -m unittest discover -s tests -p "test_*.py" -v`
  - key output: `Ran 40 tests ... OK`
  - command: `python -m unittest discover -s tests -p "test_*.py" -v`
  - key output: `Ran 41 tests ... OK`
  - command: `python -m unittest discover -s tests -p "test_*.py" -v`
  - key output: `Ran 43 tests ... OK`
  - command: `cd frontend && npm install && npm run build`
  - key output: `Next.js build success, routes / /login /watchlist /reports /docs-center /ops/* compiled`
  - command: `AShareAgentService().predict_run({'stock_codes':['SH600000'],'horizons':['5d','20d']})`
  - key output: `predict_source=source_id:tencent; predict_horizons=5d+20d; eval.status=ok`

### 2026-02-15
- Completed:
  - [AGT-003] LangGraph 运行时接入主查询链路（可回退）
  - [AGT-004] LangGraph 运行时接入流式查询链路（SSE `stream_runtime`）
  - [AGT-005] Query 显式返回运行时 + LangChain 结构化工具绑定
  - [AGT-006] 技术点能力核查接口 + LangChain Prompt 模板渲染
  - [AGT-007] 细粒度 Tool Schema + 请求级 runtime 覆盖
  - 新增前后端缺口优化文档与 LangGraph 迭代记录
- Evidence:
  - command: `.\.venv\Scripts\python -m pip install -r requirements.txt`
  - key output: `langchain/langgraph/langsmith 安装完成`
  - command: `.\.venv\Scripts\python -m pytest -q`
  - key output: `57 passed`
  - command: `cd frontend && npm run build`
  - key output: `Route(app) 全量构建通过`
  - command: `cd frontend && npx tsc --noEmit`
  - key output: `通过（先 build 生成 .next/types）`
  - command: `python - <<... for ev in s.query_stream_events(...) ...>>`
  - key output: `包含 stream_runtime={'runtime':'langgraph'} 事件`
  - command: `python - <<... s.query(...) ...>>`
  - key output: `workflow_runtime=langgraph`
  - command: `GET /v1/ops/capabilities`
  - key output: `返回 capabilities 列表与 runtime/config 快照`
  - command: `POST /v1/query {"workflow_runtime":"direct"}`
  - key output: `响应 workflow_runtime=direct`

### 2026-02-15 (Round-B)
- Completed:
  - [AGT-008] `/v1/query` 增加结构化 `analysis_brief`，并在 `/v1/query/stream` 输出 `analysis_brief` SSE 事件。
  - [PRED-006] 预测链路改为“真实历史K线优先 + 合成序列兜底”，输出 `history_data_mode/history_source/history_sample_size`。
  - [PRED-007] 增强量化因子与解释：`trend_strength/drawdown_60/atr_14/volume_stability_20` + `rationale`。
  - [FRONT-002] 首页与预测页展示置信度、数据新鲜度、数据模式与解释信息。
  - [REL-001] 公告源抓取失败统一兜底，避免网络异常导致测试链路中断。
- Evidence:
  - command: `.\.venv\Scripts\python -m pytest -q`
  - key output: `57 passed`
  - command: `cd frontend && npm run build`
  - key output: `build passed`
  - command: `cd frontend && npx tsc --noEmit`
  - key output: `typecheck passed`
  - command: `python - <<... s.query(...) ...>>`
  - key output: `包含 analysis_brief`
  - command: `python - <<... s.predict_run(...) ...>>`
  - key output: `history_data_mode=real_history`

### 2026-02-15 (Round-C)
- Completed:
  - [OPS-AGT-001] 新增多 Agent 分歧分析接口 `/v1/ops/agent/debate`，输出共识与分歧分数。
  - [OPS-RAG-001] 新增 RAG 质量接口 `/v1/ops/rag/quality`，输出 Recall/MRR/nDCG 及 case 明细。
  - [OPS-PRM-001] 新增 Prompt 版本对比回放接口 `/v1/ops/prompts/compare`，支持渲染与 diff。
  - [FRONT-OPS-001] `ops/evals` 页面接入 debate/rag/prompt compare 可视化。
  - [REL-DATA-001] 重建 `data/sources.py` 并加强外部源短超时+禁代理稳定性。
- Evidence:
  - command: `.\.venv\Scripts\python -m pytest -q tests/test_http_api.py`
  - key output: `12 passed`
  - command: `.\.venv\Scripts\python -m pytest -q`
  - key output: `58 passed`
  - command: `cd frontend && npm run build`
  - key output: `build passed`
  - command: `cd frontend && npx tsc --noEmit`
  - key output: `typecheck passed`

### 2026-02-15 (Round-D)
- Completed:
  - [PRM-REAL-001] Prompt 多版本链路落地：`fact_qa@1.0.0` 与 `fact_qa@1.1.0`，支持真实版本对比。
  - [AGT-REAL-001] 多 Agent 辩论支持 `llm_parallel`（启用外部模型时）与规则回退双模式。
  - [RAG-ONLINE-001] 增加线上 RAG 持续评测样本落库与 `offline+online` 聚合指标输出。
  - [OPS-FRONT-002] `ops/evals` 页面接入版本选择式 prompt compare。
- Evidence:
  - command: `.\.venv\Scripts\python -m pytest -q tests/test_http_api.py tests/test_prompt_engineering.py tests/test_service.py`
  - key output: `25 passed`
  - command: `.\.venv\Scripts\python -m pytest -q`
  - key output: `58 passed`
  - command: `cd frontend && npm run build`
  - key output: `build passed`
  - command: `cd frontend && npx tsc --noEmit`
  - key output: `typecheck passed`

### 2026-02-15 (Round-E)
- Completed:
  - [AGT-009] DeepThink 会话接口与轮次流式裁决 MVP：新增 `/v1/deep-think/sessions`、`/v1/deep-think/sessions/{session_id}/rounds`、`/v1/deep-think/sessions/{session_id}`、`/v1/deep-think/sessions/{session_id}/stream`。
  - [AGT-010] 内部 A2A 适配层：新增 `/v1/a2a/agent-cards`、`/v1/a2a/tasks`、`/v1/a2a/tasks/{task_id}`；内置 agent cards 与任务状态流转。
  - [GOV-004] 每轮文档化落地：新增 `docs/rounds/README.md` 与本轮日志 `docs/rounds/2026-02-15/round-E-deepthink-a2a-mvp.md`。
- Evidence:
  - command: `.\.venv\Scripts\python -m pytest -q tests/test_service.py tests/test_http_api.py`
  - key output: `25 passed`
  - command: `.\.venv\Scripts\python -m pytest -q`
  - key output: `61 passed`
  - command: `cd frontend && npm run build`
  - key output: `build passed`
  - command: `cd frontend && npx tsc --noEmit`
  - key output: `typecheck passed`

### 2026-02-15 (Round-F)
- Completed:
  - [AGT-011] DeepThink 任务规划与重规划触发：每轮新增 `task_graph`，分歧超阈值触发 `replan_triggered`。
  - [AGT-012] DeepThink 预算治理：每轮新增 `budget_usage`，超预算时输出 `stop_reason=DEEP_BUDGET_EXCEEDED`。
  - [GOV-004] 本轮交付记录：新增 `docs/rounds/2026-02-15/round-F-deepthink-planner-budget-replan.md` 与专栏记录 `docs/agent-column/10-Round-F-DeepThink-Planner-Budget-Replan实现记录.md`。
- Evidence:
  - command: `.\.venv\Scripts\python -m pytest -q tests/test_service.py tests/test_http_api.py`
  - key output: `27 passed`
  - command: `.\.venv\Scripts\python -m pytest -q`
  - key output: `63 passed`

  - command: `cd frontend && npm run build`
  - key output: `build passed`
  - command: `cd frontend && npx tsc --noEmit`
  - key output: `failed: tsconfig include '.next/types/**/*.ts' matched missing files (pre-existing config issue)`

### 2026-02-15 (Round-G)
- Completed:
  - [FRONT-003] 首页与分析页拆分：`/` 重构为导航首页，深度分析迁移至 `/deep-think`。
  - [FRONT-004] 前端稳定性：本轮改造后 `build + tsc` 均通过。
  - [GOV-004] 本轮交付文档化：新增 `docs/rounds/2026-02-15/round-G-homepage-deepthink-separation.md` 与专栏记录 `docs/agent-column/11-Round-G-首页导航化与DeepThink独立页面实现记录.md`。
- Evidence:
  - command: `cd frontend && npm run build`
  - key output: `build passed, route /deep-think generated`
  - command: `cd frontend && npx tsc --noEmit`
  - key output: `typecheck passed`
  - command: `.\.venv\Scripts\python -m pytest -q`
  - key output: `63 passed`
### 2026-02-15 (Round-H)
- Completed:
  - [FRONT-005] DeepThink 治理看板落地：`/deep-think` 新增 round timeline、conflict_sources 可视化、budget usage、replan/stop reason、SSE 回放。
  - [GOV-004] 本轮交付文档化：新增 `docs/rounds/2026-02-15/round-H-deepthink-round-visualization.md` 与专栏记录 `docs/agent-column/12-Round-H-DeepThink轮次可视化与治理看板实现记录.md`。
- Evidence:
  - command: `.\.venv\Scripts\python -m pytest -q`
  - key output: `63 passed in 28.77s`
  - command: `cd frontend && npm run build`
  - key output: `build passed (route /deep-think generated)`
  - command: `cd frontend && npx tsc --noEmit`
  - key output: `typecheck passed`

### 2026-02-15 (Round-I)
- Completed:
  - [AGT-013] DeepThink 轮次事件存档落地：新增 `deep_think_event` 持久化与 `/v1/deep-think/sessions/{session_id}/events` 查询接口。
  - [FRONT-006] `/deep-think` 新增跨轮次观点差分、冲突源下钻（evidence IDs）与会话存档加载能力。
  - [GOV-004] 本轮交付文档化：新增 `docs/rounds/2026-02-15/round-I-deepthink-diff-drilldown-archive.md` 与专栏记录 `docs/agent-column/13-Round-I-DeepThink跨轮差分与事件存档实现记录.md`。
- Evidence:
  - command: `.\.venv\Scripts\python -m pytest -q tests/test_service.py tests/test_http_api.py`
  - key output: `27 passed in 26.05s`
  - command: `.\.venv\Scripts\python -m pytest -q`
  - key output: `63 passed in 36.52s`
  - command: `cd frontend && npm run build`
  - key output: `build passed (/deep-think generated)`
  - command: `cd frontend && npx tsc --noEmit`
  - key output: `typecheck passed`

### 2026-02-15 (Round-J)
- Completed:
  - [AGT-014] DeepThink 事件归档治理增强：`/events` 支持 `event_name` 过滤，新增会话级事件保留裁剪（`archive_max_events`）。
  - [FRONT-007] `/deep-think` 新增 round/event/limit 筛选控制并支持过滤回放。
  - [GOV-004] 本轮交付文档化：新增 `docs/rounds/2026-02-15/round-J-deepthink-archive-filter-retention.md` 与专栏记录 `docs/agent-column/14-Round-J-DeepThink事件过滤与归档保留治理实现记录.md`。
- Evidence:
  - command: `.\.venv\Scripts\python -m pytest -q tests/test_service.py tests/test_http_api.py`
  - key output: `27 passed in 24.51s`
  - command: `.\.venv\Scripts\python -m pytest -q`
  - key output: `63 passed in 40.11s`
  - command: `cd frontend && npm run build`
  - key output: `build passed (/deep-think generated)`
  - command: `cd frontend && npx tsc --noEmit`
  - key output: `typecheck passed`

### 2026-02-15 (Round-K)
- Completed:
  - [AGT-015] DeepThink 存档契约增强：`/events` 支持 `cursor/created_from/created_to`，并新增 `/events/export`（`jsonl/csv`）。
  - [FRONT-008] `/deep-think` 增加时间过滤、游标翻页与导出按钮，支持分页回放与审计导出。
  - [GOV-004] 本轮交付文档化：新增 `docs/rounds/2026-02-15/round-K-deepthink-archive-pagination-export.md` 与专栏记录 `docs/agent-column/15-Round-K-DeepThink归档分页时间过滤与导出实现记录.md`。
- Evidence:
  - command: `.\.venv\Scripts\python -m pytest -q tests/test_service.py tests/test_http_api.py`
  - key output: `27 passed in 24.16s`
  - command: `.\.venv\Scripts\python -m pytest -q`
  - key output: `63 passed in 31.40s`
  - command: `cd frontend && npm run build`
  - key output: `build passed (/deep-think generated)`
  - command: `cd frontend && npx tsc --noEmit`
  - key output: `typecheck passed`

### 2026-02-15 (Round-L)
- Completed:
  - [AGT-016] DeepThink 归档新增异步导出任务链路，补齐审计日志与归档指标接口，并收紧时间过滤参数校验。
  - [FRONT-009] `/deep-think` 增加导出任务轮询下载流与存档回放导航（上一页/回到第一页/下一页）。
  - [GOV-004] 本轮交付文档化：新增 `docs/rounds/2026-02-15/round-L-deepthink-archive-async-export-audit.md` 与专栏记录 `docs/agent-column/16-Round-L-DeepThink归档异步导出任务与审计指标实现记录.md`。
- Evidence:
  - command: `.\.venv\Scripts\python -m pytest -q tests/test_service.py tests/test_http_api.py`
  - key output: `28 passed in 25.67s`
  - command: `.\.venv\Scripts\python -m pytest -q`
  - key output: `64 passed in 35.36s`
  - command: `cd frontend && npm run build`
  - key output: `build passed (all routes generated)`
  - command: `cd frontend && npx tsc --noEmit`
  - key output: `typecheck passed`

### 2026-02-15 (Round-M)
- Completed:
  - [AGT-017] DeepThink 导出任务链路增加 `attempt_count/max_attempts`，支持 claim+requeue 重试并扩展审计分位指标。
  - [FRONT-010] `/deep-think` 时间过滤改为结构化输入（datetime-local）并增加快速窗口设置，导出任务展示重试次数。
  - [GOV-004] 本轮交付文档化：新增 `docs/rounds/2026-02-15/round-M-deepthink-export-retry-metrics-ux.md` 与专栏记录 `docs/agent-column/17-Round-M-DeepThink导出重试与审计分位指标实现记录.md`。
- Evidence:
  - command: `.\.venv\Scripts\python -m pytest -q tests/test_service.py tests/test_http_api.py`
  - key output: `29 passed in 34.52s`
  - command: `.\.venv\Scripts\python -m pytest -q`
  - key output: `65 passed in 39.41s`
  - command: `cd frontend && npm run build`
  - key output: `build passed`
  - command: `cd frontend && npx tsc --noEmit`
  - key output: `typecheck passed`
