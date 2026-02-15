# 优化迭代记录：LangGraph 真接入（2026-02-15）

## 1. 迭代目标
- 将 LangGraph 从“文档声明”升级为“后端主链路真实运行时”。
- 保留 direct 回退，避免依赖异常导致服务不可用。
- 维持现有 API 与前端行为不破坏。

## 2. 实施内容
1. 新增运行时抽象与 LangGraph 执行器：
- `backend/app/agents/langgraph_runtime.py`
- 包含 `DirectWorkflowRuntime`、`LangGraphWorkflowRuntime`、`build_workflow_runtime()`

2. `AgentWorkflow` 拆出公开阶段方法：
- `prepare_prompt()`
- `apply_before_model()`
- `invoke_model()`
- `apply_after_model()`
- `finalize_with_output()`
- 文件：`backend/app/agents/workflow.py`

3. 服务主链路接入运行时选择：
- `AShareAgentService` 初始化注入 `workflow_runtime`
- `query()` 改为 `workflow_runtime.run(...)`
- trace 增加 `workflow_runtime` 事件
- 文件：`backend/app/service.py`

4. 配置扩展：
- 新增 `USE_LANGGRAPH_RUNTIME` 开关（默认 `true`）
- 文件：`backend/app/config.py`

5. 依赖升级：
- `langchain>=1.0.0,<2`
- `langgraph>=1.0.2,<1.1`
- `langsmith>=0.2.0,<1`
- 文件：`requirements.txt`

6. 测试新增：
- `tests/test_langgraph_runtime.py`
- 覆盖 direct 与 langgraph 运行时行为

7. 流式链路升级（本轮追加）：
- `WorkflowRuntime` 新增 `run_stream()` 抽象
- `LangGraphWorkflowRuntime` 流式路径采用“前置图节点 + 流式模型 + 后置图节点”
- `/v1/query/stream` 新增 `stream_runtime` 事件（标识 `langgraph/direct`）
- 前端首页新增 Engine 标签显示

8. Query 响应运行时显式化（本轮追加）：
- `QueryResponse` 新增 `workflow_runtime` 字段
- `/v1/query` 直接返回 `langgraph/direct`
- `tests/test_http_api.py` 增加断言

9. LangChain 工具绑定（本轮追加）：
- 新增 `LangChainToolRunner`（`StructuredTool`）并保持 ACL 前置鉴权
- `AgentWorkflow` 工具调用链改为 `tool_runner.call(...)`
- 无 LangChain 或绑定异常时自动回退 ACL 原始调用

10. 技术点核查能力接口（本轮追加）：
- 新增 `GET /v1/ops/capabilities` 返回 runtime/config/capabilities 快照
- 新增 `backend/app/capabilities.py` 统一维护技术点落地状态输出
- 前端 `ops/evals` 增加能力快照展示表

11. Prompt Runtime LangChain 模板渲染（本轮追加）：
- `PromptRuntime` 优先使用 `ChatPromptTemplate` 组装三层 Prompt
- 不可用时自动回退 `python format`，并在 `prompt_meta` 返回 `prompt_engine`

12. 细粒度工具 Schema 与 runtime 覆盖（本轮追加）：
- `LangChainToolRunner` 支持每个工具独立 `args_schema`（例如 quote/retrieve/graph）
- `query/query_stream` 支持请求级 `workflow_runtime=langgraph|direct|auto`
- 支持同一服务内按请求进行 runtime A/B 对比

## 3. 自测结果
1. 后端单元测试
- 命令：`\.venv\Scripts\python -m pytest -q`
- 结果：`57 passed`

2. 前端构建
- 命令：`cd frontend && npm run build`
- 结果：构建成功，`/ops/*`、`/reports`、`/docs-center` 等路由全部生成

3. 前端类型检查
- 命令：`cd frontend && npx tsc --noEmit`
- 结果：通过
- 说明：首次需先执行 build 生成 `.next/types` 再执行 tsc

4. 运行时烟测
- 命令：`python - <<... AShareAgentService().query(...) ...>>`
- 结果：trace 中存在 `workflow_runtime={'runtime':'langgraph'}` 事件

5. 流式运行时烟测
- 命令：`python - <<... for ev in s.query_stream_events(...) ...>>`
- 结果：事件包含 `stream_runtime={'runtime':'langgraph'}`，并正常输出 `stream_source/done`

6. Query 运行时字段烟测
- 命令：`python - <<... s.query(...) ...>>`
- 结果：`workflow_runtime=langgraph`，且含 citations

7. 技术点核查接口烟测
- 命令：`GET /v1/ops/capabilities`
- 结果：返回 `runtime/config/capabilities`，包含 `langgraph/langchain/prompt_engineering` 等关键项

8. 请求级 runtime 覆盖烟测
- 命令：`POST /v1/query` with `workflow_runtime=direct`
- 结果：响应 `workflow_runtime=direct`

## 4. 兼容性与回退
- 当 `langgraph` 不可用或显式关闭时，自动回退 direct 运行时。
- 回退不影响 `/v1/query`、`/v1/query/stream`、`/v1/report/*` 等现有 API。

## 5. 下一轮优化建议
1. 引入 LangChain tool 的细粒度参数 schema（分工具定义），替换当前统一 `payload` schema。
2. 将 `workflow_runtime` 作为可观测标签下钻到前端更多页面（如 reports/predict）。
3. 增加 runtime A/B 开关测试（`USE_LANGGRAPH_RUNTIME=true/false`）并纳入 CI。
