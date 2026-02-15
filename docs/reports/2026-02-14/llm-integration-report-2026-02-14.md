# LLM接入与自测报告（2026-02-14）

## 1. 本次接入目标
1. 接入真实外部大模型（Anthropic Messages 格式）。
2. 支持多模型/多供应商配置，避免单点故障导致服务宕机。
3. 失败时自动回退本地模型，保证接口可用。

## 2. 已完成实现
1. 新增多供应商LLM网关：
- `backend/app/llm/gateway.py`
- 支持 `anthropic_messages` 与 `openai_chat` 两种API风格。
- Anthropic模式强制加 `anthropic-version` 请求头。
- 支持 `stream` 字段与SSE聚合解析。

2. Agent工作流接入外部模型并保底回退：
- `backend/app/agents/workflow.py`
- 新增 `external_model_call` 注入位与 `_model_call_with_fallback`。

3. 服务层注入LLM网关：
- `backend/app/service.py`
- `AShareAgentService` 初始化时注入 `MultiProviderLLMGateway`。

4. 配置化与启动参数：
- `backend/app/config.py` 增加外部LLM配置项（开关/路径/超时/重试/回退）。
- `backend/config/llm_providers.example.json`（示例多供应商配置）。
- `backend/config/llm_providers.local.json`（本地自测配置）。
- `start-backend.bat` 默认注入LLM环境变量。

5. 安全治理：
- 新增 `.gitignore`，忽略 `backend/config/*.local.json` 与本地密钥文件。

## 3. 自测结果
1. 语法编译：
- `python -m py_compile` 通过。

2. 自动化测试：
- `pytest -q` 通过，结果：`47 passed`。

3. 外部LLM在线连通性（你的配置）：
- 启用 `LLM_EXTERNAL_ENABLED=true` + `LLM_CONFIG_PATH=backend/config/llm_providers.local.json`
- 查询链路出现事件：`llm_provider_success`
- 证明真实外部模型调用已生效。

4. 回退机制验证：
- 无可用provider时，返回 `external_model_failed` 风险标记并继续输出结果。
- 证明“外部模型故障不致服务不可用”。

## 4. 关键说明
1. 当前后端主接口仍是同步响应；虽然网关支持 `stream` 字段与SSE解析，但HTTP层尚未新增专用流式端点（例如 `/v1/query/stream`）。
2. 如需前端逐字流式展示，可下一步补充 FastAPI `StreamingResponse` 端点并联调前端EventSource/ReadableStream。

## 5. 流式接口落地（补充）
1. 已新增后端流式接口：
- `POST /v1/query/stream`（SSE）
- 事件序列：`start` -> `meta` -> `answer_delta` -> `citations` -> `done`

2. 已实现“真流式透传”路径：
- `backend/app/llm/gateway.py`：`stream_generate(...)` 直接消费上游SSE并输出增量
- `backend/app/agents/workflow.py`：`run_stream(...)` 优先走外部流式模型
- `backend/app/service.py`：`query_stream_events(...)` 接入流式工作流

3. 前端联动：
- `frontend/app/page.tsx` 改为读取 `/v1/query/stream` 的 `ReadableStream` 并实时渲染

4. 自测结果：
- 单元/接口测试通过（`tests/test_http_api.py` 含流式端点检查）
- 在上游网关偶发 `503` 时自动回退本地流式分片，接口可用性不受影响
